"""Token-level Q-function control variates for black-box DFL -- toy experiment.

Pipeline
--------
An autoregressive predictor emits T digit tokens y = (y_0 .. y_{T-1}).
An OPAQUE oracle maps y -> scalar decision loss f(y) in [0, 1]. Hidden inside
(the learner never sees this): only k of the T positions are read; they decode
to a parameter vector that enters a small optimization problem; f is the
normalized decision regret of the induced decision under the TRUE parameters.

Estimator under test ("qcv"): per-token Q control variate with exact
vocab-marginalized correction,

    g = sum_t [ (f - Qhat_t(y_{<t}, y_t)) * grad log p_t(y_t)
                + grad_theta sum_v p_theta(v | y_{<t}) Qhat_t(y_{<t}, v) ]

Unbiased for ANY Qhat (the correction term cancels the CV in expectation,
per position, conditionally on the prefix); Qhat quality only moves variance.
The ideal choice Qhat_t = E[f | y_{<t}, y_t = v] is the token-level Q under
the current policy.

The policy is an independent categorical per position (logits table T x V) so
that the exact objective, exact gradient, and exact Q are computable by
enumeration over the oracle's V^k relevant-token combos. That enumeration
lives in `Diagnostics`, which is a PRIVILEGED measurement harness: it opens
the oracle's private fields. Nothing the learner uses may touch it.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF


# ---------------------------------------------------------------------------
# Black-box oracle
# ---------------------------------------------------------------------------

class BlackBoxOracle:
    """The learner may only call `__call__` and read `.calls`.

    Everything underscored is private to the black box: which positions are
    read, how tokens decode to parameters, what the inner optimization problem
    is, and the true parameters. `Diagnostics` (measurement only) is the sole
    consumer of those fields.
    """

    def __init__(self, T, V, k, support="early", m=None, kind="topm",
                 noise=0.0, seed=0):
        self.T, self.V = T, V
        rng = np.random.default_rng(seed)
        if support == "early":
            S = np.arange(k)
        elif support == "late":
            S = np.arange(T - k, T)
        elif support == "spread":
            S = np.unique(np.linspace(0, T - 1, k).round().astype(int))
            assert len(S) == k, "spread support collided; use larger T or fewer k"
        else:
            raise ValueError(support)
        self._S = S
        self._m = m if m is not None else max(1, k // 2)
        self._c = rng.uniform(0.0, 1.0, size=k)          # hidden true costs
        self._kind = kind
        self._noise = noise
        self._rng = rng
        self.calls = 0
        order = np.argsort(self._c)
        self._opt = self._c[order[: self._m]].sum()
        self._worst = self._c[order[-self._m:]].sum()

    def _f_raw(self, sub):
        """Deterministic loss for the k relevant tokens (no noise)."""
        chat = np.asarray(sub, dtype=float) / (self.V - 1)   # decode digits
        if self._kind == "topm":
            pick = np.argsort(chat, kind="stable")[: self._m]
            val = self._c[pick].sum()
            return float((val - self._opt) / (self._worst - self._opt + 1e-12))
        if self._kind == "quadratic":
            scale = np.mean(np.maximum(self._c, 1.0 - self._c) ** 2)
            return float(np.mean((chat - self._c) ** 2) / scale)
        raise ValueError(self._kind)

    def __call__(self, y):
        self.calls += 1
        f = self._f_raw(np.asarray(y)[self._S])
        if self._noise > 0:
            f += self._noise * self._rng.standard_normal()
        return float(f)


# ---------------------------------------------------------------------------
# Privileged diagnostics (measurement only -- never available to the learner)
# ---------------------------------------------------------------------------

class Diagnostics:
    """Exact objective / gradient / Q by enumeration over the V^k relevant
    combos. Uses the oracle's private fields; its evaluations never touch the
    budgeted call counter."""

    def __init__(self, oracle):
        self.o = oracle
        self.S = np.asarray(oracle._S)
        k, V = len(self.S), oracle.V
        grid = np.stack(
            np.meshgrid(*[np.arange(V)] * k, indexing="ij"), -1
        ).reshape(-1, k)
        self.f_table = np.array([oracle._f_raw(g) for g in grid]).reshape((V,) * k)

    # -- objective ---------------------------------------------------------

    def f_of(self, Y, with_noise=False, rng=None):
        """Deterministic loss for a batch Y: (N, T) -> (N,)."""
        F = self.f_table[tuple(Y[:, self.S].T)]
        if with_noise and self.o._noise > 0:
            rng = rng or np.random.default_rng(0)
            F = F + self.o._noise * rng.standard_normal(F.shape)
        return F

    def exact_objective_torch(self, logits):
        probs = torch.softmax(logits, dim=-1)
        table = torch.as_tensor(self.f_table, dtype=probs.dtype)
        joint = probs[self.S[0]]
        for i in range(1, len(self.S)):
            joint = joint.unsqueeze(-1) * probs[self.S[i]]
        return (joint * table).sum()

    def exact_grad(self, logits_np):
        lg = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        Ef = self.exact_objective_torch(lg)
        (g,) = torch.autograd.grad(Ef, lg)
        return Ef.item(), g.numpy()

    # -- exact token-level Q ----------------------------------------------

    def contracted_tables(self, probs_np):
        """Cs[j] = f_table with S-positions j..k-1 marginalized under the
        policy; keeps j leading axes for tokens at S[0..j-1]. Cs[0] = E[f]."""
        Cs = [self.f_table]
        for j in range(len(self.S) - 1, -1, -1):
            Cs.append(np.tensordot(Cs[-1], probs_np[self.S[j]], axes=([-1], [0])))
        Cs.reverse()
        return Cs

    def exact_Q(self, Y, probs_np):
        """Q[b, t, v] = E[f | y_{<t} = Y[b, :t], y_t = v] under the policy
        with per-position marginals probs_np. Y: (B, T) -> (B, T, V)."""
        B, T = Y.shape
        V = self.o.V
        S = self.S
        Sset = set(int(s) for s in S)
        Cs = self.contracted_tables(probs_np)
        Q = np.empty((B, T, V))
        for t in range(T):
            j = int(np.searchsorted(S, t))            # = #{s in S : s < t}
            pref = tuple(Y[:, S[:j]].T)               # j index arrays, len B
            if t in Sset:
                Q[:, t, :] = Cs[j + 1][pref]          # (B, V) or (V,) bcast
            else:
                val = Cs[j][pref] if j > 0 else np.full(B, float(Cs[0]))
                Q[:, t, :] = np.asarray(val).reshape(-1, 1)
        return Q


# ---------------------------------------------------------------------------
# Policy helpers (independent categorical over positions)
# ---------------------------------------------------------------------------

def softmax_np(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sample_tokens(rng, probs, N):
    """probs: (T, V) -> Y: (N, T) int64."""
    cum = probs.cumsum(-1)
    r = rng.random((N, probs.shape[0]))
    return (r[..., None] > cum[None]).sum(-1).astype(np.int64)


# ---------------------------------------------------------------------------
# Learned Q network (this is all the learner gets: sequences and scalars)
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    """GRU credit head: out[b, t, v] = Qhat_t(y_{<t}, v). The input at step t
    is the embedding of y_{t-1} (BOS at t=0) plus a position embedding, so the
    value for position t depends only on the prefix -- the measurability
    caveat (no peeking at realized future tokens) is enforced by wiring."""

    def __init__(self, T, V, d=48):
        super().__init__()
        self.V = V
        self.tok = nn.Embedding(V + 1, d)      # index V = BOS
        self.pos = nn.Embedding(T, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, y):
        B, T = y.shape
        bos = torch.full((B, 1), self.V, dtype=torch.long, device=y.device)
        inp = torch.cat([bos, y[:, :-1]], dim=1)
        x = self.tok(inp) + self.pos.weight[None, :T]
        h, _ = self.gru(x)
        return self.head(h)


def qnet_values(qnet, Y, chunk=4096):
    """Qhat for a large numpy batch, chunked, no grad. (N,T) -> (N,T,V)."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(Y), chunk):
            yb = torch.as_tensor(Y[i:i + chunk])
            outs.append(qnet(yb).double().numpy())
    return np.concatenate(outs, 0)


def fit_qnet(qnet, Y, F, steps=1500, bs=64, lr=2e-3, seed=0):
    """Return-regression on replay: min sum_t (Qhat_t(y_{<t}, y_t) - f)^2."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    Yt = torch.as_tensor(Y)
    Ft = torch.as_tensor(F, dtype=torch.float32)
    T = Y.shape[1]
    ar = torch.arange(T)
    for _ in range(steps):
        idx = rng.integers(0, len(Y), size=bs)
        yb, fb = Yt[idx], Ft[idx]
        q = qnet(yb)
        q_taken = q[torch.arange(len(idx))[:, None], ar[None, :], yb]
        loss = ((q_taken - fb[:, None]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return qnet


# ---------------------------------------------------------------------------
# Analytic per-sample gradients (independent-categorical policy)
#
# score part, position t:      coeff_t * (onehot(y_t) - p_t)
# correction part, position t: p_t * (Q_t - p_t . Q_t)
# ---------------------------------------------------------------------------

class EstStats:
    """Streaming mean / per-position second moment of gradient estimates."""

    def __init__(self, T, V):
        self.sum_g = np.zeros((T, V))
        self.sum_sq = np.zeros(T)     # sum over estimates of ||g_t||^2
        self.n = 0

    def add(self, g):                 # g: (M, T, V)
        self.sum_g += g.sum(0)
        self.sum_sq += (g ** 2).sum(-1).sum(0)
        self.n += len(g)

    def mean(self):
        return self.sum_g / self.n

    def var_per_pos(self):
        m = self.mean()
        return self.sum_sq / self.n - (m ** 2).sum(-1)

    def var_total(self):
        return float(self.var_per_pos().sum())


def _grad_batch(Y, coeff, probs, Q=None):
    """Per-sample gradients. Y: (M,T); coeff: (M,T); probs: (T,V);
    Q: (M,T,V) or None. Returns (M,T,V)."""
    M, T = Y.shape
    V = probs.shape[1]
    onehot = np.zeros((M, T, V))
    onehot[np.arange(M)[:, None], np.arange(T)[None, :], Y] = 1.0
    g = coeff[:, :, None] * (onehot - probs[None])
    if Q is not None:
        pQ = (probs[None] * Q).sum(-1, keepdims=True)
        g = g + probs[None] * (Q - pQ)
    return g


def estimator_stats(name, Y, F, probs, diag=None, qnet=None, K=2,
                    corrupt=None, chunk=8192):
    """Streaming stats for one estimator over pre-drawn samples Y with losses
    F. `name`:
      reinforce      score with coeff f (no baseline)
      reinforce_b    score with best constant baseline mean(F)
      rloo           groups of K, leave-one-out baseline (uses K calls/est.)
      qcv            learned-Q CV + exact vocab-marginalized correction
      qcv_exact      privileged exact Q (theory floor)
      pathwise       correction term only, learned Q  (BIASED foil)
      pathwise_exact correction term only, exact Q    (all-actions PG)
    `corrupt`: optional (T,V) offsets added to Qhat -- a deliberately wrong CV.
    Returns EstStats where n = number of gradient ESTIMATES (N or N//K)."""
    T, V = probs.shape
    st = EstStats(T, V)
    ar = np.arange(T)
    b = F.mean() if name == "reinforce_b" else 0.0
    if name == "rloo":
        M = (len(Y) // K) * K
        Y, F = Y[:M], F[:M]
    for i in range(0, len(Y), chunk):
        Yb, Fb = Y[i:i + chunk], F[i:i + chunk]
        if name in ("reinforce", "reinforce_b"):
            g = _grad_batch(Yb, np.broadcast_to((Fb - b)[:, None], Yb.shape), probs)
        elif name == "rloo":
            m = (len(Yb) // K) * K
            Yb, Fb = Yb[:m], Fb[:m]
            Fg = Fb.reshape(-1, K)
            coeff = Fg - (Fg.sum(1, keepdims=True) - Fg) / (K - 1)
            gm = _grad_batch(Yb, np.repeat(coeff.reshape(-1), T).reshape(-1, T), probs)
            g = gm.reshape(-1, K, T, V).mean(1)
        else:
            if name in ("qcv_exact", "pathwise_exact"):
                Q = diag.exact_Q(Yb, probs)
            else:
                Q = qnet_values(qnet, Yb)
            if corrupt is not None:
                Q = Q + corrupt[None]
            q_taken = Q[np.arange(len(Yb))[:, None], ar[None, :], Yb]
            if name.startswith("pathwise"):
                coeff = np.zeros_like(q_taken)
            else:
                coeff = Fb[:, None] - q_taken
            g = _grad_batch(Yb, coeff, probs, Q=Q)
        st.add(g)
    return st


# ---------------------------------------------------------------------------
# End-to-end training at a metered oracle budget
# ---------------------------------------------------------------------------

def train_policy(method, T=32, V=10, k=4, m=2, support="early", kind="topm",
                 noise=0.0, budget=8000, K=2, lr=0.05, q_lr=2e-3, q_bs=64,
                 seed=0, oracle_seed=0, record_every=100, replay_cap=4096,
                 q_warmup=0, q_warmup_steps=1500, q_steps=1):
    """Train the T x V policy logits with one estimator, spending real oracle
    calls. Returns dict with the exact-regret curve vs cumulative calls.

    q_warmup: oracle calls spent (and charged to the budget) under the initial
    policy purely to pretrain Qhat before any policy update -- the cheap
    off-policy replay phase."""
    oracle = BlackBoxOracle(T, V, k, support=support, m=m, kind=kind,
                            noise=noise, seed=oracle_seed)
    diag = Diagnostics(oracle)                      # measurement only
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    logits = torch.zeros(T, V, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    ar = torch.arange(T)

    needs_qnet = method in ("qcv", "pathwise")
    if needs_qnet:
        qnet = QNet(T, V)
        qopt = torch.optim.Adam(qnet.parameters(), lr=q_lr)
        replay = deque(maxlen=replay_cap)

    def q_step():
        idx = rng.integers(0, len(replay), size=q_bs)
        Yb = torch.as_tensor(np.stack([replay[i][0] for i in idx]))
        Fb = torch.as_tensor(np.array([replay[i][1] for i in idx]),
                             dtype=torch.float32)
        q = qnet(Yb)
        q_taken = q[torch.arange(q_bs)[:, None], ar[None, :], Yb]
        qloss = ((q_taken - Fb[:, None]) ** 2).mean()
        qopt.zero_grad()
        qloss.backward()
        qopt.step()

    baseline = None
    curve_calls, curve_ef = [], []
    next_rec = 0

    if needs_qnet and q_warmup > 0:
        ef0, _ = diag.exact_grad(logits.detach().numpy().astype(np.float64))
        curve_calls.append(0)
        curve_ef.append(ef0)
        next_rec = record_every
        Yw = sample_tokens(rng, softmax_np(logits.detach().numpy()), q_warmup)
        for yw in Yw:
            replay.append((yw, oracle(yw)))
        for _ in range(q_warmup_steps):
            q_step()

    while oracle.calls < budget:
        if oracle.calls >= next_rec:
            ef, _ = diag.exact_grad(logits.detach().numpy().astype(np.float64))
            curve_calls.append(oracle.calls)
            curve_ef.append(ef)
            next_rec += record_every

        probs_np = softmax_np(logits.detach().numpy())
        logp = tF.log_softmax(logits, dim=-1)

        if method == "rloo":
            Y = sample_tokens(rng, probs_np, K)
            Fv = np.array([oracle(y) for y in Y])
            coeff = Fv - (Fv.sum() - Fv) / (K - 1)
            Yt = torch.as_tensor(Y)
            lp = logp[ar[None, :], Yt]                       # (K, T)
            surr = (torch.as_tensor(coeff, dtype=torch.float32)[:, None]
                    * lp).sum() / K
        else:
            y = sample_tokens(rng, probs_np, 1)
            f = oracle(y[0])
            yt = torch.as_tensor(y[0])
            lp = logp[ar, yt]                                # (T,)
            if method == "reinforce":
                surr = (f * lp).sum()
            elif method == "reinforce_b":
                bl = f if baseline is None else baseline
                surr = ((f - bl) * lp).sum()
                baseline = f if baseline is None else 0.99 * baseline + 0.01 * f
            elif method == "qcv":
                with torch.no_grad():
                    qall = qnet(torch.as_tensor(y))[0]       # (T, V)
                coeff = f - qall[ar, yt]
                surr = ((coeff * lp).sum()
                        + (torch.softmax(logits, -1) * qall).sum())
                replay.append((y[0], f))
            elif method == "pathwise":
                with torch.no_grad():
                    qall = qnet(torch.as_tensor(y))[0]
                surr = (torch.softmax(logits, -1) * qall).sum()
                replay.append((y[0], f))
            elif method == "qcv_exact":
                Qex = diag.exact_Q(y, probs_np)[0]           # (T, V)
                Qt = torch.as_tensor(Qex, dtype=torch.float32)
                coeff = torch.as_tensor(f - Qex[np.arange(T), y[0]],
                                        dtype=torch.float32)
                surr = ((coeff * lp).sum()
                        + (torch.softmax(logits, -1) * Qt).sum())
            else:
                raise ValueError(method)

        opt.zero_grad()
        surr.backward()
        opt.step()

        if needs_qnet and len(replay) >= q_bs:
            for _ in range(q_steps):
                q_step()

    ef, _ = diag.exact_grad(logits.detach().numpy().astype(np.float64))
    curve_calls.append(oracle.calls)
    curve_ef.append(ef)
    out = dict(calls=np.array(curve_calls), ef=np.array(curve_ef),
               logits=logits.detach().numpy())
    if needs_qnet:
        out["qnet"] = qnet
    return out
