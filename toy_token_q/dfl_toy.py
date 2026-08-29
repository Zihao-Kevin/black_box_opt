"""Predict-then-optimize (DFL) motivating toy for coordinate-level Q control
variates -- the non-autoregressive companion to token_q_toy.py.

Pipeline
--------
Instance features x  ->  stochastic predictor psi (factorized categorical over
a V-point value grid; ALL d coordinates emitted simultaneously -- no
autoregression)  ->  predicted parameter vector chat in [0,1]^d  ->  OPAQUE
decision oracle: hidden inside, chat enters a top-m selection problem and the
induced decision is charged its regret under the hidden true costs
c*(x) = sigmoid(Hx). The learner sees (x, chat, regret) and a call meter only.

Estimator: identical to the token toy with positions = coordinates of chat in
an arbitrary fixed order, so Q_j(chat_{<j}, v) = E[f | first j coords fixed,
coord j = v] is PER-PARAMETER credit: how much the decision hinges on the j-th
predicted price. Stochastic prediction is the honest gradient channel here --
regret is piecewise-constant in chat, so d f / d chat = 0 a.e.

Reuses token_q_toy: _grad_batch, EstStats, sample_tokens, softmax_np.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

from token_q_toy import sample_tokens, softmax_np


# ---------------------------------------------------------------------------
# Black-box decision oracle (contextual)
# ---------------------------------------------------------------------------

class DFLOracle:
    """The learner may only read `X` (public features), `d`, `n_train`,
    `n_test`, `.calls`, and call `__call__(i, chat)` on TRAINING instances.
    Hidden: the feature->cost map, the true costs, and the decision problem."""

    def __init__(self, d=8, m=3, p=6, n_train=6, n_test=4, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.p = d, p
        self.n_train, self.n_test = n_train, n_test
        self.X = rng.standard_normal((n_train + n_test, p))
        self._H = rng.standard_normal((d, p)) / np.sqrt(p)
        self._C = 1.0 / (1.0 + np.exp(-2.0 * (self.X @ self._H.T)))
        self._m = m
        self._noise = noise
        self._rng = rng
        srt = np.sort(self._C, axis=1)
        self._opt = srt[:, :m].sum(1)
        self._worst = srt[:, -m:].sum(1)
        self.calls = 0

    def _regret(self, i, chat):
        pick = np.argsort(np.asarray(chat, dtype=float), kind="stable")[: self._m]
        val = self._C[i, pick].sum()
        return float((val - self._opt[i]) /
                     (self._worst[i] - self._opt[i] + 1e-12))

    def __call__(self, i, chat):
        assert i < self.n_train, "oracle is only callable on training instances"
        self.calls += 1
        f = self._regret(i, chat)
        if self._noise > 0:
            f += self._noise * self._rng.standard_normal()
        return f


class QPOracle:
    """Same public interface as DFLOracle (X, d, p, n_train, n_test, calls,
    __call__(i, chat)), but the hidden decision problem is a genuine
    quadratic program instead of top-m selection:

        z*(c) = argmax_z  c^T z - (lam/2) z^T Sigma z   s.t.  1^T z = 1

    chat plays the role of a predicted return vector, Sigma a fixed,
    shared risk/covariance structure (a low-rank factor model, PSD) --
    the portfolio motivating example from the paper. Sigma is what makes
    the true objective NON-additively-separable in c (Lemma cross(i)):
    z*_k depends on every c_j through Sigma^{-1}, not just c_k. The
    predictor architecture (tied vs separate heads, see make_predictor)
    independently controls Lemma cross(ii); both are needed for
    Delta* != 0 (Theorem optimal(b)).

    The equality-constrained QP has a closed-form, AFFINE solution
    z*(c) = M @ c + b0 (from the KKT system), precomputed once so both
    oracle calls and QPDiagnostics's enumeration are cheap matrix products,
    no per-call solver.
    """

    def __init__(self, d=8, p=6, n_train=6, n_test=4, lam=1.0,
                 k_factors=None, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.p = d, p
        self.n_train, self.n_test = n_train, n_test
        self.X = rng.standard_normal((n_train + n_test, p))
        self._H = rng.standard_normal((d, p)) / np.sqrt(p)
        self._C = 1.0 / (1.0 + np.exp(-2.0 * (self.X @ self._H.T)))

        k = k_factors if k_factors is not None else max(2, d // 2)
        Fm = rng.standard_normal((d, k)) / np.sqrt(k)
        self._Sigma = Fm @ Fm.T + 0.25 * np.eye(d)          # shared PSD risk model
        Sinv = np.linalg.inv(self._Sigma)
        Sinv1 = Sinv @ np.ones(d)
        A = float(np.ones(d) @ Sinv1)
        self._M = (Sinv - np.outer(Sinv1, Sinv1) / A) / lam  # z*(c) = c @ M.T + b0
        self._b0 = Sinv1 / A
        self._lam = lam
        self._noise = noise
        self._rng = rng
        self.calls = 0

        Zopt = self._solve(self._C)                          # z*(c_true), per instance
        Zworst = self._solve(-self._C)                        # induced worst decision
        self._opt = self._obj_batch(Zopt)
        self._worst = self._obj_batch(Zworst)

    def _solve(self, C):
        """z*(c) = M c + b0, batched over the leading axis: (..., d) -> (..., d)."""
        return C @ self._M.T + self._b0

    def _obj_batch(self, Z):
        """True objective c_true^T z - (lam/2) z^T Sigma z, one (c_i, z_i)
        pair per row: Z[i] evaluated under self._C[i]."""
        lin = np.einsum("id,id->i", self._C, Z)
        quad = np.einsum("id,de,ie->i", Z, self._Sigma, Z)
        return lin - 0.5 * self._lam * quad

    def _regret(self, i, chat):
        z = self._solve(np.asarray(chat, dtype=float))
        c = self._C[i]
        val = float(c @ z - 0.5 * self._lam * (z @ self._Sigma @ z))
        return float((self._opt[i] - val) /
                     (self._opt[i] - self._worst[i] + 1e-12))

    def __call__(self, i, chat):
        assert i < self.n_train, "oracle is only callable on training instances"
        self.calls += 1
        f = self._regret(i, chat)
        if self._noise > 0:
            f += self._noise * self._rng.standard_normal()
        return f


# ---------------------------------------------------------------------------
# Privileged diagnostics (measurement only)
# ---------------------------------------------------------------------------

class _DiagnosticsBase:
    """Exact expectations/gradients/credit by enumeration over f_tables --
    shared by DFLDiagnostics and QPDiagnostics, independent of how f_tables
    was built (top-m selection or a QP solve)."""

    def f_of(self, i, Y):
        """Deterministic regret for grid combos Y: (N, d) -> (N,)."""
        return self.f_tables[i][tuple(Y.T)]

    def cascade(self, i, probs):
        """Cs[j] = f_table[i] with coordinates j..d-1 marginalized under the
        per-coordinate marginals probs (d, V). Cs[0] = E[f]."""
        Cs = [self.f_tables[i]]
        for j in range(self.o.d - 1, -1, -1):
            Cs.append(np.tensordot(Cs[-1], probs[j], axes=([-1], [0])))
        Cs.reverse()
        return Cs

    def exact_ef(self, i, probs):
        return float(self.cascade(i, probs)[0])

    def exact_grad(self, i, logits_np):
        lg = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        probs = torch.softmax(lg, -1)
        joint = probs[0]
        for j in range(1, self.o.d):
            joint = joint.unsqueeze(-1) * probs[j]
        table = torch.as_tensor(self.f_tables[i], dtype=probs.dtype)
        Ef = (joint * table).sum()
        (g,) = torch.autograd.grad(Ef, lg)
        return Ef.item(), g.numpy()

    def exact_Q(self, i, Y, probs):
        """Q[b, j, v] = E[f | chat_{<j} = Y[b, :j], chat_j = v] under the
        factorized policy with marginals probs. Y: (B, d) -> (B, d, V)."""
        B, d = Y.shape
        Cs = self.cascade(i, probs)
        Q = np.empty((B, d, self.V))
        for j in range(d):
            Q[:, j, :] = Cs[j + 1][tuple(Y[:, :j].T)]
        return Q


class DFLDiagnostics(_DiagnosticsBase):
    """Exact per-instance regret tables over the learner's V-grid, by
    enumeration of all V^d parameter vectors. The selection depends on chat
    only, so the (combo -> picked set) map is computed once and shared."""

    def __init__(self, oracle, V):
        self.o, self.V = oracle, V
        d = oracle.d
        combos = np.stack(
            np.meshgrid(*[np.arange(V)] * d, indexing="ij"), -1
        ).reshape(-1, d)
        picks = np.argsort(combos, axis=1, kind="stable")[:, : oracle._m]
        n = oracle.n_train + oracle.n_test
        self.f_tables = np.empty((n,) + (V,) * d)
        for i in range(n):
            vals = oracle._C[i][picks].sum(1)
            self.f_tables[i] = ((vals - oracle._opt[i]) /
                                (oracle._worst[i] - oracle._opt[i] + 1e-12)
                                ).reshape((V,) * d)


class QPDiagnostics(_DiagnosticsBase):
    """Exact per-instance regret tables for QPOracle, by enumeration of all
    V^d grid combos. Grid indices are rescaled to chat in [0,1]^d (unlike
    DFLDiagnostics's top-m, which only depends on rank order and so is
    scale-invariant, the QP's affine solve genuinely depends on chat's
    values). z*(chat) for every combo is one batched matrix product."""

    def __init__(self, oracle, V):
        self.o, self.V = oracle, V
        d = oracle.d
        combos = np.stack(
            np.meshgrid(*[np.arange(V)] * d, indexing="ij"), -1
        ).reshape(-1, d).astype(np.float64)
        chat = combos / max(V - 1, 1)
        Z = oracle._solve(chat)                               # (V^d, d)
        quad = np.einsum("gd,de,ge->g", Z, oracle._Sigma, Z)   # shared across i
        n = oracle.n_train + oracle.n_test
        self.f_tables = np.empty((n,) + (V,) * d)
        for i in range(n):
            val = Z @ oracle._C[i] - 0.5 * oracle._lam * quad
            reg = ((oracle._opt[i] - val) /
                  (oracle._opt[i] - oracle._worst[i] + 1e-12))
            self.f_tables[i] = reg.reshape((V,) * d)


# ---------------------------------------------------------------------------
# Contextual predictor and credit head
# ---------------------------------------------------------------------------

class TiedPredictor(nn.Module):
    """Every coordinate reads the SAME head off the trunk, distinguished only
    by a per-coordinate embedding added before the head -- the way an LLM
    reuses one output head at every position (exact_accounting's tied=True).
    Maximal parameter sharing: unlike separate per-coordinate head rows, ALL
    of the head's weights are used by every coordinate's logits."""

    def __init__(self, p, d, V, h):
        super().__init__()
        self.d, self.V = d, V
        self.trunk = nn.Linear(p, h)
        self.pos = nn.Embedding(d, h)
        self.head = nn.Linear(h, V)

    def forward(self, x):
        batched = x.dim() > 1
        t = self.trunk(x)
        if not batched:
            hid = torch.tanh(t[None, :] + self.pos.weight)         # (d,h)
            return self.head(hid).reshape(-1)                      # (d*V,)
        hid = torch.tanh(t[:, None, :] + self.pos.weight[None])    # (B,d,h)
        return self.head(hid).reshape(x.shape[0], -1)               # (B,d*V)


def make_predictor(p, d, V, h=32, tied=False):
    """x -> shared trunk -> per-coordinate logit heads, zero-initialized head
    (uniform starting policy).

    A single nn.Linear(p, d*V) (the original version) reads DISJOINT rows of
    W per coordinate -- J_j and J_k then have non-overlapping support in
    theta, so J_k J_j^T = 0 for all j != k identically (verified numerically).
    By Lemma cross(ii) that forces B = 0 and Delta* = 0 by construction, no
    matter how coupled f is: the share=0 corner of exact_accounting.py, not
    the share=1 regime the theorem's residual is about. Routing every head
    through a shared trunk gives real overlap in theta-space -- but measured
    overlap (see scratch/measure_overlap.py) shows a trained separate-head
    trunk only reaches ~0.16 (normalized ||J_jJ_k^T|| ratio) vs 0.68 for
    exact_accounting's share=1.0, i.e. far short of "one network predicts
    everything." tied=True forces maximal sharing directly.
    """
    if tied:
        net = TiedPredictor(p, d, V, h)
        nn.init.zeros_(net.head.weight)
        nn.init.zeros_(net.head.bias)
        return net
    net = nn.Sequential(nn.Linear(p, h), nn.Tanh(), nn.Linear(h, d * V))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ContextQNet(nn.Module):
    """GRU credit head conditioned on instance features: out[b, j, v] =
    Qhat_j(x, chat_{<j}, v). Step input = embedding of the previous
    coordinate's grid index (BOS at j=0) + a coordinate embedding; the
    instance enters through the initial hidden state, so credit is
    per-instance by construction."""

    def __init__(self, d, V, p, h=48):
        super().__init__()
        self.V = V
        self.tok = nn.Embedding(V + 1, h)
        self.pos = nn.Embedding(d, h)
        self.ctx = nn.Linear(p, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h, V)

    def forward(self, x, y):
        B, d = y.shape
        bos = torch.full((B, 1), self.V, dtype=torch.long, device=y.device)
        inp = torch.cat([bos, y[:, :-1]], dim=1)
        z = self.tok(inp) + self.pos.weight[None, :d]
        h0 = torch.tanh(self.ctx(x))[None]
        hseq, _ = self.gru(z, h0)
        return self.head(hseq)


def ctx_qnet_values(qnet, x, Y, chunk=4096):
    """Qhat for one instance's feature row x over a batch Y: (N, d, V)."""
    outs = []
    xt = torch.as_tensor(x, dtype=torch.float32)[None]
    with torch.no_grad():
        for s in range(0, len(Y), chunk):
            yb = torch.as_tensor(Y[s:s + chunk])
            outs.append(qnet(xt.expand(len(yb), -1), yb).double().numpy())
    return np.concatenate(outs, 0)


def fit_ctx_qnet(qnet, X, I, Y, F, steps=2000, bs=64, lr=2e-3, seed=0):
    """Return regression on replay (instance index, combo, regret)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    It = torch.as_tensor(I)
    Yt = torch.as_tensor(Y)
    Ft = torch.as_tensor(F, dtype=torch.float32)
    d = Y.shape[1]
    ar = torch.arange(d)
    for _ in range(steps):
        idx = rng.integers(0, len(Y), size=bs)
        yb = Yt[idx]
        q = qnet(Xt[It[idx]], yb)
        q_taken = q[torch.arange(len(idx))[:, None], ar[None, :], yb]
        loss = ((q_taken - Ft[idx][:, None]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return qnet


# ---------------------------------------------------------------------------
# End-to-end training at a metered oracle budget
# ---------------------------------------------------------------------------

def train_dfl(method, V=5, budget=6000, K=2, lr=0.05, q_lr=2e-3, q_bs=64,
              q_steps=1, q_warmup=200, q_warmup_steps=1200, seed=0,
              oracle_seed=0, record_every=100, replay_cap=4096, noise=0.0,
              delta_lr=2e-3, m=None, h=32, tied=False, kind="topm", lam=1.0):
    """Train the shared linear predictor with one estimator, spending real
    oracle calls round-robin over training instances. Records the exact mean
    regret on train AND held-out instances (privileged eval, uncharged).

    method="qcv_delta" additionally fits a residual head Delta on top of the
    regression-fit Qhat, c = Qhat + Delta, with Delta trained purely by
    variance descent (Section "Interpretation, and the training split"):
    minimizing E||ghat||^2 w.r.t. Delta's parameters is exactly minimizing
    Var(ghat), since E[ghat] does not depend on Delta (Lemma 1). No extra
    oracle calls: the same single sample used for the theta step is reused,
    via double backprop (torch.autograd.grad(..., create_graph=True)) --
    no reparameterization, unlike RELAX/REBAR.

    kind="qp" swaps the top-m selection oracle for QPOracle/QPDiagnostics:
    predicted c feeds a mean-variance-style QP whose shared covariance
    Sigma makes the true objective non-separable across coordinates (Lemma
    cross(i)) -- the paper's portfolio motivating example. Combined with
    tied=True/False (Lemma cross(ii), see make_predictor) this lets Delta*
    be genuinely nonzero (tied) or exactly zero (separate) by construction.
    """
    if kind == "qp":
        oracle = QPOracle(noise=noise, seed=oracle_seed, lam=lam)
        diag = QPDiagnostics(oracle, V)
    else:
        oracle = DFLOracle(noise=noise, seed=oracle_seed,
                           **({"m": m} if m is not None else {}))
        diag = DFLDiagnostics(oracle, V)
    d, p, n = oracle.d, oracle.p, oracle.n_train
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Xt = torch.as_tensor(oracle.X, dtype=torch.float32)
    lin = make_predictor(p, d, V, h=h, tied=tied)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    ar = torch.arange(d)

    def logits_of(i):
        return lin(Xt[i]).view(d, V)

    def eval_regret(idx):
        with torch.no_grad():
            return float(np.mean([
                diag.exact_ef(i, softmax_np(logits_of(i).numpy())) for i in idx]))

    train_idx = list(range(n))
    test_idx = list(range(n, n + oracle.n_test))

    if method == "qcv_delta":
        deltanet = ContextQNet(d, V, p)
        nn.init.zeros_(deltanet.head.weight)      # Delta starts at exactly 0
        nn.init.zeros_(deltanet.head.bias)
        dopt = torch.optim.Adam(deltanet.parameters(), lr=delta_lr)

    needs_q = method in ("qcv", "pathwise", "qcv_delta")
    if needs_q:
        qnet = ContextQNet(d, V, p)
        qopt = torch.optim.Adam(qnet.parameters(), lr=q_lr)
        replay = deque(maxlen=replay_cap)

        def q_step():
            sel = rng.integers(0, len(replay), size=q_bs)
            ib = torch.as_tensor([replay[s][0] for s in sel])
            yb = torch.as_tensor(np.stack([replay[s][1] for s in sel]))
            fb = torch.as_tensor([replay[s][2] for s in sel],
                                 dtype=torch.float32)
            q = qnet(Xt[ib], yb)
            q_taken = q[torch.arange(len(sel))[:, None], ar[None, :], yb]
            qloss = ((q_taken - fb[:, None]) ** 2).mean()
            qopt.zero_grad()
            qloss.backward()
            qopt.step()

    curve_calls, curve_tr, curve_te = [], [], []

    def record():
        curve_calls.append(oracle.calls)
        curve_tr.append(eval_regret(train_idx))
        curve_te.append(eval_regret(test_idx))

    next_rec = 0
    if needs_q and q_warmup > 0:
        record()
        next_rec = record_every
        for c in range(q_warmup):
            i = c % n
            probs = softmax_np(logits_of(i).detach().numpy())
            y = sample_tokens(rng, probs, 1)[0]
            replay.append((i, y, oracle(i, y / (V - 1))))
        for _ in range(q_warmup_steps):
            q_step()

    baseline = np.full(n, np.nan)
    step = 0
    while oracle.calls < budget:
        if oracle.calls >= next_rec:
            record()
            next_rec += record_every
        i = step % n
        step += 1
        logits = logits_of(i)
        probs_np = softmax_np(logits.detach().numpy())
        logp = tF.log_softmax(logits, dim=-1)

        if method == "rloo":
            Y = sample_tokens(rng, probs_np, K)
            Fv = np.array([oracle(i, y / (V - 1)) for y in Y])
            coeff = Fv - (Fv.sum() - Fv) / (K - 1)
            lp = logp[ar[None, :], torch.as_tensor(Y)]
            surr = (torch.as_tensor(coeff, dtype=torch.float32)[:, None]
                    * lp).sum() / K
        else:
            y = sample_tokens(rng, probs_np, 1)
            f = oracle(i, y[0] / (V - 1))
            yt = torch.as_tensor(y[0])
            lp = logp[ar, yt]
            if method == "reinforce":
                surr = (f * lp).sum()
            elif method == "reinforce_b":
                bl = f if np.isnan(baseline[i]) else baseline[i]
                surr = ((f - bl) * lp).sum()
                baseline[i] = f if np.isnan(baseline[i]) \
                    else 0.95 * baseline[i] + 0.05 * f
            elif method == "qcv":
                with torch.no_grad():
                    qall = qnet(Xt[i:i + 1], torch.as_tensor(y))[0]
                coeff = f - qall[ar, yt]
                surr = ((coeff * lp).sum()
                        + (torch.softmax(logits, -1) * qall).sum())
                replay.append((i, y[0], f))
            elif method == "pathwise":
                with torch.no_grad():
                    qall = qnet(Xt[i:i + 1], torch.as_tensor(y))[0]
                surr = (torch.softmax(logits, -1) * qall).sum()
                replay.append((i, y[0], f))
            elif method == "qcv_exact":
                Qex = diag.exact_Q(i, y, probs_np)[0]
                Qt = torch.as_tensor(Qex, dtype=torch.float32)
                coeff = torch.as_tensor(f - Qex[np.arange(d), y[0]],
                                        dtype=torch.float32)
                surr = ((coeff * lp).sum()
                        + (torch.softmax(logits, -1) * Qt).sum())
            elif method == "qcv_delta":
                with torch.no_grad():
                    qall = qnet(Xt[i:i + 1], torch.as_tensor(y))[0]
                dall = deltanet(Xt[i:i + 1], torch.as_tensor(y))[0]  # grad on
                call = qall + dall
                coeff = f - call[ar, yt]
                surr = ((coeff * lp).sum()
                        + (torch.softmax(logits, -1) * call).sum())
                replay.append((i, y[0], f))

                # Variance descent on Delta only: surr(theta, phi) is smooth
                # in phi even though f is not smooth in theta, so grad-of-grad
                # is well defined. E[ghat] doesn't depend on phi (Lemma 1),
                # so d/dphi E||ghat||^2 == d/dphi Var(ghat) exactly.
                g_vec = torch.autograd.grad(surr, list(lin.parameters()),
                                            create_graph=True)
                g_flat = [g.detach().clone() for g in g_vec]
                var_proxy = sum((g ** 2).sum() for g in g_vec)
                dopt.zero_grad()
                var_proxy.backward()
                dopt.step()
            else:
                raise ValueError(method)

        if method == "qcv_delta":
            opt.zero_grad()                       # reuse g_flat: no 2nd backward
            for prm, g in zip(lin.parameters(), g_flat):
                prm.grad = g
            opt.step()
        else:
            opt.zero_grad()
            surr.backward()
            opt.step()

        if needs_q and len(replay) >= q_bs:
            for _ in range(q_steps):
                q_step()

    record()
    out = dict(calls=np.array(curve_calls), train=np.array(curve_tr),
               test=np.array(curve_te), lin=lin)
    if needs_q:
        out["qnet"] = qnet
    return out
