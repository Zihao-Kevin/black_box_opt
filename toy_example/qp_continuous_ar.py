"""Exact variance accounting for the CONTINUOUS (Gaussian-policy) QP toy, all families and Delta states.

`qp_continuous.py` covers the leave-one-out family, where the residual is linear in the noise and the
closed form is one cosine per coordinate.  The autoregressive family (Q_j = E[f | b_{<=j}], the LLM
setting) leaves higher-order noise, and this module does its accounting exactly, with no sampling.

Everything is a polynomial of degree <= 3 in the standardised noise eps = (b - mu)/sigma, because the
QP's regret is exactly quadratic in chat and the score is linear in eps.  Expanding each position's
scalar residual in the (probabilists') Hermite basis,

    ghat - E ghat = sum_beta He_beta(eps) R_beta,        R_beta = sum_j r_{j,beta} J_j,   J_j = grad mu_j,

and Hermite orthogonality gives  Var = sum_beta beta! ||R_beta||^2  exactly.

A Delta_j(state, v) subtracts h_j J_j with h_j any function of (state, b_j) that is conditionally centred
given the state, i.e. any combination of He_beta with supp(beta) in {state, j} and beta_j >= 1.  For a
prefix state that is every beta whose LATEST coordinate is j -- so each beta is reachable from exactly one
position, j*(beta) = max supp(beta), and the optimum decouples:

    Var(Q + Delta*) = sum_beta beta! ||R_beta||^2 (1 - cos^2(R_beta, J_{j*(beta)}))      [reachable beta]

The AR future noise (beta touching a later coordinate) is cancellable only through the later position's
score direction: that is the continuous form of "Delta needs states AND alignment".

Residuals (G = g + 2 H mu, eps standard normal, sigma the policy std):
  rf     : (f - E f) eps_j / sigma
  loo    : 2 sigma sum_{k != j} H_jk eps_k                                      (score term is 0)
  prefix : (f - Q_j) eps_j / sigma + 2 sigma sum_{k < j} H_jk eps_k,
           f - Q_j = sigma sum_{k>j} G_k eps_k + sigma^2 [2 sum_{k<=j<l} H_kl eps_k eps_l
                     + sum_{k,l>j} H_kl eps_k eps_l - sum_{k>j} H_kk]
"""
from collections import defaultdict
from math import factorial

import numpy as np
import torch

from qp_continuous import ContinuousQP, MeanNet, dev, DT

# x^n in the Hermite basis: x = He1, x^2 = He2 + 1, x^3 = He3 + 3 He1
_HERM = {0: [(0, 1.0)], 1: [(1, 1.0)], 2: [(2, 1.0), (0, 1.0)], 3: [(3, 1.0), (1, 3.0)]}


def _mono_to_hermite(alpha):
    out = [((), 1.0)]
    for n in alpha:
        out = [(b + (m,), c * cm) for b, c in out for m, cm in _HERM[n]]
    return out


def residual(family, G, H, sigma):
    """{monomial alpha: coefficient vector over positions j} for the centred estimator residual."""
    d = len(G)
    P = defaultdict(lambda: np.zeros(d))

    def add(idxs, j, c):
        a = [0] * d
        for i in idxs:
            a[i] += 1
        P[tuple(a)][j] += c

    for j in range(d):
        if family == "rf":
            for k in range(d):
                add([k, j], j, G[k])
                for l in range(d):
                    add([k, l, j], j, sigma * H[k, l])
            add([j], j, -sigma * np.trace(H))
        elif family == "loo":
            for k in range(d):
                if k != j:
                    add([k], j, 2 * sigma * H[j, k])
        elif family == "prefix":
            for k in range(j + 1, d):
                add([j, k], j, G[k])
                add([j], j, -sigma * H[k, k])
                for l in range(j + 1, d):
                    add([k, l, j], j, sigma * H[k, l])
            for k in range(j + 1):
                for l in range(j + 1, d):
                    add([k, l, j], j, 2 * sigma * H[k, l])
            for k in range(j):
                add([k], j, 2 * sigma * H[j, k])
        else:
            raise ValueError(family)
    Ph = defaultdict(lambda: np.zeros(d))
    for alpha, vec in P.items():
        for beta, c in _mono_to_hermite(alpha):
            Ph[beta] += c * vec
    return {b: v for b, v in Ph.items() if any(b) and np.any(v != 0)}


def reachable(beta, state):
    supp = [i for i, n in enumerate(beta) if n > 0]
    js = supp[-1]
    if state == "none":
        return len(supp) == 1
    if state == "prev":
        return all(i >= js - 1 for i in supp)
    if state == "pre":
        return True
    raise ValueError(state)


class Account:
    """exact (Var family, Var family + Delta*, per-beta h coefficients) for one instance."""

    def __init__(self, cqp, net, i):
        self.cqp, self.i = cqp, i
        self.J, self.mu = cqp.jac(net, i)
        self.H = cqp.H[i].cpu().numpy()
        self.G = (cqp.g[i] + 2 * cqp.H[i] @ self.mu).cpu().numpy()
        self.Jn = self.J.cpu().numpy()
        self.sigma = cqp.sigma
        self._cache = {}

    def hermite(self, family):
        if family not in self._cache:
            Ph = residual(family, self.G, self.H, self.sigma)
            self._cache[family] = {b: (np.prod([factorial(n) for n in b]), v @ self.Jn) for b, v in Ph.items()}
        return self._cache[family]

    def var(self, family):
        return sum(w * R @ R for w, R in self.hermite(family).values())

    def var_delta(self, family, state):
        """Var(family + Delta*) with Delta's state; also returns {beta: coefficient on J_{j*}}."""
        tot, coef = 0.0, {}
        for b, (w, R) in self.hermite(family).items():
            RR = R @ R
            if reachable(b, state):
                js = max(i for i, n in enumerate(b) if n > 0)
                Jj = self.Jn[js]
                c = R @ Jj / max(Jj @ Jj, 1e-300)
                coef[b] = c
                RR -= c * c * (Jj @ Jj)
            tot += w * RR
        return tot, coef

    def rho(self):
        Jd = self.Jn / np.maximum(np.linalg.norm(self.Jn, axis=1, keepdims=True), 1e-300)
        C = Jd @ Jd.T
        d = len(C)
        return (C.sum() - np.trace(C)) / (d * (d - 1))

    # ------------------------------------------------------------------ Monte-Carlo check of the closed form
    def mc(self, family, state=None, N=200000, seed=0):
        """form the estimator as an ALGORITHM would (score term + pathwise term, explicit Delta_j polynomials)."""
        d, s = len(self.G), self.sigma
        gen = torch.Generator(device=dev).manual_seed(seed)
        eps = torch.randn(N, d, generator=gen, dtype=DT, device=dev)
        J, mu, H = self.J, self.mu, self.cqp.H[self.i]
        G = torch.as_tensor(self.G, dtype=DT, device=dev)
        b = mu[None] + s * eps
        f = self.cqp.f(self.i, b)
        Ef = self.cqp.f(self.i, mu) + s ** 2 * H.trace()
        if family == "rf":
            coef = (f - Ef)[:, None] * eps / s
        elif family == "loo":
            coef = G[None] + 2 * (b - mu[None]) @ H - 2 * H.diagonal()[None] * (b - mu[None])
        elif family == "prefix":
            fut = H.diagonal().flip(0).cumsum(0).flip(0)                           # sum_{k>=j} H_kk
            fut = torch.cat([fut[1:], fut.new_zeros(1)])                           # sum_{k>j} H_kk
            Q = torch.stack([self.cqp.f(self.i, torch.cat([b[:, :j + 1], mu[None, j + 1:].expand(N, -1)], 1))
                             + s ** 2 * fut[j] for j in range(d)], 1)
            Ltri = torch.tril(H, -1)                                               # H_jk, k < j
            path = G[None] + 2 * (b - mu[None]) @ Ltri.T
            coef = (f[:, None] - Q) * eps / s + path
        if state is not None:
            _, cf = self.var_delta(family, state)
            He = {0: torch.ones_like(eps), 1: eps, 2: eps ** 2 - 1, 3: eps ** 3 - 3 * eps}
            phi = {1: torch.ones_like(eps), 2: eps, 3: eps ** 2 - 3}                # Delta's own-coordinate factor
            D = torch.zeros(N, d, dtype=DT, device=dev)
            Epsi = torch.zeros(N, d, dtype=DT, device=dev)                         # E[psi_j | state]
            for beta, c in cf.items():
                js = max(i for i, n in enumerate(beta) if n > 0)
                other = torch.ones(N, dtype=DT, device=dev)
                for i, n in enumerate(beta):
                    if i != js and n:
                        other = other * He[n][:, i]
                D[:, js] += s * c * other * phi[beta[js]][:, js]
                if beta[js] == 2:
                    Epsi[:, js] += c * other
            coef = coef - D * eps / s + Epsi
        Gm = coef @ J
        return Gm.mean(0), float(((Gm - Gm.mean(0)[None]) ** 2).sum(1).mean())


# ---------------------------------------------------------------------- training
def train_with(cqp, net, method, budget, lr=0.01, seed=0, log=None, every=25):
    """one oracle call per step.  method: "exact" (true gradient, the infinite-sample path), "rf"
    (REINFORCE-b with EMA baseline, `ContinuousQP.train`), or (family, state) for a single-sample
    estimator with the exact per-instance Q and Delta* (a ceiling, like MCP's oracle row) -- ("rf", None)
    is REINFORCE with the exact mean baseline, the variance tables' denominator.
    `log` (a list) receives the exact regret every `every` calls."""
    if method == "rf":
        return cqp.train(net, budget=budget, lr=lr, seed=seed)
    net = net.to(dev).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(budget + 1):
        if log is not None and step % every == 0:
            log.append(cqp.regret(net))
        if step == budget:
            break
        i = step % cqp.o.n_train
        if method == "exact":
            g = cqp.true_grad(net, i)
        else:
            g, _ = Account(cqp, net, i).mc(*method, N=1, seed=seed * 100000 + step)
        opt.zero_grad(); k = 0
        for p in net.parameters():                                    # same flat order as Account.jac
            m = p.numel(); p.grad = g[k:k + m].view_as(p).float(); k += m
        opt.step()
    return net


def _make(cfg, cqp, seed):
    """cfg: a CFG key, or a callable seed -> model (any module mapping features (..., p) to (..., d))."""
    torch.manual_seed(seed)
    return cfg(seed).to(dev) if callable(cfg) else MeanNet(cqp.p, cqp.d, seed=seed, **CFG[cfg]).to(dev)


def curves(cqp, cfg, methods, seeds=(0, 1, 2), budget=800, every=25, lr=0.01):
    """regret along training for each named method: (calls, {name: array (seeds, points)})."""
    out = {}
    for name, m in methods.items():
        rows = []
        for s in seeds:
            net = _make(cfg, cqp, s); log = []
            train_with(cqp, net, m, budget, lr=lr, seed=s, log=log, every=every)
            rows.append(log)
        out[name] = np.array(rows)
    return np.arange(0, budget + 1, every), out


def shares(cqp, cfg, seeds=(0, 1, 2), family="prefix", state="pre"):
    """(Q, Q + Delta*) as fractions of REINFORCE-b's gradient variance, at the untrained predictor."""
    q = qd = 0.0
    for s in seeds:
        net = _make(cfg, cqp, s)
        acc = [Account(cqp, net, i) for i in range(cqp.o.n_train)]
        vrf = sum(a.var("rf") for a in acc)
        q += sum(a.var(family) for a in acc) / vrf / len(seeds)
        qd += sum(a.var_delta(family, state)[0] for a in acc) / vrf / len(seeds)
    return q, qd


# ---------------------------------------------------------------------- notebook-style trajectory
CFG = {"tutorial-style": dict(h=32, pos_scale=1.0, train_pos=True),
       "aligned":        dict(h=8,  pos_scale=0.1, train_pos=False)}
CMB = [("loo", "none"), ("prefix", "none"), ("prefix", "pre")]


def trajectory(cqp, cfg, seeds=(0, 1, 2), budgets=(0, 200, 400, 800), combos=CMB, lr=0.01, path="exact"):
    """variance shares along a policy path.  `path="exact"` follows the true gradient (REINFORCE-b at
    sigma = 0.15 does not move the aligned predictor at all, so its path is uninformative)."""
    out = {}
    for s in seeds:
        net = _make(cfg, cqp, s); prev = 0
        for bud in budgets:
            if bud > prev:
                net = train_with(cqp, net, path, bud - prev, lr=lr, seed=s); prev = bud
            acc = [Account(cqp, net, i) for i in range(cqp.o.n_train)]
            vrf = sum(a.var("rf") for a in acc)
            rec = out.setdefault(bud, {"regret": [], "rho": []})
            rec["regret"].append(cqp.regret(net)); rec["rho"].append(np.mean([a.rho() for a in acc]))
            for fam, st in combos:
                rec.setdefault((fam, st), []).append((sum(a.var(fam) for a in acc) / vrf,
                                                      sum(a.var_delta(fam, st)[0] for a in acc) / vrf))
    return out


def show(title, out, combos=CMB):
    print(f"\n{title}")
    print(f"{'calls':>6}{'regret':>8}{'rho':>7}   " + "".join(f"{f[:3] + '/' + s:>22}" for f, s in combos))
    print(f"{'':>21}   " + "".join(f"{'Q/rf  Q+D/rf  ratio':>22}" for _ in combos))
    for bud in sorted(out):
        r = out[bud]
        line = f"{bud:>6}{np.mean(r['regret']):>8.3f}{np.mean(r['rho']):>7.3f}   "
        for key in combos:
            a = np.array(r[key]); line += f"{a[:,0].mean():>8.4f}{a[:,1].mean():>8.4f}{(a[:,1]/a[:,0]).mean():>6.3f}"
        print(line)


if __name__ == "__main__":
    import time
    t0 = time.time()
    cqp = ContinuousQP(d=8, lam=1.3, seed=0)
    net = MeanNet(cqp.p, cqp.d, seed=0, **CFG["aligned"]).to(dev)
    net = cqp.train(net, budget=200, seed=0)
    acc = Account(cqp, net, 0)
    print("closed form vs Monte Carlo, d=8 aligned, 200 calls, instance 0")
    tg = cqp.true_grad(net, 0)
    print(f"  loo   exact Var(Q)   {acc.var('loo'):.5e}  (qp_continuous.exact: {cqp.exact(net)[0]:.5e} pooled)")
    for fam, st in (("rf", None), ("loo", None), ("loo", "none"), ("prefix", None),
                    ("prefix", "none"), ("prefix", "prev"), ("prefix", "pre")):
        ex = acc.var(fam) if st is None else acc.var_delta(fam, st)[0]
        m, v = acc.mc(fam, st)
        print(f"  {fam:>6}/{str(st):<5} exact {ex:.5e}  MC {v:.5e}  |mean - true grad| {float((m - tg).norm()):.1e}")

    cqp = ContinuousQP(d=12, lam=1.3, seed=0)
    for cfg in ("tutorial-style", "aligned"):
        show(f"d=12 continuous, sigma={cqp.sigma}, {cfg} predictor {CFG[cfg]}  (exact-gradient path)",
             trajectory(cqp, cfg))
        print(f"[{time.time() - t0:.0f}s]")

    print("\nregret when TRAINED with each estimator, one call per step, lr 0.01, 2 seeds  (calls 0/200/400/800)")
    for cfg in ("tutorial-style", "aligned"):
        for method in ("exact", "rf", ("loo", "none"), ("prefix", "pre")):
            regs = []
            for seed in (0, 1):
                net = MeanNet(cqp.p, cqp.d, seed=seed, **CFG[cfg]).to(dev); prev = 0; row = []
                for bud in (0, 200, 400, 800):
                    if bud > prev:
                        net = train_with(cqp, net, method, bud - prev, lr=0.01, seed=seed); prev = bud
                    row.append(cqp.regret(net))
                regs.append(row)
            print(f"  {cfg:>15} {str(method):>18}  " + "  ".join(f"{x:.3f}" for x in np.mean(regs, 0)))
    print(f"[{time.time() - t0:.0f}s]")


# ---------------------------------------------------------------------- figures for the story notebook
_COL = {"plain": "#2a78d6", "+ Q": "#eb6834", "+ Q + Δ": "#1baf7a", "+ Q + row-wise Δ": "#eda100"}   # categorical slots 1-4, fixed
_MORE = ["#9a8fbf", "#7f9ca8", "#b57c8c", "#6e8b5e", "#a08268"]                 # muted slots for any other series (the baselines)
_INK, _MUTED, _GRID, _REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def _clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=1); ax.set_axisbelow(True)


def plot_shares(rows, ax=None):
    """rows = {model name: (Q share, Q+Delta share[, Q+row-wise Delta share])} of the plain estimator's gradient noise."""
    import matplotlib.pyplot as plt
    nbar = 1 + max(len(v) for v in rows.values())
    ax = ax or plt.subplots(figsize=(7.2, 0.6 + 0.45 * nbar * len(rows)))[1]
    names = list(_COL); y = 0; ticks, labels = [], []
    for model, vals in rows.items():
        ax.text(0, y, model, ha="left", va="center", fontsize=10, color=_INK, fontweight="bold")
        for name, val in zip(names, (1.0, *vals)):
            y -= 1
            ax.barh(y, 100 * val, height=0.62, color=_COL[name], edgecolor="white", linewidth=1)
            ax.text(100 * val + 1.2, y, f"{100 * val:.0f}%" if val > 0.05 else f"{100 * val:.1f}%",
                    va="center", fontsize=9.5, color=_INK)
            ticks.append(y); labels.append(name)
        y -= 1.8
    ax.set_yticks(ticks); ax.set_yticklabels(labels, color=_MUTED, fontsize=9.5)
    ax.set_ylim(y + 1.2, 0.8)                                            # headroom under the title
    ax.set_xlim(0, 118); ax.set_xticks([]); ax.xaxis.grid(False); ax.yaxis.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Gradient noise left, as % of the plain estimator", loc="left", fontsize=10.5, color=_INK, pad=10)
    return ax


def plot_curves(calls, out, ref="exact gradient", ax=None, title="Regret while training"):
    """out = {series name: array (seeds, points)}; the series named `ref` is drawn as a gray reference."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7.2, 3.6))[1]
    ends, more = [], iter(_MORE * 4)
    for name, arr in out.items():
        m = arr.mean(0)
        if name == ref:
            ax.plot(calls, m, color=_REF, linewidth=1.6, linestyle=(0, (4, 3)), label=name)
        else:
            col = _COL.get(name) or next(more)
            ax.plot(calls, m, color=col, linewidth=2, solid_capstyle="round", label=name)
            ax.fill_between(calls, arr.min(0), arr.max(0), color=col, alpha=0.12, linewidth=0)
        ends.append((m[-1], name))
    # end labels: keep a minimum gap, draw a leader line to any label that had to move
    lo, hi = ax.get_ylim(); gap = 0.06 * (hi - lo)
    ends.sort(); pos = [v for v, _ in ends]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    for (v, name), yl in zip(ends, pos):
        x0, x1 = calls[-1] * 1.01, calls[-1] * 1.045
        if abs(yl - v) > 1e-9:
            ax.plot([x0, x1], [v, yl], color=_GRID, linewidth=1)
        ax.text(x1 + calls[-1] * 0.005, yl, name, va="center", fontsize=9.5, color=_INK)
    _clean(ax)
    ax.set_xlabel("calls to the black box", color=_MUTED, fontsize=9.5)
    ax.set_ylabel("regret  (0 = best possible portfolio)", color=_MUTED, fontsize=9.5)
    ax.set_xlim(0, calls[-1] * 1.22)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="lower left")
    ax.set_title(f"{title}, mean of {len(arr)} runs (band = spread)", loc="left", fontsize=10.5, color=_INK, pad=10)
    return ax
