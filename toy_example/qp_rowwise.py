"""Row-wise Δ on the continuous DFL QP toy (the setting of DFL_QP_Black_Box_new.ipynb), exact.

Scalar Δ (qp_continuous_ar.Account.var_delta): at coordinate j the control variate is Δ_j(state, b_j) times the score
((b_j - mu_j)/sigma^2) J_j, so it can only remove noise along J_j = d mu_j / d theta.  In the Hermite expansion
    ghat - E ghat = sum_beta He_beta(eps) R_beta,          Var = sum_beta beta! |R_beta|^2,
the beta term is reachable at j* = max supp(beta) (prefix state), and scalar Δ* leaves |R_beta|^2 (1 - cos^2(R_beta, J_j*)).

Row-wise Δ: one scalar per (coordinate, token/value, weight row).  For a linear layer the gradient of mu_j is
(backprop signal) x (layer input), so the slice of J_j in weight row r is parallel to that row's input at j.  A row-wise
Δ can therefore remove, row by row, the component of R_beta along J_j*'s slice in that row:
    Var(Q + row-wise Δ*) = sum_beta beta! |R_beta - Pi_j* R_beta|^2,     Pi_j = blockdiag_r ( Ĵ_{j,r} Ĵ_{j,r}' ).
Rows here: every output unit of trunk.weight (input x), trunk.bias (input 1), head.weight (input tanh(...)_j),
head.bias (input 1), and, when it is trained, pos as a lookup table (row k = pos[:, k], input the one-hot e_j).
"per-parameter Δ" (every parameter its own row) is shown as the far end of the same family.

The estimator subtracts sum_beta He_beta(eps) Pi_j*(beta) R_beta from the Q estimator: centred (beta != 0), so unbiased,
built from the exact per-instance Q (like the notebook's "+ Q" and "+ Q + Δ", which are ceilings too).
"""
import numpy as np, torch
from qp_continuous_ar import Account, _make
from qp_continuous import dev, DT


def row_groups(net):
    """a group id for every flat parameter (in named_parameters order): the weight row it belongs to."""
    ids, nxt = [], 0
    for name, prm in net.named_parameters():
        if name.endswith("pos"):                                   # (d, h) lookup table: row k = column pos[:, k]
            g = nxt + np.tile(np.arange(prm.shape[1]), prm.shape[0]); nxt += prm.shape[1]
        elif prm.dim() == 2:                                       # Linear weight (out, in): one group per output unit
            g = nxt + np.repeat(np.arange(prm.shape[0]), prm.shape[1]); nxt += prm.shape[0]
        else:                                                      # bias: every entry is its own row (input 1)
            g = nxt + np.arange(prm.numel()); nxt += prm.numel()
        ids.append(g)
    return np.concatenate(ids)


def project(J, groups, R):
    """Pi R for R (..., P): in every group r the component of R along J's slice, (R_r . J_r / |J_r|^2) J_r (0 where J_r = 0)."""
    n = groups.max() + 1; R2 = R.reshape(-1, len(J))
    num = np.stack([np.bincount(groups, weights=x * J, minlength=n) for x in R2]); den = np.bincount(groups, weights=J * J, minlength=n)
    return ((num / np.maximum(den, 1e-24) * (den > 1e-24))[:, groups] * J).reshape(R.shape)


class RowAccount(Account):
    """Account (exact Hermite accounting, prefix family) plus grouped Δ*: kind = "scalar" | "row" | "param"."""

    def __init__(self, cqp, net, i):
        super().__init__(cqp, net, i)
        self.groups = {"scalar": np.zeros(self.Jn.shape[1], int), "row": row_groups(net), "param": np.arange(self.Jn.shape[1])}
        self._proj = {}

    def proj(self, kind, j, R):
        return project(self.Jn[j], self.groups[kind], R)

    def var_grouped(self, kind, family="prefix"):
        tot = 0.0
        for b, (w, R) in self.hermite(family).items():
            r = R - self.proj(kind, max(i for i, n in enumerate(b) if n > 0), R)
            tot += w * r @ r
        return tot

    def estimator(self, kind, N=1, seed=0):
        """(N, P) single-sample gradient estimates: Q (kind None) or Q + grouped Δ*."""
        d, s = len(self.G), self.sigma
        gen = torch.Generator(device=dev).manual_seed(seed)
        eps = torch.randn(N, d, generator=gen, dtype=DT, device=dev)
        J, mu, H = self.J, self.mu, self.cqp.H[self.i]; G = torch.as_tensor(self.G, dtype=DT, device=dev); b = mu[None] + s * eps
        f = self.cqp.f(self.i, b)
        fut = H.diagonal().flip(0).cumsum(0).flip(0); fut = torch.cat([fut[1:], fut.new_zeros(1)])
        Q = torch.stack([self.cqp.f(self.i, torch.cat([b[:, :j + 1], mu[None, j + 1:].expand(N, -1)], 1)) + s ** 2 * fut[j] for j in range(d)], 1)
        coef = (f[:, None] - Q) * eps / s + G[None] + 2 * (b - mu[None]) @ torch.tril(H, -1).T
        g = (coef @ J).cpu().numpy()
        if kind is None: return g
        return g - self.hermite_values(eps.cpu().numpy()) @ self.projected(kind)

    def projected(self, kind):
        """(n_beta, P): Pi_j*(beta) R_beta for every Hermite term, in hermite() order."""
        key = ("PR", kind)
        if key not in self._proj:
            self._proj[key] = np.stack([self.proj(kind, max(i for i, n in enumerate(b) if n > 0), R) for b, (w, R) in self.hermite("prefix").items()])
        return self._proj[key]

    def hermite_values(self, e):
        """(N, n_beta): He_beta(eps) for every Hermite term, in hermite() order."""
        He = {1: e, 2: e ** 2 - 1, 3: e ** 3 - 3 * e}; cols = []
        for b in self.hermite("prefix"):
            h = np.ones(len(e))
            for i, n in enumerate(b):
                if n: h = h * He[n][:, i]
            cols.append(h)
        return np.stack(cols, 1)


def shares(cqp, cfg, seeds=(0, 1, 2)):
    """(Q, Q + Delta*, Q + row-wise Delta*) as fractions of REINFORCE-b's gradient variance, at the untrained predictor."""
    out = np.zeros(3)
    for s in seeds:
        acc = [RowAccount(cqp, _make(cfg, cqp, s), i) for i in range(cqp.o.n_train)]
        vrf = sum(a.var("rf") for a in acc)
        out += np.array([sum(a.var("prefix") for a in acc), sum(a.var_delta("prefix", "pre")[0] for a in acc), sum(a.var_grouped("row") for a in acc)]) / vrf / len(seeds)
    return tuple(out)
