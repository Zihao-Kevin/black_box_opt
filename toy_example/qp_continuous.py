"""The control variate with a CONTINUOUS predicted parameter: no grid, no vocabulary.

The toy's V-point grid is `dfl_toy`'s device, not the method's.  Real DFL predicts chat in R^d, and the
estimator carries over by replacing the vocabulary sum with an expectation under the policy:

    ghat = sum_j [ (f(b) - c_j(b_{-j}, b_j)) grad log p_j(b_j)  +  grad_theta E_{v ~ p_j}[c_j(b_{-j}, v)] ]

unbiased for any c, by the same identity (E_v[c grad log p] = grad_theta E_v[c], so the two terms cancel in
expectation).  Discrete: the second term is a sum over V.  Continuous: it is an integral -- closed form for a
Gaussian policy whenever c is polynomial in v, which is the relevant case here because the QP's regret is
exactly quadratic in chat, so the exact Q_j is exactly quadratic in v.

Policy: b_j ~ N(mu_j(theta; x), sigma^2), independent across j.  Reward: f(b) = a + g'b + b'H b  (QP regret,
exact).  Writing J_j = grad_theta mu_j in R^P:

  Q_j(b_{-j}, v) = f(b_{-j}, v),  E_v[Q_j] = ... + g_j mu_j + H_jj(mu_j^2 + sigma^2) + 2 mu_j sum_{k!=j} H_jk b_k
  => ghat_Q = sum_j [ g_j + 2 H_jj mu_j + 2 sum_{k != j} H_jk b_k ] J_j                          (score term is 0)
  => ghat_Q - E ghat_Q = sum_k (b_k - mu_k) w_k,     w_k = 2 sum_{j != k} H_jk J_j                      (*)

exactly the additive structure of the discrete case, with the score direction u_j now equal to J_j = grad mu_j.

**The rank-1 property that V = 2 was engineering is FREE here.**  A Gaussian with a learned mean has a
one-dimensional score direction per coordinate (grad log p_j = ((b_j - mu_j)/sigma^2) J_j), so there is exactly
one direction Delta must align with -- no coarse grid needed.  A Delta_j(.) perturbs the estimator by
-(psi_j(b_j) - E psi_j) J_j with psi_j(v) = Delta_j(v)(v - mu_j)/sigma^2, so the reachable set per coordinate is
{h(b_j) J_j : h centred} and the problem decouples exactly as before:

    V_Q      = sum_k sigma^2 ||w_k||^2
    V_{Q+D*} = sum_k sigma^2 ||w_k||^2 (1 - cos^2(w_k, J_k))
    ratio    = 1 - weighted-mean cos^2(w_k, J_k)                                                       (**)

-- the SAME closed form, with u_k = grad_theta mu_k.  For a quadratic f the optimal Delta_k is a per-coordinate
CONSTANT, delta_k = sigma^2 <w_k, J_k> / ||J_k||^2 (the target is linear in b_k, and a constant Delta is what
produces a linear psi); a rounder f needs a richer Delta, matching its nonlinearity.

Everything above is exact.  Monte Carlo is used only to CHECK it (unbiasedness, the variance formulas) and to
report Var(REINFORCE-b), whose Gaussian moments are messy and which is only a reference denominator.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev

from dfl_toy import QPOracle

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64


class MeanNet(nn.Module):
    """x -> trunk -> tanh(trunk(x) + pos_j) -> ONE shared head -> mu_j.  Continuous output: no vocabulary.

    train_pos=False makes positions fixed features, so every trainable parameter is shared across
    coordinates; pos_scale controls how strongly the coordinates are differentiated."""

    def __init__(self, p, d, h=32, pos_scale=1.0, train_pos=True, seed=0):
        super().__init__()
        self.d = d
        g = torch.Generator().manual_seed(seed)
        self.trunk = nn.Linear(p, h)
        pos = torch.randn(d, h, generator=g) * pos_scale
        if train_pos:
            self.pos = nn.Parameter(pos)
        else:
            self.register_buffer("pos", pos)
        self.head = nn.Linear(h, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(torch.tanh(self.trunk(x).unsqueeze(-2) + self.pos)).squeeze(-1)   # (..., d)


class ContinuousQP:
    """QPOracle with a continuous Gaussian predictor.  regret_i(chat) = a_i + g_i'chat + chat'H_i chat, exact."""

    def __init__(self, d=8, p=6, lam=1.3, k_factors=None, sigma=0.15, seed=0, n_train=6, n_test=4, oracle=None):
        self.o = oracle or QPOracle(d=d, p=p, n_train=n_train, n_test=n_test, lam=lam, k_factors=k_factors, seed=seed)
        self.d, self.p, self.sigma = self.o.d, self.o.p, sigma
        M, S, b0, lm = self.o._M, self.o._Sigma, self.o._b0, self.o._lam
        H0 = 0.5 * lm * (M.T @ S @ M)                                    # from -(lam/2) z'Sigma z, z = M chat + b0
        self.H, self.g, self.a = [], [], []
        for i in range(self.o.n_train):
            D = self.o._opt[i] - self.o._worst[i] + 1e-12
            lin = M.T @ self.o._C[i] - lm * (M.T @ S @ b0)
            cst = self.o._C[i] @ b0 - 0.5 * lm * (b0 @ S @ b0)
            self.H.append(torch.as_tensor(H0 / D, dtype=DT, device=dev))         # (opt - val)/D
            self.g.append(torch.as_tensor(-lin / D, dtype=DT, device=dev))
            self.a.append(float((self.o._opt[i] - cst) / D))

    @classmethod
    def from_data(cls, X, true_returns, Sigma, lam=1.3, sigma=0.15):
        """the same black box built from data made elsewhere: features X (n, p), hidden true returns (n, d),
        hidden risk model Sigma (d, d).  Every row is a training instance."""
        o = QPOracle.__new__(QPOracle)
        X, C, S = (np.asarray(v, dtype=float) for v in (X, true_returns, Sigma))
        o.X, o._C, o._Sigma, o._lam, o._noise, o.calls = X, C, S, lam, 0.0, 0
        o.n_train, o.n_test, o.d, o.p = len(X), 0, C.shape[1], X.shape[1]
        Sinv = np.linalg.inv(S); Sinv1 = Sinv @ np.ones(o.d); A = float(np.ones(o.d) @ Sinv1)
        o._M, o._b0 = (Sinv - np.outer(Sinv1, Sinv1) / A) / lam, Sinv1 / A
        o._opt, o._worst = o._obj_batch(o._solve(C)), o._obj_batch(o._solve(-C))
        return cls(sigma=sigma, oracle=o)

    def f(self, i, b):
        """exact regret of the QP at chat = b: (..., d) -> (...)"""
        return self.a[i] + b @ self.g[i] + torch.einsum("...j,jk,...k->...", b, self.H[i], b)

    def regret(self, net):
        """exact E[f] under the Gaussian policy, averaged over training instances (E f = f(mu) + sigma^2 tr H)."""
        mu = self.mus(net)
        return float(np.mean([self.f(i, mu[i]).item() + self.sigma ** 2 * float(self.H[i].trace())
                              for i in range(self.o.n_train)]))

    def mus(self, net):
        X = torch.as_tensor(self.o.X[:self.o.n_train], dtype=torch.float32, device=dev)
        return net(X).double()

    def jac(self, net, i):
        """J[j] = grad_theta mu_j in R^P, and mu."""
        net = net.cpu().to(torch.float32)
        x = torch.as_tensor(self.o.X[i], dtype=torch.float32)
        prm = {k: v.detach() for k, v in net.named_parameters()}
        Jd = jacrev(lambda pr: functional_call(net, pr, (x,)))(prm)
        J = torch.cat([Jd[k].reshape(self.d, -1) for k in prm], 1).double().to(dev)
        mu = net(x).detach().double().to(dev)
        net.to(dev)
        return J, mu

    # ------------------------------------------------------------------ exact quantities, from (*) and (**)
    def pieces(self, net, i):
        J, mu = self.jac(net, i)
        H = self.H[i]
        W = 2.0 * (H @ J - H.diagonal()[:, None] * J)                    # w_k = 2 sum_{j != k} H_jk J_j
        return J, mu, W

    def exact(self, net):
        """pooled (Var Q, Var Q+Delta*, rho) over training instances -- closed form, no sampling."""
        vq = vqd = 0.0; rhos = []
        for i in range(self.o.n_train):
            J, mu, W = self.pieces(net, i)
            Wn = W.norm(dim=1).clamp_min(1e-300); Jn = J.norm(dim=1).clamp_min(1e-300)
            cos2 = ((W * J).sum(1) / (Wn * Jn)) ** 2
            mass = self.sigma ** 2 * Wn ** 2
            vq += float(mass.sum()); vqd += float((mass * (1 - cos2)).sum())
            Jd = J / Jn[:, None]; Cs = Jd @ Jd.T
            rhos.append(float((Cs.sum() - Cs.diagonal().sum()) / (self.d * (self.d - 1))))
        return vq, vqd, float(np.mean(rhos))

    def delta_star(self, net, i):
        J, mu, W = self.pieces(net, i)
        return self.sigma ** 2 * (W * J).sum(1) / (J * J).sum(1).clamp_min(1e-300)          # per-coordinate constant

    # ------------------------------------------------------------------ Monte-Carlo checks
    def mc(self, net, i, N=400000, seed=0, est="Q"):
        """sample b ~ N(mu, sigma^2 I) and form the estimator as an ALGORITHM would: score term + pathwise term."""
        J, mu, W = self.pieces(net, i)
        H, g = self.H[i], self.g[i]
        gen = torch.Generator(device=dev).manual_seed(seed)
        b = mu[None] + self.sigma * torch.randn(N, self.d, generator=gen, dtype=DT, device=dev)
        score = (b - mu[None]) / self.sigma ** 2                                            # coeff on J_j
        if est == "rf":                                                                     # REINFORCE, exact mean baseline
            Ef = self.f(i, mu) + self.sigma ** 2 * H.trace()
            coef = (self.f(i, b) - Ef)[:, None] * score
        else:
            # c_j = Q_j (+ Delta_j).  Score term: f(b) - c_j(b_{-j}, b_j).  Q_j(b_{-j}, b_j) = f(b) exactly,
            # so it survives only through Delta.  Pathwise term: d/dmu_j E_v[c_j].
            path = g[None] + 2 * b @ H - 2 * H.diagonal()[None] * b + 2 * H.diagonal()[None] * mu[None]
            coef = path
            if est == "QD":
                D = self.delta_star(net, i)
                coef = coef - D[None] * score                    # -Delta_j * score_j ; E_v[Delta] is const in mu
        G = coef @ J
        return G.mean(0), float(((G - G.mean(0)[None]) ** 2).sum(1).mean()), G.shape[0]

    def true_grad(self, net, i):
        J, mu, W = self.pieces(net, i)
        return (self.g[i] + 2 * self.H[i] @ mu) @ J                                          # grad_theta E[f]

    # ------------------------------------------------------------------ training (REINFORCE with EMA baseline)
    def train(self, net, budget=200, lr=0.01, seed=0):
        net = net.to(dev).to(torch.float32)
        rng = np.random.default_rng(seed)
        X = torch.as_tensor(self.o.X[:self.o.n_train], dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        n, base, step, calls = self.o.n_train, np.full(self.o.n_train, np.nan), 0, 0
        while calls < budget:
            i = step % n; step += 1; calls += 1
            mu = net(X[i]).double()
            b = mu.detach() + self.sigma * torch.as_tensor(rng.standard_normal(self.d), dtype=DT, device=dev)
            f = float(self.f(i, b))
            lp = -((b - mu) ** 2).sum() / (2 * self.sigma ** 2)          # log p up to a theta-free constant
            bl = f if np.isnan(base[i]) else base[i]
            base[i] = f if np.isnan(base[i]) else 0.95 * base[i] + 0.05 * f
            opt.zero_grad(); ((f - bl) * lp).backward(); opt.step()
        return net
