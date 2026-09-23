"""Standard policy-gradient baselines, written once for both black boxes (the QP toy and the MCP agent).

Every estimator here has the form  g_hat = w @ Z:  Z holds the score of every "entry" (a coordinate of the Gaussian
predictor in the QP toy, a (prefix, token) pair of the MCP tree), and w is a coefficient vector built from the sampled
episodes.  The task supplies, for a group of K episodes of one instance, the rewards f (K,), the score coefficient of
every decision S (K, T) with  grad log p(y_i) = sum_t S[i, t] Z[pos[i, t]],  and the entry of every decision
pos (K, T), -1 once the episode has ended.  It gets back w (E,) and forms g_hat = w @ Z itself.

Group baselines (one sequence-level advantage per episode, or one per decision for OTB):
  RLOO  (Kool et al. 2019; Ahmadian et al. 2024)   b_i = the mean reward of the other K - 1 episodes
  GRPO  (Shao et al. 2024)                          (f_i - mean f) / std f; one on-policy step, so the clipped ratio is 1
                                                    and the KL term is 0; the 1/|y| factor is dropped (Dr. GRPO), Adam
                                                    does not see a constant scale anyway
  OTB   (Li et al. 2026)                            B_t = sum_i f_i W_it / sum_i W_it with the realised energy
                                                    W_it = sum_{j <= t} ||s_ij||^2; here ||s||^2 is exact (the paper's
                                                    logit proxy stands in for it when the parameter gradient is costly)
Learned control variates (Grathwohl et al. 2018), unbiased for every phi, phi trained to minimise ||g_hat||^2:
  LAX   for the Gaussian policy: g_hat = (f - c(b)) grad log p(b) + grad_theta c(b) with b = mu + sigma eps
  RELAX for the categorical policy, one term per decision t (the paper's RL form):
        g_hat = sum_t (f - c_t(z~_t)) grad log p(b_t | b_<t) + grad c_t(z_t) - grad c_t(z~_t),  z_t = logp_t + Gumbel
        noise with argmax b_t, z~_t a second draw conditioned on b_t, and c_t = c_phi(prefix state, softmax(z_t / tau))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_weights(method, f, S, pos, diag):
    """w (E,) for RLOO / GRPO / OTB.  f (K,), S and pos (K, T), diag (E,) = ||Z_e||^2 (OTB only)."""
    K, valid = len(f), pos >= 0
    if method == "RLOO":
        A = (f - (f.sum() - f) / (K - 1))[:, None]
    elif method == "GRPO":
        A = ((f - f.mean()) / (f.std() + 1e-8))[:, None]
    elif method == "OTB":
        W = (S ** 2 * diag[pos.clamp_min(0)] * valid).cumsum(1) * valid            # realised energy, 0 after the end
        A = f[:, None] - (f[:, None] * W).sum(0) / W.sum(0).clamp_min(1e-30)       # one baseline per decision
    else:
        raise ValueError(method)
    src = A * S * valid
    return torch.zeros(K, len(diag), dtype=src.dtype, device=src.device).scatter_add_(1, pos.clamp_min(0), src).mean(0)


class Surrogate(nn.Module):
    """c_phi: a small MLP, and RELAX's relaxation temperature.  fit() takes one Adam step on ||g_hat||^2 = w K w."""

    def __init__(self, dim, hidden=64, lr=1e-2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def fit(self, w, K):                       # w (E,) with its graph to phi; K = Z Z^T
        loss = w @ K.to(w) @ w
        self.opt.zero_grad(); loss.backward(); self.opt.step()


def lax_weights(c, mu, b, f, sigma):
    """w (d,) for a Gaussian policy b ~ N(mu, sigma^2), b (K, d):  (f - c(b)) (b - mu) / sigma^2 + dc/db,  d b / d mu = I."""
    z = b.detach().requires_grad_()
    cz = c(z)
    dc = torch.autograd.grad(cz.sum(), z, create_graph=True)[0]
    return ((f - cz)[:, None] * (b - mu) / sigma ** 2 + dc).mean(0)


def cond_gumbel(logp, b, u):
    """z ~ p(z | argmax z = b) for z = logp + Gumbel noise, differentiable in logp (Tucker et al. 2017, App. B).
    logp (..., A), b (...), u (..., A) uniform."""
    ub = u.gather(-1, b[..., None])
    z = -torch.log(-torch.log(u) / logp.exp() - torch.log(ub))
    return torch.where(F.one_hot(b, logp.shape[-1]).bool(), -torch.log(-torch.log(ub)), z)


def relax_weights(c, logp, ent, b, f, E, state, gen=None):
    """w (E,) for a categorical policy over the T nodes each episode went through, with one surrogate value per node,
    c_t = c([state_t, softmax(z_t / tau)]), as in RELAX's RL setting.  logp (K, T, A): the node log-probs, ent (K, T, A):
    the entry index of each of them (-1 where a node has fewer than A branches, or after the end), b (K, T): the branch
    taken, f (K,), E: the number of entries, state (K, T, D): what the surrogate may know about the prefix."""
    valid, A = ent >= 0, ent.shape[-1]; live = valid.any(-1)
    lp = torch.where(valid, logp, 0.0).clamp_min(-60.0).requires_grad_()     # exp(-lp) must stay finite in float32
    u, v = torch.rand(2, *ent.shape, generator=gen, device=ent.device, dtype=lp.dtype)
    feat = lambda z: torch.cat([state, torch.softmax(torch.where(valid, z, -1e4) / c.log_tau.exp(), -1)], -1)
    z = lp + (cond_gumbel(lp.detach(), b, u) - lp.detach())      # z = logp + Gumbel noise whose argmax is b: d z / d logp = I
    cz, czt = c(feat(z)) * live, c(feat(cond_gumbel(lp, b, v))) * live      # z~ | b: the conditional reparameterisation
    dlp = torch.autograd.grad((cz - czt).sum(), lp, create_graph=True)[0]
    coef = (f[:, None] - czt)[..., None] * (F.one_hot(b, A) * valid) + dlp
    w = torch.zeros(len(f), E, dtype=lp.dtype, device=ent.device)
    return w.scatter_add_(1, ent.clamp_min(0).flatten(1), (coef * valid).flatten(1)).mean(0)
