# cells 0-2 of DFL_QP_Black_Box_new.ipynb, verbatim in substance
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # toy_example/
import numpy as np, torch, torch.nn as nn
from qp_continuous import ContinuousQP
from qp_continuous_ar import Account
d, p, n_days = 12, 6, 6
lam, sigma   = 1.3, 0.15
rng = np.random.default_rng(0)
X = rng.standard_normal((n_days, p))
true_returns = 1 / (1 + np.exp(-2 * X @ rng.standard_normal((d, p)).T / np.sqrt(p)))
F = rng.standard_normal((d, d // 2)) / np.sqrt(d // 2)
Sigma = F @ F.T + 0.25 * np.eye(d)
Sinv = np.linalg.inv(Sigma); u = Sinv @ np.ones(d)
M, b0 = (Sinv - np.outer(u, u) / u.sum()) / lam, u / u.sum()
def value(day, z): return true_returns[day] @ z - lam / 2 * z @ Sigma @ z
best  = [value(i,  M @ true_returns[i] + b0) for i in range(n_days)]
worst = [value(i, -M @ true_returns[i] + b0) for i in range(n_days)]
def black_box(day, predicted_returns): return (best[day] - value(day, M @ predicted_returns + b0)) / (best[day] - worst[day])
box = ContinuousQP.from_data(X, true_returns, Sigma, lam=lam, sigma=sigma)
class Model(nn.Module):
    def __init__(self, h=8, pos_scale=0.1, train_pos=False, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.trunk, self.head = nn.Linear(p, h), nn.Linear(h, 1)
        pos = torch.randn(d, h) * pos_scale
        if train_pos: self.pos = nn.Parameter(pos)
        else: self.register_buffer("pos", pos)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x): return self.head(torch.tanh(self.trunk(x).unsqueeze(-2) + self.pos)).squeeze(-1)
shared = lambda seed: Model(h=8,  pos_scale=0.1, train_pos=False, seed=seed)
private = lambda seed: Model(h=32, pos_scale=1.0, train_pos=True,  seed=seed)
