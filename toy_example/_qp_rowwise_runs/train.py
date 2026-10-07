# the notebook's training loop (cell 5), plus "+ Q + row-wise Δ"; one run per call: model method seed
# usage: python train.py shared|private "+ Q + row-wise Δ" 0     (methods: plain, + Q, + Q + Δ, + Q + row-wise Δ, exact gradient)
import os, sys, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.py")).read()); torch.set_num_threads(1)
from qp_rowwise import RowAccount
Xt = torch.as_tensor(X, dtype=torch.float32); baseline = {}
def plain(model, day, seed):
    mu = model(Xt[day]); nudged = mu.detach() + sigma * torch.randn(d); score = black_box(day, nudged.numpy())
    b = baseline.get(day, score); baseline[day] = 0.9 * b + 0.1 * score
    log_prob = -((nudged - mu) ** 2).sum() / (2 * sigma ** 2)
    grads = torch.autograd.grad((score - b) * log_prob, list(model.parameters())); return torch.cat([g.flatten() for g in grads])
with_q       = lambda model, day, seed: Account(box, model, day).mc("prefix", None,  N=1, seed=seed)[0]
with_q_delta = lambda model, day, seed: Account(box, model, day).mc("prefix", "pre", N=1, seed=seed)[0]
with_q_row   = lambda model, day, seed: torch.as_tensor(RowAccount(box, model, day).estimator("row", N=1, seed=seed)[0])
exact        = lambda model, day, seed: box.true_grad(model, day)
METHODS = {"plain": plain, "+ Q": with_q, "+ Q + Δ": with_q_delta, "+ Q + row-wise Δ": with_q_row, "exact gradient": exact}
def train(make_model, gradient, calls=1600, lr=0.01, seed=0, every=50, K=1):
    model = make_model(seed); opt = torch.optim.Adam(model.parameters(), lr=lr); baseline.clear(); log = []
    for step in range(0, calls + 1, K):
        if len(log) <= step // every: log.append(box.regret(model))
        if step == calls: break
        g = gradient(model, (step // K) % n_days, seed * 100000 + step)
        opt.zero_grad(); k = 0
        for prm in model.parameters(): prm.grad = g[k:k + prm.numel()].view_as(prm).float(); k += prm.numel()
        opt.step()
    return log
mname, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{mname}_{method.replace(' ', '').replace('+', 'p')}_s{seed}.npy")
if not os.path.exists(out):
    torch.manual_seed(seed); t0 = time.time()
    log = train({"shared": shared, "private": private}[mname], METHODS[method], seed=seed)
    np.save(out, np.array(log)); print(f"{mname} {method} seed {seed}: regret {log[0]:.3f} -> {log[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
