"""Biased vs unbiased on the same fitted reward model, QP toy, the paper's Adam protocol (cell 13: 12,800 calls, lr LR[1], 10 seeds).
The reward model f-hat (one quadratic per day, LOO-ridge least squares) is fitted once from N_OFF offline nudges per day and FROZEN.
  DM (frozen)       grad E_theta[f-hat]  in closed form (Gaussian policy, quadratic f-hat): biased, pays nothing during training
  Q + rowΔ (frozen) the paper's estimator with f-hat as the control variate only, the paid score is the true f: unbiased
  DM (online)       the direct method with one paid call per step added to f-hat (a reference)
usage: QP_OFFLINE=50 python qp_frozen.py [n_procs]      -> toy_example/_qp_frozen_runs/n{N_OFF}_{name}_s{seed}.npy"""
import os, sys, json, numpy as np, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
nb = json.load(open("DFL_QP_Black_Box_new.ipynb")); src = lambda i: "".join(nb["cells"][i]["source"])
exec(src(0)); exec(src(1)); exec(src(2)); exec(src(5))
s = src(10); exec(s[:s.index("seeds = range(10)")])
s = src(13); s13 = s[:s.index("fitted_makers")]; exec(s13)
frozen_src = s13[s13.index("def fitted(kind, rm):"):].replace("def fitted(kind, rm):", "def fitted_frozen(kind, rm):").replace("rm.add(day, b[0], f[0]); return", "return")
assert "rm.add" not in frozen_src; exec(frozen_src)
N_OFF, OUT = int(os.environ.get("QP_OFFLINE", 50)), os.path.join(HERE, "_qp_frozen_runs"); os.makedirs(OUT, exist_ok=True)
gen = torch.Generator().manual_seed(2026)
OFF = {i: sigma * torch.randn(N_OFF, d, generator=gen) for i in range(n_days)}
OFF_F = {i: torch.tensor([black_box(i, b.numpy()) for b in OFF[i]], dtype=torch.float32) for i in range(n_days)}

def rm_offline():
    rm = RewardModel()
    for i in range(n_days):
        for b, f in zip(OFF[i], OFF_F[i]): rm.add(i, b, f)
    return rm

def dm(rm, online):
    def gradient(model, day, seed):
        if online:                                                          # one paid call per step, added to the reward model
            J, mu, b, f = nudge(model, day, seed, 1); rm.add(day, b[0], f[0])
        fb = rm.box(); J, mu = (x.double() for x in box.jac(model, day))     # grad E[f-hat] = J' (g + 2 H mu): exact under the Gaussian policy
        return ((fb.g[day] + 2 * fb.H[day] @ mu) @ J).float()
    return gradient

makers = {"DM (frozen)": lambda: dm(rm_offline(), False), "Q + rowΔ (frozen)": lambda: fitted_frozen("row", rm_offline()), "DM (online)": lambda: dm(rm_offline(), True)}
tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
fname = lambda name, seed: os.path.join(OUT, f"n{N_OFF}_{tag(name)}_s{seed}.npy")

def job(args):
    name, seed = args
    return name, seed, train(shared, makers[name](), calls=CALLS, lr=LR[1], seed=seed, every=100)

if __name__ == "__main__":
    from multiprocessing import get_context
    import time; t0 = time.time()
    rm = rm_offline(); fb = rm.box()
    err = np.mean([float(((fb.H[i] - box.H[i]) ** 2).sum() / (box.H[i] ** 2).sum()) for i in range(n_days)]) if hasattr(box, "H") else float("nan")
    print(f"offline dataset: {N_OFF} nudges per day (quadratic has {1 + d + d * (d + 1) // 2} coefficients); relative error of f-hat's H {err:.3f}", flush=True)
    seeds = range(10); todo = [(n, s) for n in makers for s in seeds if not os.path.exists(fname(n, s))]
    with get_context("fork").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 30) as pool:
        for name, seed, log in pool.imap_unordered(job, todo):
            np.save(fname(name, seed), np.array(log)); print(f"{name:18s} seed {seed}: regret {log[0]:.3f} -> min {min(log):.3f} -> final {log[-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)
