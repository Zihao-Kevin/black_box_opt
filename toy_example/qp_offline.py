"""The QP figure's reward-model rows with a fixed offline dataset (qp_offline.py).
Protocol: DFL_QP_Black_Box_new.ipynb cell 13 as it is (Adam, 12,800 calls, 10 seeds, lr 0.005, one quadratic reward model per day fitted by
least squares with a leave-one-out ridge), except that every reward model starts from a fixed offline dataset: N_OFF nudges per day drawn
once from the initial policy (the zero-initialised head predicts 0 for every asset, so b ~ N(0, sigma^2 I)) and scored by the black box.
LAX's surrogate and replay buffer start from the same dataset.  REINFORCE, RLOO, GRPO, OTB and the exact gradient use no reward model and
are copied unchanged from _qp_adam_runs/.  Output: _qp_adam_runs_offline/, the same file names as _qp_adam_runs/.
usage: QP_OFFLINE=50 python qp_offline.py [n_procs=40]"""
import os, sys, json, shutil, numpy as np, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
nb = json.load(open("DFL_QP_Black_Box_new.ipynb")); src = lambda i: "".join(nb["cells"][i]["source"])
exec(src(0)); exec(src(1)); exec(src(2)); exec(src(5))                      # data, black box, model, the estimators and train()
s = src(10); exec(s[:s.index("seeds = range(10)")])                          # nudge, reinforce, grouped, lax, CALLS, LR, makers, groups
s = src(13); exec(s[:s.index("fitted_makers")])                              # feats, RewardModel, fitted, train_sgd
N_OFF, OUT = int(os.environ.get("QP_OFFLINE", 50)), os.path.join(HERE, "_qp_adam_runs_offline"); os.makedirs(OUT, exist_ok=True)
gen = torch.Generator().manual_seed(2026)
OFF = {i: sigma * torch.randn(N_OFF, d, generator=gen) for i in range(n_days)}                                   # b ~ p_theta0, the initial policy
OFF_F = {i: torch.tensor([black_box(i, b.numpy()) for b in OFF[i]], dtype=torch.float32) for i in range(n_days)}  # scored once
np.savez(os.path.join(OUT, "offline_data.npz"), n_per_day=N_OFF, seed=2026, **{f"b{i}": OFF[i].numpy() for i in range(n_days)}, **{f"f{i}": OFF_F[i].numpy() for i in range(n_days)})

def rm_offline():                                                            # a reward model that starts from the offline dataset
    rm = RewardModel()
    for i in range(n_days):
        for b, f in zip(OFF[i], OFF_F[i]): rm.add(i, b, f)
    return rm

def lax_offline(window=128, steps=8):                                        # cell 10's lax(), with the replay buffer and the surrogate started from the offline dataset
    replay = {}
    def gradient(model, day, seed):
        J, mu, b, f = nudge(model, day, seed, 1)
        if day not in surrogates:
            surrogates[day] = Surrogate(d); replay[day] = (list(OFF[day]), list(OFF_F[day]))
            B, F = (torch.stack(v[-window:]) for v in replay[day])
            for _ in range(steps): surrogates[day].fit(lax_weights(surrogates[day], mu, B, F, sigma), J @ J.T)
        c = surrogates[day]; g = lax_weights(c, mu, b, f, sigma).detach() @ J
        replay[day][0].append(b[0]); replay[day][1].append(f[0]); B, F = (torch.stack(v[-window:]) for v in replay[day])
        for _ in range(steps): c.fit(lax_weights(c, mu, B, F, sigma), J @ J.T)
        return g
    return gradient

makers_off = {"+ Q": lambda: fitted(None, rm_offline()), "+ Q + Δ": lambda: fitted("scalar", rm_offline()), "+ Q + row-wise Δ": lambda: fitted("row", rm_offline()), "LAX": lambda: lax_offline()}
tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
fname = lambda name, seed: os.path.join(OUT, f"{'exact' if name == 'LAX' else 'fitted'}_{tag(name)}_s{seed}.npy")

def job(args):
    name, seed = args
    return name, seed, train(shared, makers_off[name](), calls=CALLS, lr=LR[1], seed=seed, every=100)

if __name__ == "__main__":
    from multiprocessing import get_context
    import time; t0 = time.time()
    print(f"offline dataset: {N_OFF} nudges per day, regret mean {np.mean([OFF_F[i].mean() for i in range(n_days)]):.3f}", flush=True)
    seeds = range(10); todo = [(n, s) for n in makers_off for s in seeds if not os.path.exists(fname(n, s))]
    with get_context("fork").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 40) as pool:
        for name, seed, log in pool.imap_unordered(job, todo):
            np.save(fname(name, seed), np.array(log)); print(f"{name:18s} seed {seed}: regret {log[0]:.3f} -> {log[-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    for f in os.listdir(os.path.join(HERE, "_qp_adam_runs")):                # the rows without a reward model, unchanged
        if any(f.startswith(f"exact_{tag(n)}_s") for n in ("REINFORCE", "RLOO", "GRPO", "OTB", "exact gradient")): shutil.copy(os.path.join(HERE, "_qp_adam_runs", f), OUT)
    print("done", flush=True)
