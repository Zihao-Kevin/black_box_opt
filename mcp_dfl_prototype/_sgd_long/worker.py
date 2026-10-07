"""Plain SGD at lr 0.06 (SGD_LR=0.03 SGD_OUT=_sgd_long/lr0.03 for the lower rate), run until the curve is flat.  One process per GPU; each claims (method, seed) jobs with a lock file.
usage: python worker.py <gpu>        (from any directory; runs in the notebook's directory, on a snapshot of its code cells)
A run stops when the mean P(success) over the last WIN steps moved by less than TOL against the WIN steps before, checked every
10 steps after MIN_STEPS, or at MAX_STEPS.  A collapsed run goes flat at 0 and stops by the same rule.  Every run's log is
(episodes, P(success), noise / REINFORCE's) per step, as in the notebook; partial logs are written every 20 steps."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.dirname(HERE)); sys.path.insert(0, os.getcwd())     # mcp_live, baselines
MIN_STEPS, MAX_STEPS, WIN, TOL = 120, int(os.environ.get("SGD_MAX_STEPS", 400)), 40, 0.01     # SGD_MAX_STEPS: a short smoke test
ORDER = ["Q + rowΔ", "Q", "Q + Δ", "REINFORCE", "exact gradient", "Q (exact)", "Q + Δ (exact)", "Q + rowΔ (exact)",
         "V", "PSB", "RELAX", "OTB", "RLOO", "GRPO"]
tag = lambda m, s: f"{m.replace(' ', '').replace('+', 'p').replace('(', '').replace(')', '')}_s{s}"

nb = json.load(open(os.path.join(HERE, "_snapshot.ipynb")))
cell = lambda i: "\n".join(l for l in "".join(nb["cells"][i]["source"]).split("\n") if "await " not in l)   # the agent re-run (RUN_AGENT = False) is a notebook-only await
job = "lint"                                                  # cell 7 ends with a check on this task

def graded_table(R, cap=False):
    """partial credit: for every configuration the largest, over the task's exact-match winners w, fraction of w's attached servers
    that the configuration also attaches.  The cap is ignored (cap=True: 0 over the cap, which makes every later slot matter again
    and leaves the table close to the lenient one).  A slot that no winner uses does not count, so the reward is decided by the
    one or two slots the task needs: nothing is left inside a prefix past them, and there V = Q."""
    out = {}
    for t, r in R.items():
        winners = [[(j, d) for j, d in enumerate(CONFIGS[i]) if d > 0] for i in np.flatnonzero(r > 0)]
        g = np.zeros(len(CONFIGS))
        for i, c in enumerate(CONFIGS):
            if cap and OVER_CAP[i]: continue
            g[i] = max((sum(c[j] == d for j, d in w) / len(w) for w in winners if w), default=0.0)
        out[t] = g
    return out

for i in (1, 2, 4, 7, 9, 12, 14):                              # setup, rewards, model, tree, tables, baselines, model-based methods
    exec(cell(i))
    if i == 1 and os.environ.get("SGD_STRICT", "1") == "0": STRICT = False     # SGD_STRICT=0: the lenient reward (the checker alone), read by cell 2's reward()
    if i == 2:
        REWARD_BASE = REWARD                                                     # the 0/1 table (strict or lenient); logged as the 4th column
        if os.environ.get("SGD_REWARD", "").startswith("graded"): REWARD = graded_table(REWARD, cap=os.environ["SGD_REWARD"] == "graded_cap")   # SGD_REWARD=graded (cap ignored) | graded_cap
    if i == 9 and os.environ.get("SGD_REWARD") == "graded":                     # the graded reward ignores the cap, so the fitted table must not zero over-cap configurations
        Seen.f_hat = lambda self, prior=1.0: self.sum / (self.n + prior)
exec(cell(17)[:cell(17).index("logs = {}")])                   # TRAIN, EPISODES, BASELINES use, and train() itself, not the run loop
import numpy as np, torch
LR_SGD, SEEDS_SGD = float(os.environ.get("SGD_LR", 0.06)), range(*[int(x) + i for i, x in enumerate(os.environ.get("SGD_SEEDS", "0-4").split("-"))])   # SGD_SEEDS=a-b inclusive   # after the notebook's cells: cell 17 sets LR and SEEDS for the Adam runs; SGD_LR / SGD_OUT pick another lr and its directory
FIXED = int(os.environ.get("SGD_FIXED_STEPS", 0))                    # > 0: a fixed budget of FIXED steps (FIXED // K for the group baselines, as the notebook does), no stopping rule
JOBS = [(float(x.split(":")[0]), x.split(":")[1]) for x in os.environ["SGD_JOBS"].split(",")] if "SGD_JOBS" in os.environ else [(LR_SGD, os.environ.get("SGD_OUT", HERE))]
ORDER = os.environ["SGD_METHODS"].split(",") if "SGD_METHODS" in os.environ else ORDER      # a subset of the methods, most important first
SEEDS_SGD = range(*[int(x) for x in os.environ["SGD_SEEDS"].split("-")]) if "SGD_SEEDS" in os.environ else SEEDS_SGD     # "5-20": seeds 5..19
OFFLINE = os.environ.get("SGD_OFFLINE")                          # offline.py's dataset (a json of [task, configuration] episodes), pre-filled into the fitted reward tables
WARM = os.environ.get("SGD_WARM")                                # a directory of warm_s{seed}.pt (warm.py): every method starts a seed from the same supervised warm start

def train_long(method, seed, lr=LR_SGD, out_dir=HERE):                           # the notebook's train(), plus the stopping rule; episodes = 1 per job per step
    K = BASELINES.get(method, EPISODES); episodes = K
    reset(); rng = np.random.default_rng(seed); gen = torch.Generator("cuda").manual_seed(seed)
    if WARM:
        wf = os.path.join(WARM, f"warm_s{seed}.pt"); wf = wf if os.path.exists(wf) else os.path.join(WARM, "warm.pt")   # one warm start per seed (warm.py), or one for all (offline.py)
        for p, w in zip(params, torch.load(wf)): p.data.copy_(w.to(p.device))
    opt = torch.optim.Adam(params, lr=lr) if os.environ.get("SGD_OPT") == "adam" else torch.optim.SGD(params, lr=lr * episodes)   # SGD_OPT=adam: the notebook's Adam runs, from a warm start
    seen = {t: Seen() for t in TRAIN}; surrogates = {}; log = []; dead = False
    if OFFLINE:                                                                    # the fixed offline dataset (offline.py): every fitted table starts from its episodes, re-scored under the active reward
        for t, conf in json.load(open(OFFLINE))["episodes"]:
            if t in seen: seen[t].add(conf, float(REWARD[t][conf]))
    psb = {t: [torch.zeros(NP, device="cuda"), torch.zeros(NP, device="cuda")] for t in TRAIN}
    out = os.path.join(out_dir, tag(method, seed))
    for step in range((FIXED or MAX_STEPS) // K + 1):
        grad, success, var, var_rf, success_base = 0.0, [], 0.0, 0.0, []
        for t in (TRAIN if not dead else []):
            tree = build_tree(t); f = leaf_rewards(tree, REWARD[t])
            if not torch.isfinite(tree.pb).all():
                print(f"  {method}, seed {seed}: the policy blew up at episode {step * episodes * len(TRAIN)}", flush=True); dead = True; del tree; break
            success.append(float(tree.pb @ f)); var_rf += noise(tree, torch.zeros_like(tree.P), f); success_base.append(float(tree.pb @ leaf_rewards(tree, REWARD_BASE[t])))
            c = xi = None
            if method in BASELINES: var = np.nan
            elif method in MODEL: var += model_noise(method, tree, f, seen[t], psb[t])
            else: c, xi = control_variate(method, tree, seen[t], f); var += noise(tree, c, f) if xi is None else noise_row(tree, c, f, xi)
            ys = sample(tree, episodes, rng)
            if method in BASELINES: g = baseline_weights(method, tree, ys, f, surrogates, t, gen).float() @ tree.Z
            elif method in MODEL: g = model_grad(method, tree, ys, f, seen[t], psb[t])
            else:
                g = weights(tree, c, f)[ys].mean(0).float() @ tree.Z
                if xi is not None: g = g - torch.stack([row_correction(tree, xi, y) for y in ys]).mean(0).float()
            grad = grad + g / len(TRAIN)
            for y in ys: seen[t].add(tree.conf[y], float(f[y]))
            del tree
        if dead: log.append((step * episodes * len(TRAIN), 0.0, np.nan, 0.0))
        else:
            log.append((step * episodes * len(TRAIN), np.mean(success), var / max(var_rf, 1e-30), np.mean(success_base)))   # 4th column: P(success) under the 0/1 table
            opt.zero_grad()
            for p, g in zip(params, (-grad).split([p.numel() for p in params])): p.grad = g.view_as(p)
            opt.step()
        P = np.array([l[1] for l in log]); n = len(log)
        if n % 20 == 0: np.save(out + ".partial.npy", np.array(log))
        if not FIXED and n * K >= MIN_STEPS and n % 10 == 0 and abs(P[-WIN // K:].mean() - P[-2 * WIN // K:-WIN // K].mean()) < TOL: break
    return np.array(log)

t0 = time.time()
jobs = [(m, lr, d, s) for m in ORDER for lr, d in JOBS for s in SEEDS_SGD]
me, n = (int(x) for x in os.environ.get("SGD_WORKER", "0/1").split("/"))     # "i/n": this worker takes every n-th job from i (a static split; lock files were not exclusive on this NFS)
for m, lr, d, s in jobs[me::n]:
            os.makedirs(d, exist_ok=True); out = os.path.join(d, tag(m, s))
            if os.path.exists(out + ".npy"): continue
            t1 = time.time(); log = train_long(m, s, lr=lr, out_dir=d); np.save(out + ".npy", log)
            if os.path.exists(out + ".partial.npy"): os.remove(out + ".partial.npy")
            print(f"lr {lr} {m:<17} seed {s}: {len(log) - 1} steps, {int(log[-1, 0])} episodes, P(success) {log[0, 1]:.3f} -> {log[-1, 1]:.3f}  ({time.time() - t1:.0f}s)", flush=True)
print(f"worker on GPU {sys.argv[1]} done ({(time.time() - t0) / 60:.0f} min)", flush=True)
os._exit(0)
