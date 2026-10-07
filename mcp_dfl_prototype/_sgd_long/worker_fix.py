"""worker.py with two fixes, for the group baselines and RELAX (a separate file so the shared worker stays as it is):
  1. the same learning rate per STEP for every method (worker.py scales it by the group size K so that a K-episode step
     spends K episodes' worth), and the same number of steps for every method, so a group method runs K times the episodes;
     the log has the episode count per step, so the equal-episode point can be read off.
  2. RELAX's surrogate starts at exactly zero (so RELAX begins as plain REINFORCE) and is fitted by least-squares regression
     of the reward on the visited nodes' features over the run's own episodes, instead of one Adam step per episode on the
     single-sample variance ||g_hat||^2 = w K w from a random init.
Same environment variables as worker.py."""
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
for i in (1, 2, 4, 7, 9, 12, 14):                              # setup, rewards, model, tree, tables, baselines, model-based methods
    exec(cell(i))
exec(cell(17)[:cell(17).index("logs = {}")])                   # TRAIN, EPISODES, BASELINES use, and train() itself, not the run loop
import numpy as np, torch
from baselines import cond_gumbel

class RegSurrogate(Surrogate):
    """fix 2: c_phi starts at 0 and is fitted by regression of f on the visited nodes' features, over all episodes so far."""
    def __init__(self, dim, hidden=64, lr=1e-2):
        super().__init__(dim, hidden, lr); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
        self.X, self.y = [], []
    def fit_reg(self, feats, f, epochs=5):
        self.X.append(feats.detach()); self.y.append(torch.full((len(feats),), float(f), device=feats.device))
        X, y = torch.cat(self.X), torch.cat(self.y)
        for _ in range(epochs):
            loss = ((self(X) - y) ** 2).mean(); self.opt.zero_grad(); loss.backward(); self.opt.step()

def relax_weights_reg(c, logp, ent, b, f, E, state, gen=None):
    """baselines.relax_weights, returning also the features the surrogate saw at the visited nodes (for the regression)."""
    valid, A = ent >= 0, ent.shape[-1]; live = valid.any(-1)
    lp = torch.where(valid, logp, 0.0).clamp_min(-60.0).requires_grad_()
    u, v = torch.rand(2, *ent.shape, generator=gen, device=ent.device, dtype=lp.dtype)
    feat = lambda z: torch.cat([state, torch.softmax(torch.where(valid, z, -1e4) / c.log_tau.exp(), -1)], -1)
    z = lp + (cond_gumbel(lp.detach(), b, u) - lp.detach())
    ft = feat(cond_gumbel(lp, b, v)); cz, czt = c(feat(z)) * live, c(ft) * live
    dlp = torch.autograd.grad((cz - czt).sum(), lp, create_graph=True)[0]
    coef = (f[:, None] - czt)[..., None] * (F.one_hot(b, A) * valid) + dlp
    w = torch.zeros(len(f), E, dtype=lp.dtype, device=ent.device)
    return w.scatter_add_(1, ent.clamp_min(0).flatten(1), (coef * valid).flatten(1)).mean(0), ft.detach()[live]

_baseline_weights = baseline_weights
def baseline_weights(method, tree, ys, f, surrogates, task, gen):
    if method != "RELAX": return _baseline_weights(method, tree, ys, f, surrogates, task, gen)
    ent, b = nodes(tree, ys); st = state(b)
    if task not in surrogates: surrogates[task] = RegSurrogate(st.shape[-1] + 5).cuda()
    c = surrogates[task]
    w, ft = relax_weights_reg(c, torch.log(tree.P.clamp_min(1e-300))[ent.clamp_min(0)].float(), ent, b, f[ys].float(), len(tree.P), st, gen)
    g = w.detach(); c.fit_reg(ft, float(f[ys][0]))
    return g
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
    opt = torch.optim.SGD(params, lr=lr)                    # fix 1: per step, not per episode
    seen = {t: Seen() for t in TRAIN}; surrogates = {}; log = []; dead = False
    if OFFLINE:                                                                    # the fixed offline dataset (offline.py): every fitted table starts from its episodes, re-scored under the active reward
        for t, conf in json.load(open(OFFLINE))["episodes"]:
            if t in seen: seen[t].add(conf, float(REWARD[t][conf]))
    psb = {t: [torch.zeros(NP, device="cuda"), torch.zeros(NP, device="cuda")] for t in TRAIN}
    out = os.path.join(out_dir, tag(method, seed))
    for step in range((FIXED or MAX_STEPS) + 1):           # fix 1: the same number of steps for every method
        grad, success, var, var_rf = 0.0, [], 0.0, 0.0
        for t in (TRAIN if not dead else []):
            tree = build_tree(t); f = leaf_rewards(tree, REWARD[t])
            if not torch.isfinite(tree.pb).all():
                print(f"  {method}, seed {seed}: the policy blew up at episode {step * episodes * len(TRAIN)}", flush=True); dead = True; del tree; break
            success.append(float(tree.pb @ f)); var_rf += noise(tree, torch.zeros_like(tree.P), f)
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
        if dead: log.append((step * episodes * len(TRAIN), 0.0, np.nan))
        else:
            log.append((step * episodes * len(TRAIN), np.mean(success), var / max(var_rf, 1e-30)))
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
