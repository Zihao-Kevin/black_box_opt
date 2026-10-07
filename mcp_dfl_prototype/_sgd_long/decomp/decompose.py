"""Per-position variance decomposition of the estimators, exact (the analogue of Tucker et al. 2018, Fig. 1).
Var[g_hat] = sum_n E||xi_n||^2 over the positions n of the answer (martingale increments); at every entry e = (prefix, branch)
the scalar residual removes the part of xi_e along the branch's own score, the row-wise residual the part inside the
row-input span.  Evaluated on the policy of the Q + rowDelta run (lr 0.045, warm start, seed 0) at 0 / 100 / 200 / 400
episodes, on the four training tasks, with the true reward table (the geometry) and the run's own fitted table.
usage: python decompose.py <gpu>      writes decomp.npz next to this file."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); os.chdir(ROOT); sys.path.insert(0, ROOT)
nb = json.load(open(os.path.join(os.path.dirname(HERE), "_snapshot.ipynb")))
cell = lambda i: "\n".join(l for l in "".join(nb["cells"][i]["source"]).split("\n") if "await " not in l)
job = "lint"

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

for i in (1, 2, 4, 7, 9, 12, 14):
    exec(cell(i))
    if i == 1 and os.environ.get("DECOMP_STRICT", "1") == "0": STRICT = False    # DECOMP_STRICT=0: the lenient reward, read by cell 2's reward()
    if i == 2 and os.environ.get("DECOMP_REWARD", "").startswith("graded"): REWARD = graded_table(REWARD, cap=os.environ["DECOMP_REWARD"] == "graded_cap")   # DECOMP_REWARD=graded (cap ignored) | graded_cap
    if i == 9 and os.environ.get("DECOMP_REWARD") == "graded": Seen.f_hat = lambda self, prior=1.0: self.sum / (self.n + prior)   # no over-cap zeroing under the cap-ignored reward
GRADED_REWARD, METHOD_PATH = os.environ.get("DECOMP_REWARD", "").startswith("graded"), os.environ.get("DECOMP_METHOD", "Q + rowΔ")   # the run whose policy path is decomposed
exec(cell(17)[:cell(17).index("logs = {}")])
import numpy as np, torch
LR, SEED, SNAP = float(os.environ.get("DECOMP_LR", 0.045)), int(os.environ.get("DECOMP_SEED", 0)), (0, 25, 50, 100)   # the run to replay (DECOMP_LR, DECOMP_SEED) and the steps (x4 episodes) to decompose at
STEPS_ = [k for k, t in enumerate(TEMPLATE) if t or FREE_TEXT]  # the scored positions, as in build_tree
NODES = [(k, d) for k in STEPS_ for d in itertools.product(range(4), repeat=TEMPLATE[:k].count(None))]
POS = np.array([STEPS_.index(k) for k, _ in NODES])               # node -> position in the answer
DIGIT = [i for i, k in enumerate(STEPS_) if TEMPLATE[k] is None]  # the positions that write a digit
METHODS = ["REINFORCE", "V", "Q", "Q + Δ", "Q + rowΔ"]

def per_entry(tree, method, f_table, f_true):
    """E-vector: this entry's contribution w_e * (what is left of ||xi_e||^2 after the method's residual), exact."""
    w = tree.pb @ tree.taken
    if method == "REINFORCE": c = torch.zeros_like(tree.P)
    else:
        Q = Q_table(tree, f_table)
        c = V_table(tree, Q) if method == "V" else Q + Delta_table(tree, f_table, Q) if method == "Q + Δ" else Q
    x = increments(tree, c, f_true); left = ((x @ tree.K) * x).sum(1)
    if method == "Q + rowΔ":                                       # the residual is built from the table's increments, applied to the true ones
        xi = increments(tree, c, f_table)
        for a, b, K in zip(row_parts(tree, x), row_parts(tree, xi), (tree.KA, tree.KB)): left = left + ((b @ K) * b).sum(1) - 2 * ((a @ K) * b).sum(1)
    return (w * left).cpu().numpy()

out = {}                                                        # (stage, table, method, task) -> per-position variance (16,)
reset(); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
WARM, OFFLINE, OUT_DIR = os.environ.get("DECOMP_WARM", os.path.join(os.path.dirname(HERE), "warm")), os.environ.get("DECOMP_OFFLINE"), os.environ.get("DECOMP_OUT", HERE)   # as SGD_WARM / SGD_OFFLINE in worker.py
wf = os.path.join(WARM, f"warm_s{SEED}.pt"); wf = wf if os.path.exists(wf) else os.path.join(WARM, "warm.pt")
for p, wv in zip(params, torch.load(wf)): p.data.copy_(wv.to(p.device))
opt = torch.optim.SGD(params, lr=LR); seen = {t: Seen() for t in TRAIN}; t0 = time.time()
if OFFLINE:
    for t, conf in json.load(open(OFFLINE))["episodes"]:
        if t in seen: seen[t].add(conf, float(REWARD[t][conf]))
os.makedirs(OUT_DIR, exist_ok=True)
for step in range(max(SNAP) + 1):
    grad, success = 0.0, []
    for t in TRAIN:
        tree = build_tree(t); f = leaf_rewards(tree, REWARD[t]); success.append(float(tree.pb @ f))
        if step in SNAP:
            fh = leaf_rewards(tree, seen[t].f_hat()); node_pos = POS[tree.node.cpu().numpy()]
            for m in METHODS:
                for name, tab in (("exact", f), ("fitted", fh)):
                    v = per_entry(tree, m, tab, f); out[step, name, m, t] = np.bincount(node_pos, weights=v, minlength=len(STEPS_))
        c, xi = control_variate(METHOD_PATH, tree, seen[t], f)         # the replayed run (DECOMP_METHOD; default Q + rowΔ with the fitted table)
        ys = sample(tree, 1, rng)
        g = weights(tree, c, f)[ys].mean(0).float() @ tree.Z - torch.stack([row_correction(tree, xi, y) for y in ys]).mean(0).float()
        grad = grad + g / len(TRAIN)
        for y in ys: seen[t].add(tree.conf[y], float(f[y]))
        del tree
    if step in SNAP: print(f"step {step} ({4 * step} episodes): E[reward] {np.mean(success):.3f}   ({time.time() - t0:.0f}s)   [{'graded' if GRADED_REWARD else 'strict' if STRICT else 'lenient'} reward, path {METHOD_PATH}]", flush=True)
    opt.zero_grad()
    for p, g in zip(params, (-grad).split([p.numel() for p in params])): p.grad = g.view_as(p)
    opt.step()
np.savez(os.path.join(OUT_DIR, ("decomp_graded" if GRADED_REWARD else "decomp" if STRICT else "decomp_lenient") + (f"_lr{LR:g}" if LR != 0.045 else "") + (f"_s{SEED}" if SEED or OUT_DIR != HERE else "") + ("_exactpath" if "exact" in METHOD_PATH else "") + ".npz"), keys=np.array([f"{s}|{n}|{m}|{t}" for s, n, m, t in out]), vals=np.stack(list(out.values())),
         positions=np.array(STEPS_), digit=np.array(DIGIT), snap=np.array(SNAP))
print("saved", len(out), "entries"); os._exit(0)
