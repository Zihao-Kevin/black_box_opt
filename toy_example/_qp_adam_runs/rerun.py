"""Re-run the Adam, 12800-call, 10-seed comparison of DFL_QP_Black_Box_new.ipynb (cells 10 and 13) and save every run as
_qp_adam_runs/<exact|fitted>_<method>_s<seed>.npy, so the paper figure can be drawn without the notebook kernel.
usage (from toy_example/): python _qp_adam_runs/rerun.py"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.dirname(HERE)); sys.path.insert(0, os.getcwd())
nb = json.load(open("DFL_QP_Black_Box_new.ipynb"))
cell = lambda i: "".join(nb["cells"][i]["source"])
for i in (0, 1, 2, 5): exec(cell(i))
c10 = cell(10); exec(c10[:c10.index("seeds = range(10)")])
c13 = cell(13); exec(c13[:c13.index("def job_f")])
import numpy as np
from multiprocessing import get_context
seeds = range(10)
EXACT = ["REINFORCE", "RLOO", "GRPO", "OTB", "LAX", "+ Q", "+ Q + row-wise Δ", "exact gradient"]
FITTED = ["+ Q", "+ Q + Δ", "+ Q + row-wise Δ"]
tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
def job(args):
    kind, name, seed = args; K = groups.get(name, 1); t0 = time.time()
    g = makers[name]() if kind == "exact" else fitted_makers[name]()
    log = train(shared, g, calls=CALLS, lr=LR[K], seed=seed, every=100, K=K)
    np.save(os.path.join(HERE, f"{kind}_{tag(name)}_s{seed}.npy"), np.array(log))
    return kind, name, seed, np.array(log), time.time() - t0
todo = [("exact", n, s) for n in EXACT for s in seeds] + [("fitted", n, s) for n in FITTED for s in seeds]
todo = [j for j in todo if not os.path.exists(os.path.join(HERE, f"{j[0]}_{tag(j[1])}_s{j[2]}.npy"))]
print(f"{len(todo)} runs to do", flush=True)
runs = {}
with get_context("fork").Pool(min(60, len(todo) or 1)) as pool:
    for kind, name, seed, log, dt in pool.imap_unordered(job, todo):
        runs.setdefault((kind, name), {})[seed] = log
        print(f"{kind:7s} {name:20s} seed {seed}: mean {log.mean():.3f} final {log[-1]:.3f}  ({dt:.0f}s)", flush=True)
print("\nregret: mean over training / final (notebook printed values in brackets)")
ref = {("exact", "REINFORCE"): "0.351 / 0.341", ("exact", "RLOO"): "0.332 / 0.319", ("exact", "GRPO"): "0.340 / 0.333", ("exact", "OTB"): "0.335 / 0.321",
       ("exact", "LAX"): "0.337 / 0.324", ("exact", "+ Q"): "0.316 / 0.293", ("exact", "+ Q + row-wise Δ"): "0.240 / 0.225", ("exact", "exact gradient"): "0.233 / 0.226",
       ("fitted", "+ Q"): "0.319 / 0.293", ("fitted", "+ Q + Δ"): "0.263 / 0.222", ("fitted", "+ Q + row-wise Δ"): "0.254 / 0.212"}
import glob
for kind, names in (("exact", EXACT), ("fitted", FITTED)):
    for n in names:
        L = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(HERE, f"{kind}_{tag(n)}_s*.npy")))])
        print(f"  {kind:7s} {n:20s} {L.mean():.3f} / {L[:, -1].mean():.3f}   [{ref[(kind, n)]}]   n={len(L)}")
