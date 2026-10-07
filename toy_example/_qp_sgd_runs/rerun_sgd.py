"""Re-run the PLAIN-SGD, 12800-call, 10-seed comparison of DFL_QP_Black_Box_new.ipynb (cell 16) and save every run as
_qp_sgd_runs/<exact|fitted>_<method>_s<seed>.npy, in the naming of _qp_adam_runs/ so the paper figure can read either
directory.  Adam is cell 10/13 and _qp_adam_runs/rerun.py; this is the same methods and the same calls, each at the
learning rate cell 16 picked for it, with the Q rows fitted from the black box ("fitted") and, for reference, the same
rows read off the true quadratic ("exact").
usage (from toy_example/): python _qp_sgd_runs/rerun_sgd.py"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.dirname(HERE)); sys.path.insert(0, os.getcwd())
nb = json.load(open("DFL_QP_Black_Box_new.ipynb"))
cell = lambda i: "".join(nb["cells"][i]["source"])
for i in (0, 1, 2, 5): exec(cell(i))
c10 = cell(10); exec(c10[:c10.index("seeds = range(10)")])           # makers, groups, CALLS, LR
c13 = cell(13); exec(c13[:c13.index("def job_f")])                   # train_sgd, fitted_makers
c16 = cell(16); exec(c16[:c16.index("def job_s")])                   # SGD_LR, sgd_makers
import numpy as np
from multiprocessing import get_context
seeds = range(10)
tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
#      file kind, file name (as _qp_adam_runs spells it), the sgd_makers key
JOBS = ([("exact", n, n) for n in ("REINFORCE", "RLOO", "GRPO", "OTB", "LAX", "exact gradient")]
        + [("fitted", n, n) for n in ("+ Q", "+ Q + row-wise Δ")]
        + [("exact", n, n + " (exact)") for n in ("+ Q", "+ Q + row-wise Δ")])


def job(args):
    kind, name, key, seed = args; K = groups.get(key, 1); t0 = time.time()
    log = train_sgd(shared(seed), sgd_makers[key](), CALLS, SGD_LR[key], seed, every=100, K=K)
    np.save(os.path.join(HERE, f"{kind}_{tag(name)}_s{seed}.npy"), np.array(log))
    return kind, name, seed, np.array(log), time.time() - t0


todo = [(k, n, key, s) for k, n, key in JOBS for s in seeds]
todo = [j for j in todo if not os.path.exists(os.path.join(HERE, f"{j[0]}_{tag(j[1])}_s{j[3]}.npy"))]
print(f"{len(todo)} runs to do", flush=True)
if todo:
    with get_context("fork").Pool(min(60, len(todo))) as pool:
        for kind, name, seed, log, dt in pool.imap_unordered(job, todo):
            print(f"{kind:7s} {name:20s} seed {seed}: mean {log.mean():.3f} final {log[-1]:.3f}  ({dt:.0f}s)", flush=True)
import glob
print("\nregret: mean over training / final     (lr, and how many of the 10 seeds blew up)")
for kind, name, key in JOBS:
    L = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(HERE, f"{kind}_{tag(name)}_s*.npy")))])
    print(f"  {kind:7s} {name:22s} {L.mean():.3f} / {L[:, -1].mean():.3f}   lr {SGD_LR[key]:<7g} blew up {int((L[:, -1] >= 1.0).sum())}/{len(L)}   n={len(L)}")
