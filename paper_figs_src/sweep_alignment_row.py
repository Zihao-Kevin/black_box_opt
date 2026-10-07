"""The alignment grid again, with the row-wise Delta* as well as the scalar one.

sweep_alignment.py records var_qd = Var(Q + Delta*) for the scalar Delta, whose gain over Q is the paper's closed form
1 - mean cos^2.  The paper now writes Delta* for the row-wise correction, so the figure has to plot that instead:
var_qd_row = Var(Q + row-wise Delta*), from qp_rowwise.RowAccount.var_grouped("row"), on exactly the same grid.
usage: python sweep_alignment_row.py     (writes alignment_grid_row.npz next to this file; loads it if present)"""
import os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); TOY = os.path.join(os.path.dirname(HERE), "toy_example")
sys.path.insert(0, HERE); sys.path.insert(0, TOY)
import sweep_alignment as sw
from qp_continuous import ContinuousQP, MeanNet
from qp_rowwise import RowAccount

OUT = os.path.join(HERE, "alignment_grid_row.npz")
KEYS = ("d", "seed", "pos_scale", "cos2", "var_rf", "var_q", "var_qd", "var_qd_row")


def one(d, seed, pos_scale):
    X, R, S = sw.data(d, seed); cqp = ContinuousQP.from_data(X, R, S, lam=sw.LAM, sigma=sw.SIGMA)
    torch.manual_seed(seed); net = MeanNet(cqp.p, cqp.d, h=8, pos_scale=pos_scale, train_pos=False, seed=seed)
    acc = [RowAccount(cqp, net, i) for i in range(sw.N_DAYS)]
    cos2 = 0.0
    for a in acc:
        Jd = a.Jn / np.maximum(np.linalg.norm(a.Jn, axis=1, keepdims=True), 1e-300); C = Jd @ Jd.T; off = ~np.eye(d, dtype=bool)
        cos2 += (C[off] ** 2).mean() / sw.N_DAYS
    return dict(cos2=cos2, var_rf=sum(a.var("rf") for a in acc), var_q=sum(a.var("prefix") for a in acc),
                var_qd=sum(a.var_grouped("scalar") for a in acc),          # == Account.var_delta("prefix","pre"), the closed form's Delta
                var_qd_row=sum(a.var_grouped("row") for a in acc))


def _job(a): return a, one(*a)


def load(workers=24):
    if os.path.exists(OUT): return dict(np.load(OUT))
    from multiprocessing import get_context
    torch.set_num_threads(1)
    grid = [(d, s, ps) for d in sw.DS for ps in sw.GRID_SCALES for s in sw.SEEDS]
    res, t0 = {}, time.time()
    with get_context("spawn").Pool(workers) as pool:
        for k, (a, r) in enumerate(pool.imap_unordered(_job, grid), 1):
            res[a] = r
            print(f"[{k:>3}/{len(grid)}] d {a[0]:>2} seed {a[1]} pos {a[2]:<4}: cos2 {r['cos2']:.3f}  "
                  f"scalar {1 - r['var_qd'] / r['var_q']:.3f}  row {1 - r['var_qd_row'] / r['var_q']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    rows = {k: [] for k in KEYS}
    for (d, s, ps) in grid:
        rows["d"].append(d); rows["seed"].append(s); rows["pos_scale"].append(ps)
        for k, v in res[(d, s, ps)].items(): rows[k].append(v)
    out = {k: np.array(v) for k, v in rows.items()}; np.savez(OUT, **out); return out


if __name__ == "__main__":
    g = load()
    print("\ngrid points:", len(g["d"]))
    print("mean gain  scalar %.3f   row-wise %.3f   mean cos2 %.3f" % ((1 - g["var_qd"] / g["var_q"]).mean(), (1 - g["var_qd_row"] / g["var_q"]).mean(), g["cos2"].mean()))
