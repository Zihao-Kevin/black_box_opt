"""Sweep the sequence length d on the QP toy (DFL_QP_Black_Box_new.ipynb setting: 6 days, 6 features, lam 1.3, sigma 0.15,
shared network h=8, pos_scale 0.1, fixed positions), at initialization, exact accounting, no training.  Per (d, seed):
  cos2      mean over coordinate pairs of cos^2 between the score directions J_j = d mu_j / d theta (mean over days)
  cos       the same without the square (the paper's rho)
  var_rf    Var(REINFORCE with the exact mean baseline), pooled over days
  var_q, var_qd          Var(Q), Var(Q + Delta*) for the prefix (autoregressive) family, Delta with the prefix state, as in the notebook
  var_q_loo, var_qd_loo  the same for the leave-one-out family, the paper's eq. cos closed form
  cos2_w    the closed form's own weighted mean cos^2(w_k, J_k) over coordinates (loo), = 1 - var_qd_loo / var_q_loo
usage: python sweep_alignment.py   (writes alignment_sweep.npz next to this file; loads it if present)"""
import os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); TOY = os.path.join(os.path.dirname(HERE), "toy_example"); sys.path.insert(0, TOY)
torch.set_num_threads(4)
from qp_continuous import ContinuousQP, MeanNet
from qp_continuous_ar import Account
OUT, OUT_POS = os.path.join(HERE, "alignment_sweep.npz"), os.path.join(HERE, "alignment_sweep_pos.npz")
POS_SCALES = (0.1, 0.3, 1.0, 3.0)                                            # the paper's own sweep (tab:toy text), at d = 12
DS, SEEDS, P, N_DAYS, LAM, SIGMA = (4, 6, 8, 12, 16, 24), (0, 1, 2), 6, 6, 1.3, 0.15

def data(d, seed):                                                          # the notebook's cell 0 with d and the seed free
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_DAYS, P))
    true_returns = 1 / (1 + np.exp(-2 * X @ rng.standard_normal((d, P)).T / np.sqrt(P)))
    F = rng.standard_normal((d, d // 2)) / np.sqrt(d // 2)
    return X, true_returns, F @ F.T + 0.25 * np.eye(d)

def one(d, seed, pos_scale=0.1):
    X, R, S = data(d, seed); cqp = ContinuousQP.from_data(X, R, S, lam=LAM, sigma=SIGMA)
    torch.manual_seed(seed); net = MeanNet(cqp.p, cqp.d, h=8, pos_scale=pos_scale, train_pos=False, seed=seed)   # the notebook's shared network
    acc = [Account(cqp, net, i) for i in range(N_DAYS)]
    cos2 = cos = 0.0
    for a in acc:
        Jd = a.Jn / np.maximum(np.linalg.norm(a.Jn, axis=1, keepdims=True), 1e-300); C = Jd @ Jd.T; off = ~np.eye(d, dtype=bool)
        cos2 += (C[off] ** 2).mean() / N_DAYS; cos += C[off].mean() / N_DAYS
    vq_loo, vqd_loo, _ = cqp.exact(net)
    return dict(cos2=cos2, cos=cos, var_rf=sum(a.var("rf") for a in acc), var_q=sum(a.var("prefix") for a in acc),
                var_qd=sum(a.var_delta("prefix", "pre")[0] for a in acc), var_q_loo=vq_loo, var_qd_loo=vqd_loo, cos2_w=1 - vqd_loo / vq_loo)

def load(path=OUT, grid=None):
    """grid: list of (d, seed, pos_scale); cached in `path`"""
    if os.path.exists(path): return dict(np.load(path))
    grid = grid or [(d, s, 0.1) for d in DS for s in SEEDS]
    rows = {k: [] for k in ("d", "seed", "pos_scale", "cos2", "cos", "var_rf", "var_q", "var_qd", "var_q_loo", "var_qd_loo", "cos2_w")}; t0 = time.time()
    for d, s, ps in grid:
        r = one(d, s, ps); rows["d"].append(d); rows["seed"].append(s); rows["pos_scale"].append(ps)
        for k, v in r.items(): rows[k].append(v)
        print(f"d {d:>2} seed {s} pos {ps:<4}: cos2 {r['cos2']:.3f}  cos {r['cos']:.3f}  removed prefix {1 - r['var_qd'] / r['var_q']:.3f}  loo {r['cos2_w']:.3f}  "
              f"Q/rf {r['var_q'] / r['var_rf']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = {k: np.array(v) for k, v in rows.items()}; np.savez(path, **out); return out

def load_pos():
    return load(OUT_POS, [(12, s, ps) for ps in POS_SCALES for s in SEEDS])

OUT_GRID, GRID_SCALES = os.path.join(HERE, "alignment_grid.npz"), (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
def _job(a): return a, one(*a)
def load_grid(workers=24):
    """the full grid d x position scale x seed, computed in parallel and cached"""
    if os.path.exists(OUT_GRID): return dict(np.load(OUT_GRID))
    from multiprocessing import get_context
    grid = [(d, s, ps) for d in DS for ps in GRID_SCALES for s in SEEDS]; res = {}; t0 = time.time()
    torch.set_num_threads(1)
    with get_context("fork").Pool(workers) as pool:
        for k, (a, r) in enumerate(pool.imap_unordered(_job, grid)):
            res[a] = r
            if k % 20 == 0: print(f"{k + 1}/{len(grid)} ({time.time() - t0:.0f}s)", flush=True)
    rows = {k: [] for k in ("d", "seed", "pos_scale", "cos2", "cos", "var_rf", "var_q", "var_qd", "var_q_loo", "var_qd_loo", "cos2_w")}
    for d, s, ps in grid:
        rows["d"].append(d); rows["seed"].append(s); rows["pos_scale"].append(ps)
        for k, v in res[(d, s, ps)].items(): rows[k].append(v)
    out = {k: np.array(v) for k, v in rows.items()}; np.savez(OUT_GRID, **out); return out

if __name__ == "__main__":
    if "grid" in sys.argv:
        g = load_grid()
        print(f"{'pos':>5} {'d':>3} {'cos2':>6} {'cos':>6} {'rm prefix':>10} {'rm loo':>7} {'(d-1)r^2/(1+(d-2)r)':>20}")
        for ps in GRID_SCALES:
            for d in DS:
                m = (g["pos_scale"] == ps) & (g["d"] == d); r = g["cos"][m].mean()
                print(f"{ps:>5} {d:>3} {g['cos2'][m].mean():>6.3f} {r:>6.3f} {(1 - g['var_qd'][m] / g['var_q'][m]).mean():>10.3f} {g['cos2_w'][m].mean():>7.3f} {(d - 1) * r ** 2 / (1 + (d - 2) * r):>20.3f}")
        sys.exit()
    z = load(); zp = load_pos()
    print(f"\n{'pos_scale':>9} {'cos2':>7} {'cos':>7} {'removed (prefix)':>17} {'removed (loo)':>14}   (d = 12)")
    for ps in POS_SCALES:
        m = zp["pos_scale"] == ps
        print(f"{ps:>9} {zp['cos2'][m].mean():>7.3f} {zp['cos'][m].mean():>7.3f} {(1 - zp['var_qd'][m] / zp['var_q'][m]).mean():>17.3f} {zp['cos2_w'][m].mean():>14.3f}")
    print(f"\n{'d':>3} {'cos2':>7} {'cos':>7} {'removed (prefix)':>17} {'removed (loo)':>14} {'Q/rf prefix':>12}")
    for d in DS:
        m = z["d"] == d
        print(f"{d:>3} {z['cos2'][m].mean():>7.3f} {z['cos'][m].mean():>7.3f} {(1 - z['var_qd'][m] / z['var_q'][m]).mean():>17.3f} {z['cos2_w'][m].mean():>14.3f} {(z['var_q'][m] / z['var_rf'][m]).mean():>12.3f}")
