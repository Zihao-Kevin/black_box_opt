import numpy as np, glob, re, collections, sys, os
from scipy import stats
os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_main")); R = collections.defaultdict(dict)     # pass runs_g4 for 4 episodes per step
for f in sorted(glob.glob("*.npz")):
    m = re.match(r"(.+)_(\d+)jobs_lr([\d.e-]+)_(.+)_s(\d+)\.npz", f); d = np.load(f, allow_pickle=True)
    R[(m.group(1)[-28:], m.group(3), m.group(4))][int(m.group(5))] = d
AUC = {}
for k, v in R.items():
    S = np.stack([v[s]["success"].mean(1) for s in sorted(v)]); T = S.shape[1] - 1
    done = np.stack([np.array(v[s]["done"][-1][0], float) for s in sorted(v)]) if len(v[min(v)]["done"]) else np.zeros((1, 1))
    D = np.stack([v[s]["diag"] for s in sorted(v)]); ratio = D[:, :, 4] / np.maximum(D[:, :, 3], 1e-300); AUC[k] = dict(zip(sorted(v), S.mean(1)))
    print(f"{k[0]:<28} lr{k[1]:<7}{k[2]:<12} n={len(v):<3} reward@0,T/4,T/2,T: " + " ".join(f"{S[:, t].mean():.3f}" for t in (0, T // 4, T // 2, T)) + f"  AUC {S.mean():.3f}  done>=k: " + " ".join(f"{x:.2f}" for x in done.mean(0))
          + "  Δ*/Q: " + " ".join(f"{np.nanmedian(ratio[:, t]):.2f}" for t in np.linspace(0, ratio.shape[1] - 1, 6).astype(int)))
FIN = {k: dict(zip(sorted(v), [v[s_]["success"][-1].mean() for s_ in sorted(v)])) for k, v in R.items()}
for k in FIN: print(f"  {k[0][-10:]:<10} {k[2]:<14} collapsed (final < 0.02): {sum(x < 0.02 for x in FIN[k].values())}/{len(FIN[k])}   solved (final > 0.9): {sum(x > 0.9 for x in FIN[k].values())}/{len(FIN[k])}   final mean {np.mean(list(FIN[k].values())):.3f}")
for a, b in [("QprowΔ_exact", "Q_exact"), ("QprowΔ_exact", "QpΔ_exact"), ("QpΔ_exact", "Q_exact"), ("Q0prowΔ", "Q0"), ("Q0prowΔ", "Q0pΔ"), ("Q0pΔ", "Q0"), ("Q0prowΔ", "REINFORCE"), ("Q0", "REINFORCE"),
             ("Q0prowΔ", "DR"), ("Q0prowΔ", "V0"), ("Q0prowΔ", "PSB"), ("Q0prowΔ", "RLOO"), ("Q0prowΔ", "GRPO"), ("Q0prowΔ", "OTB"), ("Q0prowΔ", "RELAX"), ("DR", "Q0"), ("DM", "Q0prowΔ"), ("DM", "DR"), ("QprowΔ_exact", "PSB_exact"), ("exactgradient", "QprowΔ_exact")]:
    for k in AUC:
        if k[2] == a and (k[0], k[1], b) in AUC:
            o = AUC[(k[0], k[1], b)]; common = sorted(set(AUC[k]) & set(o)); d = np.array([AUC[k][s] - o[s] for s in common])
            if len(d) > 2: print(f"  {k[0]:<28} lr{k[1]:<7} AUC {a} − {b}: {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f}  (n={len(d)}, better in {(d>0).sum()}, p={stats.ttest_1samp(d, 0).pvalue:.3g})")
