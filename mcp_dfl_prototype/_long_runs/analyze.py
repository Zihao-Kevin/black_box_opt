import numpy as np, glob, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats
TAG = {"REINFORCE": "REINFORCE", "Q": "Q", "QpΔ": "Q + Δ", "Q0": "Q0", "Q0pΔ": "Q0 + Δ", "Q_exact": "Q (exact)", "QpΔ_exact": "Q + Δ (exact)"}
R = {}
for f in glob.glob(os.path.join(HERE, "manifest_10jobs_lr0.001_*.npz")):
    m = re.match(r"manifest_10jobs_lr0.001_(.+)_s(\d+)\.npz", os.path.basename(f)); d = np.load(f)
    R.setdefault(TAG[m.group(1)], {})[int(m.group(2))] = (d["success"], d["diag"], d["jobs"])
def summary(m, seeds):
    A = np.stack([R[m][s][0] for s in seeds]); mean = A.mean(2)
    return f"{m:<15} n={len(seeds):>2}  step25 {mean[:,25].mean():.3f}  step50 {mean[:,50].mean():.3f}  step100 {mean[:,100].mean():.3f}  final {mean[:,-1].mean():.3f}  AUC {mean.mean():.3f}  solved {(A[:,-1]>0.4).sum(1).mean():.2f}  collapsed {(mean[:,-1]<0.02).sum()}/{len(seeds)}"
def paired(a, b, seeds, what):
    fa = lambda m: np.array([R[m][s][0].mean() if what == "AUC" else R[m][s][0][-1].mean() for s in seeds]); d = fa(a) - fa(b)
    return f"  {what:<5} {a} − {b}: {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}  better {(d>1e-9).sum()}/{len(d)} worse {(d<-1e-9).sum()}  t p={stats.ttest_1samp(d,0).pvalue:.3g}  Wilcoxon p={stats.wilcoxon(d).pvalue:.3g}"
if __name__ == "__main__":
    new, s40 = list(range(20, 40)), list(range(40))
    print("manifest, 10 jobs, lr 1e-3, 200 steps, all 40 seeds")
    for m in TAG.values(): print(summary(m, s40))
    print("\nFRESH seeds 20-39 only (replication):")
    for a, b in [("Q0 + Δ", "Q0"), ("Q0 + Δ", "Q"), ("Q0 + Δ", "REINFORCE"), ("Q + Δ", "Q"), ("Q0", "Q"), ("Q", "REINFORCE")]:
        print(paired(a, b, new, "AUC")); print(paired(a, b, new, "final"))
    print("\nall 40 seeds:")
    for a, b in [("Q0 + Δ", "Q0"), ("Q0 + Δ", "Q"), ("Q0 + Δ", "REINFORCE"), ("Q + Δ", "Q"), ("Q", "REINFORCE"), ("Q + Δ (exact)", "Q (exact)"), ("Q + Δ (exact)", "Q0 + Δ")]:
        print(paired(a, b, s40, "AUC")); print(paired(a, b, s40, "final"))
    jobs = R["Q"][0][2]; print("\nfinal P(success) per job, 40 seeds:")
    for m in TAG.values(): print(f"  {m:<15}" + " ".join(f"{j[:5]}={np.mean([R[m][s][0][-1, i] for s in s40]):.2f}" for i, j in enumerate(jobs)))
