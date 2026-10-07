"""QP toy: both estimators refit the same LINEAR reward model online (misspecified: the regret is quadratic).  Regret vs calls."""
import glob, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/nethome/zzhao628/blogs/black_box_opt/toy_example"; D = sys.argv[2] if len(sys.argv) > 2 else "_qp_online_linear_runs_c51200"
def load(pat):
    fs = sorted(glob.glob(pat)); return np.stack([np.minimum(np.load(f), 1.0) for f in fs]) if fs else None
curves = [("direct method ∇E[f̂], linear f̂ refit every call  (biased)", f"{T}/{D}/n50_DMonline_s*.npy", "#e2711d", "-"),
          ("Q + Δ*, same linear f̂ refit every call  (unbiased)", f"{T}/{D}/n50_QprowΔonline_s*.npy", "#1f5fbf", "-"),
          ("REINFORCE  (no f̂)", f"{T}/_qp_adam_runs/exact_REINFORCE_s*.npy", "#444444", "-"),
          ("exact gradient  (reference)", f"{T}/_qp_adam_runs/exact_exactgradient_s*.npy", "#999999", "--")]
fig, ax = plt.subplots(figsize=(5.4, 4.0)); xmax = 0
for lab, pat, c, ls in curves:
    Y = load(pat)
    if Y is None: print("missing", lab); continue
    X = np.arange(Y.shape[1]) * 100; mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(Y)); xmax = max(xmax, X[-1])
    ax.plot(X, mu, color=c, ls=ls, lw=1.9, label=f"{lab}  [n={len(Y)}]")
    if ls == "-": ax.fill_between(X, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    k = [np.searchsorted(X, e) for e in (0, 400, 1600, 6400, 12800, 25600, 51200) if e <= X[-1]]
    print(f"{lab[:28]:<28} n {len(Y):>2}  regret at {[int(X[i]) for i in k]}: {[round(float(mu[i]), 3) for i in k]}  final {mu[-1]:.3f}±{se[-1]:.3f}  min of mean {mu.min():.3f} at {int(X[mu.argmin()])}")
ax.set(xlabel="black-box calls during training", ylabel="regret (true)", xscale="log", xlim=(90, xmax * 1.05), ylim=(0.15, 0.4), title="QP: linear f̂ refit online by both estimators")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, frameon=False, loc="lower left"); fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/nethome/zzhao628/blogs/black_box_opt/fig_qp_linear_online.png"; fig.savefig(out, dpi=160); print("wrote", out)
