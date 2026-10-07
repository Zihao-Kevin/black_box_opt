"""QP toy: biased vs unbiased on the same frozen reward model.  Regret (true, capped at 1 = the worst portfolio) vs black-box calls."""
import glob, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/nethome/zzhao628/blogs/black_box_opt/toy_example"; N = int(os.environ.get("QP_OFFLINE", 50))
def load(pat):
    fs = sorted(glob.glob(pat)); runs = [np.minimum(np.load(f), 1.0) for f in fs]
    return (np.stack(runs) if runs else None)
curves = [("DM (frozen)", f"{T}/_qp_frozen_runs/n{N}_DMfrozen_s*.npy", "#e2711d", "-", "direct method ∇E[f̂], f̂ frozen  (biased)"),
          ("Q + rowΔ (frozen)", f"{T}/_qp_frozen_runs/n{N}_QprowΔfrozen_s*.npy", "#1f5fbf", "-", "Q + Δ*, same frozen f̂  (unbiased)"),
          ("DM (online)", f"{T}/_qp_frozen_runs/n{N}_DMonline_s*.npy", "#e2711d", ":", "direct method, f̂ refit every call"),
          ("REINFORCE", f"{T}/_qp_adam_runs/exact_REINFORCE_s*.npy", "#444444", "-", "REINFORCE  (no f̂)"),
          ("exact gradient", f"{T}/_qp_adam_runs/exact_exactgradient_s*.npy", "#999999", "--", "exact gradient  (reference)")]
fig, ax = plt.subplots(figsize=(5.2, 3.9))
for name, pat, c, ls, lab in curves:
    Y = load(pat)
    if Y is None: print("missing", name); continue
    X = np.arange(Y.shape[1]) * 100; mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(Y))
    ax.plot(X, mu, color=c, ls=ls, lw=1.9, label=f"{lab}  [n={len(Y)}]")
    if ls == "-": ax.fill_between(X, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    k = [np.searchsorted(X, e) for e in (0, 400, 800, 1600, 3200, 6400, 12800)]
    print(f"{name:<18} n {len(Y):>2}  regret at calls {[int(X[i]) for i in k]}: {[round(float(mu[i]), 3) for i in k]}  min of mean {mu.min():.3f} at {int(X[mu.argmin()])}")
ax.set(xlabel="black-box calls during training", ylabel="regret (true), capped at 1", xscale="log", xlim=(90, 13000), ylim=(0, 1.02),
       title=f"QP: same f̂ from {N} offline calls per day, frozen")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.2, frameon=False, loc="upper left"); fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/nethome/zzhao628/blogs/black_box_opt/fig_qp_fhat_crossover.png"; fig.savefig(out, dpi=160); print("wrote", out)
