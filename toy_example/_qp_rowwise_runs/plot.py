"""Figure: regret while training on the DFL QP toy, shared vs private predictor, 20 seeds (mean ± s.e.)."""
import numpy as np, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); calls = np.arange(0, 1601, 50)
INK, MUTED, GRID, SURFACE, REF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#8a8984"
SERIES = [("plain", "plain", "#2a78d6"), ("pQ", "+ Q", "#eb6834"), ("pQpΔ", "+ Q + Δ", "#1baf7a"), ("pQprow-wiseΔ", "+ Q + row-wise Δ", "#eda100"), ("exactgradient", "exact gradient", REF)]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True, facecolor=SURFACE)
for ax, (model, title) in zip(axes, [("shared", "shared network (8 hidden units, fixed asset codes)"), ("private", "private parameters (32 hidden units, trained asset codes)")]):
    ax.set_facecolor(SURFACE); ends = []
    for tag, name, col in SERIES:
        S = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(HERE, f"{model}_{tag}_s*.npy")))]); m = S.mean(0); se = S.std(0, ddof=1) / np.sqrt(len(S))
        ls = (0, (4, 3)) if tag == "exactgradient" else "-"
        if m.max() > 0.4:                                            # plain REINFORCE on the private model blows up: say so, don't draw it
            ax.annotate(f"plain (not shown) diverges: mean regret {m[-1]:.1f} at 1600", (0.98, 0.97), xycoords="axes fraction", ha="right", va="top", color=MUTED, fontsize=8.5); continue
        if tag != "exactgradient": ax.fill_between(calls, m - se, m + se, color=col, alpha=0.12, lw=0)
        ax.plot(calls, m, color=col, lw=2 if tag != "exactgradient" else 1.6, ls=ls, solid_capstyle="round", label=name)
        ends.append((m[-1], f"{name}  {m[-1]:.3f}"))
    ends.sort(); ys = []
    for y, lab in ends:
        y2 = y if not ys or y - ys[-1] > 0.018 else ys[-1] + 0.018; ys.append(y2)
        ax.annotate(lab, (calls[-1], y2), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_title(title, loc="left", color=INK, fontsize=10, pad=8); ax.set_xlabel("calls to the black box", color=MUTED); ax.set_xlim(0, 1600); ax.set_ylim(0.08, 0.38)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("regret (0 = best portfolio)", color=MUTED)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9.5, ncol=5, loc="upper left", bbox_to_anchor=(0.005, 0.925), labelcolor=MUTED)
fig.suptitle("DFL QP toy: row-wise Δ in training (exact per-instance Q and Δ*, Adam lr 0.01, 20 seeds, mean ± s.e.)", x=0.01, ha="left", color=INK, fontsize=11.5, fontweight="bold")
fig.subplots_adjust(left=0.06, right=0.855, top=0.8, bottom=0.12, wspace=0.62)
out = os.path.join(os.path.dirname(HERE), "_qp_rowwise_training.png"); fig.savefig(out, dpi=160, facecolor=SURFACE); print(out)
