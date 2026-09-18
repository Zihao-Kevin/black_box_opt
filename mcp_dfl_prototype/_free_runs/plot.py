"""Figure for LIVE_FREE.md: mean reward over training, 20 seeds, free-text queue of four tasks (all-done reward)."""
import numpy as np, glob, os, re, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); PREFIX = sys.argv[1] if len(sys.argv) > 1 else "queue_defines-lint-port-digest_product"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COLOR = {"REINFORCE": "#8a8983", "Q": "#2a78d6", "Q + Δ": "#eb6834", "Q + row-wise Δ": "#1baf7a"}          # colour follows the method in both panels
PANELS = [("reward table learned from the run's own episodes", {"REINFORCE": "REINFORCE", "Q": "Q0", "Q + Δ": "Q0pΔ", "Q + row-wise Δ": "Q0prowΔ"}),
          ("true reward table (a ceiling)", {"Q": "Q_exact", "Q + Δ": "QpΔ_exact", "Q + row-wise Δ": "QprowΔ_exact"})]

def load(tag):
    runs = [np.load(f, allow_pickle=True)["success"].mean(1) for f in sorted(glob.glob(os.path.join(HERE, "runs_main", f"{PREFIX}_*jobs_lr0.001_{tag}_s*.npz")))]
    return np.stack(runs)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
for ax, (title, methods) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE); ends = []
    for name, tag in methods.items():
        S = load(tag); m = S.mean(0); se = S.std(0, ddof=1) / np.sqrt(len(S)); x = np.arange(len(m))
        ax.fill_between(x, m - se, m + se, color=COLOR[name], alpha=0.10, lw=0); ax.plot(x, m, color=COLOR[name], lw=2, solid_capstyle="round", label=name)
        ax.plot(x[-1], m[-1], "o", ms=8, color=COLOR[name], mec=SURFACE, mew=2, clip_on=False); ends.append((m[-1], name))
    ends.sort(); ys = []
    for y, name in ends:                                                   # direct labels at the line ends, nudged apart
        y2 = y if not ys or y - ys[-1] > 0.07 else ys[-1] + 0.07; ys.append(y2); ax.annotate(f"{name}  {y:.2f}", (len(m) - 1, y2), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=10); ax.set_xlabel("episodes", color=MUTED); ax.set_ylim(-0.02, 1.04); ax.set_xlim(0, len(m) - 1)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("P(all four tasks done), exact", color=MUTED)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9.5, ncol=4, loc="upper left", bbox_to_anchor=(0.005, 0.925), labelcolor=MUTED, handlelength=1.6, columnspacing=1.8)
fig.suptitle("Free-text answer to a queue of four tasks: a row-wise Δ keeps training from collapsing   (mean ± s.e., 20 seeds)", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.subplots_adjust(left=0.06, right=0.84, top=0.76, bottom=0.13, wspace=0.42)
out = os.path.join(os.path.dirname(HERE), "_free_queue_training.png"); fig.savefig(out, dpi=160, facecolor=SURFACE); print(out)
