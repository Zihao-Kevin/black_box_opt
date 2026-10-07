"""Second figure for LIVE_FREE.md: every estimator on the free-text queue (all-done reward), mean success over training ± s.e."""
import numpy as np, glob, os, re, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
LEARNED, CEILING, ROW = "#2a78d6", "#b5b4ad", "#1baf7a"
NAME = {"REINFORCE": "REINFORCE (running mean)", "RLOO": "RLOO", "GRPO": "GRPO", "OTB": "OTB", "RELAX": "RELAX", "V0": "V baseline", "Q0": "Q baseline",
        "Q0pΔ": "Q + Δ", "Q0prowΔ": "Q + row-wise Δ", "DR": "doubly robust", "DM": "direct method (biased)", "PSB": "per-parameter baseline",
        "Q_exact": "Q, true table", "QpΔ_exact": "Q + Δ, true table", "QprowΔ_exact": "Q + row-wise Δ, true table", "PSB_exact": "per-parameter, true table", "exactgradient": "exact gradient"}
PANELS = [("runs_main", "1 episode per step, 400 steps"), ("runs_g4", "4 episodes per step, 100 steps")]

def load(folder, tag):
    fs = sorted(glob.glob(os.path.join(HERE, folder, f"queue_defines-lint-port-digest_product_1jobs_lr0.001_{tag}_s*.npz")))
    return np.array([np.load(f, allow_pickle=True)["success"].mean() for f in fs])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), facecolor=SURFACE)
for ax, (folder, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE); rows = []
    for tag, name in NAME.items():
        v = load(folder, tag)
        if len(v): rows.append((v.mean(), v.std(ddof=1) / np.sqrt(len(v)), name, tag, len(v)))
    rows.sort(); y = np.arange(len(rows))
    for i, (m, se, name, tag, n) in enumerate(rows):
        col = ROW if "rowΔ" in tag else CEILING if ("exact" in tag) else LEARNED
        ax.barh(i, m, height=0.55, color=col, xerr=se, error_kw=dict(ecolor=INK, elinewidth=1, capsize=0)); ax.text(m + se + 0.015, i, f"{m:.2f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([r[2] + ("" if r[4] == 20 else f"  (n={r[4]})") for r in rows], fontsize=9, color=INK); ax.set_xlim(0, 1.08); ax.set_xlabel("mean P(all four tasks done) over training", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=8); ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=LEARNED, label="reward table learned from the run's own episodes"), Patch(color=ROW, label="row-wise Δ"), Patch(color=CEILING, label="true reward table (ceiling)")],
           frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0.005, 0.93), labelcolor=MUTED)
fig.suptitle("Free-text queue of four tasks: all estimators, 20 seeds, same LoRA and learning rate", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.subplots_adjust(left=0.2, right=0.98, top=0.83, bottom=0.1, wspace=0.75)
out = os.path.join(os.path.dirname(HERE), "_free_queue_baselines.png"); fig.savefig(out, dpi=160, facecolor=SURFACE); print(out)
