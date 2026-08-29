#!/usr/bin/env python
"""Final E3 figure: hard regime (T=32, k=6, m=3), zoomed to the decisive
first 2000 oracle calls, log regret. Rebuilt from saved curves."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_toy import COL, LAB, style_ax, direct_labels, FIGS, RES

d = np.load(os.path.join(RES, "train_curves_T32_k6_w250.npz"))
grid = d["grid"]
sel = grid <= 2000
methods = ["reinforce_b", "rloo", "qcv", "qcv_exact", "pathwise"]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
style_ax(ax)
series = []
for meth in methods:
    C = np.maximum(d[f"early|{meth}"][:, sel], 2e-4)
    mean = C.mean(0)
    ax.fill_between(grid[sel], C.min(0), C.max(0), color=COL[meth],
                    alpha=0.13, lw=0, zorder=2)
    ax.plot(grid[sel], mean, color=COL[meth], lw=2, zorder=3)
    series.append((meth, grid[sel].astype(float), mean))
ax.set_yscale("log")
ax.set_xlabel("cumulative oracle calls (Q-CV/pathwise include the "
              "250-call Qhat warmup)")
ax.set_ylabel("exact expected regret  E[f]")
ax.set_title("E3 - metered oracle budget, hard regime "
             "(T=32, k=6, m=3; 4 seeds)", loc="left")
direct_labels(ax, series)
fig.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.savefig(os.path.join(FIGS, "e3_training_budget_final.png"),
            bbox_inches="tight")
print("saved figs/e3_training_budget_final.png")
