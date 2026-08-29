#!/usr/bin/env python
"""Fig. 1 -- one point per method: how fast it gets good vs. how good it gets.

x = mean oracle calls to reach regret < THRESH (efficiency)
y = final regret at the full budget (quality ceiling)

Reuses the D2 predict-then-optimize run already saved in
results/dfl_train_curves.npz (see run_dfl.py:train) -- no retraining, no new
oracle calls.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_toy import COL, LAB, style_ax, FIGS, RES

METHODS = ["qcv_exact", "qcv_delta", "qcv", "reinforce_b", "rloo"]
THRESH = 0.03

d = np.load(os.path.join(RES, "dfl_train_curves.npz"))
grid = d["grid"]

fig, ax = plt.subplots(figsize=(5.6, 4.2))
style_ax(ax)
for meth in METHODS:
    C = d[f"tr|{meth}"]
    hits = [grid[np.argmax(c < THRESH)] if (c < THRESH).any() else np.nan
            for c in C]
    calls, final = np.nanmean(hits), C[:, -1].mean()
    if np.isnan(calls):
        ax.scatter(grid[-1], final, s=90, marker="x", color=COL[meth], zorder=3)
        ax.annotate(f"{LAB[meth]}\n(never reached <{THRESH})",
                    (grid[-1], final), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=COL[meth])
    else:
        ax.scatter(calls, final, s=90, color=COL[meth], zorder=3)
        ax.annotate(LAB[meth], (calls, final), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=COL[meth])
ax.set_yscale("log")
ax.set_xlabel(f"mean oracle calls to reach regret < {THRESH}")
ax.set_ylabel(f"final regret at budget={int(grid[-1])}")
ax.set_title("Fig. 1 -- efficiency (calls to target) vs. quality ceiling"
              " (final regret)", loc="left")
fig.tight_layout()
out = os.path.join(FIGS, "fig1_regret_vs_calls.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
