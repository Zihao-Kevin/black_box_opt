"""The cost of the reward model: the exact variance of the plug-in estimators against the size of the offline dataset, at the
warm start every run of the paper starts from.  Runs: mcp_dfl_prototype/_sgd_long/rm_cost.ipynb, which writes
_sgd_long/rmcost/rmcost_{strict,graded}.npz.  The palette and the page metrics come from make_paper_figs.py (it is
__main__-guarded, so importing it draws nothing).
usage: python fig_rmcost.py [strict] [graded]        -> fig_rmcost.pdf / .png
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_figs import COL, LS, LW, MK, INK, MUTED, GRID, COLW, TEXTW, clean, legend_below, save

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "mcp_dfl_prototype/_sgd_long/rmcost")
NAME = {"V": "V", "Q": "Q", "Q + Δ": "Q + Δ", "Q + rowΔ": "Q + row-wise Δ"}          # the npz's keys -> the palette's
ORDER = ["V", "Q", "Q + Δ", "Q + rowΔ"]
TITLE = {"strict": "exact-match reward", "graded": "graded reward"}


def load(kind):
    """(method -> (N, seed) total variance over the tasks), the exact-table total, Var(REINFORCE), the paper's own dataset,
    and the share of the draws whose dataset holds at least one success on every task."""
    z = np.load(os.path.join(RUNS, f"rmcost_{kind}.npz"), allow_pickle=False)
    N, seed, method, var = z["N"], z["seed"], z["method"], z["var"]
    ns, seeds = [int(n) for n in z["ns"]], sorted({int(s) for s in seed if s >= 0})
    tot = {}                                                                          # (method, N, seed) -> sum over the tasks
    for n, s, m, v in zip(N, seed, method, var): tot[m, int(n), int(s)] = tot.get((m, int(n), int(s)), 0.0) + v
    G = {m: np.array([[tot[m, n, s] for s in seeds] for n in ns]) for m in ORDER}
    ex = {m: sum(v for k, v in zip(z["exact_keys"], z["exact"]) if k.split("|")[0] == m) for m in ORDER}
    paper = {m: tot[m, 50, -1] for m in ORDER} if any(s == -1 for s in seed) else None
    covered = {}                                                                      # N -> P(the draw holds a success on all four tasks)
    if "draw_wins" in z:
        for n in ns:
            w = z["draw_wins"][(z["draw_N"] == n) & (z["draw_seed"] >= 0)]
            covered[n] = float((w.min(1) > 0).mean()) if len(w) else np.nan
    return np.array(ns, dtype=float), G, ex, float(z["rf"].sum()), paper, covered


def panels(ax, kind, legend=False, titles=True):
    ns, G, ex, rf, paper, cov = load(kind)
    x = ns.copy(); x[x == 0] = 0.55                                                   # N = 0 (the empty table = REINFORCE) on a log axis
    knee = next((n for n in ns if cov.get(int(n), 0) >= 0.5), None)                   # half the datasets hold a success on every task
    if knee and kind == "strict":
        for a in ax: a.axvline(knee, color=MUTED, ls=(0, (2, 3)), lw=0.9, zorder=0)
        ax[0].annotate("one success\nper task", xy=(knee, 0.02), xycoords=("data", "axes fraction"), xytext=(4, 0),
                       textcoords="offset points", color=MUTED, fontsize=7.5, va="bottom", ha="left", linespacing=1.15,
                       bbox=dict(fc="white", ec="none", pad=0.8), zorder=6)
    span = [0.0, 0.0]
    for m in ORDER:
        c, g, lab = COL[NAME[m]], G[m], NAME[m]
        med, lo, hi = np.median(g, 1), np.quantile(g, 0.25, axis=1), np.quantile(g, 0.75, axis=1)
        mk = dict(marker=MK.get(lab, "o"), ms=3.0, mfc="white", mew=1.0, markevery=(1, 2))
        ax[0].fill_between(x, lo / rf, hi / rf, color=c, alpha=0.14, lw=0)
        ax[0].plot(x, med / rf, color=c, ls=LS[lab], lw=LW[lab], label=lab, solid_capstyle="round", zorder=3, **mk)
        ax[0].axhline(ex[m] / rf, color=c, ls=(0, (1, 2)), lw=1.1, zorder=1)
        if m == "V": continue                                                          # V's exact table is worse than REINFORCE, so its gap is a large negative: left panel only
        ax[1].fill_between(x, (lo - ex[m]) / rf, (hi - ex[m]) / rf, color=c, alpha=0.14, lw=0)
        ax[1].plot(x, (med - ex[m]) / rf, color=c, ls=LS[lab], lw=LW[lab], solid_capstyle="round", zorder=3, **mk)
        span[0], span[1] = min(span[0], float((lo - ex[m]).min()) / rf), max(span[1], float((hi - ex[m]).max()) / rf)
        if paper: ax[0].plot([50], [paper[m] / rf], marker="*", ms=8.5, color=c, mec="white", mew=0.7, zorder=5, clip_on=False)
    ax[0].axhline(1.0, color=COL["REINFORCE"], lw=1.2, zorder=1)
    ax[0].annotate("REINFORCE", xy=(0.99, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points",
                   color=COL["REINFORCE"], va="bottom", ha="right", fontsize=7.5, zorder=6)
    ax[1].axhline(0.0, color=COL["REINFORCE"], lw=1.0, zorder=1)
    ax[0].set_ylabel(f"{TITLE[kind]}\nVar / Var(REINFORCE)", linespacing=1.5)
    ax[1].set_ylabel("gap / Var(REINFORCE)")
    ticks = [n for n in ns if n in (1, 3, 8, 20, 50, 200, 800)]
    for a, ti in zip(ax, ("variance at the warm start", "the cost of the reward model")):
        a.set_xscale("log"); a.set_xlabel("offline episodes per task, $N$", labelpad=1); clean(a, "both")
        if titles: a.set_title(ti, pad=5)
        a.set_xticks([0.55] + ticks); a.set_xticklabels(["0"] + [f"{int(n)}" for n in ticks]); a.minorticks_off()
    ax[0].set_yscale("log"); ax[1].set_yscale("symlog", linthresh=3e-3, linscale=0.35)
    ax[1].set_ylim(min(1.5 * span[0], -6e-3), 1.7 * span[1])
    ax[1].set_yticks([t for t in (-1e-1, -1e-2, 0, 1e-2, 1e-1, 1) if 1.5 * span[0] <= t <= 1.7 * span[1]])
    if legend: legend_below(ax[0], 4, y=-0.38)


if __name__ == "__main__":
    kinds = [k for k in (sys.argv[1:] or ["strict"]) if os.path.exists(os.path.join(RUNS, f"rmcost_{k}.npz"))]
    fig, AX = plt.subplots(len(kinds), 2, figsize=(TEXTW, 2.55 * len(kinds)), squeeze=False)
    for r, k in enumerate(kinds): panels(AX[r], k, legend=(r == len(kinds) - 1), titles=(r == 0))
    fig.tight_layout(w_pad=1.6, h_pad=1.6); save(fig, "fig_rmcost")
