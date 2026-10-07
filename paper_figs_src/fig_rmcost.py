"""The cost of the reward model: the exact variance of the plug-in estimators against the size of the offline dataset, at the
warm start every run of the paper starts from.  Runs: mcp_dfl_prototype/_sgd_long/rm_cost.ipynb, which writes
_sgd_long/rmcost/rmcost_{strict,graded}.npz.  The palette and the page metrics come from make_paper_figs.py (it is
__main__-guarded, so importing it draws nothing).
usage: python fig_rmcost.py [strict] [graded] [ablation]
  strict / graded  -> fig_rmcost.pdf, the four-panel diagnostic (both rewards, the scalar Delta included)
  ablation         -> fig_data_ablation.pdf, the paper's two panels in the row figure's pattern
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.lines import Line2D
from make_paper_figs import COL, LS, LW, MK, INK, MUTED, GRID, COLW, TEXTW, clean, legend_below, save

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "mcp_dfl_prototype/_sgd_long/rmcost")
NAME = {"V": "V", "Q": "Q", "Q + Δ": "Q + Δ", "Q + rowΔ": "Q + Δ*"}                  # the npz's keys -> the palette's (Δ* is the row-wise correction)
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


# ---- the paper's ablation: two panels in the row figure's pattern -------------------------------------------------------
ABL_RC = {"axes.titlesize": 13.5, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 12, "legend.fontsize": 11.5}   # larger than the row figure's fonts
ABL_NOTE = 9.5                                                           # the in-plot notes (REINFORCE, exact table, one success per task)

def fig_ablation(kind="strict", name="fig_data_ablation", star_N=50):
    with plt.rc_context(ABL_RC): return _fig_ablation(kind, name, star_N)

def _fig_ablation(kind, name, star_N):
    """How much offline data the reward model needs, on the live MCP agent at the warm start, exactly.  Left: the variance of
    each plug-in estimator against the dataset size N, in units of Var(REINFORCE) - an empty table IS REINFORCE, and every
    method walks from there down to its own exact-table floor (dotted).  Right: the gap each estimator still has to that
    floor, i.e. what the fitted table costs - Q's closes by N ~ 30, the correction's needs N ~ 200.  The scalar Δ is left out
    (the paper writes Δ* for the row-wise correction) and V only appears on the left, where its floor is above REINFORCE."""
    from make_paper_figs import row_axes, row_legend, OURS
    ns, G, ex, rf, paper, cov = load(kind)
    x = ns.copy(); x[x == 0] = 0.3                                                    # N = 0 (the empty table = REINFORCE) on a log axis, far enough left that its label clears the 1
    MS = ["V", "Q", "Q + rowΔ"]
    fig, (a, b) = row_axes(2, TOP_IN=0.36, BOT_IN=0.54, WS=0.40, L=0.08)
    knee = next((n for n in ns if cov.get(int(n), 0) >= 0.5), None)                   # half the datasets hold a success on every task
    for ax in (a, b):
        if knee: ax.axvline(knee, color=MUTED, ls=(0, (2, 3)), lw=0.9, zorder=0)
    if knee:
        a.annotate("one success\nper task", xy=(knee, 0.02), xycoords=("data", "axes fraction"), xytext=(-4, 0), textcoords="offset points",
                   color=MUTED, fontsize=ABL_NOTE, va="bottom", ha="right", linespacing=1.15, bbox=dict(fc="white", ec="none", pad=0.8), zorder=6)
    span = [0.0, 0.0]
    for m in MS:
        c, lab = COL[NAME[m]], NAME[m]
        med, lo, hi = np.median(G[m], 1), np.quantile(G[m], 0.25, axis=1), np.quantile(G[m], 0.75, axis=1)
        mk = dict(marker=MK.get(lab, "o"), ms=3.0, mfc="white", mew=1.0, markevery=(1, 2))
        a.fill_between(x, lo / rf, hi / rf, color=c, alpha=0.14, lw=0)
        a.plot(x, med / rf, color=c, ls=LS[lab], lw=LW[lab], solid_capstyle="round", zorder=3, **mk)
        a.axhline(ex[m] / rf, color=c, ls=(0, (1, 2)), lw=1.1, zorder=1)              # the floor this estimator reaches with the exact table
        if paper: a.plot([star_N], [paper[m] / rf], marker="*", ms=8.5, color=c, mec="white", mew=0.7, zorder=5, clip_on=False)
        if m == "V": continue                                                          # V's floor is worse than REINFORCE, so its gap is a large negative
        b.fill_between(x, (lo - ex[m]) / rf, (hi - ex[m]) / rf, color=c, alpha=0.14, lw=0)
        b.plot(x, (med - ex[m]) / rf, color=c, ls=LS[lab], lw=LW[lab], solid_capstyle="round", zorder=3, **mk)
        span[0], span[1] = min(span[0], float((lo - ex[m]).min()) / rf), max(span[1], float((hi - ex[m]).max()) / rf)
    a.axhline(1.0, color=COL["REINFORCE"], lw=1.2, zorder=1)
    a.annotate("REINFORCE", xy=(0.99, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points",
               color=COL["REINFORCE"], va="bottom", ha="right", fontsize=ABL_NOTE, zorder=6)
    b.axhline(0.0, color=COL["REINFORCE"], lw=1.0, zorder=1)
    b.annotate("exact table", xy=(0.04, 0.0), xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points",
               color=MUTED, va="bottom", ha="left", fontsize=ABL_NOTE, zorder=6)
    ticks = [1, 10, 100, 1000]                                                              # decades, not a pick of the data points
    for ax, ti in ((a, "Variance at the Warm Start"), (b, "Cost of the Reward Model")):
        ax.set_xscale("log"); ax.set_xlabel("offline episodes per task, $N$", labelpad=1); clean(ax, "both")
        ax.set_title(ti, color=INK, pad=6)
        ax.set_xticks([0.3] + ticks); ax.set_xticklabels(["0"] + [f"{int(n)}" for n in ticks]); ax.minorticks_off(); ax.set_xlim(0.22, 1250)   # room for the 1000 tick
    a.set_yscale("log"); a.set_ylabel("Var / Var(REINFORCE)")
    a.set_yticks([0.2, 0.5, 1, 2, 3]); a.set_yticklabels(["0.2", "0.5", "1", "2", "3"])
    b.set_yscale("symlog", linthresh=3e-3, linscale=0.35); b.set_ylabel("gap to the exact table")
    b.set_ylim(min(1.5 * span[0], -6e-3), 1.7 * span[1])
    b.set_yticks([t for t in (-1e-1, -1e-2, 0, 1e-2, 1e-1, 1) if 1.5 * span[0] <= t <= 1.7 * span[1]])
    h = [Line2D([], [], color=COL[NAME[m]], ls=LS[NAME[m]], lw=LW[NAME[m]], marker=MK.get(NAME[m], "o"), ms=3.0, mfc="white",
                mew=1.0, label=OURS.get(NAME[m], NAME[m])) for m in MS]
    h.append(Line2D([], [], color=MUTED, ls=(0, (1, 2)), lw=1.1, label="exact table"))
    if paper: h.append(Line2D([], [], color=MUTED, ls="none", marker="*", ms=8.5, mec="white", mew=0.7, label=f"our runs ($N = {star_N}$)"))
    row_legend(fig, h, ncol=5, columnspacing=1.2)
    save(fig, name); return {m: float(np.median(G[m], 1)[list(ns).index(star_N)] / rf) for m in MS}


if __name__ == "__main__":
    args = sys.argv[1:] or ["strict", "ablation"]
    kinds = [k for k in args if os.path.exists(os.path.join(RUNS, f"rmcost_{k}.npz"))]
    if kinds:                                                                   # the four-panel diagnostic, only when a reward is named
        fig, AX = plt.subplots(len(kinds), 2, figsize=(TEXTW, 2.55 * len(kinds)), squeeze=False)
        for r, k in enumerate(kinds): panels(AX[r], k, legend=(r == len(kinds) - 1), titles=(r == 0))
        fig.tight_layout(w_pad=1.6, h_pad=1.6); save(fig, "fig_rmcost")
    if "ablation" in args: print("median Var / Var(REINFORCE) at the paper's dataset:", fig_ablation())
