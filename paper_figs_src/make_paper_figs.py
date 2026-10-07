"""The three paper figures, one style.  usage (from anywhere): python make_paper_figs.py
  fig_live_mcp.pdf              Live_agent_rowwise.ipynb, warm start, plain SGD lr 0.045, 20 seeds (_sgd_long/warm_lr0.045[_fix])
  fig_qp_adam.pdf               DFL_QP_Black_Box_new.ipynb, Adam, 12800 calls, 10 seeds, Q rows fitted from the black box (_qp_adam_runs/)
  fig_variance_by_decision.pdf  _sgd_long/decomp/decomp.npz, the per-decision variance decomposition
Every figure is drawn at its final printed size (a 3.3 in column or a 6.75 in text width) so the fonts are the paper's fonts."""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIGS, PNGS = os.path.join(ROOT, "paper_figs"), os.path.join(HERE, "png")   # the pdfs go to paper_figs/ (the paper copies them), the pngs stay beside the scripts
LIVE, LIVE_FIX = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/warm_lr0.045"), os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/warm_lr0.045_fix")
LIVE_LENIENT = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long", os.environ.get("LENIENT_DIR", "warm_lr0.045_lenient"))   # the same runs under the lenient reward (SGD_STRICT=0); LENIENT_DIR picks another rate's directory
QP_SGD = os.path.join(ROOT, "toy_example/_qp_sgd_runs")        # the same comparison under plain SGD (notebook cell 16, _qp_sgd_runs/rerun_sgd.py)
QP_SGD_Y = dict(ylim=(0.19, 0.40), yticks=(0.2, 0.25, 0.3, 0.35, 0.4))   # plain SGD starts a little higher and reaches a little lower than Adam
QP_SGD_R = dict(ylim=(0.60, 0.81), yticks=(0.6, 0.65, 0.7, 0.75, 0.8), reward=True)   # the same panel as reward = 1 - regret (the row figure)
QP, DECOMP = os.path.join(ROOT, "toy_example/_qp_adam_runs"), os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/decomp/decomp.npz")
DECOMP_LENIENT = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/decomp/decomp_lenient_s1.npz")   # the lenient reward, seed 1 (seed 0 collapses at 100 episodes)
DECOMP_GRADED = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/decomp/decomp_graded_exactpath.npz")   # the graded reward, along the exact-table Q + rowΔ run, seed 0
LIVE_GRADED = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long/warm_lr0.045_graded")
if os.environ.get("OFFLINE"):                                                                    # the fixed-offline-dataset protocol (offline.py, launch_offline.sh, qp_offline.py)
    _lr = os.environ.get("OFFLINE_LR", "0.045")
    LIVE, LIVE_FIX, LIVE_GRADED = (os.path.join(ROOT, f"mcp_dfl_prototype/_sgd_long/offline_lr{_lr}{sfx}") for sfx in ("", "_fix", "_graded"))
    QP = os.path.join(ROOT, "toy_example/_qp_adam_runs_offline")
    _sub, _t = ("offline", "") if _lr == "0.045" else (f"offline_lr{_lr}", f"_lr{_lr}")                  # decompose.py names the files by lr, 0.045 being its default
    DECOMP, DECOMP_GRADED = (os.path.join(ROOT, f"mcp_dfl_prototype/_sgd_long/decomp/{_sub}/{pat}") for pat in (f"decomp{_t}_s*.npz", f"decomp_graded{_t}_s*_exactpath.npz"))
    FIGS, PNGS = (os.path.join(p, _sub) for p in (FIGS, PNGS))                                            # the figures go to paper_figs/offline[_lr<lr>]/

# ---- one style for the three figures -------------------------------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
COL = {"Q + Δ*": "#1f5fbf", "Q": "#e2711d", "Q + Δ": "#1baf7a",                            # ours: two saturated slots, fixed (blue/orange also separate in print)
       "REINFORCE": "#1f1f1f", "exact gradient": "#9a9994",                                  # the two references: near-black, and grey dashed
       "V": "#c2648a", "RLOO": "#7a6ea8", "OTB": "#5f8496", "GRPO": "#b8925a", "LAX": "#6e8b5e"}   # the other baselines, muted; they also carry a marker
MK = {"RLOO": "s", "OTB": "D", "GRPO": "^", "LAX": "v", "V": "^", "REINFORCE": "s"}                  # second encoding for the muted colours
LS = {m: "-" for m in COL}; LS["exact gradient"] = (0, (4, 2.5))
LW = {m: 1.5 for m in COL}; LW.update({"Q + Δ*": 2.2, "Q": 2.0, "Q + Δ": 2.0, "REINFORCE": 1.7, "exact gradient": 1.5})
OURS = {m: m + "  $\\it{(ours)}$" for m in ("Q + Δ*",)}   # the estimator this paper proposes; italic keeps the mark quieter than the name
ORDER = {"live": ["REINFORCE", "RLOO", "GRPO", "OTB", "Q", "Q + Δ*", "exact gradient"],                      # legend order = adjacency for the colour check
         "qp": ["REINFORCE", "RLOO", "GRPO", "OTB", "LAX", "Q", "Q + Δ*", "exact gradient"],
         "decomp": ["REINFORCE", "V", "Q", "Q + Δ*"]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "legend.fontsize": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 100, "savefig.dpi": 300})
TITLE_ROW = 9.5                                                          # a one-line panel title in the row figure: the widest one has to clear its neighbour
COLW, TEXTW = 3.3, 6.75                                                  # inches: ICML column / text width; NeurIPS text width is 5.5 in
PANEL_ASPECT = (2.75 * (plt.rcParams["figure.subplot.top"] - plt.rcParams["figure.subplot.bottom"])
                / (COLW * (plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"])))   # height/width of a (COLW, 2.75) figure's axes box


def row_axes(ncols, aspect=None, L=0.065, R=0.995, WS=0.36, TOP_IN=0.44, BOT_IN=0.46):
    """A row of panels at text width, every axes box carrying the one-column figures' height/width so the curves are not
    squeezed: the panel width follows from the margins, and the figure height follows from that aspect.  TOP_IN / BOT_IN are
    the inches the titles and the x labels need; the legend goes under the row with row_legend()."""
    w = TEXTW * (R - L) / (ncols + (ncols - 1) * WS)                     # one panel's axes width, in inches
    H = (aspect or PANEL_ASPECT) * w + TOP_IN + BOT_IN
    fig, axes = plt.subplots(1, ncols, figsize=(TEXTW, H), gridspec_kw=dict(wspace=WS), squeeze=False)
    fig.subplots_adjust(left=L, right=R, top=1 - TOP_IN / H, bottom=BOT_IN / H)
    return fig, axes[0]


def row_legend(fig, handles, ncol, y=-0.02, columnspacing=1.5, handlelength=2.0):
    """one legend for the whole row, centred under it"""
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, handlelength=handlelength, columnspacing=columnspacing, handletextpad=0.6, labelcolor=INK, borderaxespad=0)

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7); ax.set_axisbelow(True)

def curve(ax, x, S, m, label=None, band=True, lw=None):
    """mean over seeds (rows of S) with a standard-error band"""
    mean, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
    if band and m != "exact gradient": ax.fill_between(x, mean - se, mean + se, color=COL[m], alpha=0.14, lw=0)
    mk = dict(marker=MK[m], ms=3.6, markevery=(len(x) // 16, max(1, len(x) // 8)), mfc="white", mew=1.1) if m in ("RLOO", "OTB", "GRPO", "LAX", "V") else {}
    ax.plot(x, mean, color=COL[m], ls=LS[m], lw=lw or LW[m], label=label or m, solid_capstyle="round", zorder=3 if m in ("Q + Δ*", "Q") else 2, **mk)

def legend_below(ax, ncol, y=-0.27, handles=None):
    h, l = handles or ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=2.0, columnspacing=1.3, handletextpad=0.6, labelcolor=INK, borderaxespad=0)

def save(fig, name):
    for ext, d in (("pdf", FIGS), ("png", PNGS)):
        os.makedirs(d, exist_ok=True); fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("wrote", name)

# ---- 1. live MCP agent: warm start, plain SGD, lr 0.045, 20 seeds --------------------------------------------------------
def panel_live(ax, title="Success Rate for MCP Agent", compact=False):
    """the live-agent curves on one axis - drawn alone by fig_live(), and as the first panel of fig_row()"""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    src = {"REINFORCE": (LIVE, "REINFORCE"), "Q": (LIVE, "Q"), "Q + Δ*": (LIVE, "Q + rowΔ"), "exact gradient": (LIVE, "exact gradient"),
           "RLOO": (LIVE_FIX, "RLOO"), "GRPO": (LIVE_FIX, "GRPO"), "OTB": (LIVE_FIX, "OTB")}   # the group baselines with the same step per update (worker_fix.py)
    n = None
    for m in ORDER["live"]:
        d, name = src[m]; L = load(d, name); keep = L[0, :, 0] <= 400; n = n or len(L)     # the same 400 episodes for every method
        curve(ax, L[0, keep, 0], L[:, keep, 1], m)
    ax.set(xlim=(0, 400), ylim=(0, 1), xticks=[0, 200, 400] if compact else [0, 100, 200, 300, 400],
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="agent episodes", ylabel="P(success)")
    ax.set_title(title, color=INK, pad=6, **({"fontsize": 10} if compact else {})); clean(ax)
    return n, ORDER["live"]

def fig_live():
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n, _ = panel_live(ax)
    legend_below(ax, ncol=3); save(fig, "fig_live_mcp"); return n

# ---- 2. quadratic program: Adam, 12800 calls, 10 seeds, Q rows fitted from the black box ---------------------------------
def panel_qp(ax, title="Regret for DFL Black-box", compact=False, runs=None, ylim=(0.19, 0.375), yticks=(0.2, 0.25, 0.3, 0.35), reward=False):
    """the quadratic-program curves on one axis - drawn alone by fig_qp(), and as the first panel of fig_row().
    runs: the directory of .npy runs, QP (Adam) by default, QP_SGD for the plain-SGD version.
    reward=True draws 1 - regret, the normalised portfolio value (1 = best portfolio, 0 = worst); ylim/yticks are then in reward."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    src = {"REINFORCE": ("exact", "REINFORCE"), "RLOO": ("exact", "RLOO"), "GRPO": ("exact", "GRPO"), "OTB": ("exact", "OTB"), "LAX": ("exact", "LAX"),
           "Q": ("fitted", "+ Q"), "Q + Δ*": ("fitted", "+ Q + row-wise Δ"), "exact gradient": ("exact", "exact gradient")}
    n = None
    for m in ORDER["qp"]:
        kind, name = src[m]; S = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(runs or QP, f"{kind}_{tag(name)}_s*.npy")))]); n = n or len(S)
        curve(ax, np.arange(0, 100 * (S.shape[1] - 1) + 1, 100), 1 - S if reward else S, m)
    ax.set(xlim=(0, 12800), ylim=ylim, xticks=[0, 6400, 12800] if compact else [0, 3200, 6400, 9600, 12800],
           yticks=list(yticks), xlabel="calls to the black box", ylabel="expected reward" if reward else "regret")
    ax.set_title(title, color=INK, pad=6, **({"fontsize": 10} if compact else {})); clean(ax)
    return n, ORDER["qp"]

def fig_qp(runs=None, name="fig_qp_adam", **kw):
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n, _ = panel_qp(ax, runs=runs, **kw)
    legend_below(ax, ncol=3); save(fig, name); return n

# ---- 3. variance by decision (exact decomposition along one run) ---------------------------------------------------------
def fig_decomp(path=DECOMP, out="fig_variance_by_decision", ylim=(2e-5, 3e2), yticks=(1e-4, 1e-2, 1e0, 1e2), show_rm=False):
    """path: one decomp npz, or a glob / list of them (one per seed): then the lines are the geometric mean over the seeds and the
    band is one standard error of log10 of the term, so a single-seed path draws exactly as before."""
    paths = sorted(glob.glob(path)) if isinstance(path, str) else list(path)
    zs = [np.load(pth) for pth in paths]; z = zs[0]
    Ds = [{tuple(k.split("|")): v for k, v in zip(zz["keys"], zz["vals"])} for zz in zs]
    snap, digit, npos = [int(s) for s in z["snap"]], [int(d) for d in z["digit"]], len(z["positions"]); tasks = sorted({k[3] for k in Ds[0]})
    slot_of = np.searchsorted(np.array(digit), np.arange(npos), side="right") - 1
    per_slot_one = lambda D, s, tab, m: np.bincount(slot_of, weights=sum(D[str(s), tab, m, t] for t in tasks), minlength=4)
    def per_slot(s, tab, m):                                     # geometric mean over the seeds, and its log10 standard error
        L = np.log10(np.clip(np.stack([per_slot_one(D, s, tab, m) for D in Ds]), 1e-300, None))
        return 10 ** L.mean(0), (L.std(0, ddof=1) / np.sqrt(len(L)) if len(L) > 1 else np.zeros(4))
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + Δ*": "Q + rowΔ"}         # the npz's method keys
    has_rm = show_rm and any(k[1] == "fitted_rm" for k in Ds[0])          # decompose_rm.py also carries the repaired reward model; drawn only on request
    FIT = [("fitted", (0, (2, 1.6)), 1.1, "o", "untried = fails")] + ([("fitted_rm", (0, (5, 1.5)), 1.3, "^", "fitted prior + design")] if has_rm else [])
    fig, axes = plt.subplots(1, len(snap), figsize=(TEXTW, 2.1), sharey=True); x = np.arange(1, 5)
    for ax, s in zip(axes, snap):
        for m in ORDER["decomp"]:
            y, se = per_slot(s, "exact", name[m])
            if len(Ds) > 1: ax.fill_between(x, y / 10 ** se, y * 10 ** se, color=COL[m], alpha=0.16, lw=0)
            ax.plot(x, y, color=COL[m], lw=LW[m] if m != "REINFORCE" else 1.5, marker=MK.get(m, "o"), ms=4.2, label=OURS.get(m, m), zorder=3 if m.startswith("Q") else 2)
            if s > 0 and m.startswith("Q"):
                for tab, ls, lw, mk, _ in FIT:
                    ax.plot(x, per_slot(s, tab, name[m])[0], color=COL[m], ls=ls, lw=lw, marker=mk, ms=3.4, mfc="white", mew=1.0, zorder=3)
        ax.set(yscale="log", xticks=x, xlim=(0.7, 4.3), title=f"{4 * s} episodes"); clean(ax)
        ax.tick_params(axis="y", which="minor", left=False)
    axes[0].set_ylim(*ylim); axes[0].set_yticks(list(yticks)); axes[0].set_ylabel("variance term")
    h, l = axes[0].get_legend_handles_labels()
    fit_h = [Line2D([], [], color=MUTED, ls=ls, lw=lw, marker=mk, ms=3.4, mfc="white") for _, ls, lw, mk, _ in FIT]
    fig.supxlabel("decision in the answer", y=0.02, color=INK, fontsize=10.5)
    kw = dict(loc="upper center", frameon=False, handlelength=2.0, columnspacing=1.3, handletextpad=0.6, labelcolor=INK)
    fig.legend(h + fit_h[:1], l + ["fitted reward function"], bbox_to_anchor=(0.5, -0.06), ncol=5, **kw)   # the same row in every version of this figure
    if len(FIT) > 1:                                         # the repaired reward model gets its own row underneath, so the first row stays identical
        fig.legend(fit_h[1:], [f"fitted reward function: {lab}" for *_, lab in FIT[1:]], bbox_to_anchor=(0.5, -0.15), ncol=len(FIT) - 1, **kw)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.2, wspace=0.12)
    save(fig, out)

# ---- 4. alignment decides the gain of Delta* (QP toy, at initialization, exact, d x position scale) -----------------------
def fig_alignment():
    """Why Δ* pays off at all, on the QP toy at initialization, exactly.  Left: the alignment of the per-position scores is set
    by the scale of the position features, not by the number of positions - the six values of d lie on one curve.  Right: that
    alignment is what the gain buys; the dashed diagonal is the closed form 1 - mean cos², exact for the scalar Δ and a floor
    for the row-wise Δ* the paper uses, so the points sit on it at high alignment and above it at low alignment."""
    import sweep_alignment as sw, sweep_alignment_row as swr
    g = swr.load(); ds = sorted(set(g["d"].tolist())); scales = sorted(set(g["pos_scale"].tolist()))
    gain = 1 - g["var_qd_row"] / g["var_q"]                    # prefix family, the row-wise Δ* the paper plots; var_qd is the scalar the closed form predicts
    BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]                     # one hue, light -> dark with d
    fig, (a, b) = row_axes(2, TOP_IN=0.30)
    for c, d in zip(BLUES, ds):                                                                    # A: alignment against the sharing scale, one line per d
        m = [g["cos2"][(g["d"] == d) & (g["pos_scale"] == ps)].mean() for ps in scales]
        a.plot(scales, m, color=c, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.5, label=f"d = {d}", solid_capstyle="round")
    a.set(xscale="log", xlim=(0.085, 3.5), ylim=(0, 1), xticks=scales, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          xlabel="scale of the position features", ylabel="alignment (mean cos²)")
    a.set_xticklabels([f"{ps:g}" for ps in scales]); a.tick_params(axis="x", which="minor", bottom=False); clean(a)
    a.set_title("Alignment vs. Feature Scale", color=INK, pad=6)
    b.plot([0, 1], [0, 1], color=COL["exact gradient"], ls=LS["exact gradient"], lw=1.4, zorder=1)   # B: the gain against the alignment; exact for the scalar Δ, a floor for Δ*
    for c, d in zip(BLUES, ds):
        k = g["d"] == d; b.scatter(g["cos2"][k], gain[k], s=20, color=c, edgecolor="white", linewidth=0.6, zorder=3)
    b.text(0.04, 0.95, "above the line:\nΔ* beats the closed form", transform=b.transAxes, color=MUTED,     # the low-alignment cloud sits under this corner
           fontsize=7.8, va="top", ha="left", linespacing=1.25)
    b.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
          xlabel="alignment (mean cos²)", ylabel="1 − Var(Q + Δ*) / Var(Q)"); clean(b, grid="both")
    b.set_title("Gain vs. Alignment", color=INK, pad=6)
    h = [Line2D([], [], color=c, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.5, label=f"d = {d}") for c, d in zip(BLUES, ds)]
    h.append(Line2D([], [], color=COL["exact gradient"], ls=LS["exact gradient"], lw=1.4, label="closed form (scalar Δ)"))
    row_legend(fig, h, ncol=7, columnspacing=1.1, handlelength=1.8)
    save(fig, "fig_alignment"); return len(gain)

# ---- 5. the live MCP agent under two rewards: exact-match (strict) and lenient ----------------------------------------
def fig_two_rewards(right=("lenient reward", None), out=None, exact_table=False, ylabel="P(success)", title="Success Rate for MCP Agent, Two Rewards"):
    """left: the exact-match runs; right: another reward's runs (title, directory).  exact_table=True draws the rows fitted from the
    exact reward table (V, Q, Q + Δ* with '(exact)') instead of the plug-in rows, where those runs exist."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + Δ*": "Q + rowΔ", "exact gradient": "exact gradient"}
    panels = [("exact-match reward", LIVE), (right[0], right[1] or LIVE_LENIENT)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.75), sharey=True, gridspec_kw=dict(wspace=0.1)); n = {}
    for ax, (ttl, d) in zip(axes, panels):
        for m in ["REINFORCE", "V", "Q", "Q + Δ*", "exact gradient"]:
            src = name[m] + (" (exact)" if exact_table and m in ("V", "Q", "Q + Δ*") and glob.glob(f"{d}/{tag(name[m] + ' (exact)')}_s*.npy") else "")
            L = load(d, src); keep = L[0, :, 0] <= 400; n[ttl, src] = len(L)                       # the same 400 episodes for every method
            curve(ax, L[0, keep, 0], L[:, keep, 1], m)
        ax.set(xlim=(0, 400), ylim=(0, 1), xticks=[0, 100, 200, 300, 400], xlabel="agent episodes"); ax.set_title(ttl + (", exact table" if exact_table else ""), fontsize=10.5, color=INK, pad=5); clean(ax)
    axes[0].set(yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ylabel=ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False, handlelength=2.0, columnspacing=1.4, handletextpad=0.6, labelcolor=INK)
    fig.suptitle(title, y=1.04, color=INK, fontsize=11)
    suffix = "" if "LENIENT_DIR" not in os.environ else "_" + os.environ["LENIENT_DIR"].replace("warm_", "").replace("_lenient", "")   # e.g. _lr0.03 for the lower-rate lenient panel
    save(fig, out or "fig_two_rewards" + suffix); return n

# ---- 6. the live MCP agent under the graded reward, one panel (the third panel of the paper's curve figure) ---------------
def fig_live_graded(exact_table=True, out=None):
    """The graded reward alone, drawn like fig_live: REINFORCE, V, Q, Q + Δ*, exact gradient.  exact_table=True draws the
    rows built from the exact reward table (the estimator on its own); False draws the plug-in rows fitted from the run."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + Δ*": "Q + rowΔ", "exact gradient": "exact gradient"}
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n = {}
    for m in ["REINFORCE", "V", "Q", "Q + Δ*", "exact gradient"]:
        src = name[m] + (" (exact)" if exact_table and m in ("V", "Q", "Q + Δ*") else "")
        L = load(LIVE_GRADED, src); keep = L[0, :, 0] <= 400; n[src] = len(L)
        curve(ax, L[0, keep, 0], L[:, keep, 1], m)
    ax.set(xlim=(0, 400), ylim=(0, 1), xticks=[0, 100, 200, 300, 400], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="agent episodes", ylabel="expected reward")
    ax.set_title("MCP Agent, Graded Reward", color=INK, pad=6); clean(ax)
    legend_below(ax, ncol=3); save(fig, out or ("fig_live_mcp_graded" if exact_table else "fig_live_mcp_graded_fitted")); return n

# ---- 7. twelve decisions per episode: the manifest format (live_long.py, _long_runs/), for the appendix ----------------------
LONG = os.path.join(ROOT, "mcp_dfl_prototype/_long_runs")
def fig_manifest():
    """Left: training on the 12-decision manifest format, all 10 tasks, Adam lr 1e-3, 200 steps of one episode per task, 40 seeds:
    REINFORCE, Q and Q + Δ* with the fitted table (untried = fails), and the same two rows with the exact table (dashed).
    Right: Var(Q + Δ*) / Var(Q), exact, at the policies the exact-table Q + Δ* run visits, 12 decisions against 4 (geometric mean over seeds).
    Δ* is the row-wise correction throughout, as everywhere else in the paper (live_long.py "row" methods; diag column 7)."""
    load = lambda name, seeds: np.stack([np.load(f"{LONG}/manifest_10jobs_lr0.001_{name}_s{s}.npz")["success"].mean(1) for s in seeds])
    rows = [("REINFORCE", "REINFORCE", "-"), ("Q", "Q0", "-"), ("Q + Δ*", "Q0prowΔ", "-"), ("Q", "Q_exact", (0, (4, 2.5))), ("Q + Δ*", "QprowΔ_exact", (0, (4, 2.5)))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28)); eps = np.arange(201) * 10; n = None
    for m, name, ls in rows:
        S = load(name, range(40)); n = n or len(S); mean, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S)); exact = ls != "-"
        if not exact: a.fill_between(eps, mean - se, mean + se, color=COL[m], alpha=0.14, lw=0)
        a.plot(eps, mean, color=COL[m], ls=ls, lw=1.6 if exact else LW[m], label=m + (" (exact table)" if exact else ""), solid_capstyle="round", zorder=3)
    a.set(xlim=(0, 2000), ylim=(0, 0.7), xticks=[0, 500, 1000, 1500, 2000], yticks=[0, 0.2, 0.4, 0.6], xlabel="agent episodes", ylabel="mean P(success), 10 tasks")
    a.set_title("Twelve decisions per episode", color=INK, pad=6); clean(a)
    a.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, handlelength=2.0, labelspacing=0.3, columnspacing=1.0, borderaxespad=0.3)
    def ratio(pattern, seeds):
        out = []
        for s in seeds:
            d = np.load(pattern.format(s))["diag"]; live = d[:, 1] > 1e-9 * d[0, 1]
            out.append(np.where(live & (d[:, 3] > 0), d[:, 7] / np.maximum(d[:, 3], 1e-300), np.nan))     # column 7: the row-wise Delta*
        out = np.array(out); k = (~np.isnan(out)).sum(0)
        return np.load(pattern.format(0))["diag"][:, 0] * 10, np.exp(np.nanmean(np.log(out), 0)), k
    for pat, seeds, c, mk, lab in [(f"{LONG}/manifest_10jobs_lr0.001_QprowΔ_exact_s{{}}.npz", range(40), COL["Q + Δ*"], "o", "12 decisions (manifest)"),
                                   (f"{LONG}/slots_10jobs_lr0.001_QprowΔ_exact_s{{}}.npz", range(20), MUTED, "^", "4 decisions (slots)")]:
        x, y, k = ratio(pat, seeds); ok = k >= 5
        b.plot(x[ok], y[ok], color=c, lw=1.8, marker=mk, ms=4, label=lab, solid_capstyle="round")
    b.axhline(1, color=GRID, lw=0.8)
    b.set(xlim=(0, 1200), ylim=(0, 1.05), xticks=[0, 400, 800, 1200], yticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel="agent episodes", ylabel="Var(Q + Δ*) / Var(Q)")
    b.set_title("Share left by Δ* along training", color=INK, pad=6); clean(b)
    b.legend(frameon=False, fontsize=8, loc="lower right", handlelength=2.0, borderaxespad=0.3)
    save(fig, "fig_manifest"); return n

# ---- 8. the learning-rate sweep of the live agent under the offline protocol (appendix) ----------------------------------------
def fig_lr_sweep(lrs=(0.03, 0.045, 0.06)):
    """Final exact P(success) at 400 episodes, mean and standard error over the 20 seeds, against the plain-SGD learning rate, one line
    per method; the group baselines from the _fix directories.  Reads mcp_dfl_prototype/_sgd_long/offline_lr<lr>{,_fix}/."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    base = os.path.join(ROOT, "mcp_dfl_prototype/_sgd_long")
    src = {"REINFORCE": ("", "REINFORCE"), "Q": ("", "Q"), "Q + Δ*": ("", "Q + rowΔ"), "exact gradient": ("", "exact gradient"),
           "RLOO": ("_fix", "RLOO"), "GRPO": ("_fix", "GRPO"), "OTB": ("_fix", "OTB")}
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n = {}
    for m in ORDER["live"]:
        sfx, name = src[m]; mean, se = [], []
        for lr in lrs:
            fs = [f for f in sorted(glob.glob(f"{base}/offline_lr{lr}{sfx}/{tag(name)}_s*.npy")) if "partial" not in f]
            if not fs: mean.append(np.nan); se.append(np.nan); continue
            L = np.stack([np.load(f) for f in fs]); keep = L[0, :, 0] <= 400; fin = L[:, keep, 1][:, -1]; n[m, lr] = len(fs)
            mean.append(fin.mean()); se.append(fin.std(ddof=1) / np.sqrt(len(fin)))
        mean, se = np.array(mean), np.array(se)
        mk = dict(marker=MK[m], ms=4.5, mfc="white", mew=1.1) if m in MK and m != "REINFORCE" else dict(marker="o", ms=4.5)
        if m != "exact gradient": ax.fill_between(lrs, mean - se, mean + se, color=COL[m], alpha=0.14, lw=0)
        ax.plot(lrs, mean, color=COL[m], ls=LS[m], lw=LW[m], label=m, zorder=3 if m.startswith("Q") else 2, **mk)
    ax.set(xlim=(min(lrs) - 0.003, max(lrs) + 0.003), ylim=(0, 1), xticks=list(lrs), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="learning rate (plain SGD)", ylabel="P(success) at 400 episodes")
    ax.set_xticklabels([f"{lr:g}" for lr in lrs]); ax.set_title("MCP Agent, Step Size", color=INK, pad=6); clean(ax)
    legend_below(ax, ncol=3); save(fig, "fig_lr_sweep"); return n

# ---- 7. JobShop: the same estimator, the reward moved off the first branch ------------------------------------------------
JOBSHOP = os.path.join(ROOT, "experiments/jobshop/results")

def fig_jobshop(name="fig_jobshop"):
    """Two runs of the SAME estimators on the SAME task; only the order of the four JSON fields differs.
    Left, dispatch first: it settles ~98% of the reward in the root branch, so Q is already near-exact and the
    residual a row-wise residual can cancel is 0.2% of REINFORCE's variance; the curves coincide. Right,
    dispatch last: that residual is 6.6% and the curves separate (AUC +0.0035, p 0.047, 4/4 seeds).
    Matched 16-epoch window; the left run continues flat to epoch 32. Q is drawn wide and Delta*
    narrow on top, so a coincidence reads as an orange halo rather than one curve hiding the other."""
    import json
    RD = "Q + \u0394*"
    v1 = json.load(open(os.path.join(JOBSHOP, "validation-curves.json")))
    v2 = json.load(open(os.path.join(JOBSHOP, "v2-pilot-curves.json")))
    S1 = {m: np.array([[x["mean_reward"] for x in r["validation"]] for r in v1[k]])[:, :17] for m, k in (("Q", "qcv"), (RD, "row_delta"))}
    S2 = {m: np.array(v2["sampled"][k]) for m, k in (("Q", "qcv"), (RD, "row_delta"))}
    const = v2["reference"]["best_constant_dispatch"]
    panels = [("dispatch first  (published)", S1, v2["headroom"]["v1"]["q_residual_share"]),
              ("dispatch last", S2, v2["headroom"]["v2"]["q_residual_share"])]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.8), sharey=True)
    x = np.arange(17)
    for i, (ax, (title, S, share)) in enumerate(zip(axes, panels)):
        ax.axhline(const, color=MUTED, ls=(0, (2, 2.5)), lw=1.0, zorder=1)
        curve(ax, x, S["Q"], "Q", lw=3.4)                                         # wide underneath
        curve(ax, x, S[RD], RD, lw=1.5)                                           # narrow on top: coincidence shows as a halo
        ax.set_title(title, color=INK, fontsize=10, pad=4)
        ax.text(0.97, 0.04, f"Q leaves {100 * share:.2g}% of REINFORCE variance\n{len(S['Q'])} seeds",
                transform=ax.transAxes, color=MUTED, fontsize=8.2, va="bottom", ha="right", linespacing=1.35)
        ax.set(xlim=(0, 16), xticks=[0, 4, 8, 12, 16], xlabel="epoch"); clean(ax)
    for m, dy in ((RD, 5), ("Q", -5)):                                            # direct labels on the right panel only
        axes[1].annotate(m, (16, S2[m].mean(0)[-1]), xytext=(5, dy), textcoords="offset points",
                         color=COL[m], fontsize=8.5, va="center", ha="left", annotation_clip=False)
    axes[0].set_ylim(0.60, 0.838); axes[0].set_yticks([0.62, 0.68, 0.74, 0.80]); axes[0].set_ylabel("validation reward")
    axes[0].annotate("best constant dispatch rule", (0.3, const), xytext=(0, -3), textcoords="offset points",
                     color=MUTED, fontsize=8, va="top", ha="left")
    h = [Line2D([], [], color=COL["Q"], lw=3.4, label="Q"), Line2D([], [], color=COL[RD], lw=1.5, label=RD),
         Line2D([], [], color=MUTED, ls=(0, (2, 2.5)), lw=1.0, label="best constant dispatch rule")]
    axes[0].legend(h, [t.get_label() for t in h], loc="upper center", bbox_to_anchor=(1.07, -0.20), ncol=3, frameon=False,
                   handlelength=2.0, columnspacing=1.6, handletextpad=0.6, labelcolor=INK, borderaxespad=0)
    fig.subplots_adjust(wspace=0.10, right=0.86)
    save(fig, name); return {t: len(S["Q"]) for t, S, _ in panels}


def fig_jobshop_sgd(name="fig_jobshop_sgd"):
    """JobShop, dispatch-last grammar, plain SGD lr 0.03, no entropy bonus, 2 samples x 2 instances per update, 8 seeds.
    Left: on-policy training reward per update (8-update mean). Right: validation reward per epoch. The
    variance ordering shows in the curves - REINFORCE bounces between rules, OTB lags, the Q family settles on
    the best rule within ~20 updates - but Q and Q + Delta* coincide: with an accurate table Q is exact
    on the one decision that carries the reward, and the decisions that leave Q a residual carry ~1% of it."""
    import json
    d = json.load(open(os.path.join(JOBSHOP, "v2-sgd-lr0.03-curves.json")))
    NM = {"reinforce": "REINFORCE", "otb": "OTB", "qcv": "Q", "row_delta": "Q + \u0394*"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    box = np.ones(8) / 8
    for k, m in NM.items():
        S = np.array(d["train"][k])[:, :192]; mean = np.convolve(S.mean(0), box, "valid"); se = np.convolve(S.std(0, ddof=1) / np.sqrt(len(S)), box, "valid"); x = np.arange(len(mean)) + 4
        a1.fill_between(x, mean - se, mean + se, color=COL[m], alpha=0.14, lw=0)
        a1.plot(x, mean, color=COL[m], lw=(3.4 if m == "Q" else 1.5 if m.startswith("Q +") else LW[m]), label=m, zorder=3 if m.startswith("Q") else 2)
        V = np.array(d["val"][k]); curve(a2, np.arange(V.shape[1]), V, m, lw=(3.4 if m == "Q" else 1.5 if m.startswith("Q +") else None))
    for ax in (a1, a2): ax.axhline(0.8201, color=MUTED, ls=(0, (2, 2.5)), lw=1.0, zorder=1); clean(ax)
    a1.set(xlim=(0, 192), ylim=(0.5, 0.86), xlabel="update  (2 samples \u00d7 2 instances)", ylabel="reward"); a1.set_title("training reward, 8-update mean", color=INK, fontsize=10, pad=4)
    a2.set(xlim=(0, 8), ylim=(0.5, 0.86), xticks=[0, 2, 4, 6, 8], xlabel="epoch"); a2.set_title("validation reward", color=INK, fontsize=10, pad=4); a2.set_yticklabels([])
    a2.annotate("best constant\ndispatch rule", (7.9, 0.8201), xytext=(0, -4), textcoords="offset points", color=MUTED, fontsize=7.8, va="top", ha="right", linespacing=1.2)
    a1.text(0.97, 0.04, "plain SGD, lr 0.03, no entropy bonus\n8 seeds", transform=a1.transAxes, color=MUTED, fontsize=8.2, va="bottom", ha="right", linespacing=1.35)
    h = [Line2D([], [], color=COL["REINFORCE"], lw=LW["REINFORCE"], label="REINFORCE"), Line2D([], [], color=COL["OTB"], lw=LW["OTB"], label="OTB"),
         Line2D([], [], color=COL["Q"], lw=3.4, label="Q"), Line2D([], [], color=COL["Q + \u0394*"], lw=1.5, label="Q + \u0394*")]
    a1.legend(h, [t.get_label() for t in h], loc="upper center", bbox_to_anchor=(1.12, -0.22), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.5, handletextpad=0.6, labelcolor=INK, borderaxespad=0)
    fig.subplots_adjust(wspace=0.08); save(fig, name)


def fig_jobshop_groups(name="fig_jobshop_groups", curve_lr=None, exclude=("vbase",)):
    """JobShop with one dispatch rule per machine group (4 groups x 5 rules), exact reward table, plain SGD, no
    entropy bonus, 2 samples per update.  Left: validation reward per epoch at the learning rate where Q sits at its
    stability edge.  Right: across the learning-rate sweep, the fraction of seeds that locked onto the best plan -
    the noisier the estimator, the lower the rate at which it survives, and the ordering is the paper's claim."""
    import json
    d = json.load(open(os.path.join(JOBSHOP, "v3-exact-lr-sweep.json")))
    NM = {k: v for k, v in {"reinforce": "REINFORCE", "rloo": "RLOO", "grpo": "GRPO", "otb": "OTB", "relax": "LAX", "vbase": "V", "qcv": "Q", "row_delta": "Q + \u0394*"}.items()
          if k not in exclude and all(k in d["by_lr"][lr] for lr in d["by_lr"])}            # methods present at every step size; V there is the oracle-table one, so it is left out
    lrs = sorted(d["by_lr"], key=float); lr0 = curve_lr or d.get("curve_lr", lrs[-1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ex = json.load(open(os.path.join(JOBSHOP, f"v3-exact-curves-lr{float(lr0):g}.json")))            # exact expected validation reward per saved actor (no sampling)
    for k, m in NM.items():
        runs = [v for key, v in ex.items() if key.endswith("/" + k)]
        if not runs: continue
        L = min(len(v["updates"]) for v in runs); x = np.array(runs[0]["updates"][:L]); S = np.array([v["val"][:L] for v in runs]); nseed = len(S)
        curve(a1, x, S, m, lw=(3.4 if m == "Q" else 1.5 if m.startswith("Q +") else None))
        ok = [d["by_lr"][lr][k]["correct_fraction"] for lr in lrs]; n = [d["by_lr"][lr][k]["n"] for lr in lrs]
        mk = dict(marker=MK[m], ms=4, mfc="white", mew=1.1) if m in MK else {}
        a2.plot(range(len(lrs)), ok, color=COL[m], lw=(3.4 if m == "Q" else 1.5 if m.startswith("Q +") else LW[m]), label=m, zorder=3 if m.startswith("Q") else 2, **mk)
    for ax in (a1, a2): clean(ax)
    a1.axhline(d["reference"]["all_est_spt"], color=MUTED, ls=(0, (2, 2.5)), lw=1.0, zorder=1)
    a1.set(xlim=(0, 100), ylim=(0.76, 0.825), xlabel="update  (2 samples each)", ylabel="expected validation reward"); a1.set_title(f"learning curve, lr {float(lr0):g}", color=INK, fontsize=10, pad=4)
    a1.text(0.97, 0.04, f"plain SGD, no entropy bonus\nexact reward table, {nseed} seeds", transform=a1.transAxes, color=MUTED, fontsize=8.2, va="bottom", ha="right", linespacing=1.35)
    a1.annotate("best constant plan", (98, d["reference"]["all_est_spt"]), xytext=(0, -3), textcoords="offset points", color=MUTED, fontsize=8, va="top", ha="right")
    a2.set(xticks=range(len(lrs)), ylim=(-0.03, 1.06), yticks=[0, 0.25, 0.5, 0.75, 1], xlabel="learning rate", ylabel="seeds on the best plan")
    a2.set_xticklabels([f"{float(l):g}".lstrip("0") if float(l) < 1 else f"{float(l):g}" for l in lrs], fontsize=8.5)
    a2.text(0.03, 0.05, "20 seeds at lr .5, .7, 1, 2;  8 elsewhere", transform=a2.transAxes, color=MUTED, fontsize=7.8, va="bottom", ha="left")
    a2.set_title("survival across step sizes", color=INK, fontsize=10, pad=4)
    h = [Line2D([], [], color=COL[m], lw=(3.4 if m == "Q" else 1.5 if m.startswith("Q +") else LW[m]), label=m, **(dict(marker=MK[m], ms=4, mfc="white", mew=1.1) if m in MK else {})) for m in NM.values()]
    a1.legend(h, [t.get_label() for t in h], loc="upper center", bbox_to_anchor=(1.12, -0.22), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.3, handletextpad=0.6, labelcolor=INK, borderaxespad=0)
    fig.subplots_adjust(wspace=0.32); save(fig, name)


def panel_jobshop_curve(ax, part="val", data="v3-exact-curves-lr0.5.json", lr=0.5, table="exact reward table", seeds=None,
                        title=None, subtitle=True, compact=False, halo=True, ceiling=True):
    """JobShop, one dispatch rule per machine group, plain SGD, no entropy bonus, 2 samples per update.  y = the policy's
    EXACT expected reward on the 128 validation instances (sum over the 625 plans of p(plan | instance) x reward), computed
    for the actor saved every 2 updates - no sampling anywhere, so every method starts from the same initial policy and each
    converges to the plan its seeds locked onto.  subtitle=True writes the run settings under the title (the standalone
    figure); fig_row() turns them off and takes a plain task title instead, and drops the halo so the row's shared legend
    matches every panel."""
    import json
    d = json.load(open(os.path.join(JOBSHOP, data)))
    if seeds is not None:                                            # draw a chosen subset of the seeds; the file keeps all of them
        keep = {f"seed-{i}" for i in seeds}; d = {k: v for k, v in d.items() if k.split("/")[0] in keep}
    L = max(len(v["updates"]) for v in d.values())
    if L < 2: raise ValueError(f"{data}: every run is a single point - those trainers died before saving an actor")
    d = {k: v for k, v in d.items() if len(v["updates"]) == L}          # a run that died early is dropped, not silently truncated
    NM = {"reinforce": "REINFORCE", "rloo": "RLOO", "grpo": "GRPO", "otb": "OTB", "relax": "LAX", "vbase": "V", "qcv": "Q", "row_delta": "Q + Δ*",
          "exact_gradient": "exact gradient"}                        # scripts/curve/exact_gradient.py: true_grad on the exhaustive table, same schedule and step
    n = None; present = []
    for k, m in NM.items():
        runs = [v for key, v in d.items() if key.endswith("/" + k)]
        if not runs: continue
        x = np.array(runs[0]["updates"]) * 2; S = np.array([v[part] for v in runs]); n = len(S); present.append(m)
        curve(ax, x, S, m, lw=((3.0 if m == "Q" else 1.6 if m.startswith("Q +") else None) if halo else None))   # halo: Q wide, Δ* narrow on top, so a coincidence still reads
    if ceiling:                                          # the best of the 625 group plans, from the exhaustive table for that split; the row figure leaves it out
        best = 0.8163 if part == "val" else 0.8176
        ax.axhline(best, color=MUTED, ls=(0, (2, 2.5)), lw=1.0, zorder=1)
        ax.annotate("best plan", (196, best), xytext=(0, 2), textcoords="offset points", color=MUTED, fontsize=8, va="bottom", ha="right")   # above the ceiling: the exact-gradient line runs just under it
    ax.set(xlim=(0, 200), xticks=[0, 100, 200] if compact else [0, 50, 100, 150, 200], ylim=(0.760, 0.8225 if ceiling else 0.820),
           xlabel="episodes (2 per update)", ylabel="expected reward" if compact else "expected validation reward")
    ax.set_title(title or "JobShop, rule per machine group", color=INK, pad=28 if subtitle else 6,
                 **({"fontsize": 10} if compact else {})); clean(ax)   # the run settings sit under the title, clear of the curves
    if subtitle:
        ax.text(0.5, 1.055, f"plain SGD lr {lr:g}, no entropy bonus", transform=ax.transAxes, color=MUTED, fontsize=8.2, va="bottom", ha="center")
        ax.text(0.5, 1.008, f"{table} · {n} seeds", transform=ax.transAxes, color=MUTED, fontsize=8.2, va="bottom", ha="center")
    return n, present

def fig_jobshop_curve(name="fig_jobshop_curve", part="val", data="v3-exact-curves-lr0.5.json", lr=0.5, table="exact reward table", seeds=None):
    fig, ax = plt.subplots(figsize=(COLW, 2.75))
    n, present = panel_jobshop_curve(ax, part=part, data=data, lr=lr, table=table, seeds=seeds)
    h = [Line2D([], [], color=COL[m], lw=(3.0 if m == "Q" else 1.6 if m.startswith("Q +") else LW[m]), ls=LS[m], label=m,
                **(dict(marker=MK[m], ms=3.6, mfc="white", mew=1.1) if m in MK and m != "REINFORCE" else {})) for m in present]
    legend_below(ax, ncol=(4 if len(present) > 4 else 2), y=-0.27, handles=(h, [t.get_label() for t in h])); save(fig, name); return n

# ---- the three tasks in one row, one shared legend underneath --------------------------------------------------------------
ROW = ["REINFORCE", "RLOO", "GRPO", "OTB", "LAX", "V", "Q", "Q + Δ*", "exact gradient"]   # legend order for the row: baselines, ours, then the reference

def fig_row(name="fig_main_row", jobshop=dict(data="v3-fitted400-curves-lr0.5.json", lr=0.5, table="fitted table, 400 labels/instance", seeds=range(10, 30))):
    """The three tasks side by side at text width: the DFL quadratic program, the live MCP agent, and JobShop.  Each panel is
    the curve its standalone figure draws, with a plain task title and thinned x ticks; the legend is the union of the methods
    drawn, once, under the row.  Each panel's axes box carries the same height/width as the one-column figures, so the
    curves are not squeezed: the panel width follows from the margins, and the figure height follows from that aspect."""
    fig, axes = row_axes(3)
    n_qp, m_qp = panel_qp(axes[0], title="Reward for DFL Black-box", compact=True, runs=QP_SGD, **QP_SGD_R)   # plain SGD, as fig_qp_sgd, drawn as reward like the other two panels
    n_live, m_live = panel_live(axes[1], compact=True)
    n_js, m_js = panel_jobshop_curve(axes[2], title="Reward for JobShop Scheduling", subtitle=False, compact=True, halo=False, ceiling=False, **jobshop)
    for ax in axes: ax.title.set_fontsize(TITLE_ROW)      # one line each, so they are sized to clear the gap between panels
    drawn = set(m_live) | set(m_qp) | set(m_js)
    h = [Line2D([], [], color=COL[m], ls=LS[m], lw=LW[m], label=OURS.get(m, {"LAX": "(RE)LAX"}.get(m, m)),   # the row names LAX as (RE)LAX
                **(dict(marker=MK[m], ms=3.6, mfc="white", mew=1.1) if m in ("RLOO", "OTB", "GRPO", "LAX", "V") else {}))
         for m in ROW if m in drawn]
    row_legend(fig, h, ncol=5)
    save(fig, name); return {"qp": n_qp, "live": n_live, "jobshop": n_js}


def fig_jobshop_budget(name="fig_jobshop_budget"):
    """Why the JobShop ordering needs a good enough reward model: the per-update gradient variance of each
    table-based estimator against the offline label budget that the table was fitted from (625 = every plan
    labelled, i.e. the exact table).  Measured at the initial policy under the TRUE reward, 2 episodes per
    update, exactly as the trainer forms an update.  OTB is table-free, so it is flat - and it is the lowest
    of all, because its expected update is only 0.35x the true gradient (it is biased, not better)."""
    import json
    d = json.load(open(os.path.join(JOBSHOP, "v3-variance-ladder.json")))
    xs = sorted(int(k) for k in d["by_labels"]); rows = [d["by_labels"][str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(COLW, 2.75))
    for k, m in (("reinforce", "REINFORCE"), ("rloo", "RLOO"), ("otb", "OTB"), ("vbase", "V"), ("qcv", "Q"), ("row_delta", "Q + \u0394*")):
        y = np.array([r[k] for r in rows])
        mk = dict(marker=MK[m], ms=4.2, mfc="white", mew=1.2) if m in MK else dict(marker="o", ms=4.2, mfc="white", mew=1.2)
        ax.plot(xs, y, color=COL[m], ls=LS[m], lw=LW[m], label=m, solid_capstyle="round", zorder=3 if m.startswith("Q") else 2, **mk)
    ax.set_yscale("log"); ax.set(xlim=(0, 660), xticks=xs, xlabel="offline labels per instance (of 625 plans)", ylabel="gradient variance per update")
    ax.set_title("What the reward model costs", color=INK, pad=6); clean(ax)
    ax.axhline(d["true_gradient_sq"], color=MUTED, ls=(0, (2, 2.5)), lw=1.0, zorder=1)
    ax.annotate("$|\\nabla J|^2$", (300, d["true_gradient_sq"]), xytext=(0, -3), textcoords="offset points", color=MUTED, fontsize=8, va="top", ha="center")
    legend_below(ax, ncol=3); save(fig, name); return xs

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["live", "qp", "decomp", "decomp_lenient", "align", "rewards"]
    if "live" in which: print("live seeds", fig_live())
    if "qp" in which: print("qp seeds", fig_qp())
    if "qp_sgd" in which: print("qp sgd seeds", fig_qp(runs=QP_SGD, name="fig_qp_sgd", **QP_SGD_Y))
    if "decomp" in which: fig_decomp()
    if "decomp_lenient" in which: fig_decomp(DECOMP_LENIENT, "fig_variance_by_decision_lenient")
    if "align" in which: print("alignment points", fig_alignment())
    if "rewards" in which: print("seeds per panel and method", fig_two_rewards())
    if "graded" in which:                                                                          # the graded (partial-credit) reward: plug-in rows, and the exact-table rows
        print(fig_two_rewards(("graded reward", LIVE_GRADED), "fig_two_rewards_graded", ylabel="expected reward"))
        print(fig_two_rewards(("graded reward", LIVE_GRADED), "fig_two_rewards_graded_exact", exact_table=True, ylabel="expected reward"))   # both panels with the exact reward table
        _g = DECOMP_GRADED.replace("decomp_graded", "decomp_rm_graded")                     # the run that also carries the repaired reward model
        fig_decomp(_g if glob.glob(_g) else DECOMP_GRADED, "fig_variance_by_decision_graded",
                   ylim=(3e-6, 3e2), yticks=(1e-5, 1e-3, 1e-1, 1e1))      # the repaired table reaches lower than the strict figure's floor
    if "sweep" in which: print(fig_lr_sweep())
    if "manifest" in which: print("manifest seeds", fig_manifest())
    if "jobshop" in which: print("jobshop seeds per panel", fig_jobshop())
    if "jobshop_sgd" in which: fig_jobshop_sgd()
    if "jobshop_groups" in which: fig_jobshop_groups()
    if "jobshop_curve" in which: print("jobshop curve seeds", fig_jobshop_curve())
    if "row" in which: print("row seeds", fig_row())
    if "jobshop_budget" in which: print("label budgets", fig_jobshop_budget())
    if "jobshop_curve_fitted" in which:                                                          # the paper protocol: Q, V and Delta* all from the FITTED table
        import glob, re
        LAB = {"fitted": "fitted table, 64 labels/instance", "fitted200": "fitted table, 200 labels/instance", "fitted400": "fitted table, 400 labels/instance"}
        for f in sorted(glob.glob(os.path.join(JOBSHOP, "v3-fitted*-curves-lr*.json"))):
            tag, lr = re.match(r"v3-(fitted\d*)-curves-lr([\d.]+)\.json", os.path.basename(f)).groups()
            seeds = range(10, 30) if (tag, lr) == ("fitted400", "0.5") else None   # this panel is drawn on seeds 10-29; the file holds 40
            print(f"{tag} lr {lr} seeds", fig_jobshop_curve(f"fig_jobshop_curve_{tag}_lr{float(lr):g}", data=os.path.basename(f), lr=float(lr), table=LAB[tag], seeds=seeds))
    if "graded_single" in which:                                                                   # one panel per table, for the three-panel curve figure
        print(fig_live_graded(True)); print(fig_live_graded(False))
