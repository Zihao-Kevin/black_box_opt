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
COL = {"Q + row-wise Δ": "#2a78d6", "Q + Δ": "#1baf7a", "Q": "#eb6834",                     # ours: the three saturated slots, fixed
       "REINFORCE": "#1f1f1f", "exact gradient": "#9a9994",                                  # the two references: near-black, and grey dashed
       "V": "#c2648a", "RLOO": "#8c6a8f", "OTB": "#7f9ca8", "GRPO": "#b8925a", "LAX": "#6e8b5e"}   # the other baselines, muted; they also carry a marker
MK = {"RLOO": "s", "OTB": "D", "GRPO": "^", "LAX": "v", "V": "^", "REINFORCE": "s"}                  # second encoding for the muted colours
LS = {m: "-" for m in COL}; LS["exact gradient"] = (0, (4, 2.5))
LW = {m: 1.5 for m in COL}; LW.update({"Q + row-wise Δ": 2.2, "Q + Δ": 2.0, "Q": 2.0, "REINFORCE": 1.7, "exact gradient": 1.5})
ORDER = {"live": ["REINFORCE", "RLOO", "GRPO", "OTB", "Q", "Q + Δ", "Q + row-wise Δ", "exact gradient"],          # legend order = adjacency for the colour check
         "qp": ["REINFORCE", "RLOO", "GRPO", "OTB", "LAX", "Q", "Q + Δ", "Q + row-wise Δ", "exact gradient"],
         "decomp": ["REINFORCE", "V", "Q", "Q + Δ", "Q + row-wise Δ"]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "legend.fontsize": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 100, "savefig.dpi": 300})
COLW, TEXTW = 3.3, 6.75                                                  # inches: ICML column / text width; NeurIPS text width is 5.5 in

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7); ax.set_axisbelow(True)

def curve(ax, x, S, m, label=None, band=True):
    """mean over seeds (rows of S) with a standard-error band"""
    mean, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
    if band and m != "exact gradient": ax.fill_between(x, mean - se, mean + se, color=COL[m], alpha=0.14, lw=0)
    mk = dict(marker=MK[m], ms=3.6, markevery=(len(x) // 16, max(1, len(x) // 8)), mfc="white", mew=1.1) if m in ("RLOO", "OTB", "GRPO", "LAX", "V") else {}
    ax.plot(x, mean, color=COL[m], ls=LS[m], lw=LW[m], label=label or m, solid_capstyle="round", zorder=3 if m in ("Q + row-wise Δ", "Q + Δ", "Q") else 2, **mk)

def legend_below(ax, ncol, y=-0.27, handles=None):
    h, l = handles or ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=2.0, columnspacing=1.3, handletextpad=0.6, labelcolor=INK, borderaxespad=0)

def save(fig, name):
    for ext, d in (("pdf", FIGS), ("png", PNGS)):
        os.makedirs(d, exist_ok=True); fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("wrote", name)

# ---- 1. live MCP agent: warm start, plain SGD, lr 0.045, 20 seeds --------------------------------------------------------
def fig_live():
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    src = {"REINFORCE": (LIVE, "REINFORCE"), "Q": (LIVE, "Q"), "Q + Δ": (LIVE, "Q + Δ"), "Q + row-wise Δ": (LIVE, "Q + rowΔ"), "exact gradient": (LIVE, "exact gradient"),
           "RLOO": (LIVE_FIX, "RLOO"), "GRPO": (LIVE_FIX, "GRPO"), "OTB": (LIVE_FIX, "OTB")}   # the group baselines with the same step per update (worker_fix.py)
    fig, ax = plt.subplots(figsize=(COLW, 2.75))
    n = None
    for m in ORDER["live"]:
        d, name = src[m]; L = load(d, name); keep = L[0, :, 0] <= 400; n = n or len(L)     # the same 400 episodes for every method
        curve(ax, L[0, keep, 0], L[:, keep, 1], m)
    ax.set(xlim=(0, 400), ylim=(0, 1), xticks=[0, 100, 200, 300, 400], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="agent episodes", ylabel="P(success)")
    ax.set_title("Success Rate for MCP Agent", color=INK, pad=6); clean(ax)
    legend_below(ax, ncol=3); save(fig, "fig_live_mcp"); return n

# ---- 2. quadratic program: Adam, 12800 calls, 10 seeds, Q rows fitted from the black box ---------------------------------
def fig_qp():
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    src = {"REINFORCE": ("exact", "REINFORCE"), "RLOO": ("exact", "RLOO"), "GRPO": ("exact", "GRPO"), "OTB": ("exact", "OTB"), "LAX": ("exact", "LAX"),
           "Q": ("fitted", "+ Q"), "Q + Δ": ("fitted", "+ Q + Δ"), "Q + row-wise Δ": ("fitted", "+ Q + row-wise Δ"), "exact gradient": ("exact", "exact gradient")}
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n = None
    for m in ORDER["qp"]:
        kind, name = src[m]; S = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(QP, f"{kind}_{tag(name)}_s*.npy")))]); n = n or len(S)
        curve(ax, np.arange(0, 100 * (S.shape[1] - 1) + 1, 100), S, m)
    ax.set(xlim=(0, 12800), ylim=(0.19, 0.375), xticks=[0, 3200, 6400, 9600, 12800], yticks=[0.2, 0.25, 0.3, 0.35], xlabel="calls to the black box", ylabel="regret")
    ax.set_title("Regret for DFL Black-box", color=INK, pad=6); clean(ax)
    legend_below(ax, ncol=3); save(fig, "fig_qp_adam"); return n

# ---- 3. variance by decision (exact decomposition along one run) ---------------------------------------------------------
def fig_decomp(path=DECOMP, out="fig_variance_by_decision"):
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
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + Δ": "Q + Δ", "Q + row-wise Δ": "Q + rowΔ"}         # the npz's method keys
    fig, axes = plt.subplots(1, len(snap), figsize=(TEXTW, 2.1), sharey=True); x = np.arange(1, 5)
    for ax, s in zip(axes, snap):
        for m in ORDER["decomp"]:
            y, se = per_slot(s, "exact", name[m])
            if len(Ds) > 1: ax.fill_between(x, y / 10 ** se, y * 10 ** se, color=COL[m], alpha=0.16, lw=0)
            ax.plot(x, y, color=COL[m], lw=LW[m] if m != "REINFORCE" else 1.5, marker=MK.get(m, "o"), ms=4.2, label=m, zorder=3 if m.startswith("Q") else 2)
            if s > 0 and m.startswith("Q"): ax.plot(x, per_slot(s, "fitted", name[m])[0], color=COL[m], ls=(0, (2, 1.6)), lw=1.1, marker="o", ms=3.4, mfc="white", mew=1.0, zorder=3)
        ax.set(yscale="log", xticks=x, xlim=(0.7, 4.3), title=f"{4 * s} episodes"); clean(ax)
        ax.tick_params(axis="y", which="minor", left=False)
    axes[0].set_ylim(2e-5, 3e2); axes[0].set_yticks([1e-4, 1e-2, 1e0, 1e2]); axes[0].set_ylabel("variance term")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=MUTED, ls=(0, (2, 1.6)), lw=1.1, marker="o", ms=3.4, mfc="white")); l.append("fitted reward table")
    fig.supxlabel("decision in the answer", y=0.02, color=INK, fontsize=10.5)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=6, frameon=False, handlelength=2.0, columnspacing=1.3, handletextpad=0.6, labelcolor=INK)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.2, wspace=0.12)
    save(fig, out)

# ---- 4. alignment decides the gain of Delta* (QP toy, at initialization, exact, d x position scale) -----------------------
def fig_alignment():
    import sweep_alignment as sw
    g = sw.load_grid(); ds = sorted(set(g["d"].tolist())); scales = sorted(set(g["pos_scale"].tolist())); gain = 1 - g["var_qd"] / g["var_q"]   # prefix family, as in the QP figure
    BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]                     # one hue, light -> dark with d
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.3))
    for c, d in zip(BLUES, ds):                                                                    # A: alignment against the sharing scale, one line per d
        m = [g["cos2"][(g["d"] == d) & (g["pos_scale"] == ps)].mean() for ps in scales]
        a.plot(scales, m, color=c, lw=1.8, marker="o", ms=4.5, label=f"d = {d}", solid_capstyle="round")
    a.set(xscale="log", xlim=(0.085, 3.5), ylim=(0, 1), xticks=scales, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="scale of the position features", ylabel="alignment (mean cos²)")
    a.set_xticklabels([f"{ps:g}" for ps in scales]); a.tick_params(axis="x", which="minor", bottom=False); clean(a)
    a.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.6, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)
    b.plot([0, 1], [0, 1], color=COL["exact gradient"], ls=LS["exact gradient"], lw=1.4, label="closed form", zorder=1)   # B: the gain against the alignment
    for c, d in zip(BLUES, ds):
        k = g["d"] == d; b.scatter(g["cos2"][k], gain[k], s=22, color=c, edgecolor="white", linewidth=0.6, zorder=3)
    b.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], xlabel="alignment (mean cos²)", ylabel="1 − Var(Q + Δ*) / Var(Q)"); clean(b, grid="both")
    b.set_aspect("equal"); b.legend(frameon=False, fontsize=8, loc="upper left", handlelength=2.0, borderaxespad=0.3)
    fig.suptitle("Alignment vs. Gain for DFL Black-box", y=1.0, color=INK, fontsize=11)
    save(fig, "fig_alignment"); return len(gain)

# ---- 5. the live MCP agent under two rewards: exact-match (strict) and lenient ----------------------------------------
def fig_two_rewards(right=("lenient reward", None), out=None, exact_table=False, ylabel="P(success)", title="Success Rate for MCP Agent, Two Rewards"):
    """left: the exact-match runs; right: another reward's runs (title, directory).  exact_table=True draws the rows fitted from the
    exact reward table (V, Q, Q + row-wise Δ with '(exact)') instead of the plug-in rows, where those runs exist."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + row-wise Δ": "Q + rowΔ", "exact gradient": "exact gradient"}
    panels = [("exact-match reward", LIVE), (right[0], right[1] or LIVE_LENIENT)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.75), sharey=True, gridspec_kw=dict(wspace=0.1)); n = {}
    for ax, (ttl, d) in zip(axes, panels):
        for m in ["REINFORCE", "V", "Q", "Q + row-wise Δ", "exact gradient"]:
            src = name[m] + (" (exact)" if exact_table and m in ("V", "Q", "Q + row-wise Δ") and glob.glob(f"{d}/{tag(name[m] + ' (exact)')}_s*.npy") else "")
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
    """The graded reward alone, drawn like fig_live: REINFORCE, V, Q, Q + row-wise Δ, exact gradient.  exact_table=True draws the
    rows built from the exact reward table (the estimator on its own); False draws the plug-in rows fitted from the run."""
    tag = lambda m: m.replace(" ", "").replace("+", "p").replace("(", "").replace(")", "")
    load = lambda d, m: np.stack([np.load(f) for f in sorted(glob.glob(f"{d}/{tag(m)}_s*.npy")) if "partial" not in f])
    name = {"REINFORCE": "REINFORCE", "V": "V", "Q": "Q", "Q + row-wise Δ": "Q + rowΔ", "exact gradient": "exact gradient"}
    fig, ax = plt.subplots(figsize=(COLW, 2.75)); n = {}
    for m in ["REINFORCE", "V", "Q", "Q + row-wise Δ", "exact gradient"]:
        src = name[m] + (" (exact)" if exact_table and m in ("V", "Q", "Q + row-wise Δ") else "")
        L = load(LIVE_GRADED, src); keep = L[0, :, 0] <= 400; n[src] = len(L)
        curve(ax, L[0, keep, 0], L[:, keep, 1], m)
    ax.set(xlim=(0, 400), ylim=(0, 1), xticks=[0, 100, 200, 300, 400], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="agent episodes", ylabel="expected reward")
    ax.set_title("MCP Agent, Graded Reward", color=INK, pad=6); clean(ax)
    legend_below(ax, ncol=3); save(fig, out or ("fig_live_mcp_graded" if exact_table else "fig_live_mcp_graded_fitted")); return n

# ---- 7. twelve decisions per episode: the manifest format (live_long.py, _long_runs/), for the appendix ----------------------
LONG = os.path.join(ROOT, "mcp_dfl_prototype/_long_runs")
def fig_manifest():
    """Left: training on the 12-decision manifest format, all 10 tasks, Adam lr 1e-3, 200 steps of one episode per task, 40 seeds:
    REINFORCE, Q and Q + Δ with the fitted table (untried = fails), and the same two rows with the exact table (dashed).
    Right: Var(Q + Δ*) / Var(Q), exact, at the policies the exact-table Q + Δ run visits, 12 decisions against 4 (geometric mean over seeds)."""
    load = lambda name, seeds: np.stack([np.load(f"{LONG}/manifest_10jobs_lr0.001_{name}_s{s}.npz")["success"].mean(1) for s in seeds])
    rows = [("REINFORCE", "REINFORCE", "-"), ("Q", "Q0", "-"), ("Q + Δ", "Q0pΔ", "-"), ("Q", "Q_exact", (0, (4, 2.5))), ("Q + Δ", "QpΔ_exact", (0, (4, 2.5)))]
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
            out.append(np.where(live & (d[:, 3] > 0), d[:, 4] / np.maximum(d[:, 3], 1e-300), np.nan))
        out = np.array(out); k = (~np.isnan(out)).sum(0)
        return np.load(pattern.format(0))["diag"][:, 0] * 10, np.exp(np.nanmean(np.log(out), 0)), k
    for pat, seeds, c, mk, lab in [(f"{LONG}/manifest_10jobs_lr0.001_QpΔ_exact_s{{}}.npz", range(40), COL["Q + Δ"], "o", "12 decisions (manifest)"),
                                   (f"{LONG}/slots_10jobs_lr0.001_QpΔ_exact_s{{}}.npz", range(20), MUTED, "^", "4 decisions (slots)")]:
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
    src = {"REINFORCE": ("", "REINFORCE"), "Q": ("", "Q"), "Q + Δ": ("", "Q + Δ"), "Q + row-wise Δ": ("", "Q + rowΔ"), "exact gradient": ("", "exact gradient"),
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

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["live", "qp", "decomp", "decomp_lenient", "align", "rewards"]
    if "live" in which: print("live seeds", fig_live())
    if "qp" in which: print("qp seeds", fig_qp())
    if "decomp" in which: fig_decomp()
    if "decomp_lenient" in which: fig_decomp(DECOMP_LENIENT, "fig_variance_by_decision_lenient")
    if "align" in which: print("alignment points", fig_alignment())
    if "rewards" in which: print("seeds per panel and method", fig_two_rewards())
    if "graded" in which:                                                                          # the graded (partial-credit) reward: plug-in rows, and the exact-table rows
        print(fig_two_rewards(("graded reward", LIVE_GRADED), "fig_two_rewards_graded", ylabel="expected reward"))
        print(fig_two_rewards(("graded reward", LIVE_GRADED), "fig_two_rewards_graded_exact", exact_table=True, ylabel="expected reward"))   # both panels with the exact reward table
        fig_decomp(DECOMP_GRADED, "fig_variance_by_decision_graded")
    if "sweep" in which: print(fig_lr_sweep())
    if "manifest" in which: print("manifest seeds", fig_manifest())
    if "graded_single" in which:                                                                   # one panel per table, for the three-panel curve figure
        print(fig_live_graded(True)); print(fig_live_graded(False))
