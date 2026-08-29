#!/usr/bin/env python
"""Experiment runner for the token-level Q control-variate toy.

Subcommands:
  selftest   sanity checks (exact tables, unbiasedness z-tests, zero-corrections)
  bias       E1: bias table -- the "insurance" claim (unbiased for any Qhat)
  variance   E2: gradient variance vs sequence length T at fixed k (T/k separation)
  train      E3: end-to-end training at a metered oracle budget vs RLOO
  diagnose   E4: per-position credit diagnostics + support recovery
  all        everything, in that order
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_q_toy import (BlackBoxOracle, Diagnostics, QNet, EstStats,
                         estimator_stats, fit_qnet, qnet_values, sample_tokens,
                         softmax_np, train_policy)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
RES = os.path.join(HERE, "results")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Fixed categorical slots (validated reference palette, light mode); color
# follows the method across every figure, never its rank in one chart.
COL = {
    "qcv":         "#2a78d6",   # slot 1 blue  -- the protagonist
    "rloo":        "#eb6834",   # slot 2 orange
    "qcv_exact":   "#1baf7a",   # slot 3 aqua  -- privileged theory floor
    "reinforce_b": "#eda100",   # slot 4 yellow (low contrast -> direct labels)
    "reinforce":   "#e87ba4",   # slot 5 magenta
    "pathwise":    "#008300",   # slot 6 green -- the biased foil
    "qcv_delta":   "#7b52ab",   # slot 7 purple -- Q + variance-optimal Delta
}
LAB = {
    "qcv":         "Q-CV (learned)",
    "rloo":        "RLOO (K=2)",
    "qcv_exact":   "Q-CV (exact Q)",
    "reinforce_b": "REINFORCE+baseline",
    "reinforce":   "REINFORCE",
    "pathwise":    "pathwise-only",
    "qcv_delta":   "Q-CV + $\\Delta^\\star$ (ours)",
}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#c9c8bf", "axes.labelcolor": "#40403a",
    "xtick.color": "#5f5e56", "ytick.color": "#5f5e56",
    "text.color": "#33322e", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False,
})


def style_ax(ax):
    ax.grid(True, color="#ecebe4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def direct_labels(ax, series, xpos=None, fs=8):
    """series: list of (name, x, y) -- label each line at its right end,
    nudging labels apart vertically (log-aware if the y-scale is log)."""
    ends = []
    for name, x, y in series:
        ends.append([name, x[-1] if xpos is None else xpos,
                     y[-1] if xpos is None else np.interp(xpos, x, y)])
    logy = ax.get_yscale() == "log"
    key = (lambda v: np.log10(max(v, 1e-12))) if logy else (lambda v: v)
    ends.sort(key=lambda e: key(e[2]))
    lo, hi = ax.get_ylim()
    span = (np.log10(hi) - np.log10(max(lo, 1e-12))) if logy else (hi - lo)
    min_gap = 0.045 * span
    prev = None
    for e in ends:
        v = key(e[2])
        if prev is not None and v - prev < min_gap:
            v = prev + min_gap
        prev = v
        yl = 10 ** v if logy else v
        ax.annotate(f" {LAB[e[0]]}", (e[1], yl), color=COL[e[0]],
                    fontsize=fs, va="center", ha="left",
                    annotation_clip=False)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

TMAX = 64


def master_theta(V=10):
    rng = np.random.default_rng(123)
    return 0.7 * rng.standard_normal((TMAX, V))


def theta_slice(theta, T, support):
    """Keep the relevant positions' marginals identical across T: 'early'
    anchors to the sequence start, 'late' to the end."""
    return theta[:T] if support != "late" else theta[TMAX - T:]


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest(args):
    T, V, k = 16, 10, 4
    oracle = BlackBoxOracle(T, V, k, support="early", m=2, seed=0)
    diag = Diagnostics(oracle)
    theta = theta_slice(master_theta(V), T, "early")
    probs = softmax_np(theta)
    rng = np.random.default_rng(7)

    # 1. exact objective: contracted tables == torch enumeration == MC
    ef_t, gstar = diag.exact_grad(theta)
    ef_c = float(diag.contracted_tables(probs)[0])
    Y = sample_tokens(rng, probs, 200_000)
    F = diag.f_of(Y)
    assert abs(ef_t - ef_c) < 1e-10, (ef_t, ef_c)
    assert abs(F.mean() - ef_t) < 4 * F.std() / np.sqrt(len(F)) + 1e-9
    print(f"[1] exact E[f] consistent: torch {ef_t:.6f} == tables {ef_c:.6f}"
          f" ~= MC {F.mean():.6f}")

    # 2. after the last relevant position (early support), exact Q == f
    Q = diag.exact_Q(Y[:2000], probs)
    q_taken = Q[np.arange(2000)[:, None], np.arange(T)[None, :], Y[:2000]]
    resid_post = np.abs(F[:2000, None] - q_taken)[:, k:]
    assert resid_post.max() < 1e-9
    print(f"[2] exact-Q residual after support: max {resid_post.max():.2e} (=0)")

    # 3. a Q constant in v contributes exactly zero correction gradient
    from token_q_toy import _grad_batch
    Qc = np.repeat(rng.random((500, T, 1)), V, axis=2)
    g = _grad_batch(Y[:500], np.zeros((500, T)), probs, Q=Qc)
    assert np.abs(g).max() < 1e-12
    print(f"[3] constant-in-v Q gives zero pathwise term: max |g| {np.abs(g).max():.2e}")

    # 4. unbiasedness z-tests against the exact gradient
    torch.manual_seed(0)
    qrand = QNet(T, V)
    checks = [("reinforce", {}), ("rloo", {"K": 2}),
              ("qcv random-Qhat", {"qnet": qrand}),
              ("qcv_exact", {}), ("pathwise_exact", {})]
    gn = np.linalg.norm(gstar)
    for label, kw in checks:
        name = label.split()[0]
        st = estimator_stats(name, Y, F, probs, diag=diag, **kw)
        err = np.linalg.norm(st.mean() - gstar)
        clt = np.sqrt(st.var_total() / st.n)
        assert err < 4 * clt + 1e-9, (label, err, clt)
        print(f"[4] {label:18s} unbiased: |mean-g*|/|g*| = {err/gn:.4f} "
              f"(CLT scale {clt/gn:.4f})")
    print("selftest OK")


# ---------------------------------------------------------------------------
# E1: bias table -- the insurance claim
# ---------------------------------------------------------------------------

def bias(args):
    T, V, k = 16, 10, 4
    N1, N2 = args.n_small, args.n_big
    oracle = BlackBoxOracle(T, V, k, support="spread", m=2, seed=0)
    diag = Diagnostics(oracle)
    theta = theta_slice(master_theta(V), T, "spread")
    probs = softmax_np(theta)
    rng = np.random.default_rng(11)
    _, gstar = diag.exact_grad(theta)
    gn = np.linalg.norm(gstar)

    torch.manual_seed(1)
    qrand = QNet(T, V)
    qtrain = QNet(T, V)
    Yfit = sample_tokens(rng, probs, 4096)
    fit_qnet(qtrain, Yfit, diag.f_of(Yfit), steps=1500, seed=1)
    corrupt = np.random.default_rng(5).standard_normal((T, V))

    rows = [
        ("reinforce",            "reinforce",   {}),
        ("reinforce + baseline", "reinforce_b", {}),
        ("RLOO (K=2)",           "rloo",        {"K": 2}),
        ("qcv, random Qhat",     "qcv",         {"qnet": qrand}),
        ("qcv, trained Qhat",    "qcv",         {"qnet": qtrain}),
        ("qcv, corrupted Qhat",  "qcv",         {"qnet": qtrain, "corrupt": corrupt}),
        ("qcv, exact Q",         "qcv_exact",   {}),
        ("pathwise, trained Qhat", "pathwise",  {"qnet": qtrain}),
        ("pathwise, exact Q",    "pathwise_exact", {}),
    ]
    Y = sample_tokens(rng, probs, N2)
    F = diag.f_of(Y)
    out = []
    for label, name, kw in rows:
        st1 = estimator_stats(name, Y[:N1], F[:N1], probs, diag=diag, **kw)
        st2 = estimator_stats(name, Y, F, probs, diag=diag, **kw)
        e1 = np.linalg.norm(st1.mean() - gstar) / gn
        e2 = np.linalg.norm(st2.mean() - gstar) / gn
        clt2 = np.sqrt(st2.var_total() / st2.n) / gn
        verdict = "BIASED" if e2 > 5 * clt2 else "unbiased"
        out.append(dict(label=label, err_small=e1, err_big=e2, clt=clt2,
                        var=st2.var_total(), verdict=verdict))

    # stochastic-oracle variant: unbiasedness survives oracle noise
    Fz = diag.f_of(Y, with_noise=False) + 0.3 * rng.standard_normal(len(Y))
    st = estimator_stats("qcv", Y, Fz, probs, diag=diag, qnet=qtrain)
    e2 = np.linalg.norm(st.mean() - gstar) / gn
    clt2 = np.sqrt(st.var_total() / st.n) / gn
    out.append(dict(label="qcv, trained Qhat, noisy oracle (sigma=.3)",
                    err_small=float("nan"), err_big=e2, clt=clt2,
                    var=st.var_total(),
                    verdict="BIASED" if e2 > 5 * clt2 else "unbiased"))

    w = max(len(r["label"]) for r in out)
    print(f"\nE1 bias check  (T={T}, k={k}, |g*|={gn:.4f}; rel. errors)")
    print(f"{'estimator':{w}s}  err@{N1//1000}k  err@{N2//1000}k  CLT@{N2//1000}k"
          f"   variance  verdict")
    for r in out:
        print(f"{r['label']:{w}s}  {r['err_small']:7.4f}  {r['err_big']:7.4f}"
              f"  {r['clt']:7.4f}  {r['var']:9.4f}  {r['verdict']}")
    with open(os.path.join(RES, "bias.json"), "w") as fh:
        json.dump(out, fh, indent=1)


# ---------------------------------------------------------------------------
# E2: variance vs T at fixed k
# ---------------------------------------------------------------------------

def variance(args):
    V, k = 10, 4
    Ts = [8, 16, 32, 64]
    N = args.n_var
    theta_m = master_theta(V)
    res = {}
    for support in ("early", "late"):
        for T in Ts:
            t0 = time.time()
            oracle = BlackBoxOracle(T, V, k, support=support, m=2, seed=0)
            diag = Diagnostics(oracle)
            theta = theta_slice(theta_m, T, support)
            probs = softmax_np(theta)
            rng = np.random.default_rng(100 + T)
            torch.manual_seed(T)
            qnet = QNet(T, V)
            Yfit = sample_tokens(rng, probs, 4096)
            fit_qnet(qnet, Yfit, diag.f_of(Yfit), steps=1200, seed=T)
            Y = sample_tokens(rng, probs, N)
            F = diag.f_of(Y)
            for name, kw, K in [("reinforce_b", {}, 1), ("rloo", {"K": 2}, 2),
                                ("qcv", {"qnet": qnet}, 1),
                                ("qcv_exact", {}, 1)]:
                st = estimator_stats(name, Y, F, probs, diag=diag, **kw)
                res[(support, T, name)] = st.var_total() * K   # per oracle call
            print(f"  variance {support} T={T} done ({time.time()-t0:.0f}s)")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, support in zip(axes, ("early", "late")):
        style_ax(ax)
        series = []
        for name in ("reinforce_b", "rloo", "qcv", "qcv_exact"):
            ys = [res[(support, T, name)] for T in Ts]
            ax.plot(Ts, ys, color=COL[name], lw=2, marker="o", ms=4.5,
                    zorder=3)
            series.append((name, np.array(Ts, float), np.array(ys)))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ts, [str(t) for t in Ts])
        ax.set_xlabel("sequence length T  (k = 4 relevant positions)")
        ax.set_title(f"relevant tokens {support}")
        if support == "early":
            ax.set_ylabel("gradient variance per oracle call")
        direct_labels(ax, series)
    fig.suptitle("E2 - variance vs sequence length at fixed k", x=0.02,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.90, 0.97))
    fig.savefig(os.path.join(FIGS, "e2_variance_vs_T.png"),
                bbox_inches="tight")
    with open(os.path.join(RES, "variance.json"), "w") as fh:
        json.dump({f"{s}|{t}|{n}": v for (s, t, n), v in res.items()}, fh,
                  indent=1)
    print("saved figs/e2_variance_vs_T.png")


# ---------------------------------------------------------------------------
# E3: end-to-end training at a metered oracle budget
# ---------------------------------------------------------------------------

def train(args):
    methods = args.methods.split(",")
    budget, seeds = args.budget, list(range(args.seeds))
    T, k, m = args.T, args.k, args.m
    tag = (f"T{T}_k{k}" + (f"_w{args.q_warmup}" if args.q_warmup else "")
           + (f"_qs{args.q_steps}" if args.q_steps > 1 else ""))
    grid = np.arange(0, budget + 1, 100)
    all_res = {}
    for support in args.supports.split(","):
        for meth in methods:
            curves = []
            for sd in seeds:
                t0 = time.time()
                wu = args.q_warmup if meth in ("qcv", "pathwise") else 0
                out = train_policy(meth, T=T, V=10, k=k, m=m,
                                   support=support, budget=budget, K=2,
                                   lr=args.lr, seed=1000 + sd, q_warmup=wu,
                                   q_steps=args.q_steps)
                curves.append(np.interp(grid, out["calls"], out["ef"]))
                print(f"  train {support} {meth} seed{sd}: "
                      f"final regret {out['ef'][-1]:.4f} "
                      f"({time.time()-t0:.0f}s)")
            all_res[(support, meth)] = np.stack(curves)
    np.savez(os.path.join(RES, f"train_curves_{tag}.npz"), grid=grid,
             **{f"{s}|{m}": c for (s, m), c in all_res.items()})

    sups = args.supports.split(",")
    fig, axes = plt.subplots(1, len(sups), figsize=(4.6 * len(sups), 3.5),
                             sharey=True, squeeze=False)
    for ax, support in zip(axes[0], sups):
        style_ax(ax)
        series = []
        for meth in methods:
            C = all_res[(support, meth)]
            mean = C.mean(0)
            ax.fill_between(grid, C.min(0), C.max(0), color=COL[meth],
                            alpha=0.13, lw=0, zorder=2)
            ax.plot(grid, mean, color=COL[meth], lw=2, zorder=3)
            series.append((meth, grid.astype(float), mean))
        ax.set_xlabel("cumulative oracle calls")
        ax.set_title(f"relevant tokens {support}")
        direct_labels(ax, series)
    axes[0][0].set_ylabel("exact expected regret  E[f]")
    wtxt = (f"; {args.q_warmup}-call Qhat warmup charged"
            if args.q_warmup else "")
    fig.suptitle(f"E3 - training at a metered oracle budget "
                 f"(T={T}, k={k}{wtxt}; mean and seed range)", x=0.02,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    fig.savefig(os.path.join(FIGS, f"e3_training_budget_{tag}.png"),
                bbox_inches="tight")
    print(f"saved figs/e3_training_budget_{tag}.png")


# ---------------------------------------------------------------------------
# E4: per-position credit diagnostics
# ---------------------------------------------------------------------------

def diagnose(args):
    # Measured at a FIXED near-uniform policy: at a converged policy the
    # gradient (and so any credit signal) vanishes exactly on the decided
    # positions, which confounds the exhibit.
    T, V, k = 32, 10, 4
    support = "spread"
    oracle = BlackBoxOracle(T, V, k, support=support, m=2, seed=0)
    diag = Diagnostics(oracle)
    S = diag.S
    theta = theta_slice(master_theta(V), T, support)
    probs = softmax_np(theta)
    rng = np.random.default_rng(3)
    torch.manual_seed(9)
    qnet = QNet(T, V)
    Yfit = sample_tokens(rng, probs, 4096)
    fit_qnet(qnet, Yfit, diag.f_of(Yfit), steps=2500, seed=9)
    Y = sample_tokens(rng, probs, args.n_var)
    F = diag.f_of(Y)

    prof = {}
    for name, kw in [("reinforce_b", {}), ("qcv", {"qnet": qnet}),
                     ("qcv_exact", {})]:
        st = estimator_stats(name, Y, F, probs, diag=diag, **kw)
        prof[name] = st.var_per_pos()

    # support recovery: mean pathwise-term magnitude per position
    Q = qnet_values(qnet, Y[:4096])
    pQ = (probs[None] * Q).sum(-1, keepdims=True)
    corr = probs[None] * (Q - pQ)
    mag = np.linalg.norm(corr, axis=2).mean(0)
    topk = np.sort(np.argsort(mag)[-k:])
    print(f"  support recovery: top-{k} pathwise positions {topk.tolist()} "
          f"vs true S {S.tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
    ax = axes[0]
    style_ax(ax)
    for s in S:
        ax.axvspan(s - 0.5, s + 0.5, color="#f1efe7", zorder=0)
    series = []
    for name in ("reinforce_b", "qcv", "qcv_exact"):
        y = np.maximum(prof[name], 1e-12)
        ax.plot(np.arange(T), y, color=COL[name], lw=2, zorder=3)
        series.append((name, np.arange(T, dtype=float), y))
    ax.set_yscale("log")
    ax.set_xlabel("position t   (shaded = oracle-relevant S)")
    ax.set_ylabel("per-position variance contribution")
    ax.set_title("where the variance lives")
    direct_labels(ax, series)

    ax = axes[1]
    style_ax(ax)
    bars = ax.bar(np.arange(T), mag, color=COL["qcv"], width=0.72, zorder=3)
    for s in S:
        bars[s].set_edgecolor("#33322e")
        bars[s].set_linewidth(1.2)
    ax.set_xlabel("position t   (outlined = true S)")
    ax.set_ylabel("mean pathwise-term magnitude")
    ax.set_title("learned Qhat: credit concentrates on S")
    fig.suptitle("E4 - per-position credit diagnostics "
                 f"(T={T}, k={k}, support {support})", x=0.02, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    fig.savefig(os.path.join(FIGS, "e4_credit_diagnostics.png"),
                bbox_inches="tight")
    json.dump({"topk": topk.tolist(), "S": S.tolist(),
               "profiles": {n: p.tolist() for n, p in prof.items()},
               "pathwise_mag": mag.tolist()},
              open(os.path.join(RES, "diagnose.json"), "w"), indent=1)
    print("saved figs/e4_credit_diagnostics.png")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "bias", "variance", "train",
                                    "diagnose", "all"])
    ap.add_argument("--n-small", type=int, default=20_000)
    ap.add_argument("--n-big", type=int, default=200_000)
    ap.add_argument("--n-var", type=int, default=30_000)
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--supports", type=str, default="early,late")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--q-warmup", type=int, default=0)
    ap.add_argument("--q-steps", type=int, default=1)
    ap.add_argument("--methods", type=str,
                    default="reinforce_b,rloo,qcv,qcv_exact,pathwise")
    args = ap.parse_args()
    steps = ([args.cmd] if args.cmd != "all"
             else ["selftest", "bias", "variance", "train", "diagnose"])
    for s in steps:
        print(f"=== {s} ===")
        globals()[s](args)


if __name__ == "__main__":
    main()
