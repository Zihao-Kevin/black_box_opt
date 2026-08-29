#!/usr/bin/env python
"""Runner for the predict-then-optimize (DFL) motivating toy.

Subcommands:
  selftest   exact-table sanity + unbiasedness z-tests (coordinate estimator)
  credit     D1: per-coordinate credit -- exact vs learned, per instance
  train      D2: end-to-end budget race + held-out generalization
  all        everything, in that order
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_q_toy import EstStats, _grad_batch, sample_tokens, softmax_np
from dfl_toy import (DFLOracle, DFLDiagnostics, ContextQNet, ctx_qnet_values,
                     fit_ctx_qnet, train_dfl)
from run_toy import COL, LAB, style_ax, direct_labels, FIGS, RES

V = 5   # the learner's value grid {0, .25, .5, .75, 1}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest(args):
    oracle = DFLOracle(seed=0)
    diag = DFLDiagnostics(oracle, V)
    d = oracle.d
    rng = np.random.default_rng(1)

    # 1) table matches the public oracle interface (real-valued chat)
    Yr = rng.integers(0, V, size=(200, d))
    tab = diag.f_of(0, Yr)
    ora = np.array([oracle(0, y / (V - 1)) for y in Yr])
    assert np.allclose(tab, ora), "table/oracle mismatch"
    print("[1] f_table == oracle(chat) on 200 random grid vectors  OK")

    # 2) uniform-policy E[f] via cascade == table mean
    u = np.full((d, V), 1.0 / V)
    ef = diag.exact_ef(0, u)
    assert abs(ef - diag.f_tables[0].mean()) < 1e-10
    print(f"[2] cascade E[f] at uniform == table mean ({ef:.4f})  OK")

    # 3) unbiasedness z-tests at a random policy, instance 0
    logits = 0.7 * rng.standard_normal((d, V))
    probs = softmax_np(logits)
    N = 120_000
    Y = sample_tokens(rng, probs, N)
    F = diag.f_of(0, Y)
    _, gex = diag.exact_grad(0, logits)
    gnorm = np.linalg.norm(gex)

    qnet = ContextQNet(d, V, oracle.p)          # untrained = random Qhat
    ar = np.arange(d)
    ests = {}
    ests["reinforce"] = ("score, coeff f", None)
    ests["qcv_random"] = ("random Qhat + correction", qnet)
    ests["qcv_exact"] = ("exact Q + correction", "exact")
    ests["pathwise_exact"] = ("correction only, exact Q", "exact")
    for name, (desc, q) in ests.items():
        st = EstStats(d, V)
        for s in range(0, N, 8192):
            Yb, Fb = Y[s:s + 8192], F[s:s + 8192]
            if q is None:
                g = _grad_batch(Yb, np.broadcast_to(Fb[:, None], Yb.shape),
                                probs)
            else:
                Q = (diag.exact_Q(0, Yb, probs) if q == "exact"
                     else ctx_qnet_values(qnet, oracle.X[0], Yb))
                q_taken = Q[np.arange(len(Yb))[:, None], ar[None, :], Yb]
                coeff = (np.zeros_like(q_taken) if name.startswith("pathwise")
                         else Fb[:, None] - q_taken)
                g = _grad_batch(Yb, coeff, probs, Q=Q)
            st.add(g)
        err = np.linalg.norm(st.mean() - gex) / gnorm
        clt = np.sqrt(st.var_total() / N) / gnorm
        tag = "OK" if err < 5 * max(clt, 1e-9) else "BIASED?"
        print(f"[3] {name:<15} rel err {err:.4f}  (CLT scale {clt:.4f}) "
              f"var {st.var_total():.4f}  {tag}")
        assert tag == "OK", name


# ---------------------------------------------------------------------------
# D1: per-coordinate credit, exact vs learned, per instance
# ---------------------------------------------------------------------------

def credit(args):
    oracle = DFLOracle(seed=0)
    diag = DFLDiagnostics(oracle, V)
    d, n = oracle.d, oracle.n_train
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    u = np.full((d, V), 1.0 / V)                 # fixed near-uniform predictor

    # learner side: on-policy replay across instances, one shared Qhat
    per = 700
    I = np.repeat(np.arange(n), per)
    Y = sample_tokens(rng, u, n * per)
    F = np.concatenate([diag.f_of(i, Y[i * per:(i + 1) * per])
                        for i in range(n)])
    qnet = ContextQNet(d, V, oracle.p)
    fit_ctx_qnet(qnet, oracle.X, I, Y, F, steps=2500)

    def spread(Q):                               # (B, d, V) -> (d,)
        return (Q.max(-1) - Q.min(-1)).mean(0)

    ex, le, rho = [], [], []
    B = 2000
    for i in range(n):
        Yb = sample_tokens(rng, u, B)
        ex.append(spread(diag.exact_Q(i, Yb, u)))
        le.append(spread(ctx_qnet_values(qnet, oracle.X[i], Yb)))
        rho.append(spearman(ex[-1], le[-1]))
        top_ex = np.argsort(ex[-1])[::-1][:3]
        top_le = np.argsort(le[-1])[::-1][:3]
        print(f"instance {i}: spearman(exact, learned) = {rho[-1]:+.2f}   "
              f"top-3 credit exact {sorted(top_ex.tolist())} "
              f"learned {sorted(top_le.tolist())}")
    print(f"mean spearman over {n} instances: {np.mean(rho):.2f}")

    # pick the instance pair with the most different exact-credit rankings
    pair, worst = (0, 1), 2.0
    for a in range(n):
        for b in range(a + 1, n):
            r = spearman(ex[a], ex[b])
            if r < worst:
                worst, pair = r, (a, b)
    print(f"most-different profiles: instances {pair} "
          f"(cross-instance spearman {worst:+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    w = 0.6
    for ax, i in zip(axes, pair):
        style_ax(ax)
        ax.bar(np.arange(d), ex[i], w, color=COL["qcv_exact"], zorder=3,
               label="exact credit  E[max$_v$Q $-$ min$_v$Q]")
        ax.plot(np.arange(d), le[i], "o", color=COL["qcv"], ms=5, zorder=4,
                label="learned $\\hat{Q}$ (replay only)")
        ax.set_title(f"instance {i}  ($\\rho_s$ = {spearman(ex[i], le[i]):+.2f})",
                     loc="left")
        ax.set_xlabel("coordinate of predicted $\\hat{c}$")
        ax.set_xticks(np.arange(d))
    axes[0].set_ylabel("per-coordinate credit")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("D1 -- which price predictions does the hidden problem "
                 "hinge on? (per-instance credit)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "dfl_credit.png"), bbox_inches="tight")
    print("saved figs/dfl_credit.png")

    json.dump({"spearman": rho, "pair": list(pair),
               "exact": [e.tolist() for e in ex],
               "learned": [l.tolist() for l in le]},
              open(os.path.join(RES, "dfl_credit.json"), "w"), indent=1)


# ---------------------------------------------------------------------------
# D2: end-to-end budget race + held-out generalization
# ---------------------------------------------------------------------------

def train(args):
    methods = args.methods.split(",")
    seeds = list(range(args.seeds))
    grid = np.arange(0, args.budget + 1, 100)
    curves, curves_te, finals = {}, {}, {}
    for meth in methods:
        Ctr = np.zeros((len(seeds), len(grid)))
        Cte = np.zeros((len(seeds), len(grid)))
        for s in seeds:
            out = train_dfl(meth, V=V, budget=args.budget, seed=s,
                            oracle_seed=0, lr=args.lr,
                            q_warmup=args.q_warmup if meth in ("qcv", "pathwise")
                            else 0)
            Ctr[s] = np.interp(grid, out["calls"], out["train"])
            Cte[s] = np.interp(grid, out["calls"], out["test"])
            print(f"{meth} seed {s}: train {Ctr[s, -1]:.4f}  "
                  f"held-out {Cte[s, -1]:.4f}", flush=True)
        curves[meth], curves_te[meth] = Ctr, Cte
        finals[meth] = dict(train=float(Ctr[:, -1].mean()),
                            test=float(Cte[:, -1].mean()))

    np.savez(os.path.join(RES, "dfl_train_curves.npz"), grid=grid,
             **{f"tr|{m}": curves[m] for m in methods},
             **{f"te|{m}": curves_te[m] for m in methods})

    thr = [0.10, 0.03, 0.01]
    print(f"\nmean oracle calls to train-regret threshold "
          f"({len(seeds)} seeds):")
    for meth in methods:
        row = []
        for th in thr:
            hits = [grid[np.argmax(C < th)] if (C < th).any() else np.nan
                    for C in curves[meth]]
            row.append(np.nanmean(hits))
        print(f"  {LAB[meth]:<22} " +
              "  ".join(f"<{t}: {r:6.0f}" for t, r in zip(thr, row)) +
              f"   final train {finals[meth]['train']:.4f} "
              f"held-out {finals[meth]['test']:.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    style_ax(ax)
    series = []
    for meth in methods:
        C = np.maximum(curves[meth], 2e-4)
        ax.fill_between(grid, C.min(0), C.max(0), color=COL[meth],
                        alpha=0.13, lw=0, zorder=2)
        ax.plot(grid, C.mean(0), color=COL[meth], lw=2, zorder=3)
        series.append((meth, grid.astype(float), C.mean(0)))
    ax.set_yscale("log")
    ax.set_xlabel("cumulative oracle calls (Q-CV includes the "
                  f"{args.q_warmup}-call $\\hat{{Q}}$ warmup)")
    ax.set_ylabel("exact mean regret, training instances")
    ax.set_title("D2 -- predict-then-optimize with an opaque oracle "
                 f"(d=8, m=3, {args.seeds} seeds)", loc="left")
    direct_labels(ax, series)
    fig.tight_layout(rect=(0, 0, 0.82, 1.0))
    fig.savefig(os.path.join(FIGS, "dfl_budget.png"), bbox_inches="tight")
    print("saved figs/dfl_budget.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "credit", "train", "all"])
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--q-warmup", type=int, default=200)
    ap.add_argument("--methods", type=str,
                    default="reinforce_b,rloo,qcv,qcv_exact")
    args = ap.parse_args()
    steps = ([args.cmd] if args.cmd != "all"
             else ["selftest", "credit", "train"])
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        globals()[s](args)


if __name__ == "__main__":
    main()
