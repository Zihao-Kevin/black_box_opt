# Token-level Q control variates for black-box DFL — toy experiment

End-to-end toy for the credit-assignment estimator (project notes 2026-08-05,
"Q-choice convergence"). A predictor emits `T` digit tokens; an **opaque
oracle** maps the sequence to a scalar decision loss. The learner sees only
`(sequence, scalar)` pairs and a call counter — not the inner optimization
problem, not which tokens are read, not how they decode.

Hidden inside the oracle (learner never sees this): only `k` of the `T`
positions decode to a parameter vector `ĉ`; the oracle solves a top-m
selection problem under `ĉ` and returns the **normalized decision regret**
under the true parameters — piecewise-constant in `ĉ`, the classic hard case
for surrogate gradients.

## Estimator

Per-token Q control variate with exact vocab-marginalized correction:

$$
\widehat{g} \;=\; \sum_{t=1}^{T}
\underbrace{\bigl(f(y) - \widehat{Q}_t(y_{\lt t}, y_t)\bigr)\,
\nabla_\theta \log \pi_\theta(y_t \mid y_{\lt t})}_{\text{score term (insurance)}}
\;+\;
\underbrace{\nabla_\theta \sum_{v \in \mathcal{V}}
\pi_\theta(v \mid y_{\lt t})\, \widehat{Q}_t(y_{\lt t}, v)}_{\text{pathwise term (credit)}}
$$

- **Score term** = insurance: unbiased for ANY Q̂ (the correction cancels the
  CV conditionally on the prefix, per position). Q̂ error costs variance, never bias.
- **Pathwise term** = credit: with exact `Q_t = E[f | y_<t, y_t=v]` it is the
  Sutton all-actions gradient; positions whose Q is constant in `v` contribute
  exactly zero.
- Q̂ = GRU credit head trained by return regression on **replayed** oracle
  calls (no fresh calls needed) — the bias-free "shadow solver" living inside the CV.

Policy = independent categorical per position (logits table `T×V`), so exact
objective / gradient / Q are computable by enumeration (`Diagnostics`, a
privileged measurement harness — never available to the learner).

## Files

- `token_q_toy.py` — oracle, privileged diagnostics, QNet, analytic per-sample
  gradients, metered training loop.
- `run_toy.py` — experiments: `selftest | bias | variance | train | diagnose | all`.
- `make_e3_final_fig.py` — rebuilds the zoomed hard-regime E3 figure from saved curves.
- `dfl_toy.py`, `run_dfl.py` — predict-then-optimize companion toy (no
  autoregression): features → factorized-categorical predictor → ĉ → opaque
  oracle; see "DFL motivating example" below.
- `figs/`, `results/` — outputs.

Run with the `py39` env: `python run_toy.py all`.

## Results

### E1 — the insurance claim (bias table, T=16, k=4, N=200k)

| estimator | rel. err of mean | variance | verdict |
|---|---|---|---|
| REINFORCE | 0.067 | 6.64 | unbiased |
| REINFORCE + baseline | 0.035 | 1.46 | unbiased |
| RLOO (K=2) | 0.045 | 1.47 (×2 calls) | unbiased |
| Q-CV, random Q̂ | 0.066 | 6.27 | unbiased |
| Q-CV, trained Q̂ | 0.025 | **0.98** | unbiased |
| Q-CV, corrupted Q̂ | 0.079 | 11.14 | **still unbiased** |
| Q-CV, exact Q | 0.024 | 0.86 | unbiased |
| pathwise-only, trained Q̂ | **0.398 (stuck)** | 0.004 | **BIASED** |
| pathwise-only, exact Q | 0.002 | 0.005 | unbiased |
| Q-CV, trained Q̂, noisy oracle σ=0.3 | 0.041 | 2.21 | unbiased |

All errors are consistent with CLT shrinkage except pathwise-only with an
inexact Q̂ — the actor-critic / surrogate-descent foil, whose error does not
move with N. Corrupting Q̂ multiplies variance by ~11 but leaves the mean
exactly on target. Oracle noise is absorbed the same way. **No unbiased credit
assignment without a model; no safe model without the correction term.**

### E2 — T/k separation (variance per oracle call, k=4 fixed)

Relevant tokens **early** (decisive block + long irrelevant padding):

| T | REINFORCE+b | RLOO K=2 | Q-CV learned | Q-CV exact |
|---|---|---|---|---|
| 8 | 0.74 | 1.50 | 0.20 | 0.15 |
| 16 | 1.47 | 2.97 | 0.25 | 0.15 |
| 32 | 2.97 | 6.01 | 0.34 | 0.15 |
| 64 | 5.92 | 11.95 | 0.49 | **0.15** |

Score-function baselines grow **linearly in T** (uniform credit *is* the
variance); exact-Q is **flat** — at T=64 that is 40× less than
REINFORCE+baseline and 80× less per call than RLOO, and the learned Q̂ keeps
12–25× of it. The separation grows with sequence length, as predicted.

Relevant tokens **late** (honest boundary): all methods grow ~linearly and
Q-CV ≈ REINFORCE+baseline. Positions *before* the decisive tokens cannot beat
the conditional mean — `Var(f | y_≤t)` is still the full `Var(f)` there.
Conditioning only drains variance *after* decisive tokens are fixed. The sweet
spot is therefore: long sequences whose oracle-relevant tokens come early/mid
— and per-position irreducible variance `E[Var(f|y_≤t)·‖∇log p_t‖²]` is the
principled statement of this (see E4).

### E3 — end-to-end training at a metered oracle budget

**Easy regime (T=32, k=4, no Q̂ warmup)** — mean oracle calls to regret < 0.03
(4 seeds, early support): exact-Q **250**, REINFORCE+b 500, Q-CV learned 575,
RLOO(2) 775. Exact-Q wins 2–3×, learned Q-CV beats RLOO ~1.4×, but *trails
plain REINFORCE+baseline*: the task is solved in ~500 calls, exactly the
window where the online-trained Q̂ is still bad. **The economics claim in
reverse — a task cheap enough to solve with a mean baseline never amortizes a
credit model.** This regime boundary belongs in the paper.

**Hard regime (T=32, k=6, m=3, 250-call Q̂ warmup charged to the budget)** —
mean oracle calls to threshold (4 seeds, early support):

| method | <0.10 | <0.03 | <0.01 | regret@1k |
|---|---|---|---|---|
| Q-CV exact Q | **200** | **325** | **525** | 0.0028 |
| REINFORCE+baseline | 300 | 525 | 925 | 0.0079 |
| Q-CV learned (qs=1) | 500 | 725 | 1100 | 0.0116 |
| Q-CV learned (qs=4) | 500 | 750 | 1125 | 0.0131 |
| RLOO (K=2) | 500 | 900 | 1550 | 0.0225 |
| pathwise-only | 325 | 400 | 400 | 0.0005 |

Readings (see `figs/e3_training_budget_final.png`):

1. **Make-or-break vs RLOO: passes.** Learned Q-CV needs 1.25–1.4× fewer
   calls than RLOO at every threshold, in the forced K=1–2 regime.
2. **The estimator's ceiling wins outright**: exact-Q beats
   REINFORCE+baseline 1.6–1.8× and RLOO 2.8–3×. The per-position variance
   theory (E2) translates into real budget savings.
3. **The learned credit model is the whole game.** Learned Q-CV trails
   REINFORCE+baseline on *total* budget — but the deficit is exactly the
   warmup: post-warmup it reaches each threshold in ~475/850 marginal calls
   vs 525/925 from scratch (~10% faster per call), and in the training curve
   it crosses below REINFORCE+baseline around 1400 calls. With warmup
   amortized (across instances/contexts, or replayed from logged data — which
   the estimator permits bias-free) the comparison flips.
4. **More regression steps don't help** (qs=4 ≈ qs=1): the gap to exact-Q is
   NOT an optimization-budget problem. Remaining levers: replay
   freshness-weighting (two-timescale tracking, challenge 5), the stage-2
   variance fine-tune, and above all *structure in the CV* — a generic GRU
   value head is precisely the Mirage-objection baseline; the paper's thesis
   that the differentiator must be oracle structure now has a number attached.
5. **pathwise-only is fastest here and silently wrong in general**: with a
   deterministic small-support oracle the Q̂ fit is good enough that biased
   surrogate descent lands on the optimum (the honest-limit note in the
   project memory, exhibited). E1 shows the same construction is
   irreparably biased the moment Q̂ is imperfect — and without the score
   term there is no diagnostic. The correction is the insurance you cannot
   see you need.

### E4 — per-position credit diagnostics (T=32, k=4, spread support)

At a fixed near-uniform policy with an on-policy-fitted Q̂:

- **Where the variance lives**: exact-Q per-position variance is a decreasing
  staircase — it drops each time a decisive token gets fixed and collapses to
  ~0 at the last relevant position (there `Q_T = f`: the zero-variance exact
  all-actions gradient). REINFORCE+baseline is flat. This staircase is the
  irreducible credit-uncertainty profile `E[Var(f|y_≤t)‖∇log p_t‖²]`.
- **Support recovery**: ranking positions by mean pathwise-term magnitude of
  the *learned* Q̂ recovers the hidden support exactly:
  top-4 = {0, 10, 21, 31} = S. The estimator grades its own credit assigner.
- Caveat found while building this: measure credit at a *fixed* policy. At a
  mid-training policy the gradient (hence any credit signal) vanishes on
  already-decided positions — an identifiability limit of gradient-based
  credit diagnostics, worth a remark in the paper.

## DFL motivating example (`dfl_toy.py`, `run_dfl.py`)

Companion toy in classic **predict-then-optimize** form — no autoregression.
Instance features `x` → stochastic predictor (factorized categorical over a
5-point value grid; all `d=8` coordinates emitted **simultaneously**) →
predicted cost vector `ĉ ∈ [0,1]^8` → opaque oracle. Hidden inside: true
costs `c*(x) = σ(Hx)` and a top-3 selection problem; the oracle returns
normalized decision regret. 6 training instances + 4 held-out (evaluated only
through privileged diagnostics — the oracle is never called on them).

The estimator is unchanged: positions = coordinates of `ĉ` in an arbitrary
fixed order, so `Q_j(ĉ_<j, v) = E[f | first j coords fixed, coord j = v]` is
**per-parameter credit** — how much the hidden decision problem hinges on the
j-th predicted price, *for this instance*. Stochastic prediction is the honest
gradient channel: regret is piecewise-constant in `ĉ`, so `∂f/∂ĉ = 0` a.e.
and any smoothed-surrogate gradient is a modeling choice; the score-function
channel is exact.

Run: `python run_dfl.py all` (selftest | credit | train).

### Algorithm (coordinate-level Q-CV for black-box predict-then-optimize)

```
Input:  features {x_i}, i = 1..n;  opaque oracle O(i, ĉ) -> regret f (the ONLY
        access to the decision problem);  value grid G = {0, 1/(V-1), .., 1};
        oracle budget B, warmup N0;  step sizes lr (predictor), q_lr (credit).
Output: predictor ψ (and credit head φ, reusable as a credit diagnostic).

Model:  π_ψ(ĉ | x) = Π_j Cat(v_j ; softmax(g_ψ(x)_j)),   ĉ_j = G[v_j]
        Q̂_φ(x, ĉ_<j , v): contextual credit head (GRU; x enters via h0)

Warmup (N0 oracle calls, charged):
    sample ĉ ~ π_ψ(·|x_i) round-robin;  store (i, ĉ, O(i, ĉ)) in replay D
    fit φ by return regression:  min_φ Σ_j (Q̂_φ(x, ĉ_<j, ĉ_j) − f)²  over D

Loop until B oracle calls spent:
    1  i <- next instance (round-robin);  compute logits g_ψ(x_i)
    2  sample ĉ ~ π_ψ(·|x_i);   f <- O(i, ĉ)          # ONE oracle call
    3  gradient estimate (unbiased for ANY φ):
         g = Σ_j [ (f − Q̂_φ(x_i, ĉ_<j, ĉ_j)) ∇_ψ log π_j(ĉ_j | x_i)     # insurance
                   + ∇_ψ Σ_{v∈G} π_j(v | x_i) Q̂_φ(x_i, ĉ_<j, v) ]       # credit
    4  ψ <- Adam(ψ, g, lr)
    5  append (i, ĉ, f) to D;  one regression step on φ over D (no new calls)
```

Line 3 is the whole method: the score term keeps the estimator exactly
unbiased whatever φ is; the pathwise term routes gradient mass to the
coordinates the oracle actually hinges on. Baselines replace line 3 with
f·∇logπ (REINFORCE), (f−EMA_i)·∇logπ (per-instance baseline), or a
leave-one-out group baseline (RLOO, K calls per step); the exact-Q ceiling
replaces Q̂_φ by the enumerated E[f | ĉ_<j, ĉ_j = v].

### Selftest (instance 0, d=8, N=120k)

Unbiased with a *random* Q̂ (rel. err at CLT scale), and exact-Q variance is
**24×** below REINFORCE (0.11 vs 2.65) already at d=8 — every coordinate is
oracle-relevant here, so this gap is pure conditional-variance cancellation,
not filler-token suppression.

### D1 — per-instance credit (`figs/dfl_credit.png`)

At a fixed uniform predictor with a replay-trained contextual Q̂ (GRU + feature
conditioning, 4200 samples):

- Exact per-coordinate credit `E[max_v Q_j − min_v Q_j]` vs the same
  functional of the learned Q̂: mean Spearman **0.81** over 6 instances;
  top-3 credited coordinates match exactly on 3/6 instances, 2/3 overlap
  otherwise.
- **Credit is instance-dependent**: instances 0 and 1 have near-*opposite*
  exact credit rankings (cross-instance Spearman **−0.86**). No static/global
  feature-importance scheme can represent this — credit must be a function of
  `x`, which is exactly what the contextual Q̂ provides from scalar feedback
  alone.

### D2 — budget race + generalization (`figs/dfl_budget.png`)

Mean oracle calls to train-regret threshold (3 seeds, budget 6000, round-robin
over instances; Q-CV charged a 200-call Q̂ warmup):

| method | <0.10 | <0.03 | <0.01 | final train | final held-out |
|---|---|---|---|---|---|
| Q-CV exact Q | **200** | **367** | **600** | 0.0011 | 0.22 |
| REINFORCE+baseline | 400 | 800 | 1233 | 0.0028 | 0.20 |
| Q-CV learned | 533 | 1500 | — | 0.0265 | 0.20 |
| RLOO (K=2) | 500 | 2167† | 1800† | 0.0180 | 0.20 |

† different seed subsets: only 1/3 RLOO seeds ever reached <0.01.

Readings:

1. **The estimator's ceiling wins 2×+ at every threshold** in genuine
   predict-then-optimize form — the E2/E3 variance story is not an artifact of
   the autoregressive toy.
2. **The easy-regime boundary reappears, now contextual**: six per-instance
   EMA scalars solve this task in ~1000 calls, and the online contextual Q̂
   (which has the strictly harder job — credit must generalize across
   instances) never amortizes inside 6000. Same lesson as E3-easy; the
   hard-regime economics live in the token toy's E3.
3. **Held-out regret ~0.20 for every method** (vs 0.56 at init): decision
   quality partially transfers from 6 instances, and the estimator choice
   moves *training efficiency*, not the generalization gap — as it should;
   generalization is the model's job, credit assignment is the estimator's.

## Mapping to the paper's claims

- **C1 (bias-free model-based DFL)** → E1: model error = variance only;
  replay-trained Q̂; biased foil = pathwise-only.
- **T/k separation** (credit-assignment reading) → E2: separation vs
  RLOO/GRPO-style baselines grows with T at fixed k.
- **Make-or-break vs RLOO at matched budget** → E3.
- **Support recovery / credit discovery** → E4 (seeds the C4
  identifiability/query-complexity theory).
- Challenge 3 (stochastic oracle) → E1 noisy row: conditioning integrates
  oracle noise out; unbiasedness untouched.

## Deliberate toy simplifications

- Independent-categorical policy (exact enumeration available); no context
  features `x`. A contextual/autoregressive policy changes nothing in the
  estimator (the correction differentiates the conditional at fixed prefix).
- Filler positions are *exactly* irrelevant. In CoT-style generation, earlier
  tokens influence `f` through the future policy, so `Var(f|y_≤t)` shrinks
  gradually rather than in steps.
- Q̂ trained by plain return regression (stage 1 only); the variance-descent
  fine-tune of φ (stage 2, RELAX-style `∇_φ E[g²]`) is not implemented yet.
- Single fixed lr across methods in E3 (Adam, 0.05); no per-method tuning.
