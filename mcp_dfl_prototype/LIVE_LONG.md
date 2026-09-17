# Does Q + Δ beat Q in training, not only at the start? (`live_long.py`)

Two levers were tried on the strict live-agent reward, with zero new agent runs: **more decisions per episode** and
**more jobs per run, including the hard ones**. 480 training runs, exact P(success) logged at every step.
Figure: `_long_manifest_training.png`. Raw runs and the analysis: `_long_runs/` (`analyze.py`, `plot.py`).

## The two levers

**More decisions: the manifest format.** The answer is one line per server instead of one per slot,

    filesystem: <yes|no>\ngrep: <yes|no>\n ... \nticket: <yes|no>          (12 decisions instead of 4)

same 12 servers, same cached strict rewards. The client keeps its rules (at most 2 servers, at most one per slot) and
stops reading, with reward 0, at the first line that breaks one. `filesystem:` is prefilled; every later token is sampled
from the full 128k vocabulary with an aggregated *anything else* branch. 994 prefixes, 2228 entries per job.

**More jobs.** All 10 jobs in every step (one episode per job per step), instead of the 4 the base model can reach.

## Making 480 runs affordable

The LoRA is on the last layer's `down_proj`, after the last attention, so nothing before it depends on the LoRA. `prep`
runs the transformer once per job and caches, for every prefix, the LoRA's input `x` and `v0` (residual stream + frozen
`down_proj` output). Training is then `h = RMSNorm(v0 + s·B A x)`, `logits = W h`, with no transformer in the loop. Each
score is rank one in each LoRA factor, so the score Gram is `K = s²[(UUᵀ)∘(XA XAᵀ) + (BU BUᵀ)∘(XXᵀ)]` and the 2228 × 90k
score matrix is never formed. TF32 for the vocabulary matmuls gives the same numbers to four digits, 5× faster.
Check: on the slot format this reproduces `tutorial_live_strict.ipynb` exactly (REINFORCE 10 → Q 6.12 → Q + Δ 3.61).

## Result 1: with 12 decisions, Δ does most of the work, all through training

Exact noise at the start, pooled over the 10 jobs (REINFORCE = 10):

| format | decisions | align | Q | Q + Δ* | Δ*/Q |
| --- | --- | --- | --- | --- | --- |
| slots | 4 | 0.65-0.84 | 6.12 | 3.61 | 0.59 |
| manifest | 12 | 0.70-0.95 | 8.74 | **1.12** | **0.13** |

With 12 decisions a prefix-only Q barely beats REINFORCE, and on some jobs it is far worse (digest 23, shipped 28): the
later tokens' noise leaks into the early tokens' credit 12 times over. Δ removes 87% of what Q leaves.
The ratio measured *at the policies visited during training* stays low: 0.2-0.4 (manifest) against 0.6-0.8 (slots),
for as long as the policy is still random (middle panel of the figure).

## Result 2: in training, Q + Δ beats Q at every checkpoint (manifest, all 10 jobs, lr 1e-3, 200 steps, 40 seeds)

Mean P(success) over the 10 jobs:

| method | 250 episodes | 500 | 1000 | 2000 | AUC | jobs solved | collapsed runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REINFORCE | 0.09 | 0.13 | 0.14 | 0.10 | 0.113 | 1.2 | 28/40 |
| Q, fitted | 0.12 | 0.16 | 0.17 | 0.18 | 0.157 | 2.0 | 20/40 |
| Q + Δ, fitted | 0.19 | 0.26 | 0.27 | 0.27 | 0.247 | 3.1 | 11/40 |
| Q, exact | 0.21 | 0.25 | 0.28 | 0.29 | 0.258 | 3.3 | 11/40 |
| Q + Δ, exact | **0.35** | **0.40** | **0.44** | **0.44** | **0.403** | **4.9** | **0/40** |

("fitted" = reward table with *untried = fails*, see below; "exact" = the true reward table, a ceiling.)

Paired by seed:

| comparison | AUC | final | note |
| --- | --- | --- | --- |
| Q + Δ − Q, exact tables | +0.145 ± 0.033, p = 1e-4 | +0.151 ± 0.042, p = 1e-3 | first 20 seeds +0.133 (p 0.009), 20 fresh seeds +0.157 (p 0.005) |
| Q + Δ − Q, fitted | +0.090 ± 0.029, p = 0.003 | +0.097 ± 0.034, p = 0.007 | first 20 seeds +0.120 (p 0.005), 20 fresh seeds +0.060 ± 0.043 (p 0.18) |
| Q + Δ fitted − REINFORCE | +0.134 ± 0.028, p = 3e-5 | +0.172 ± 0.036 | fresh seeds alone p = 0.001 |
| Q fitted − REINFORCE | +0.044 (n.s.) | | Q alone does not help here |

Why the gain lasts past the start: with 12 decisions a noisy step does not just slow learning, it **collapses the run**. The
shared LoRA sharpens onto a wrong all-`no`-like answer and every job goes to 0, after which nothing is sampled that could
undo it. Exact Q + Δ never collapsed in 40 runs; exact Q collapsed in 11, REINFORCE in 28. So lower variance changes where
training ends, not only how fast it starts, and it solves more jobs (4.9 vs 3.3).

**Caveats.**
- The exact-table result is solid (replicated on fresh seeds). With exact rewards the exact gradient is also available
  in a setting this small, so it is a diagnostic of the estimator, not a method.
- The fitted result is weaker: same sign on fresh seeds but half the size and not significant on its own. The
  *untried = fails* prior was chosen after the first fitted runs showed no gain (+0.05 ± 0.03 with the old prior,
  pooled), then tested on the same seeds and replicated on fresh ones; read +0.09 as an upper-ish estimate.
- The reward model is still the gap: exact Q + Δ − fitted Q + Δ = +0.156 ± 0.034. Fitted Q + Δ ≈ exact Q.
- It depends on the step size. At lr 3e-4 (400 steps, 20 seeds) nothing collapses, noise averages out over steps, and no
  method differs significantly (REINFORCE 0.419, Q + Δ exact 0.435). Δ pays when steps are large relative to the noise.

## The reward-table prior

`Seen.f_hat` shrank every untried configuration toward the mean reward of the *tried* ones. The policy oversamples good
configurations, so that mean is biased up (0.2-0.3 when the truth is 1-3 winners in 67), and the tables built on it are
wrong exactly where the policy has not been. *Untried = fails* (`"Q0"`, `"Q0 + Δ"` in `live_long.py`: shrink toward 0) is
the natural prior under a strict reward. It does nothing for Q (−0.006) and helps Q + Δ (+0.036, n.s.): Δ is the part
that uses the table at untried configurations.

## More jobs did not help

On the slot format, going from 4 jobs to 10 changed nothing (Q + Δ − Q: +0.017 ± 0.020 fitted, +0.010 ± 0.008 exact).
The hard jobs (`shipped`, `oncall`, `digest` in the manifest) end at 0.00 for **every** method, the exact ones included:
the shared LoRA sharpens onto the easy jobs within ~40 steps and the hard jobs' answers become deterministic and wrong
before a success is ever sampled. That is an exploration problem; no control variate touches it.

## Commands

```bash
C=/local/scratch/dir                                        # ~2 GB of cached forward passes; keep it off a full home disk
python live_long.py prep  --format manifest --gpu 1 --cache $C
python live_long.py stats --format manifest --gpu 1 --cache $C        # the table of Result 1
python live_long.py train --format manifest --gpu 1 --cache $C --jobs all --methods "Q0,Q0 + Δ" --seeds 0,1,2 --steps 200
python _long_runs/analyze.py                                # run from a folder that holds long_runs/ (or edit the glob)
```
One run is 1-3 minutes on an H100. Methods: `REINFORCE`, `Q`, `Q + Δ` (prior = tried mean), `Q0`, `Q0 + Δ` (untried = fails),
`Q (exact)`, `Q + Δ (exact)`.
