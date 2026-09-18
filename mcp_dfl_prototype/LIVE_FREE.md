# Free-text answers: why Q + Δ stalls, and the row-wise Δ that fixes it (`live_free.py`)

The 12-line yes/no manifest of `LIVE_LONG.md` is not how an LLM answers. This note replaces it with free text on the same
strict live-agent reward (cached, nothing re-run), every token sampled from the full 128k vocabulary.
Figures: `_free_queue_training.png`, `_free_queue_baselines.png`. Runs and analysis: `_free_runs/` (`runs_main/` = 1 episode per step,
`runs_g4/` = 4 episodes per step; `analyze.py`, `plot.py`, `plot_baselines.py`).

## 1. Natural formats, standard LoRA: Δ adds almost nothing

| answer format | example | Δ*/Q at the start |
| --- | --- | --- |
| list | `servers: grep, filesystem` | 0.97 |
| plan of calls (repeats, ≤ 6) | `calls: notes, notes, notes, mail` | 0.83 - 0.86 |
| one line per task, 10 tasks, one-shot prompt | `servers for task 1: grep` | 0.94 |
| queue of 4 tasks in ONE generation | `servers for task 1: filesystem\nservers for task 2: lint\n...` | 0.93 (share done) / 0.89 (all done) |
| queue of 7 tasks in ONE generation | | 0.94 / 0.95 |
| *(reference)* 4 slots, digits 0-3 | `files: 1\nknowledge: 0\n...` | 0.59 |
| *(reference)* 12-line manifest | `grep: yes\n...` | 0.13 |

The queue is the natural way to a long answer: the prompt lists K tasks, the model writes one line of server names per task,
the agent works through the queue and the client stops at the first task that fails. With a one-shot demonstration the base
model writes this format unaided (greedy: `filesystem / lint / grep / python` for defines, lint, port, digest) and the base
rates are healthy (P(success) 0.13 - 0.55 per task). Length is not the problem: the 7-task queue is as bad as one line.

## 2. Why: Δ is one scalar per branch

At a prefix n the control variate that is realized when token v is drawn is `c_n − Δ(n, v) · s_n(v)`: a common vector minus a
multiple of **that branch's own score**. The noise it has to cancel on branch v is the martingale increment of the Q estimator,

    ξ_v = A_v · S_n + D_v − D̄_n          (A_v = Q_v − V_n, S_n = summed score of the path to n, D = expected local gradients below)

so only the component of ξ_v along s_n(v) can go.
- yes / no: the two scores at a prefix are collinear, and the same contrast recurs on every line, so S_n is parallel to
  s_n(v): nearly everything cancels (0.13).
- a 13-way menu of server names: twelve different directions, and the names on the path are other names: almost nothing
  cancels, however long the answer. The old "fraction inside the span of another position's scores" overstated what Δ can do.

Check. Make every score at a prefix collinear by brute force (LoRA started the other way round, B random and A = 0, rank 1, so
that a score is `(B'u_v) · x_n`): list 0.97 → 0.35, slots 0.59 → 0.27, 4-task queue 0.93 → 0.44, 7-task queue 0.94 → 0.32, and the
manifest does not move (0.13 → 0.10). This is a diagnostic, not a fix: with B frozen a rank-1 adapter cannot learn the task
(0.29 → 0.33), and with B trained the ratio is back at 0.7 - 0.85 within 20 - 60 steps.

## 3. The fix: a row-wise Δ

The gradient of a linear layer is (backprop signal) ⊗ (layer input). Inside one weight row, the score of **every** token is
therefore a multiple of the same vector: z_n = A x_n for the rows of B, x_n for the rows of A. Give Δ one scalar per
(prefix, token, row) and it can cancel, row by row, the whole component of ξ_v along that input. In closed form the estimator is

    ĝ = ĝ_Q − Σ_{n on the path}  Π_n ξ_{y_n},          Π_n = projection of each row onto the layer input at n,

unbiased for any reward table because Σ_v P_v ξ_v = 0. Cost: the same two tree passes as the scalar Δ, and no linear solve.
Exact noise at the start, standard LoRA (r = 8, last-layer `down_proj`):

| format | Q + Δ / Q | Q + row-wise Δ / Q |
| --- | --- | --- |
| 10 free-text lines | 0.94 | **0.38** |
| 4-task queue, share done / all done | 0.93 / 0.89 | **0.43 / 0.32** |
| 7-task queue, share done / all done | 0.94 / 0.95 | **0.42 / 0.31** |
| list `servers: a, b` (two menus: names, separators) | 0.97 | 0.68 |
| 4 slots | 0.59 | 0.33 |
| manifest | 0.13 | 0.12 |

Checks: the increments ξ sum to exactly the Q noise (ratio 1.0000 on every tree); Monte Carlo (3000 episodes, both LoRA factors
active): bias² at the noise/N level, noise 0.80 measured against 0.92 exact.

## 4. Training (20 seeds; learning rate and reward-table prior carried over from `LIVE_LONG.md`, nothing tuned here)

Free-text queue of 4 tasks (defines, lint, port, digest), reward 1 only if all four get done (P = 0.015 at the start, SNR of one
Q step 0.14), standard LoRA, Adam lr 1e-3, 400 episodes, one episode per step. "Fitted" = reward table learned from the run's own
episodes with *untried = fails*; "exact" = the true table, a ceiling.

| method | mean over training | final | runs that collapse to 0 | runs that solve it |
| --- | --- | --- | --- | --- |
| REINFORCE | 0.13 | 0.10 | 16/20 | 0/20 |
| Q, fitted | 0.35 | 0.37 | 9/20 | 4/20 |
| Q + Δ, fitted | 0.42 | 0.45 | 8/20 | 6/20 |
| **Q + row-wise Δ, fitted** | **0.65** | **0.71** | **1/20** | **9/20** |
| Q, exact | 0.38 | 0.40 | 9/20 | 5/20 |
| Q + Δ, exact | 0.40 | 0.40 | 10/20 | 6/20 |
| **Q + row-wise Δ, exact** | **0.95** | **0.97** | **0/20** | **19/20** |

Paired by seed (mean over training): row-wise Δ − Q = +0.30 ± 0.09 fitted (p = 0.003, better in 17/20) and +0.57 ± 0.09 exact
(p = 7e-6, 20/20); row-wise Δ − scalar Δ = +0.23 ± 0.08 fitted (p = 0.013); scalar Δ − Q = +0.07 ± 0.07 fitted, +0.02 ± 0.11 exact
(nothing). Same mechanism as in the manifest: a noisy step collapses the run, lower variance changes where training ends.

**Where it does not matter.** Ten separate one-line tasks (one episode per task per step, SNR 0.5 - 2.4 after Q): row-wise Δ − Q
= +0.05 ± 0.02 exact (p = 0.01) and +0.00 ± 0.02 fitted; REINFORCE is as good as anything. Variance reduction pays when one
step is noisier than the gradient is long, not otherwise.

**Caveats.** One queue, one model, one learning rate. With the exact table the exact gradient is available too, so the exact rows
are a ceiling. As rows shrink to single parameters a row-wise Δ* becomes full Rao-Blackwellization; what is left to learn is the
reward table, and the fitted-to-exact gap (0.65 vs 0.95) is again the reward model.

## 5. Baselines (same queue, LoRA, Adam lr 1e-3, seeds; 20 seeds each; `_free_queue_baselines.png`)

Mean P(all four tasks done) over training. Group baselines need several episodes per step, so everything was also run with 4 episodes
per step (same 400 episodes). RLOO / GRPO / OTB / RELAX are `../baselines.py`, as in `Live_agent_w_baselines.ipynb`.

| estimator | uses | 1 episode × 400 steps | 4 episodes × 100 steps | collapsed runs (1 ep / 4 ep) |
| --- | --- | --- | --- | --- |
| REINFORCE, running-mean baseline | rewards | 0.13 | 0.47 | 16 / 3 |
| RLOO | group of 4 | - | 0.31 | - / 9 |
| GRPO | group of 4 | - | 0.32 | - / 7 |
| OTB (per-decision, energy-weighted) | group of 4 | - | 0.27 | - / 9 |
| RELAX (learned surrogate) | rewards | 0.09 | 0.27 | 18 / 11 |
| per-parameter constant baseline | rewards | 0.12 | 0.41 | 17 / 5 |
| V baseline | reward table | 0.06 | 0.40 | 19 / 4 |
| Q baseline | reward table | 0.35 | 0.65 | 9 / 2 |
| Q + Δ | reward table | 0.42 | 0.60 | 8 / 2 |
| **Q + row-wise Δ** | reward table | **0.65** | **0.77** | **1 / 0** |
| doubly robust, ∇E[f̂] + (f − f̂)∇log p | reward table | 0.65 | 0.72 | 1 / 0 |
| direct method, ∇E[f̂] alone (**biased**) | reward table | 0.89 | 0.83 | 0 / 0 |
| *ceilings, true reward table:* Q / Q + row-wise Δ / exact gradient | | 0.38 / 0.95 / 0.98 | 0.63 / 0.89 / 0.93 | 9, 0, 0 / 2, 0, 0 |

Paired by seed:
- **Against every model-free baseline row-wise Δ wins clearly**: +0.51 REINFORCE, +0.56 RELAX, +0.53 per-parameter (1 episode per step);
  +0.30 REINFORCE, +0.46 RLOO, +0.45 GRPO, +0.50 OTB, +0.50 RELAX, +0.36 per-parameter (4 per step); every p < 1e-3.
  Group baselines are below plain REINFORCE here: with P(success) = 0.015 a group of four is almost always all zeros.
- **Against the doubly-robust estimator built on the same reward table it is a tie**: −0.00 ± 0.08 (p = 0.98) and +0.05 ± 0.04 (p = 0.17).
  That is what the algebra says: a row-wise Δ built from a reward table removes the part of the noise the table can predict,
  which is what the doubly-robust correction does at the sequence level. With the true table it reaches the exact gradient
  (0.95 vs 0.98, p = 0.23).
- **The biased direct method beats both**: +0.24 ± 0.05 over row-wise Δ (p = 3e-4) at 1 episode per step, +0.06 ± 0.03 (p = 0.06) at
  4. It has no sampling noise at all, and its bias is harmless here: rewards are a lookup table over 67 configurations per task, a
  tried configuration is known exactly, and the base model finds a success for every task within a few episodes. The unbiased
  estimators pay for the term (f − f̂)∇log p, a REINFORCE-sized kick each time a new success is drawn; that kick is what still
  collapses 1 run in 20. Nothing here shows the direct method failing, so this experiment does **not** show that an unbiased
  estimator is needed; it shows that among unbiased estimators row-wise Δ and doubly robust are the best, and far above Q + Δ.
- On the ten one-line tasks (high SNR) all reward-table methods tie: Q 0.68, Q + row-wise Δ 0.68, doubly robust 0.70, direct 0.69.

RELAX had to run in float64: its double backward through exp(−log p)² overflows float32 once a branch falls below e⁻⁴⁴.

## 6. Scalable exact accounting (what made 80k-entry trees possible)

Control-variate terms of different prefixes are uncorrelated, so the Δ* system is block diagonal: one k × k solve per prefix
(k = its branches), with `b_n[v] = −P_v (A_v ⟨s_v, S_n⟩ + ⟨s_v, D_v − D̄_n⟩)`. S comes from one top-down pass, D from one
bottom-up pass, the exact noise of any table from one more top-down pass, all level by level with weight-shaped vectors:
O(entries × 90k) instead of O(entries³). It reproduces the dense code exactly (slots 6.12 / 3.61, manifest 8.74 / 1.11), and a
79k-entry tree takes 0.8 s.

## Commands

```bash
Q="defines,lint,port,digest"                                   # caches go to /tmp/zzhao628_free (--cache)
python live_free.py prep  --format queue --queue $Q --gpu 1
python live_free.py stats --format queue --queue $Q --reward product --rowwise --gpu 1
python live_free.py stats --format queue --queue $Q --init B --r 1 --gpu 1          # the collinearity check of section 2
python live_free.py train --format queue --queue $Q --reward product --methods "Q0,Q0 + Δ,Q0 + rowΔ" --seeds 0,1,2 --steps 400 --gpu 1
python live_free.py train --format queue --queue $Q --reward product --methods "RLOO,GRPO,OTB,RELAX,DR,DM,PSB,V0" --episodes 4 --steps 100 --gpu 1
python _free_runs/analyze.py [_free_runs/runs_g4] ; python _free_runs/plot.py ; python _free_runs/plot_baselines.py
```
Other formats: `--format list | plan --L 6 | slots | manifest`; several prompts per step: `--queue "port;lint;digest"`.
