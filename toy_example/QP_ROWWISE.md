# Row-wise Δ on the DFL QP toy (`qp_rowwise.py`)

Setting: `DFL_QP_Black_Box_new.ipynb` unchanged (12 assets, 6 days, Gaussian nudges σ = 0.15, prefix Q, Adam lr 0.01, 1600 calls).
"+ Q", "+ Q + Δ" and "+ Q + row-wise Δ" all use the exact per-instance Q and Δ* (ceilings, as in the notebook).
Figure: `_qp_rowwise_training.png`. Runs and scripts: `_qp_rowwise_runs/` (`train.py`, `setup.py`, `plot.py`, 200 runs).
The notebook now has the same comparison with 10 seeds (row-wise Δ bar, shared and private training cells); seeds 0-9 there reproduce these runs exactly.

**Row-wise Δ here.** One scalar per (asset j, weight row r) instead of one per asset: the row-r slice of ∇μ_j is
(backprop) × (row input), so Δ can remove, row by row, the component of the noise along J_j's slice. Rows: every output unit of
`trunk.weight` (input x, the same for all assets), `trunk.bias`, `head.weight` (input tanh(...)_j, different per asset), `head.bias`,
and `pos` as a lookup table. Exact accounting: Var = Σ_β β! |R_β − Π_j*(β) R_β|², Π_j = blockdiag_r of rank-1 projections.
Checks: one group reproduces the notebook's scalar Δ* exactly; Monte Carlo variance matches within 1%; unbiased.

## Training, 20 seeds (regret, lower is better)

| method | shared: mean over training | shared: final | private: mean over training | private: final |
| --- | --- | --- | --- | --- |
| plain | 0.351 | 0.347 | diverges | diverges (11.4) |
| + Q | 0.342 | 0.332 | 0.228 | 0.200 |
| + Q + Δ | 0.299 | 0.266 | 0.236 | 0.182 |
| **+ Q + row-wise Δ** | **0.276** | **0.244** | **0.160** | **0.130** |
| exact gradient | 0.249 | 0.230 | 0.120 | 0.108 |

Paired, row-wise Δ − Δ: shared −0.022 ± 0.002 (20/20 seeds, p = 5e-10), private −0.077 ± 0.003 (20/20, p = 1e-16).
On the private model scalar Δ is no better than Q (+0.008, p = 0.03, slightly worse), and row-wise Δ closes 66% of the gap from
Q to the exact gradient (mean over training).

## Why, when the total variance barely moves

Total noise at the start, as a share of REINFORCE's: shared Q 0.47, Q + Δ 0.029, Q + row-wise Δ 0.025; private 0.42 / 0.39 / 0.37.
Along training, row-wise removes 12-77% of what scalar Δ leaves on the shared model and ~5% on the private model.

The gain is in the small blocks, which Adam rescales one parameter at a time. Per-parameter noise fraction Var_p / (g_p² + Var_p):

| block | Q | Q + Δ | Q + row-wise Δ |
| --- | --- | --- | --- |
| trunk.weight, trunk.bias | 0.97-1.00 | 0.95-0.99 | **0.00** |
| head.bias | 1.00 | 1.00 | **0.00** |
| head.weight, pos | 0.94-0.99 | same | same |

The trunk rows are cancelled exactly because their input is the day's features x, identical for all assets. `head.weight` holds
90%+ of the total noise and its input tanh(trunk(x) + pos_j) differs per asset, so row-wise cannot touch it. Under Adam the trunk
then takes clean steps instead of noise. (`head.bias` has true gradient exactly 0: the optimizer ignores a shift of all predicted
returns, M·1 = 0.)

Caveats: exact Q and Δ* (a ceiling; a fitted version is not tested); the training gain goes through Adam's per-parameter scaling
(plain SGD not tested); the finest grouping (one scalar per parameter) gives the exact gradient (Var 0 on the shared model), so
row-wise is a middle point whose rows are the layers' output units.
