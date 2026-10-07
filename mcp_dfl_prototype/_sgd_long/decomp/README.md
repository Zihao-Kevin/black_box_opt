# Per-position variance decomposition (the analogue of Tucker et al. 2018, Fig. 1)

`decompose.py <gpu>` replays the Q + rowΔ run (lr 0.045, warm start, seed 0) and, at 0 / 100 / 200 / 400 episodes, computes for
every entry e = (prefix, branch) of every training task the exact contribution w_e ||xi_e||^2 of that entry to the estimator's
variance, Var[g_hat] = sum_n E||xi_n||^2 (martingale increments over the positions n of the answer), for REINFORCE, V, Q, Q + Δ
and Q + rowΔ, with the true reward table and with the run's own fitted table.  `decomp.npz` holds the per-position sums.

Figures: `decomposition.*` (x = the 16 scored positions, shaded = the four digit positions), `decomposition_slots.*` (x = the four
decisions, a decision's tokens summed; dashed = fitted table), `decomposition_shares.*` (the share of Q's term each residual removes).

Draft caption.  Variance of the gradient estimator by decision in the answer, exactly computed from the policy (no sampling), at four
stages of one training run.  The residuals act on every decision: the scalar Δ removes the component of each increment along the
branch's own score, the row-wise Δ the component inside the row-input span, and the terms nest at every position.  A state baseline V
is above plain REINFORCE and grows with training (5.8x at the warm start, 42x at 400 episodes) even with the true reward table: under a
0/1 reward REINFORCE's failed episodes contribute nothing, so its noise is second order in 1 - p on a peaked policy, while a baseline
makes every rare failed path pay first order.  The Q table does not pay there because it knows those branches fail.
