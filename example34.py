r"""Every number in Example 3.4 of black_box_opt_my.tex, by hand-checkable enumeration.

b1, b2 ~ Bernoulli(p), ONE shared parameter theta with p = sigmoid(theta); f(b) = b1 b2.
At theta = 0: p = 1/2, dp/dtheta = p(1-p) = 1/4, score s_j = b_j - p, true gradient g = dE[f]/dtheta = 1/4.

The recipe is always the same three steps:
    1. list the 4 outcomes and their probabilities,
    2. write down what the estimator outputs at each outcome,
    3. Var = sum_b p(b) ghat(b)^2  -  ( sum_b p(b) ghat(b) )^2.

Only step 2 differs between estimators.  The estimator family is

    ghat = sum_j [ (f(b) - c_j(.)) s_j(b_j)  +  sum_v p(v) c_j(., v) s_j(v) ]
                   \-------- score term -------/  \------ pathwise term ------/

and the two control-variate families differ only in what c_j is allowed to look at:
    leave-one-out  c_j(b_{-j}, v)   -- may see every OTHER coordinate
    autoregressive c_j(b_{<j}, v)   -- may see only the PAST
"""
import itertools
import numpy as np

p = 0.5
OUT = list(itertools.product([0, 1], repeat=2))          # the 4 outcomes
PR = np.array([0.25] * 4)                                # each has probability 1/4
f = lambda b: b[0] * b[1]
s = lambda v: v - p                                      # score of one coordinate, d/dtheta log p(v)
g = 2 * p * (p * (1 - p))                                # true gradient dE[f]/dtheta = d(p^2)/dtheta = 1/4


def var(vals):
    vals = np.asarray(vals, float)
    return float(PR @ vals ** 2 - (PR @ vals) ** 2), float(PR @ vals)


def pathwise(c_of_v):
    """sum_v p(v) c(v) s(v): the 'credit' term.  c_of_v is a length-2 list [c(0), c(1)]."""
    return sum(p * c_of_v[v] * s(v) for v in (0, 1))


# ---------------------------------------------------------------- the estimators, outcome by outcome
def reinforce(b, baseline=0.0):
    return (f(b) - baseline) * (s(b[0]) + s(b[1]))


def ghat_loo(b, D=(0.0, 0.0)):
    """c_j(b_{-j}, v) = f(b_{-j}, v) + Delta_j   (Delta constant, as in the Example)."""
    tot = 0.0
    for j in (0, 1):
        other = b[1 - j]
        c = [other * v + D[j] for v in (0, 1)]           # Q_j(b_-j, v) = v * b_other, plus Delta_j
        tot += (f(b) - c[b[j]]) * s(b[j]) + pathwise(c)
    return tot


def ghat_ar(b, D1=(0.0, 0.0), D2=None):
    """c_1(v) = E[f | b1 = v] = v*p   (must average over the FUTURE b2)
       c_2(b1, v) = f(b1, v) = b1*v  (may use the PAST b1)
       D1 = [Delta_1(0), Delta_1(1)];  D2(b1) -> [Delta_2(b1,0), Delta_2(b1,1)]."""
    D2 = D2 or (lambda b1: (0.0, 0.0))
    c1 = [v * p + D1[v] for v in (0, 1)]
    c2 = [b[0] * v + D2(b[0])[v] for v in (0, 1)]
    return ((f(b) - c1[b[0]]) * s(b[0]) + pathwise(c1)
            + (f(b) - c2[b[1]]) * s(b[1]) + pathwise(c2))


# ---------------------------------------------------------------- optimal Delta, the manuscript's way
def optimal_delta(base, psi):
    """Var = V_Q - 2 B'Delta + Delta' C Delta, so Delta* = C^+ B and the gain is B' C^+ B.

    base[i]   = ghat[Q] at outcome i
    psi[i, r] = d ghat / d Delta_r at outcome i, i.e. -u_r (u is linear in Delta)."""
    base = np.asarray(base, float); psi = np.asarray(psi, float)
    C = psi.T @ (PR[:, None] * psi)                      # E[u u'] -- reward-free
    B = -psi.T @ (PR * (base - PR @ base))               # E[u' (ghat[Q] - g)] -- linear in f
    D = np.linalg.pinv(C) @ B
    return D, float(PR @ (base + psi @ D) ** 2 - (PR @ (base + psi @ D)) ** 2)


rows = []
rows.append(("REINFORCE", [reinforce(b) for b in OUT]))
rows.append(("REINFORCE + mean baseline", [reinforce(b, baseline=p * p) for b in OUT]))
rows.append(("Q  (leave-one-out)", [ghat_loo(b) for b in OUT]))
rows.append(("Q + Delta* = 1/4  (loo)", [ghat_loo(b, D=(0.25, 0.25)) for b in OUT]))
rows.append(("Q  (autoregressive)", [ghat_ar(b) for b in OUT]))

# AR + state-free Delta: Delta_1(v) = d1, Delta_2(v) = d2, both constant in v -> 2 unknowns
base_ar = [ghat_ar(b) for b in OUT]
psi_sf = [[ghat_ar(b, D1=(e, e)) - ghat_ar(b) for e in (1.0,)] + [ghat_ar(b, D2=lambda _: (1.0, 1.0)) - ghat_ar(b)]
          for b in OUT]
D_sf, v_sf = optimal_delta(base_ar, psi_sf)
rows.append((f"AR Q + state-free Delta {np.round(D_sf, 3)}", [base_ar[i] + np.dot(psi_sf[i], D_sf) for i in range(4)]))

# AR + prefix Delta: Delta_2 may depend on b1 -> 3 unknowns (d1, d2 at b1=0, d2 at b1=1)
def psi_prefix(b):
    e1 = ghat_ar(b, D1=(1.0, 1.0)) - ghat_ar(b)
    e20 = ghat_ar(b, D2=lambda b1: (1.0, 1.0) if b1 == 0 else (0.0, 0.0)) - ghat_ar(b)
    e21 = ghat_ar(b, D2=lambda b1: (1.0, 1.0) if b1 == 1 else (0.0, 0.0)) - ghat_ar(b)
    return [e1, e20, e21]

psi_pf = [psi_prefix(b) for b in OUT]
D_pf, v_pf = optimal_delta(base_ar, psi_pf)
rows.append((f"AR Q + prefix Delta {np.round(D_pf, 3)}", [base_ar[i] + np.dot(psi_pf[i], D_pf) for i in range(4)]))

print(f"outcomes (b1,b2): {OUT}   each with probability 1/4;  true gradient g = {g}\n")
print(f"{'estimator':<38}{'ghat at the 4 outcomes':<34}{'mean':>7}{'variance':>12}")
for name, vals in rows:
    v, m = var(vals)
    cells = "  ".join(f"{x:+.3f}" for x in vals)
    print(f"{name:<38}{cells:<34}{m:>7.3f}{v:>9.4f} = {round(v * 64)}/64")


# ================================================================================================
# Where  (f-hat - f)' M (f-hat - f)  comes from.
#
# ghat is AFFINE in the table c and LINEAR in the reward vector f, so write it on the 4 outcomes as
#     ghat = A_f f + A_c c          (A_f: 4x4, A_c: 4x8 for the loo family with 2 binary coordinates)
# With W = diag(P) - P P' the centring matrix, Var(c) = (A_f f + A_c c)' W (A_f f + A_c c), i.e.
#     Var(c) = V_0 - 2 b(f)' c + c' C c,     C = A_c' W A_c  (NO f),   b(f) = -A_c' W A_f f  (linear in f).
# Variance is therefore an exact quadratic in c, so completing the square gives, for ANY c,
#     Var(c) - Var(c*) = (c - c*)' C (c - c*),        c* = C^+ b(f) =: T_theta f.
# Plugging the plug-in c-hat = T_theta f-hat -- built from a reward MODEL, scored under the TRUE f --
#     Var(c-hat) - Var(c*) = (T f-hat - T f)' C (T f-hat - T f) = (f-hat - f)' M (f-hat - f),
#     M = T_theta' C T_theta,  symmetric PSD, and reward-free (policy and scores only).
# ================================================================================================
IDX = {b: i for i, b in enumerate(OUT)}
CIDX = {(j, o, v): 4 * j + 2 * o + v for j in (0, 1) for o in (0, 1) for v in (0, 1)}   # c_j(b_other, v)

A_f = np.zeros((4, 4)); A_c = np.zeros((4, 8))
for b, i in IDX.items():
    A_f[i, i] = s(b[0]) + s(b[1])                        # ghat depends on f only through f(b)
    for j in (0, 1):
        o = b[1 - j]
        A_c[i, CIDX[(j, o, b[j])]] -= s(b[j])            # -c_j(b_-j, b_j) s_j   (score term)
        for v in (0, 1):
            A_c[i, CIDX[(j, o, v)]] += p * s(v)          # + sum_v p(v) c_j(b_-j, v) s_j(v)  (pathwise)

W = np.diag(PR) - np.outer(PR, PR)
Cm = A_c.T @ W @ A_c                                     # C: reward-free
T = -np.linalg.pinv(Cm) @ (A_c.T @ W @ A_f)              # c* = T f
M = T.T @ Cm @ T                                         # the quadratic form on the reward model

fvec = np.array([f(b) for b in OUT], float)
var_of = lambda c, fv: float((A_f @ fv + A_c @ c) @ W @ (A_f @ fv + A_c @ c))
v_star = var_of(T @ fvec, fvec)
print(f"\nc* = T f  reproduces the oracle: Var = {v_star:.2e}   (Example 3.4 says 0)")
print(f"M is symmetric PSD: eigenvalues {np.round(np.linalg.eigvalsh(M), 4)}\n")
print(f"{'f-hat':<34}{'Var(T f-hat) - Var(c*)':>24}{'(f-hat-f) M (f-hat-f)':>24}")
rng = np.random.default_rng(0)
for lab, fh in [("f (oracle)", fvec), ("f + 0.3 on (1,1)", fvec + np.array([0, 0, 0, .3])),
                ("0.5 * f (shrunk)", .5 * fvec), ("f + 1 (constant shift)", fvec + 1),
                ("random draw", fvec + rng.normal(0, .4, 4)), ("random draw", fvec + rng.normal(0, .4, 4))]:
    lhs = var_of(T @ fh, fvec) - v_star
    rhs = float((fh - fvec) @ M @ (fh - fvec))
    print(f"{lab:<34}{lhs:>24.6f}{rhs:>24.6f}")
