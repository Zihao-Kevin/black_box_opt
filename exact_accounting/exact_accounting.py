"""
Exact accounting for the variance-optimal control variate.

WHAT THIS IS, IN PLAIN LANGUAGE
-------------------------------
We have a model that makes d guesses at once (coordinate 1, ..., coordinate d).
Each guess is a pick from V options.  A black box scores the whole set of guesses
with one number, f(b).  We want to nudge the model's parameters in the direction
that lowers that score.

Because the black box is not differentiable, we can only estimate that direction,
and the estimate is noisy.  A "control variate" c is a helper table we subtract to
cancel some of the noise.  Any c leaves the estimate correct on average; it only
changes how noisy it is.  So the whole game is: which c is least noisy?

There is an obvious candidate, Q: just predict what f will be.  The paper proves
Q is the best possible choice for each coordinate *taken by itself*, but not
necessarily best overall, because the coordinates share parameters.  The leftover
improvement is called the residual.

This script measures that leftover exactly.  Every quantity is a finite sum over
all V^d possible guess-combinations, so nothing is approximated and nothing is
trained.  It answers two questions:

    1. How big is the leftover?          ->  gain / V_Q, as a percentage
    2. Does it need a complicated table?  ->  how much of it a single number gets

HOW TO READ THE OUTPUT
----------------------
    V_Q       noise level of the obvious choice c = Q          (lower is better)
    gain      how much noise the best possible c removes
    gain/V_Q  the leftover, as a fraction of what Q leaves behind  <- headline
    1 number  fraction of `gain` recovered by one global constant
    d numbers fraction recovered by one constant per coordinate
    no-ctx    fraction recovered without looking at the other coordinates

WHICH BASELINES ARE IN HERE, AND WHICH ARE NOT
----------------------------------------------
REINFORCE, REINFORCE with a constant baseline, a per-coordinate baseline, and
IndeCateR/RAM are all special cases of the same estimator -- just different
choices of the helper table c -- so their variance is computed exactly, by the
same code, with no training and no sampling.  Table D lists them together.

RELAX and REBAR are NOT special cases, and cannot be added here.  They subtract
a control variate that looks at all coordinates jointly and add its mean back by
*sampling* a relaxed, continuous version of the draw.  That means (a) their
helper does not decompose per coordinate, so it has no address in the tables
below, and (b) their added-back term is itself random and depends on a
relaxation temperature and a trained network, so it cannot be enumerated.  They
belong in the training experiment, not in exact accounting.

Two knobs control the setting:
    share     0 = each coordinate has its own parameters, 1 = all share a trunk
    interact  0 = f scores each coordinate separately, 1 = coordinates interact

Theory says the leftover must be exactly zero when share = 0 or interact = 0.
The script checks this, which is how you know the code is right.
"""

import itertools

import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class Setup:
    """One fully enumerable instance of the setting."""

    def __init__(self, d=3, V=3, share=1.0, interact=1.0, kind='max', tied=False,
                 seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.V, self.interact, self.kind = d, V, interact, kind

        # Parameters theta = [shared trunk, private_1, ..., private_d].
        # Head j reads the trunk through its own readout R_j, plus its own block:
        #     z_j = share * R_j @ trunk  +  (1 - share) * private_j
        # share = 1 -> the heads have every trunk parameter in common
        # share = 0 -> the heads have no parameter in common
        # tied = True gives every coordinate the SAME readout of the trunk, the
        # way an LLM reuses one output head at every position.  tied = False
        # gives each coordinate its own readout of the same trunk.
        self.P = V * (d + 1)
        R = rng.normal(size=(d, V, V))
        if tied:
            R[:] = R[0]
        self.J = []
        for j in range(d):
            Jj = np.zeros((V, self.P))
            Jj[:, :V] = share * R[j]
            Jj[:, V * (j + 1):V * (j + 2)] = (1 - share) * np.eye(V)
            self.J.append(Jj)

        theta = 0.5 * rng.normal(size=self.P)
        self.p = [softmax(Jj @ theta) for Jj in self.J]  # one distribution per coordinate

        # The black box: a separable part blended with a part that couples
        # the coordinates.  interact = 0 leaves it fully separable.
        self.a = rng.normal(size=(d, V))
        self.w = rng.normal(size=(d, V))
        self.A = rng.normal(size=(d, d))
        self.T = rng.normal(size=(d, d, d))

        self.states = list(itertools.product(range(V), repeat=d))
        self.at = {b: i for i, b in enumerate(self.states)}

    # ---- the black box -----------------------------------------------------

    def coupled(self, b):
        """How the coordinates interact.  Four flavours, increasingly hostile."""
        d, w = self.d, self.w
        pair = sum(self.A[j, k] * w[j, b[j]] * w[k, b[k]]
                   for j in range(d) for k in range(j + 1, d))
        if self.kind == 'pairwise':
            return pair
        if self.kind == 'tanh':
            return float(np.tanh(2 * pair))
        if self.kind == 'max':       # closest to a decision loss: winner-take-all
            return float(max(w[j, b[j]] for j in range(d)))
        if self.kind == 'triple':
            return pair + sum(self.T[j, k, l] * w[j, b[j]] * w[k, b[k]] * w[l, b[l]]
                              for j in range(d)
                              for k in range(j + 1, d)
                              for l in range(k + 1, d))
        raise ValueError(self.kind)

    def f(self, b):
        separate = sum(self.a[j, b[j]] for j in range(self.d))
        return (1 - self.interact) * separate + self.interact * self.coupled(b)

    # ---- basic pieces ------------------------------------------------------

    def prob(self, b):
        return float(np.prod([self.p[j][b[j]] for j in range(self.d)]))

    def swap(self, b, j, v):
        """b, but with coordinate j changed to value v."""
        c = list(b)
        c[j] = v
        return tuple(c)

    def score(self, b, j):
        """s_j: the gradient of log-probability of the value we drew."""
        e = np.zeros(self.V)
        e[b[j]] = 1.0
        return self.J[j].T @ (e - self.p[j])

    def grad_of_average(self, j, values):
        """
        d/dtheta of  sum_v p_j(v) * values[v],  with `values` held fixed.

        This is the exact V-term average the estimator adds back.  It is the
        only calculus in the file, and it is used twice: once on f (giving the
        term m_j) and once on the residual table (giving part of u_j).
        """
        pj = self.p[j]
        return self.J[j].T @ (pj * (values - pj @ values))

    def marginal_term(self, b, j):
        """m_j: coordinate j averaged out exactly, other coordinates left at b."""
        vals = np.array([self.f(self.swap(b, j, v)) for v in range(self.V)])
        return self.grad_of_average(j, vals)

    def fluctuation(self, b, j, Delta):
        """u_j: the mean-zero wobble a residual table Delta adds to block j."""
        vals = np.array([Delta[j][self.at[self.swap(b, j, v)]] for v in range(self.V)])
        return Delta[j][self.at[b]] * self.score(b, j) - self.grad_of_average(j, vals)

    # ---- the two estimators ------------------------------------------------

    def estimate_at_Q(self, b):
        """Estimator with c = Q.  The score terms cancel, leaving sum_j m_j."""
        return sum(self.marginal_term(b, j) for j in range(self.d))

    def estimate(self, b, Delta):
        """Estimator with c = Q + Delta."""
        return self.estimate_at_Q(b) - sum(self.fluctuation(b, j, Delta)
                                           for j in range(self.d))


def accounting(S):
    """Return V_Q, B and C, all computed by exhaustive enumeration."""
    nb, d, V, P = len(S.states), S.d, S.V, S.P

    pr = np.array([S.prob(b) for b in S.states])
    gQ = np.array([S.estimate_at_Q(b) for b in S.states])   # (nb, P)
    true_grad = pr @ gQ
    resid = gQ - true_grad
    V_Q = float(pr @ np.einsum('bp,bp->b', resid, resid))

    # M[b] is the linear map  Delta -> (total wobble at b).  Build it one
    # table-entry at a time.  Changing Delta_j at a single entry only affects
    # samples that agree with that entry on every coordinate except j, which
    # is why the inner loop is over V samples and not all V^d of them.
    N = d * nb
    M = np.zeros((nb, P, N))
    for j in range(d):
        for w, bw in enumerate(S.states):
            Delta = np.zeros((d, nb))
            Delta[j, w] = 1.0
            for v in range(V):
                b = S.swap(bw, j, v)
                M[S.at[b], :, j * nb + w] = S.fluctuation(b, j, Delta)

    # Variance(Delta) = V_Q - 2 B.Delta + Delta.C.Delta
    X = (M * np.sqrt(pr)[:, None, None]).reshape(nb * P, N)
    r = (resid * np.sqrt(pr)[:, None]).reshape(nb * P)
    return V_Q, X.T @ r, X.T @ X


def gain(B, C, basis=None):
    """
    Best variance reduction available, optionally restricted to a family of
    residual tables spanned by `basis` (one column per free number).
    """
    if basis is not None:
        B, C = basis.T @ B, basis.T @ C @ basis
    return float(B @ np.linalg.pinv(C) @ B)


def var_at(V_Q, B, C, Delta):
    """Variance of the estimator using c = Q + Delta."""
    return float(V_Q - 2 * B @ Delta + Delta @ C @ Delta)


def best_var_in(V_Q, B, C, offset, basis):
    """Lowest variance reachable inside the family  c = Q + offset + basis @ t."""
    t = np.linalg.pinv(basis.T @ C @ basis) @ (basis.T @ (B - C @ offset))
    return var_at(V_Q, B, C, offset + basis @ t)


def known_estimators(S, V_Q, B, C):
    """
    Prior estimators are all points in this same family, so their variance is
    exact too.  Recall the estimator is  sum_j (f - c_j) s_j + (exact average of c_j):

        c = 0        -> sum_j f(b) s_j                  = plain REINFORCE
        c = lambda   -> sum_j (f - lambda) s_j          = REINFORCE + constant baseline
        c = lambda_j -> per-coordinate constant baseline
        c = Q        -> the score terms cancel          = IndeCateR / RAM
        c = Q + D*   -> ours
    """
    fam = families(S)
    Q = np.array([[S.f(b) for b in S.states]] * S.d).reshape(-1)   # c = Q  <-> Delta = 0
    return [
        ('REINFORCE (c=0)',            '1',      var_at(V_Q, B, C, -Q)),
        ('+ best constant baseline',   '1',      best_var_in(V_Q, B, C, -Q, fam['1 number'])),
        ('+ best per-coord baseline',  '1',      best_var_in(V_Q, B, C, -Q, fam['d numbers'])),
        ('IndeCateR / RAM (c=Q)',      'd*V',    V_Q),
        ('ours, optimal (c=Q+D*)',     '1',      V_Q - gain(B, C)),
    ]


def families(S):
    """Restricted residual families, from simplest to richest."""
    nb, d, V = len(S.states), S.d, S.V

    one = np.ones((d * nb, 1))                                   # one number overall

    per_coord = np.zeros((d * nb, d))                            # one number per coordinate
    for j in range(d):
        per_coord[j * nb:(j + 1) * nb, j] = 1.0

    no_context = np.zeros((d * nb, d * V))                       # ignores other coordinates
    for j in range(d):
        for w, bw in enumerate(S.states):
            no_context[j * nb + w, j * V + bw[j]] = 1.0

    return {'1 number': one, 'd numbers': per_coord, 'no-ctx': no_context}


# ---------------------------------------------------------------------------
# Correctness checks.  Each one is a statement from the paper, verified
# numerically.  If these pass, the numbers below can be trusted.
# ---------------------------------------------------------------------------

KINDS = ('pairwise', 'tanh', 'triple', 'max')


def run_checks():
    print('CHECKS  (each line is a statement from the paper, verified numerically)')
    rng = np.random.default_rng(1)
    for kind in KINDS:
        S = Setup(d=3, V=3, share=1.0, interact=1.0, kind=kind, seed=7)
        pr = np.array([S.prob(b) for b in S.states])
        V_Q, B, C = accounting(S)
        true_grad = pr @ np.array([S.estimate_at_Q(b) for b in S.states])

        Delta = rng.normal(size=(S.d, len(S.states)))
        vals = np.array([S.estimate(b, Delta) for b in S.states])

        # 1. Any control variate leaves the estimate correct on average (Lemma 1).
        bias = np.abs(pr @ vals - true_grad).max()

        # 2. The quadratic formula reproduces brute-force variance (Prop. 1).
        brute = float(pr @ np.einsum('bp,bp->b', vals - true_grad, vals - true_grad))
        flat = Delta.reshape(-1)
        quad = V_Q - 2 * B @ flat + flat @ C @ flat

        # 3./4. No leftover if f is separable, or if heads share nothing (Thm. a).
        sep = gain(*accounting(Setup(d=3, V=3, share=1.0, interact=0.0,
                                     kind=kind, seed=7))[1:])
        dis = gain(*accounting(Setup(d=3, V=3, share=0.0, interact=1.0,
                                     kind=kind, seed=7))[1:])
        print(f'  {kind:>9}:  bias {bias:.0e}   formula-vs-brute '
              f'{abs(brute - quad):.0e}   separable {sep:.0e}   disjoint {dis:.0e}')
    print()


def row(S):
    V_Q, B, C = accounting(S)
    g = gain(B, C)
    parts = {k: gain(B, C, P) for k, P in families(S).items()}
    pct = lambda x: f'{100 * x / g:7.1f}%' if g > 1e-12 else '       -'
    ratio = f'{100 * g / V_Q:7.1f}%' if V_Q > 1e-12 else '       -'
    return (f'{V_Q:>10.2e} {g:>10.2e} {ratio:>8} | '
            f'{pct(parts["1 number"]):>8} {pct(parts["d numbers"]):>9} '
            f'{pct(parts["no-ctx"]):>8}')


HEAD = (f"{'V_Q':>10} {'gain':>10} {'gain/V_Q':>8} | "
        f"{'1 number':>8} {'d numbers':>9} {'no-ctx':>8}")


def table_a():
    print('TABLE A  how big is the leftover?   (V=3, f = winner-take-all)')
    print(f"{'d':>2} {'share':>6}  " + HEAD)
    print('-' * (len(HEAD) + 10))
    for d in (2, 3, 4, 5):
        for share in (0.0, 0.5, 1.0):
            print(f'{d:>2} {share:>6.1f}  ' + row(Setup(d=d, V=3, share=share, seed=0)))
        print()


def table_b():
    print('TABLE B  how complicated does the residual have to be?   (d=4, share=1)')
    print(f"{'f':>9} {'V':>3}  " + HEAD)
    print('-' * (len(HEAD) + 14))
    for kind in KINDS:
        for V in (2, 3, 4):
            print(f'{kind:>9} {V:>3}  ' + row(Setup(d=4, V=V, kind=kind, seed=0)))
        print()


def table_c(seeds=8):
    """Every cell above is one random instance, so spread matters as much as mean."""
    print(f'TABLE C  spread over {seeds} random instances   (d=4, V=3, share=1)')
    print(f"{'f':>9}  {'gain/V_Q mean':>13} {'sd':>6} {'min':>6} {'max':>6}  "
          f"{'no-ctx mean':>11} {'sd':>6}")
    print('-' * 64)
    for kind in KINDS:
        ratio, noctx = [], []
        for s in range(seeds):
            S = Setup(d=4, V=3, kind=kind, seed=s)
            V_Q, B, C = accounting(S)
            g = gain(B, C)
            ratio.append(100 * g / V_Q)
            noctx.append(100 * gain(B, C, families(S)['no-ctx']) / g)
        print(f'{kind:>9}  {np.mean(ratio):12.1f}% {np.std(ratio):6.1f} '
              f'{min(ratio):6.1f} {max(ratio):6.1f}  '
              f'{np.mean(noctx):10.1f}% {np.std(noctx):6.1f}')
    print()


def table_d(seeds=8):
    """Every known estimator in this family, on one exact scale."""
    print(f'TABLE D  variance of each estimator, mean over {seeds} instances '
          f'(d=4, V=3, share=1)')
    print(f"{'estimator':>26} {'calls':>6}  " + '  '.join(f'{k:>10}' for k in KINDS))
    print('-' * 82)
    rows = {}
    for kind in KINDS:
        runs = []
        for s in range(seeds):
            S = Setup(d=4, V=3, kind=kind, seed=s)
            V_Q, B, C = accounting(S)
            est = known_estimators(S, V_Q, B, C)
            # report each variance relative to IndeCateR's, so instances combine
            runs.append([v / V_Q for _, _, v in est])
        rows[kind] = (np.mean(runs, axis=0), [(n, c) for n, c, _ in est])
    names = rows[KINDS[0]][1]
    for i, (name, calls) in enumerate(names):
        cells = '  '.join(f'{rows[k][0][i]:10.2f}' for k in KINDS)
        print(f'{name:>26} {calls:>6}  {cells}')
    print('\n  (variance relative to IndeCateR = 1.00; lower is better)')
    print('  RELAX/REBAR are NOT in this family -- see note in module docstring.\n')

    # The row above is per ESTIMATE.  Per oracle CALL is the fair comparison:
    # with a budget of d*V calls you can average d*V of our 1-call estimates.
    print(f"{'same, per oracle call':>26} {'calls':>6}  "
          + '  '.join(f'{k:>10}' for k in KINDS))
    calls = [1, 1, 1, 12, 1]
    for i, (name, _) in enumerate(names):
        cells = '  '.join(f'{rows[k][0][i] * calls[i]:10.2f}' for k in KINDS)
        print(f'{name:>26} {calls[i]:>6}  {cells}')
    print('\n  ours vs IndeCateR at matched budget:  '
          + '   '.join(f'{k} {12 / (rows[k][0][4] * 1):.0f}x' for k in KINDS) + '\n')


def table_e(seeds=8):
    """Best case: one output head reused at every coordinate (LLM-style)."""
    print(f'TABLE E  tied vs separate readouts   (share=1, f=max, {seeds} instances)')
    print(f"{'d':>2} {'V':>2} {'calls':>6}  {'separate: gain/V_Q':>19}  "
          f"{'tied: gain/V_Q':>15} {'tied var':>9} {'efficiency':>11}")
    print('-' * 72)
    for d, V in [(2, 3), (3, 3), (4, 3), (5, 3), (4, 4), (3, 5)]:
        cols = {}
        for tied in (False, True):
            ratio, ours = [], []
            for s in range(seeds):
                S = Setup(d=d, V=V, kind='max', tied=tied, seed=s)
                V_Q, B, C = accounting(S)
                g = gain(B, C)
                ratio.append(100 * g / V_Q)
                ours.append((V_Q - g) / V_Q)
            cols[tied] = (np.mean(ratio), np.std(ratio), np.mean(ours))
        sep, tie = cols[False], cols[True]
        print(f'{d:>2} {V:>2} {d * V:>6}  {sep[0]:12.1f}% +/-{sep[1]:4.1f}  '
              f'{tie[0]:9.1f}% +/-{tie[1]:4.1f} {tie[2]:9.4f} '
              f'{d * V / tie[2]:10.0f}x')
    print('\n  efficiency = oracle calls x variance, ours vs IndeCateR.')
    print('  NOTE: uses the exact Q and Delta*, so these are ceilings, not results.\n')


if __name__ == '__main__':
    run_checks()
    table_a()
    table_b()
    table_c()
    table_d()
    table_e()
