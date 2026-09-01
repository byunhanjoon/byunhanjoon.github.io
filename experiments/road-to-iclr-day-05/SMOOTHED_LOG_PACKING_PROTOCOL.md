# Frozen protocol: interior-supported log-loss packing

## Motivation

The raw six-panel packing experiment uses log clipping at `1e-12`, and the
support audit finds rare exact zeros.  This prevents the smooth Taylor bias
bound from applying everywhere.  Uniformly smoothing every quotient
prediction provides an explicit positive lower bound and tests whether the
empirical gain survives a theoretically clean log-score target.

## Frozen transformation and population

For each exact or sampled class-probability prediction `p` with `C` classes,
use

`p_alpha = (1-alpha) p + alpha/C`

at `alpha in {1e-6, 1e-4, 1e-2}`.  Apply smoothing after prediction averaging;
linearity makes this identical to smoothing each member before averaging.
Use all binary and multiclass candidates in the six frozen log-loss panels,
the same 1,024 actions, and the same pair32/fourpack64 versus independent-pair
and two-pair controls as `analyze_disjoint_log_loss.py`.

The exact target for each alpha is the log loss of the smoothed complete
Cartesian quotient.  No clipping should be active because the true-class
probability is at least `alpha/C`.

## Outcomes and gates

For each alpha, method, candidate, and panel record score RMSE, absolute bias,
exact-winner agreement, and exact validation regret.

- pair32 passes an alpha when panel-mean score RMSE is strictly lower on at
  least 5/6 panels, regret is no higher on at least 4/6, and every candidate
  whose product has at most 32 cells closes below `1e-12` error;
- fourpack64 passes an alpha when score RMSE is no higher on 6/6 panels and
  strictly lower on at least 3/6, regret is no higher on at least 4/6, and
  every candidate whose product has at most 64 cells closes below `1e-12`;
- the robustness claim passes only if both comparisons pass at all three
  smoothing levels.

This changes the score estimand slightly and is not presented as a free repair
of the raw log score.  It is a sensitivity analysis with an explicit theorem
assumption.
