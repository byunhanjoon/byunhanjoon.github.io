# Approximate log-quotient jackknife protocol

Status: protocol frozen before inspecting outcomes; analysis complete.

## Motivation

The independent-cover cross-score is exact for squared Hilbert losses, but log
loss is nonlinear in the quotient probability. Test whether classical
two-block jackknife bias cancellation can give a useful approximate extension,
without claiming exact unbiasedness.

## Design

Use every binary or multiclass candidate in the five main selection panels and
the two-source multiclass panel. For two independent 16-fit estimators `A,B`,
compare the ordinary 32-fit log score

`L_mean = L((A+B)/2)`

with the two-block jackknife

`L_J = 2 L((A+B)/2) - [L(A)+L(B)]/2`.

Evaluate strength-2 and IID blocks over 1,024 deterministic actions against
exact log loss of the full finite quotient. Report score bias/RMSE, exact
log-quotient winner agreement, validation regret, and quotient test loss.
Clip probabilities only at `1e-12` for numerical evaluation.

## Frozen interpretation

- **Full pass:** cover jackknife has lower panel-mean RMSE and no higher
  validation regret than IID jackknife on at least 5/6 panels, and lower mean
  absolute bias than the ordinary cover score on at least 4/6.
- **Efficiency-only:** the first two clauses pass but bias correction does not.
- **Fail:** either cover-versus-IID clause fails.

Any positive result remains an approximate log-score extension; the exact
unbiased theorem and covariance identity remain specific to quadratic Hilbert
scores.

## Outcome

The frozen interpretation is **full pass**, with all three clauses passing on
all 6/6 panels.

- Cover-jackknife/IID-jackknife RMSE ratios are 0.360 (confirmation), 0.149
  (menu), 0.348 (external), 0.691 (multiclass), 0.501 (task-balanced), and
  0.275 (subsample).
- Cover jackknife has no higher exact log-quotient validation regret on every
  panel; it reaches zero regret on external, task-balanced, menu, multiclass,
  and subsample.
- Mean absolute bias falls versus the ordinary cover mean on all six panels,
  though the multiclass change is small (`5.58e-5` to `5.47e-5`).
- External selected test log loss is worse for the more faithful selector,
  reproducing the validation/test boundary under another score.

Proposition 24 supplies second-order bias cancellation under smoothness and a
probability-away-from-zero condition. It is asymptotic/approximate, not the
exact finite-randomization identity available for quadratic scores.
