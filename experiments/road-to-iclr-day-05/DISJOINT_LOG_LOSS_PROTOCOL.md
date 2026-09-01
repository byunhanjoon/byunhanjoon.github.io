# Frozen protocol: disjoint packing under log loss

## Question

Does the variance reduction from disjoint strength-2 covers survive a nonlinear,
unbounded proper score, or is it specific to squared loss?  This is a scope test,
not a new tuning round.

## Data and estimands

Use all binary and multiclass datasets from the five previously frozen selection
panels plus the independent two-source multiclass panel.  The target for each
candidate is the log loss of its complete Cartesian-quotient prediction.  Clip
only inside the logarithm at `1e-12`, exactly as in the prior frozen log-loss
experiment.  Reuse 1,024 deterministic Monte Carlo draws per dataset.

## Comparisons

1. At 32 fits, compare the log loss of the mean prediction from a uniformly
   sampled disjoint pair of 16-cell strength-2 covers with two independently
   sampled covers.
2. At 64 fits, compare a mutually disjoint four-cover pack with two independent
   disjoint pairs, using the already frozen equivariant sequential sampler.

No jackknife correction or post-hoc tuning is allowed.  Report candidate-level
score RMSE and absolute bias, panel-level exact-winner agreement and quotient
regret, and the maximum absolute error whenever the fitted cells exhaust the
product.

## Frozen gates

- **Pair-32 scope pass:** panel-mean score RMSE is lower on at least 5/6 panels,
  validation regret is no higher on at least 4/6 panels, and all products with at
  most 32 cells are numerically exact (`<1e-12`).
- **Pack-64 scope pass:** panel-mean score RMSE is no higher on all 6 panels and
  strictly lower on at least 3, validation regret is no higher on at least 4/6,
  and all products with at most 64 cells are numerically exact (`<1e-12`).

Both comparisons, failures included, enter the ledger.
