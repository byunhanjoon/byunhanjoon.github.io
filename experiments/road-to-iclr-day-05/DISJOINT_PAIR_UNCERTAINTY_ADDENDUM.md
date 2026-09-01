# Disjoint-pair source and non-partition addendum

Status: frozen after the candidate/panel gates and before inspecting this
source-level aggregation.

To ensure exact 32-cell partitions do not hide behavior on larger products:

- compute paired candidate RMSE and direct prediction-residual differences at
  the dataset/source level, with equal-source bootstrap intervals;
- repeat descriptive ratios on only candidates with more than 32 nuisance
  cells;
- count favorable candidate cells and sources separately;
- retain selection agreement/regret as descriptive because many sources are
  at exact-winner ceiling.

The addendum passes if RMSE and direct-residual source intervals exclude zero
favorably in at least four of five panels, and the >32-cell subset has lower
mean RMSE and direct residual. Exact-partition and non-partition results must
be reported separately.
