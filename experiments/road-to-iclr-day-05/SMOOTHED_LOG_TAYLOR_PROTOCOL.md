# Frozen protocol: Taylor calibration for packed smoothed log loss

## Question

Does the smooth-score theory explain the *magnitude* of packed log-loss error,
or merely supply a loose worst-case inequality?

## Population and estimators

Use the 23 classification candidate tensors with a 128-cell nuisance product.
For each of the four methods in the smoothed-log packing audit and each
`alpha in {1e-6,1e-4,1e-2}`, let `q` be the exact smoothed true-class
probability and `q_hat=q+delta` its action estimate.  For each of the same
1,024 frozen actions calculate:

- exact score error `mean[-log(q_hat)+log(q)]`;
- first-order term `-mean[delta/q]`;
- second-order term `-mean[delta/q] + 0.5 mean[(delta/q)^2]`;
- the Proposition-30 global MSE upper bound using `epsilon=alpha/C`.

## Diagnostics and frozen interpretation

At candidate-method level record actual/first-order correlation, first- and
second-order relative approximation RMSE, observed score bias versus the mean
quadratic correction, and bound slack.  Zero-variance cases are excluded only
from correlation counts and retained everywhere else.

The local Taylor mechanism passes at `alpha=1e-2` if at least 80% of
nondegenerate candidate-method cells have correlation above 0.99 and
second-order relative RMSE below 0.10.  The formal-bound audit passes if no
cell at any alpha exceeds its global bound beyond `1e-10` relative numerical
tolerance.  Smaller-alpha diagnostics are descriptive and may expose the
expected near-boundary deterioration.
