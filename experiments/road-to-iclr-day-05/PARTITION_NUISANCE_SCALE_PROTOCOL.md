# Partition-shift versus nuisance-error scale

Status: post-outcome mechanism diagnostic.

For the original and three alternate HistGB/CatBoost splits, compute the exact
validation and test quotient-loss gap `CatBoost - HistGB`. Define partition-gap
movement as the absolute difference between those two gaps. Compare it with the
quadrature score-RMSE scale of the two candidates under the exactly unbiased
pair-cross64 estimator on validation.

The ratio is descriptive: candidate score errors can be coupled, and
validation/test partitions are not Monte Carlo replicates from a fully
specified population. It is not a confidence bound. Its purpose is to show
whether further nuisance-action optimization is plausibly the dominant source
of remaining rank error.
