# Cross-quotient source uncertainty addendum

Status: post-gate uncertainty audit; no new action is selected.

Using the frozen cross-quotient cell table, compare `strength2_cross32` with
the 32-fit IID U-statistic at the dataset/source level.  For agreement, quotient
validation regret, selected quotient test loss, and realized test loss, report
the paired source differences, favorable-source counts, and deterministic
100,000-draw source bootstrap intervals.  Do not pool the confirmation,
menu-repeat, and subsample-repeat datasets as independent sources.

For the candidate-score calibration table, report the mean bias and the
fraction of standardized biases inside ±1.96.  These intervals quantify action
Monte Carlo error conditional on each fixed tensor; dataset bootstrap intervals
quantify source heterogeneity within a panel.
