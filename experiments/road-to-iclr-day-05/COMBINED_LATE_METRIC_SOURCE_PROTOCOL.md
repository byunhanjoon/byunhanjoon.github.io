# Post-failure protocol: source-cluster metric scope

Combine the two untouched four-source blocks. For pair32 and pack64, average
candidate metric RMSE within each of the eight dataset identities, then compute
the source-level percentage reduction versus the frozen control for Brier,
clipped log loss, ROC-AUC, and accuracy. Use 100,000 equal-source bootstrap
resamples and exact sign tests; exact ties are non-wins.

Brier/log source scope passes when at least 7/8 sources are strictly favorable
and the 95% bootstrap interval for equal-source mean reduction is above zero.
AUC and accuracy are descriptive because their candidate gates have already
failed. This is a post-failure aggregation and cannot reverse those failures.
