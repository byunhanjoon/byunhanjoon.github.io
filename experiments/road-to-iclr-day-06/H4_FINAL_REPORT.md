# H4 final report — Semantic Shadowing Forecast

Status: **FINAL**

## Verdict

**FALSIFIED AS WRITTEN.**

All 324 frozen bundles completed: three datasets, three models, three seeds,
and twelve optimizer configurations, with one canonical and two matched schema
paths per bundle.  The primary FT gates are within-dataset epoch-2/epoch-20
Spearman at least .70 in 0/3 datasets: Bank .650, Credit .413, and FreMTPL
.119.  Rank-pooled epoch-2 improvement over the constant epoch-zero control is
.394, and pooled material-configuration AUROC is .916; both pass.  The stable-
control fraction is .861, below .90, and the maximum initialization gap is
`4.77e-7`, inside the `1e-6` integrity threshold.

## Interpretation

Ranks are computed within dataset before pooling, so scale differences cannot
create the association.  Constant epoch-zero scores have the prospectively
declared association zero; all other ties use average ranks.  Optimizer
configurations and seeds are repeated measurements and do not expand the
scientific replication scope beyond three datasets.

The shadow is a matched, label-preserving metamorphic perturbation.  A positive
result would show finite-horizon same-source forecasting, not a universal
stability certificate or an accuracy improvement.  A negative result would
mean the two-epoch schema response is not a reliable ranking proxy under this
menu despite H1's clean causal mechanism.

## Decision

Discard the two-epoch shadow as a reliable same-source ranking forecast.  The
high pooled AUROC is retained as a mechanism clue, not promoted over the failed
datasetwise correlation and control gates.
