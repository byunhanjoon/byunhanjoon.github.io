# H6 final report — Semantic Lyapunov Screen

Status: **FINAL**

## Verdict

**FALSIFIED AS AN INCREMENTAL SCREEN.**

All 33 prospective bundles are distinct from the three named development
bundles.  The frozen extrapolated score reaches pooled AUROC 1.000, dataset
AUROC at least .80 in 3/3 datasets, fixed-rule sensitivity 1.000, specificity
.800, and equal-dataset rank-pooled Spearman .701.  Its AUROC improvement over
the raw epoch-20 log-MSE level is exactly zero, below the required 0.10.

## Interpretation

The early growth fit is highly predictive in this matrix, but the simpler
epoch-20 discrepancy is already at least as discriminating.  A derived slope
that supplies no prospective incremental information does not justify the
extra Lyapunov-style interpretation.  Seeds are repeated measurements and do
not turn this into 33 independent dataset replications; the scientific scope
is three datasets.

The result does not say finite-time growth rates are meaningless.  It says the
specific OLS extrapolation through epochs 5, 10, and 20 fails its frozen
incremental-utility test.  Scores are uncalibrated ranks and should not be read
as global neural-network Lyapunov exponents.

## Decision

Discard H6 as a paper claim and retain the raw epoch-20 semantic-orbit level as
the honest baseline for any future stability screen.
