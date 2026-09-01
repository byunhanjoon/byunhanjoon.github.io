# Source-cluster generalization uncertainty

Status: frozen before outcomes.

The covariance analyses are exact conditional on saved tensors, but source
breadth is finite. For the validation-screened confirmation (six independent
source groups) and untouched OpenML panel (eight sources), average cells within
source first, then resample source means with replacement 100,000 times.
Report percentile intervals for the pooled strength-2 residual reduction
against IID-16, SRSWOR-16, four strength-1 blocks, and four seed blocks. These
are descriptive fixed-panel cluster-bootstrap intervals, not population-valid
confidence guarantees. No cell-level resampling is allowed.

Post-gate scope addendum (2026-08-28): apply the same equal-source bootstrap
to the eight-source task-balanced classification/regression panel. This is a
descriptive extension and does not alter the original two-panel assessment.
