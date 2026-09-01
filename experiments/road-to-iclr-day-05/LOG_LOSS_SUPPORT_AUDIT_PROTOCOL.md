# Frozen protocol: support conditions behind the log-loss claim

## Question

The smooth-score theorem assumes that the exact true-class probabilities and
their packed estimates lie in `[epsilon, 1]`.  The empirical log-loss scripts,
however, clip only inside the logarithm at `1e-12`.  Do the evaluated
classification predictions actually satisfy the theorem's interior-support
assumption, or are the empirical results partly statements about clipped log
loss?

## Frozen population and estimators

Use every binary and multiclass candidate in the six frozen panels used by
`analyze_log_quotient_jackknife.py`.  For each candidate, audit validation
true-class probabilities for:

1. the exact Cartesian quotient;
2. the 1,024 disjoint-pair32 estimates and their independent-pair controls;
3. the 1,024 mutually-disjoint-fourpack64 estimates and their two-pair
   controls.

The action generators, seeds, and `1e-12` clipping threshold are inherited
unchanged from the frozen disjoint-log-loss experiment.  Test predictions are
not inspected because the theorem and estimator calibration concern the
validation score.

## Quantities and decision rule

For each candidate and method record the minimum true-class probability, the
number and fraction at or below `1e-12`, the number and fraction exactly zero,
and the 0.001, 0.01, and 0.05 quantiles.  Summarize candidate-level and
panel-level incidence without treating observations or action draws as
independent scientific replicates.

- `empirical_interior_support`: no audited exact or randomized estimate is at
  or below `1e-12`.  This verifies the assumption only on the finite audited
  population, with empirical epsilon equal to the observed minimum.
- `clip_active_boundary`: any audited probability is at or below `1e-12`.
  The smooth un-clipped-log theorem is then not a complete explanation of the
  empirical result.  The globally Lipschitz bound for the *clipped* log map
  remains valid but has constant `1e12` and may be vacuous.

This audit cannot establish a distribution-wide positive lower bound.  It is
an assumption check, not a new performance gate.
