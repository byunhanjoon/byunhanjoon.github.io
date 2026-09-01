# Completion-protocol deviations

## D1 — broad reference upgraded to full finite product

Recorded before any broad-mode tensor completed or was inspected. The frozen
protocol requested a deterministic 64-action uniform reference plus
equal-budget designed methods. The initial runner stored only those 64 random
actions, which would not contain every configuration required by arbitrary
strength-1/2/3, LHS, Sobol, or packing realizations. This was an implementation
design error, not an outcome failure.

Correction: every broad neural cell now enumerates the full declared finite
product (128 classification actions or 64 regression actions; fewer only when
a transformation is mathematically unavailable). The quotient is therefore
exact rather than a 64-action Monte Carlo reference, and every sampling method
can be evaluated from the same stored tensor. This strictly increases compute
and reference quality. No dataset, model, split, hyperparameter, metric, or
success criterion changes.

## D2 — matched-control telemetry repair

Recorded before consolidated matched-control analysis. The first matched-run
writer discarded the wall-time and peak-memory values returned by the common
fit routine, although predictions and initial-gap checks were stored correctly.
All sixteen matched dataset×model cells are rerun from the same frozen config
with per-arm telemetry retained. The original artifacts are moved to a
recoverable pre-telemetry directory. No prediction, intervention, seed,
hyperparameter, metric, or criterion changes.

## D3 — mixed-level GF(4) trace degeneracy repaired

Recorded after the first consolidated analysis exposed a structural, not
outcome-dependent, defect: when a category/class factor collapsed, the
nominal 16-row strength-2 construction repeated eight unique actions, and the
64-row strength-3 construction repeated half its available actions on the
128-cell products. Both arrays passed formal pairwise/triple balance but
wasted equal-compute budgets.

Correction: the final binary column uses an independent GF(4) trace functional
(`trace(2v)` for strength 2 and `trace(2w)` for strength 3). Tests now require
the maximum possible number of unique rows for every observed mixed-level
shape in addition to formal strength. All trained full-product prediction
tensors are unchanged; only the prospective finite-tensor estimator analysis,
tables, figures, and report are regenerated. The flawed analysis output is not
retained as a scientific result, but its discovery and correction remain
documented here.
