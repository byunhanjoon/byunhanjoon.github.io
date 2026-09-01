# Frozen diagnostic: dataset-cross-fitted competence calibration

Frozen: 2026-09-01, after the synthetic-tuned OpenML breadth result and before computing
any dataset-cross-fitted routing outcome.

## Question

The synthetic temperature transfers strongly to regression but is slightly worse than a
fixed mixture in classification. Is this a failure of context competence, or a transport
failure of the global temperature/degree of adaptation?

## Immutable panel and candidates

Combine every numeric real dataset already evaluated: 9 classification and 11 regression
identities across the small cached panel and unseen-identity OpenML breadth panel. Reuse
immutable expert predictions and context-CV losses; no expert is refit.

For each task type, candidate weights are

`w = (1 - alpha) softmax(-CV_loss / T) + alpha w_fixed`,

with `T in {0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1, 2}` and
`alpha in {0, 0.25, 0.50, 0.75, 1}`. `w_fixed` remains the synthetic-development fixed
mixture. The grid is frozen before outcomes.

## Dataset-level cross-fitting

For each held-out dataset, select `(T, alpha)` by the dataset-balanced query loss across
all other datasets of the same task type, then evaluate every episode of the held-out
dataset. The held-out dataset contributes neither labels nor identity features to its
parameter choice. Ties use lexicographic `(loss, T, alpha)` order.

Compare the cross-fitted router with the fixed mixture and original synthetic-tuned
competence router. Use 10,000 hierarchical paired bootstrap draws over datasets then
episodes.

## Precision addendum, frozen before routing outcomes

The initial analysis stopped before parameter selection because parent losses were
computed from float64 predictions while the immutable bundles store predictions as
float32. The maximum baseline difference was `2.864e-6`, exceeding the original `1e-6`
guard; no result file was written. Version 1 is invalid under its literal guard.

Version 2 changes only the representation-parity tolerance to `1e-5`, a standard
float32-scale audit bound. The candidate grid, folds, losses, bootstrap, seeds, and gates
are unchanged. A difference above `1e-5` still invalidates the diagnostic.

Strong/scoped gates inherit the OpenML breadth thresholds. A pass is evidence for
task-library calibration, not a deployment result for a task with no related labeled
datasets, and not novelty for stacking/meta-learning. Any independent confirmation of
the calibrated rule requires new dataset identities.
