# Metric Trust Router exploratory protocol

Status: frozen post-outcome feasibility protocol

Frozen: 2026-08-29 Asia/Seoul, after the Metric-Field Transport E1 outcome and
before computing any five-fold routing score.

## Scientific status

This experiment was motivated by observed development outcomes: standardized
raw landmark distances helped ACS, Citi Bike, and NYC TLC neural cells but were
catastrophically harmful for the string-based medical field. The original MPE
test outcomes are already known, and the current outer validation outcomes are
also known. Consequently, this experiment is exploratory only. It cannot
confirm a method or support a test-set claim.

Original test targets remain sealed. Only the original training states are
used to fit and choose a representation. Previously observed validation scores
are joined only after every router decision has been written.

## Fixed question

Can state-group cross-validation decide whether to expose normalized landmark
weights or information-preserving landmark distances, avoiding negative
transfer from an externally supplied but task-irrelevant metric?

## Fixed representations

- fallback: `weights_m32`;
- candidate: `distance_m32`, selected by the already-complete E1a rule.

No other representation, width, mixture, or learned gate is tested.

## Cross-fitted router

For each of the nine runnable tasks and each of the five original partitions:

1. Take only the original training states.
2. Sort them by SHA-256 of
   `mtr-fivefold | task | partition | state_id`, then assign them round-robin
   to five folds.
3. For each fold, fit ordinary preprocessing, target standardization,
   landmarks, and landmark-coordinate standardization on the other four folds
   only. Predict the held-out training states.
4. For each representation and each Ridge alpha in
   `[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`, average the five state-balanced
   standardized MSE values. Select the alpha with minimum mean; ties choose the
   smaller alpha.
5. Route to `distance_m32` iff its selected cross-fitted MSE is strictly lower
   than the selected `weights_m32` MSE. Otherwise route to the fallback.

There is no improvement threshold and no threshold sweep. Router JSON files
are written before joining any outer validation result.

## Fixed retrospective evaluations

After all 45 decisions are complete:

- join the corresponding E1a full-table outer Ridge result for all nine tasks;
- for the four E1b tasks and partitions 0/1, deploy the same decision across
  all three neural seeds and join the already-complete E1b outcomes.

These joins estimate feasibility only. They are not prospective evaluations.

## Feasibility gate

Recommend a new-data confirmation of metric-trust routing only if all hold:

1. neural source-balanced improvement versus always using weights is at least
   5%;
2. no neural source degrades by more than 1%;
3. the router rejects raw distance in both medical neural partitions;
4. broad nine-task Ridge source-balanced performance is no worse than weights;
5. no broad Ridge source degrades by more than 2%.

Failure kills routing as the lead. Passing only authorizes a new-source,
separately frozen confirmation; it does not authorize inspecting original test
targets.
