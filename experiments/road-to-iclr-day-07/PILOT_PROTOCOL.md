# Frozen exploratory protocol — residual geometry trust

Status: **FROZEN BEFORE TRUST-REPLAY OUTCOMES**

Freeze date: 2026-08-30 (Asia/Seoul).

## Evidence label

This is `POST-OUTCOME EXPLORATORY REPLAY`. The outer Geometry Transfer Law
results, including aggregate help/harm counts, were known before this protocol.
No result here is prospective confirmation. The purpose is to decide whether a
continuous trust mechanism is strong enough to justify a fresh-source neural
experiment.

## Fixed question

Given a target-independent metric on one semantic table field and a fixed base
predictor that excludes that field, can state-held-out validation choose a
nonnegative trust coefficient for a metric residual expert that improves on:

1. always applying a selected geometry operator at full strength; and
2. binary use-or-fallback selection?

## Data and fixed base

- Tasks: the nine runnable `geometry_transfer` retrospective tasks.
- Outer partitions: the existing five state-disjoint splits per task.
- Base: the existing three-fold row-OOF CatBoost residual cache.
- Metric: the already frozen, target-independent task metric.
- Operators: the existing nine-member `operator_family` without modification.
- Independent summary unit: source family. Splits and operators are repeated
  measurements, not independent datasets.

The replay does not refit the base inside each inner state fold. It therefore
estimates trust conditional on the stored fixed base, not the risk of a fully
nested end-to-end procedure. This is a deliberate feasibility shortcut and a
required limitation.

## State-held-out rule

Training states are assigned deterministically to five folds. For each fold,
each operator is rebuilt from the other four folds, training-state residual
means are estimated on those states, and residual predictions are made for the
held states. Candidate trust values are

`lambda in {0.0, 0.1, ..., 1.0}`.

Every candidate is scored by state-balanced residual MSE across the five held
folds. Ties choose smaller `lambda`, then the lexicographically earlier
operator. Three deployable rules are fixed:

- `full`: choose the best operator with `lambda=1`;
- `binary`: choose jointly over all operators and `lambda in {0,1}`;
- `shrink`: choose jointly over all operators and the full eleven-point grid.

The selected operator is rebuilt using all outer-training states and evaluated
on the untouched outer-test states. Test outcomes never enter selection.

## Diagnostic oracles

Two nondeployable test-outcome oracles are reported only to measure headroom:

- best operator at full strength;
- best operator and continuous `lambda` on `[0,1]`, using the exact realized
  quadratic optimum.

They cannot support a method claim.

## Feasibility gates

Recommend a fresh-source neural confirmation only if all hold:

1. source-balanced outer gain of `shrink` is positive;
2. `shrink` improves source-balanced residual MSE relative to `full`;
3. `shrink` has no more harmful task-split cells than `binary`;
4. median selected trust is strictly between zero and one, showing that the
   result is not merely a renamed binary router;
5. analysis identities and deterministic regeneration tests pass.

These gates authorize a new experiment only; they do not turn this replay into
confirmatory evidence.

