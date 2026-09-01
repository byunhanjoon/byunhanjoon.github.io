# Semantic Symmetries / Representation Orbits — Frozen Protocol

This protocol implements the adjacent Kill Experiment 2 specification. It was frozen
before model outcomes were observed.

## Estimand and sampling

- OpenML datasets and versions are fixed in `configs/semantic_orbits.yaml`.
- A single split seed (`20260901`) fixes train/validation/test row identities. Model seeds
  `0, 1, 2` therefore vary inference/training randomness without changing evaluated rows.
- At most 2,048 train, 512 validation, and 512 test rows are used per dataset. Classification
  subsampling is stratified.
- Every orbit contains eight pre-generated members. Transformation seeds depend on dataset,
  family, scope, and member, but not on model seed.
- Train, validation, and test inputs receive the same fitted transformation. Targets and row
  order never change.

## Models

- TabICLv2: official `tabicl-*-v2-20260212.ckpt`, one estimator per model seed.
- TabPFN-2.6: official v2.6 classifier/regressor checkpoints, one estimator per model seed.
- CatBoost 1.2.10: 200 iterations, depth 7, learning rate 0.05, separate refit for every
  representation.
- TabM is an optional bounded follow-up on the strongest cells because training it afresh for
  every member of the complete orbit grid is not compute-comparable to frozen-TFM inference.

One estimator is intentional: the three declared model seeds provide the replication axis,
and avoiding hidden preprocessing ensembles makes representation effects easier to interpret.

## Representation families

- T0: eight whole-table column permutations (control).
- T1: bijective relabeling of one nominal feature and all nominal features, using both native
  categorical metadata and numeric-code pipelines where supported.
- T2: positive scaling and positive-affine recoding, one numerical feature and all numerical
  features. Factors have `log10(a) ~ U(-1,1)` and offsets use `c * train_std`, `c ~ U(-3,3)`.
- T3: signed-log, empirical quantile, and strictly increasing random piecewise-linear recodings.
- T4: known diamonds orders under categorical, naive numeric-spacing, and rank-canonicalized
  treatments, for one and all ordinal features.
- T5: bike hour/weekday/month shifts, metadata-aware canonical sin/cos frontends, and invertible
  two-dimensional rotations.
- T6: an eight-RBF basis for one declared numerical feature and orthogonal, condition <=3, and
  condition <=10 invertible changes of basis.

## Repair comparisons

Training-only z-scoring is paired with T2, training empirical-quantile canonicalization with
T3, target-free frequency/distribution signatures with T1, known ranks with T4, and
metadata-aware sin/cos canonicalization with T5. Orbit ensembles use three and eight member
averages. The three-member orbit ensemble is compared with a three-seed ordinary ensemble on
the identical split.

## Metrics and decision

Classification uses probability MAD, mean JS divergence, label flips, log loss, ROC-AUC, and
accuracy. Regression uses prediction RMSE divided by test-target standard deviation, Pearson
and Spearman correlations, RMSE, and MAE. Orbit mean/worst/span are computed only after all
eight members exist. The verdict follows the kill criteria in the authoritative specification;
no thresholds are changed after seeing results.

## Conditional follow-ups

The authoritative protocol predeclares both steps below, but their dataset selection is made
only after auditing the frozen primary grid; neither can change its kill verdict.

- TabM-D (pytabkit 1.7.3) is separately refit for the reference and all 24 T6 members on the
  three datasets with repeated basis sensitivity, for 3 datasets × 3 seeds × 25 fits.
- Training-time repair uses a three-layer, width-256 MLP on California Housing and Wine Quality,
  the two strongest clean regression panels. It tests raw training, Gaussian-noise control,
  orbit augmentation, consistency at lambda 0.1 and 1.0, canonical-only, and raw+canonical
  dual-view training. Only orthogonal condition-one basis transforms are used in this test.
- Missing continuous values are filled with training medians before orbit construction. The
  serialized basis inverse is numerically audited before the canonical follow-up is accepted.
