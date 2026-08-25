# Day 3 continuation protocol — equivalent-basis orbit ensembles

## Motivation and novelty gate

Day 2 found that separately trained cumulative/local PLE models have
complementary errors even though their input coordinates span the same affine
function space. Day 3 then showed that ordinary initialization and AdamW send
such coordinate descriptions along different function-space trajectories.

This extension tests a method consequence of those two observations:

> Use information-equivalent coordinate systems as a structured diversity axis
> inside a parameter-efficient ensemble.

The proposed model is **Orbit-TabM**. It retains TabM's eight members and all
trainable parameter counts. Before the shared dense stem, four members receive
the block-whitened cumulative/Helmert representation and four receive its exact
blockwise local/adjacent recoding. Each member therefore receives all of the
same information and has the same predictor class. Only the coordinate chart
differs.

This is not presented as a new encoder, additional feature information, or a
new invariance theorem. It is a structured ensemble intervention motivated by
the measured non-equivariant function priors and trajectories.

## Freeze status

This protocol and `experiments/day3/configs/orbit_ensemble_preregistered.json`
are written before any Orbit-TabM training result is inspected. The datasets
have been used by earlier Day 3 studies, so this is outcome-blind only with
respect to the new method; it is not a newly untouched dataset benchmark.

## Representations and controls

All transforms are fit on training rows only and reused unchanged on validation
and test rows.

- `cumulative`: ordinary dense-stem TabM on the block-whitened
  cumulative/Helmert representation.
- `local`: ordinary dense-stem TabM on the exactly equivalent block-whitened
  local/adjacent representation.
- `orbit_natural`: eight TabM members alternate between the cumulative and
  local charts before a shared dense affine stem.
- `orbit_random`: member 0 uses the cumulative chart and the remaining members
  use fixed seed-derived orthogonal rotations of it. This tests whether any
  invertible coordinate diversity is sufficient or whether the sparse natural
  charts matter.

The ordinary and orbit models use the same TabM backbone, ensemble size,
trainable parameter count, loss, optimizer, stopping rule, and seed. Orbit
models do more first-stem FLOPs because they pass member-specific inputs; wall
time and peak memory are therefore reported rather than called compute-matched.

## Stages

### Development screen

Use the nine released TabPack datasets, three paired seeds, and all four arms.
No hyperparameter is selected from test outcomes. The random-orbit arm is a
mechanism control and cannot replace the primary natural-orbit method after
outcomes are known.

### Method confirmation

Use the remaining 16 datasets from the frozen broad benchmark plus its five
separately frozen extension datasets, three paired seeds, and the cumulative,
local, and natural-orbit arms. These datasets are confirmation for the method,
not a claim that the underlying datasets were never previously inspected.

## Primary analysis

The common primary outcome is proper predictive loss:

- binary classification: log loss of averaged member probabilities;
- multiclass classification: multiclass log loss of averaged probabilities;
- regression: MSE on the standardized target.

For each dataset/seed, report Orbit-TabM's relative proper-loss reduction from:

1. cumulative TabM;
2. local TabM;
3. a validation-selected single basis (cumulative or local, selection made
   without test labels).

Aggregate by dataset first and use a dataset-cluster bootstrap interval. Also
report the task score (accuracy/AUC or RMSE), paired wins, member prediction
correlation, member disagreement, train time, and peak memory.

The primary method gate passes only if, on the 21-dataset confirmation tier:

1. the 95% dataset-bootstrap interval for relative proper-loss reduction over
   cumulative TabM excludes zero;
2. at least 60% of datasets have a positive mean reduction;
3. mean reduction is at least 0.5%;
4. mean relative proper-loss reduction versus the validation-selected single
   basis is positive;
5. there are no excess training failures.

## Integrity requirements

- cumulative-to-local relative reconstruction error below `1e-8` on train,
  validation, and test;
- every orbit transform square and full rank;
- ordinary 2-D input and repeated identical 3-D member input give matching
  TabM predictions below `1e-6` in evaluation mode;
- identical trainable parameter counts for ordinary and orbit models;
- no test target is read for representation construction, training, early
  stopping, or single-basis selection.

## Stop rule

If the confirmation gate fails, retain the result as a mechanistic negative:
basis-induced diversity exists in separately trained models but does not
translate into a parameter-efficient same-model method. Do not tune the method
on Adult or silently substitute random rotations as the headline arm.
