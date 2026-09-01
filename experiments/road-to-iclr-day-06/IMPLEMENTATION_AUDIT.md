# Day-6 implementation audit

Audit date: 2026-08-29 Asia/Seoul

## Paired construction

- `render` returns transformed dense coordinate `j` from canonical coordinate
  `coordinate_map[j]`.
- `matched_state` consequently sets transformed first-layer column `j` to
  canonical column `coordinate_map[j]`, the correct conjugacy direction.
- category-ID relabeling is converted into a one-hot coordinate permutation;
  unknown-category coordinates remain aligned.
- classification outputs and training targets are mapped back through the same
  declared class action.  Day 6 currently uses the identity class action so it
  isolates schema-coordinate arithmetic.
- each path reloads the same canonical initial state and resets initialization,
  minibatch-order, and PyTorch dropout seeds.  TF32 is disabled and PyTorch
  deterministic algorithms are enabled.

## Interface intervention

`ExactAccumLinear` keeps parameters and outputs in float32.  Only its affine
operands and bias are converted to float64 for the call to `linear`, and the
result is cast immediately to float32.  MLP's aliased `network[0]` reference is
updated with the same replacement layer.  ResNet and dense-stem FT-Transformer
consume `model.first` directly.

## State-level update audit

`test_iea64_one_update_conjugates_parameters_and_adam_state` constructs a
nontrivial 13-coordinate schema permutation, matched MLP/ResNet/FT-Transformer
states, identical classification targets, and identical dropout tape.  After
one AdamW update on CPU it verifies bitwise equality of:

- every transformed parameter against the conjugated canonical parameter;
- AdamW step counters;
- first moments; and
- second moments, including the conjugated first-layer columns.

This directly checks the update-operator premise for one deterministic batch.
Long H1/H3 artifacts store aligned predictions at checkpoints, not every
intermediate parameter or optimizer state.  Reports therefore distinguish
state-level one-update closure from long-horizon checkpoint-prediction closure.

## Artifact audit

`audit_day6_integrity.py` verifies current protocol/config hashes, frozen
dataset/model/seed menus, expected precision/path ordering, checkpoint menus,
rank-four prediction shapes, target row counts, paired NPZ/JSON stems, and
finite predictions.  At final closure:

- H1: 72 bundles / 576 paths / 0 errors;
- H2: 18 bundles / 216 paths / 0 errors;
- H3: 36 bundles / 288 paths / 0 errors;
- H4: 324 bundles / 972 paths / 0 errors.

## Timing limitation

Within H3 each FP32 arm always runs before its IEA64 arm.  Timing is measured
per training trajectory but checkpoint evaluation is excluded.  The frozen
≤25% overhead gate is retained as an engineering feasibility check; it is not
an order-randomized microbenchmark or a precise kernel performance claim.

## H4/H5 pre-run smoke

Before the permanent H4 matrix, a temporary CPU smoke exercised one
Bank/MLP optimizer configuration across all three H4 seeds.  It produced three
bundles / nine paths, ran both H4 and H5 analyzers, and passed the H4 artifact
hash/menu/shape/finiteness audit.  Outputs were removed and their metrics were
not inspected or used as evidence.  This checks serialization and seed-pair
plumbing without changing the frozen 324-bundle matrix.

Before any permanent H4 artifact, the dual-GPU enumeration was also corrected
to avoid perfect device/batch-size aliasing.  A parity design over the three
optimizer-axis indices assigns each GPU exactly half of every batch-size,
weight-decay, and learning-rate level within every seed/dataset/model cell.
The partition remains disjoint and exhaustive over all 324 bundles.

The H4/H5 rank analyzers now explicitly map a constant score to zero Spearman
association.  This is required because optimizer configuration cannot affect
epoch-zero predictions, so that negative control is constant by construction
and raw Spearman would return `NaN`.  Two temporary bundles at opposite
optimizer settings verified bitwise-identical epoch-zero prediction tensors;
they were excluded and removed.  Average-rank handling at the H5 top-quartile
boundary is also tested.  The complete local suite passes 16/16 tests.

At H3 bundle 16, rerunning the H6, H7, H8, and H9 analyzers produced byte-for-
byte identical SHA-256 hashes for all four JSON summaries and their primary
prospective/dataset CSVs.  The final audit must repeat this check after the
complete matrix; this partial check establishes deterministic analysis code,
not final scientific stability.

The final reproducibility audit reruns all eight H3–H9 analyzers and byte-
matches all 28 declared JSON/CSV outputs with zero mismatches.  The final local
suite passes 16/16 tests.  All eight declared figures were regenerated in PNG
and PDF; visual inspection confirms readable titles, axes, legends, thresholds,
dataset/model panels, and no clipping or blank output.
