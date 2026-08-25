# Day 3 prospective broad basis-invariance benchmark

Version: 1.0, frozen before any broad-benchmark model outcome was generated on
2026-08-25 (Asia/Seoul).

## Question

Does the equivalent-basis sensitivity found on the three Day 3 anchors persist
across a broad tabular collection, architectures, and naturally motivated exact
encodings? Which existing or proposed optimizer intervention removes it without
damaging the unperturbed task?

The study is a mechanism and robustness benchmark. It is not a claim that an
adversarially transformed table is a normal deployment distribution.

## Dataset set

The broad tier contains all locally available, protocol-compatible datasets:
nine official TabPack tasks from Day 1 and sixteen separately prepared
finance/credit tasks. No dataset was selected using a Day 3 remedy outcome.
Training is capped at 50,000 rows and validation/test at 20,000 rows using a
fixed seed when an existing split is larger. Existing random, grouped, or
chronological splits are preserved.

The 25 tasks and ten expensive-confirmation tasks are enumerated in
`experiments/day3/configs/broad_preregistered.json`.

## Representations

All fitting uses training inputs only.

The controlled reference contains:

- full-rank piecewise-linear ramp blocks for numerical fields, individually
  whitened after training-quantile knots;
- standardized binary fields;
- full-rank Helmert contrasts for categorical fields with at most 100 training
  levels.

The number of numerical bins is chosen deterministically before training so
that numerical representation width is at most approximately 256. An exact
synthetic endpoint is `X' = XB`, where `B` has target condition number 1,000
and is normalized to preserve training activation energy. The map, rank,
reconstruction error, trace, and achieved condition number are recorded.

The natural exact pair uses the same spline and categorical spaces but changes
coordinates without a random stress matrix:

- cumulative PLE ramps + Helmert categorical contrasts;
- local adjacent ramp differences + adjacent categorical contrasts.

Each semantic block is separately whitened in both families to stabilize wide,
nearly rank-deficient spline blocks while retaining the orthogonal ambiguity
induced by the natural basis choice. Blocks are never mixed at this stage.

Affine equivalence, including the first-layer bias, must be below `1e-8`
relative reconstruction error on every split or the cell fails closed.
Standardized raw and quantile-normalized raw coordinates are non-equivalent
preprocessing baselines and are labeled accordingly.

## Architectures

- three-layer MLP;
- three-block ResNet;
- official RTDL FT-Transformer backbone after a dense affine stem;
- official TabM backbone after a dense affine stem.

The dense stem is necessary for an arbitrary full input-basis map to remain an
exact reparameterization of the first layer. Results are therefore labeled
`dense_stem_ft_transformer` and `dense_stem_tabm`, not presented as native
feature-token invariance.

## Optimizer and preprocessing comparisons

- AdamW;
- diagonal standardization + AdamW;
- full whitening + AdamW;
- invariant anchor canonicalization + whitening + AdamW;
- progressively sketched invariant-anchor canonicalization + whitening +
  AdamW, which avoids a full-row SVD when 512–8,192 deterministic training rows
  already span the retained feature space;
- inverse-input-second-moment first-layer natural gradient with compatible
  initialization;
- first-layer K-FAC (both input and preactivation-gradient factors);
- full-model Adam-grafted Shampoo;
- full-model SOAP using Adam in Shampoo eigenbases, following the public ICLR
  2025 algorithm (first-step initialization and QR eigenbasis refresh).

Implementations record whether they are exact published algorithms, scoped
first-layer variants, or controls. Hyperparameters are selected on the three
calibration datasets Adult, California, and Otto from the fixed grids in the
JSON configuration, then frozen. Test metrics are not used for selection.

## Tiers and seeds

1. Broad sensitivity: all 25 tasks, four architectures, AdamW, κ in `{1,1000}`,
   seeds `[0,1,2]`.
2. Broad remedy: all 25 tasks, MLP, all nine remedies/controls, both κ
   endpoints, seeds `[0,1,2]`.
3. Architecture confirmation: ten prespecified representative tasks, all four
   architectures, AdamW and the four strongest mechanistically distinct
   remedies selected by mean validation sensitivity subject to no more than a
   1% normalized unperturbed-performance loss, both endpoints, seeds
   `[0,1,2,3,4]`.
4. Natural encoding: all 25 tasks, MLP and ResNet, the exact natural pair and
   two standard preprocessing baselines, seeds `[0,1,2]`.
5. Robustness/efficiency: representative low-, medium-, and high-dimensional
   tasks; rank duplication levels `{0, .25, .5}`, covariance ridge values
   `{1e-10,1e-8,1e-6,1e-4}`, temporal versus random/grouped splits, and measured
   wall time and peak CUDA memory.

Failures remain in the result table. Missing cells are not silently dropped.

## Metrics and inference

- binary classification: ROC-AUC (accuracy is secondary);
- multiclass: accuracy;
- regression: RMSE in original target units.

Within a task, positive utility always means better. Basis sensitivity is the
paired κ=1000 utility minus κ=1 utility, so negative is harmful. Cross-task
aggregation first standardizes regression differences by the κ=1 RMSE and
reports source-group means, median, wins/ties/losses, bootstrap intervals, and
two-sided Wilcoxon tests. Five-seed confidence intervals are reported for the
confirmation tier.

## Gates

The paper-level basis-sensitivity claim requires a negative median paired
AdamW sensitivity, a 95% dataset-bootstrap interval excluding zero, and at
least 60% harmful task/model pairs.

A general remedy requires at least 80% reduction in absolute harmful
sensitivity relative to AdamW, no more than 1% mean normalized unperturbed
performance loss, and no material divergence excess. Exact canonicalization is
reported separately from deployable optimizer remedies.

The novelty verdict must explicitly compare the result with natural gradient,
K-FAC, whitening, Shampoo/SOAP, and general canonicalization. Existing affine
invariance theory is prior art, not claimed as new.
