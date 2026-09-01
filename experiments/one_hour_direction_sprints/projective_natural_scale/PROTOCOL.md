# Frozen natural-scale projectivity protocol

Frozen before any outcomes from this panel were computed on 2026-08-31.

## Question

Does static projective prediction retain useful aggregate uncertainty on natural
tabular regression tasks when it is compared with strong point predictors and
exact coherent Gaussian baselines, rather than only with a separately trained
direct neural control?

The experiment deliberately separates two hypotheses:

1. **Original-network hypothesis.** The synthetic-prior projective checkpoint
   transfers well enough to natural data to beat its capacity-matched direct
   checkpoint on unseen linear functionals.
2. **Strong-mean hypothesis.** Given TabPFN's marginal predictions, a single
   projective covariance assembled from ensemble-view correlations improves
   aggregate proper scores without changing any point prediction or marginal
   variance.

The second hypothesis is the important mechanism check. It prevents a weak
zero-shot mean model from being mistaken for evidence against projectivity.

## Frozen panel

- Natural regression datasets (12): `fremtpl_claim_count`,
  `kdd17_stock_return`, `abalone`, `kin8nm`, `pol`, `puma32h`, `cpu_act`,
  `elevators`, `bike_sharing`, `sulfur`, `superconduct`, and `wine_quality`.
- Prospective split seeds: 20260921, 20260922, 20260923.
- Each split uses 60% train, 20% validation, and 20% test, capped at 2,048,
  512, and 512 rows after splitting.
- Four independently sampled 16-row labelled contexts per dataset/split.
- Thirty-two disjoint evaluation groups of 12 query rows per context and
  partition. Validation and test labels are never shared.
- Functionals: point, subset mean, signed difference, unit-norm dense, and
  scaled dense.
- Numeric/categorical preprocessing is fit only on the training covariates.
  The original neural checkpoints receive a standardized PCA-8 view; all other
  models receive the full imputed/one-hot standardized view.
- Every learner sees only the 16 context labels. Context mean and scale are used
  for target normalization; the full training-label scale is used only to put
  reported errors into comparable per-dataset units.

## Frozen models

1. `neural_projective`: moment ensemble of the three already-trained
   projective checkpoints. Between-checkpoint mean variation is included in
   the joint covariance.
2. `neural_direct`: moment ensemble of the three already-trained direct
   checkpoints; queried separately for every functional.
3. `bayes_linear`: Bayesian ridge with an explicit intercept; its posterior
   predictive covariance is projected analytically.
4. `gp_rbf`: Gaussian process with constant-times-RBF plus white-noise kernel;
   its posterior predictive covariance is projected analytically.
5. `tabpfn_independent`: TabPFN mean and marginal variance with zero off-diagonal
   covariance.
6. `tabpfn_projective`: the same TabPFN mean and exact same marginal variance,
   with correlation estimated from its eight inference ensemble views. The
   correlation shrinkage parameter is selected from {0, .25, .5, .75, 1}.
7. `tabicl_independent`: TabICL-v2 mean with marginal scale estimated from its
   0.16 and 0.84 quantiles and zero off-diagonal covariance.

All Gaussian methods receive one scalar covariance temperature, estimated by
minimum validation NLL over all validation contexts and functional families for
that dataset/split. Scaling an entire covariance preserves projectivity.
TabPFN correlation shrinkage and temperature are selected jointly on validation
and then frozen for test.

For fixed raw variances, the NLL-optimal variance multiplier is the mean
standardized squared residual, clipped only for numerical safety to
`[1e-3, 1e3]`.

## Metrics

- Point RMSE.
- Gaussian NLL and Gaussian CRPS for every functional family.
- 90% Gaussian interval coverage.
- Dataset-level paired win counts and mean advantages.
- Covariance PSD, symmetry, scaling, additivity, and marginal-preservation
  numerical audits.

The Gaussian scoring approximation is intentionally common to every method. It
tests whether the learned first two moments are useful for arbitrary linear
functionals; it does not claim that TabPFN or TabICL marginals are Gaussian.

## Precommitted interpretation gates

### Original-network transfer

- Dense/scaled-dense test NLL beats `neural_direct` in at least 60% of matched
  cells and has positive mean advantage.
- Point RMSE is within 25% of `tabpfn_independent` on at least 6/12 datasets.

Both are required to call the original zero-shot network broadly viable.

### Strong-mean projectivity

- Point predictions and uncalibrated marginal variances of
  `tabpfn_projective` and `tabpfn_independent` agree to numerical precision.
- Across dense, scaled-dense, difference, and subset-mean queries,
  `tabpfn_projective` improves both mean test NLL and mean test CRPS.
- It wins dataset-level aggregate NLL on at least 7/12 datasets.
- Its absolute 90%-coverage error is no worse on average.

All four are required for a positive strong-mean mechanism result. A positive
result motivates learning a dedicated amortized joint covariance on top of a
strong tabular foundation model. A negative result retires ensemble-view
correlation as the immediate mechanism, not the projectivity principle itself.

### Competitive breadth

For context, report (without turning them into post-hoc hard gates) whether the
strong-mean projective model beats `bayes_linear`, `gp_rbf`, and
`tabicl_independent` in point and aggregate metrics, dataset by dataset.

## Integrity rule

No gate is interpreted unless all 36 dataset/split cells and all four context
replicates complete, all test rows are finite, validation and test indices are
disjoint, covariance audits pass at 1e-5 tolerance, and the saved protocol hash
matches this file.
