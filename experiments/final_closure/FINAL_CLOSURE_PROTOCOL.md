# OrbitCover final ICLR closure protocol

Status: **FROZEN BEFORE FINAL-CLOSURE OUTCOMES** on 2026-08-28 (Asia/Seoul).

This protocol prospectively governs the final closure experiments requested in
`experiments/road-to-iclr-day-05/AGENT.md — FINAL ICLR CLOSURE EXPERIMENTS.md`.
It does not change the evidence grade of any earlier experiment.  Corrections
after this freeze must be recorded in `PROTOCOL_DEVIATIONS.md`; this file and
`final_closure_config.json` must not be edited after final-closure outcomes are
examined.

## 1. Scientific question and decision rule

The primary question is whether interaction-balanced semantic randomization
estimates a randomized learning-pipeline expectation more efficiently than
training the canonical representation with genuinely independent full-pipeline
randomness.

The final verdict is:

- `SUPPORTED` only if OrbitCover beats canonical-independent ensembling in a
  dataset-balanced aggregate, the advantage survives realistic/converged
  training, SRSWOR heterogeneity is meaningfully associated with interaction
  order, and no major architecture has an unexplained systematic failure;
- `PARTIALLY SUPPORTED` if the finite/coupled design remains useful but the
  independent-canonical comparison, convergence comparison, or interaction
  explanation fails;
- `NOT SUPPORTED` if canonical-independent ensembling broadly matches or
  dominates and neither a distinct target nor a useful coupling advantage
  remains.

No dataset, model, metric, or threshold may be changed in response to an
outcome.

## 2. Shared definitions

### 2.1 Datasets and splits

Experiment A uses the complete frozen modern-neural panel:

- classification: Australian Credit, Bank Marketing, Credit Card Default,
  German Credit, HELOC, and LendingClub;
- regression: FREMtpl Claim Count, KDD17 Stock Return, OpenML Abalone 183,
  OpenML Kin8nm 189, OpenML Pol 201, and OpenML Puma32H 308.

It uses split seeds `2026082801`, `2026082811`, and `2026082821`, with the
same deterministic re-pooling, splitting, preprocessing, and 2,048/512/512
train/validation/test caps as the previous completion panel.

Experiment B uses a prospectively selected balanced six-source panel:

- classification: Bank Marketing, Credit Card Default, and HELOC;
- regression: FREMtpl Claim Count, KDD17 Stock Return, and OpenML Abalone 183.

These were selected before final-closure outcomes to span mixed/numerical
schema, source size, nuisance risk, interaction fraction, and prior
OrbitCover/SRS behavior.  HELOC is retained as a prior source-level SRSWOR
loss.  Experiment B uses split `2026082801` for the full scaling grid.  The
three-split evidence remains supplied by Experiment A and the prior panel.

Experiment D uses Bank Marketing, HELOC, FREMtpl, and KDD17 on all three fixed
splits.

### 2.2 Models

All primary cells use MLP, ResNet, FT-Transformer, and TabM with the same
architectural widths as the frozen completion panel.  Experiment A also runs
the already-stable CatBoost and XGBoost implementations as a secondary
first-split scope check on all 12 datasets.  Secondary classical cells do not
replace or dilute the 144-cell neural primary panel.  TabPFN remains prior
external-cover evidence and is not forced into independent retraining.

### 2.3 Semantic nuisance actions

Schema actions use four fixed feature-block orders, up to four valid within-
field category-ID maps, and two target-ID maps for binary classification (one
for regression).  Actions are fixed by the earlier view seed `2026082849` and
aligned back to canonical prediction coordinates.  No new action is selected
from its outcome.

### 2.4 Full-pipeline master RNG

Every independent fit receives a unique signed-63-bit integer `master_seed`.
Sub-seeds are derived deterministically by SHA-256 domain separation:

```text
SHA256("orbitcover-final-closure-v1" || master_seed || domain)
```

with domains `initialization`, `dataloader`, `dropout`, `worker`,
`preprocessing`, and `model_operation`.  The first eight bytes, masked to 63
bits, define each sub-seed.  Initialization uses only the initialization
sub-seed.  Per-epoch minibatch permutations use the dataloader sub-seed.
Dropout and other torch stochastic operations use the dropout/model-operation
sub-seeds.  Worker seeds are recorded even though the reference runner uses no
dataloader workers.  Preprocessing is deterministic in the primary recipes;
its derived seed is nevertheless recorded.  Master seeds never repeat within
an estimator, cell, or persisted independent pool.

### 2.5 Training and metrics

Unless Experiment B changes an explicitly indexed scale/budget condition,
training uses AdamW for 20 epochs, batch 256, learning rate `1e-3`, weight
decay `1e-4`, and the prior completion-panel widths/dropout.  Classification
uses Brier residual in aligned probability space; regression uses standardized
prediction MSE.  Quadratic prediction-space residual to the appropriate
reference is primary.  Secondary outcomes are Brier/MSE, log loss, accuracy,
AUROC, ECE, MAE, RMSE, and R-squared where defined.

## 3. Experiment A — independent canonical-seed showdown

### 3.1 Prediction pools and references

For every primary dataset×split×model cell, train a joint pool with eight
independent master seeds for every valid schema action.  The master seeds are
unique across all pool entries, rather than a finite reused seed menu.  Extend
the canonical schema to 128 independent master seeds total.  Thus the minimum
canonical reference has 128 genuinely independent fits and the joint reference
has 512 regression or 1,024 classification fits in the usual 64/128-action
products.

Define separately:

- `Q_CANONICAL`: the 128-fit canonical independent-seed mean;
- `Q_JOINT`: the schema-balanced mean of the eight independent predictions at
  every schema action;
- `Q_COUPLED`: the exact finite quotient of the prior declared
  schema×two-initialization×two-order product.

Monte Carlo uncertainty in `Q_CANONICAL` and `Q_JOINT` is estimated by
seed-block bootstrap and reported with their squared prediction distance.  No
claim of equal expectations is made merely because a point estimate is small.

### 3.2 Equal-budget methods

At budgets 4, 8, 16, 32, and 64 compare:

- `CANONICAL-INDEPENDENT`: canonical schema and fresh master seed per fit;
- `IID-JOINT`: IID schema and fresh master seed per fit;
- `SRS-JOINT`: schema without replacement and fresh master seed per fit;
- `OC1-INDEPENDENT`: marginally balanced schema and fresh master seed per fit;
- `OC2-INDEPENDENT`: strength-2-balanced schema and fresh master seed per fit;
- `OC2-COUPLED`: the prior finite schema×initialization×order construction;
- `OC2-PACKED`: prior disjoint block construction at budgets 32 and 64 where
  the finite product supports it.

All methods have exactly B fits.  Cached-pool resampling uses 512 deterministic
estimator draws per cell/method/budget.  A draw never repeats a master seed.
Overlapping cached-pool draws estimate conditional expected residuals only;
they are never treated as independent inferential units.  Dataset is the
primary statistical unit and dataset×split is secondary.

Each residual is measured against its declared estimand: canonical against
`Q_CANONICAL`, independent joint schema methods against `Q_JOINT`, and coupled
methods against `Q_COUPLED`.  Cross-target residuals are additionally reported
so an apparent variance advantage cannot conceal a changed expectation.

### 3.3 Primary Experiment-A tests

At B=16 answer, without a post-hoc threshold change:

1. Does `OC2-INDEPENDENT` beat `CANONICAL-INDEPENDENT`?
2. Does `OC2-COUPLED` beat `CANONICAL-INDEPENDENT`?
3. Does joint balancing add benefit beyond schema balancing with fresh RNG?
4. Does `OC2-COUPLED` beat `OC2-INDEPENDENT`?
5. Are `Q_CANONICAL` and `Q_JOINT` materially/statistically distinguishable?
6. What canonical-independent fit budget matches 16 OrbitCover fits?
7. How heterogeneous are these answers across the four architectures?

For each headline comparison report cell and source wins, equal-source mean
and median relative reduction, dataset-clustered 95% bootstrap interval, and
architecture-stratified reduction.  The bootstrap uses 10,000 deterministic
resamples of datasets.  IID-equivalent budgets are obtained by monotone
interpolation of the empirical canonical residual curve and are suppressed
when the curve does not bracket the target.

## 4. Experiment B — realistic scale and convergence

### 4.1 Nested training sizes

For each selected source create one deterministic training ordering from the
full training partition, stratified for classification.  Evaluate nested sizes
2,048, 8,192, 32,768, and the full available training partition whenever
distinct and feasible.  Sizes above the available count are omitted; examples
are never duplicated.  Validation and test partitions are fixed across nested
training sizes.

### 4.2 Optimization budgets

At every feasible size run 20, 50, 100, and 200 epochs plus a convergence arm.
The convergence arm uses a maximum of 500 epochs, validation-loss early
stopping with patience 30, relative minimum improvement `1e-4`, restores the
best checkpoint, and evaluates validation loss after every epoch.  Ties prefer
the earlier epoch.  Record epoch-level training loss, validation loss, global
gradient norm, parameter-update norm, best epoch, and stopped epoch.

### 4.3 Nuisance design and analysis

Each trajectory condition uses a fixed 128-run mixed-level strength-3 schedule
over four feature actions, four category actions when valid, two target actions
for classification, and eight independent master seeds.  Collapsed factors are
recorded and the construction must have maximum possible row uniqueness and
formal balance for every estimable one-, two-, and three-factor projection.

At the four mandatory corners—small/20, small/convergence, largest/20, and
largest/convergence—the full valid nuisance product is used.  The already
trained Experiment-A pool supplies small/20 where hashes match.  Exact corner
products and orthogonal contrasts on trajectory arrays estimate total nuisance
variance, schema-only variance, stochastic-only variance, schema×stochastic
interaction, main/pair/triple/higher fractions, effective interaction order,
and OC2/SRS and OC2/canonical-independent ratios.  Non-corner high-order
components from the fractional trajectory are explicitly labeled estimates.

At B=16 compare strength-2, SRSWOR, IID joint, and canonical independent.
B=32 is secondary.  Raw trajectories are primary; log-risk slopes versus log N
or optimization budget are descriptive with dataset-clustered intervals and
are not promoted as asymptotic exponents.

### 4.4 Matched-function convergence subexperiment

Use Bank Marketing (classification) and FREMtpl (regression), all four neural
models, and 20 epochs, 100 epochs, and convergence.  Across the complete valid
schema menu compare ordinary initialization with the exact transformed state.
The maximum aligned initial-function difference must be at most `1e-6`.
Report ordinary variance, matched variance, and fraction removed.  This test is
mandatory even if the ordinary nuisance variance becomes small.

### 4.5 Experiment-B questions

Report whether absolute nuisance variance shrinks with N/training, whether
relative OrbitCover efficiency survives, whether interaction order evolves,
and whether FT-Transformer matched residuals survive more strongly than
MLP/ResNet.  For every model the report must include nuisance variance and
OC2/SRS at all four mandatory corners.

## 5. Experiment C — interaction spectrum and SRSWOR failures

The inclusion rule is outcome-independent: include every retained exact/broad
cell for which aligned fANOVA components and equal-budget OC2/SRSWOR B=16
residuals can be regenerated.  This includes all 144 completion neural cells,
the 57 earlier cells in `without_replacement_baseline_cells.csv` when they join
uniquely to their panel fANOVA row, and compatible new A/B cells.  Duplicates
with identical dataset/split/model/config hash are included once, favoring the
rawest exact tensor.

For every cell compute main, pair, main+pair, triple, higher-order, effective
interaction order, total nuisance variance, finite population, B/N, factor
count, architecture, dataset, and task.  Define gain as
`1 - residual_OC2 / residual_SRS`; degenerate zero/zero cells are retained in
the failure inventory but excluded from ratios and rank correlations.

Primary analysis is Spearman correlation of gain with main+pair fraction;
secondary is gain with higher-order fraction.  Both use a 10,000-resample
dataset-clustered bootstrap.  Provide architecture-stratified estimates and
plots.  A transparent linear model
`gain ~ main_pair_fraction + sampling_fraction + architecture` uses
leave-one-dataset-out predictions; no black-box model is allowed.

List every cell where SRSWOR strictly beats OC2 and every non-positive source
mean.  The prior completion-panel source comparison must explicitly account
for all four non-wins (HELOC plus the three exact/degenerate regression source
ties).  Where a strength-3 B=64 result exists, test whether it recovers the
strength-2/SRS loss and retain both successes and failures.

## 6. Experiment D — coupling-mechanism decomposition

Experiment D is planned, not optional in execution unless a documented
technical impossibility survives repair attempts.  Use Bank Marketing, HELOC,
FREMtpl, and KDD17, all four neural models, and all three splits at B=16.

Use the finite factors schema, initialization (four fixed seeds), and
dataloader/dropout order (four fixed seeds).  At equal fit count compare:

- none/IID;
- schema balanced only;
- initialization balanced only;
- order balanced only;
- schema×initialization balanced;
- schema×order balanced;
- initialization×order balanced;
- all factors jointly strength-2 balanced.

The common full finite prediction registry is reused whenever possible.
Residuals are measured to its exact finite quotient.  Relate each ablation's
gain to the corresponding fANOVA component mass and report source-clustered
uncertainty.

## 7. Persistent registry and failure handling

Every fit is keyed by dataset, split, model, model-config SHA-256, schema-action
SHA-256, master RNG seed (or declared finite init/order pair), training size,
training-budget label, and matched arm.  Prediction files are written
atomically and registered only after finite/alignment/shape checks.  Complete
identical keys are reused; partial or corrupt keys are rerun.  Logs and
manifests are restartable.  OOM repairs may reduce physical batch size while
preserving the optimizer recipe and must be recorded as deviations.  A model
or dataset is not silently dropped or replaced.

## 8. Statistical aggregation

Fits and cached estimator draws are not independent scientific evidence.
Primary estimates average within each dataset and then equally across
datasets.  Secondary estimates use dataset×split.  Report paired effects,
dataset-clustered 95% intervals, task-balanced means, architecture strata, win
counts, medians, and heterogeneity.  No sign test is the sole claim criterion.

## 9. Integrity gates

Before analysis, tests must verify semantic preservation, unique master seeds,
deterministic domain-separated sub-seeds, canonical schema invariance,
OC2-independent fresh RNG, exact coupled construction, equal budgets, nested
training subsets, matched initial functions, prediction alignment, fANOVA
reconstruction, formal strength-2/3 balance, maximum row uniqueness, and no
test leakage.  Final output reports tests passed/total.

The final read-only audit must additionally prove no missing mandatory cells,
duplicate independent master seeds, corrupt predictions, unequal comparisons,
unrecorded deviations, or non-regenerating summaries/figures/tables.

## 10. Frozen deliverables

Required figures (PNG and PDF) are:

1. independent-seed residual curves;
2. architecture comparison at B=16;
3. canonical versus joint/coupled expectation distance;
4. nuisance variance versus optimization progress;
5. nuisance variance versus training size;
6. OC2/SRS ratio versus optimization progress;
7. main+pair fraction versus OC2/SRS gain;
8. fANOVA spectra for strongest OC2 and SRS wins;
9. ordinary versus matched path over convergence;
10. coupling-mechanism ablation.

Required tables, each as CSV and Markdown, are:

- `table_A_independent_seed_comparison`;
- `table_B_convergence`;
- `table_C_interaction_prediction`;
- `table_D_coupling_ablation`;
- `table_E_final_claims`.

Raw outputs, manifests, summary CSV/JSON, compute accounting, registry,
protocol deviations, and `FINAL_AUDIT.md` live under
`experiments/final_closure/`.  The standalone final report is regenerated from
audited artifacts to `experiments/final_closure/results.md`,
`experiments/road-to-iclr-day-05/results.md`, and repository-root `results.md`.
Its section structure and exact verdict/readiness/recommendation choices follow
the closure runbook verbatim.
