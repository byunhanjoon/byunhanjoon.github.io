# Day-5 remaining-program completion protocol

Status: frozen before inspecting outcomes from the new completion runs on
2026-08-28. This protocol prospectively covers only work missing after the
existing Day-5 evidence. It does not retroactively make earlier adaptive work
prospective.

## Primary panel

Twelve public datasets, balanced by task:

- classification: Australian Credit, Bank Marketing, Credit Card Default,
  German Credit, HELOC, LendingClub;
- regression: FREMtpl Claim Count, KDD17 Stock Return, OpenML Abalone 183,
  OpenML Kin8nm 189, OpenML Pol 201, OpenML Puma32H 308.

Use three fixed split seeds `2026082801`, `2026082811`, and `2026082821`.
Local pre-split sources are deterministically re-pooled before re-splitting.
OpenML IDs are immutable and cached with metadata. Cap each split at 2,048
training, 512 validation, and 512 test rows using task-stratified subsampling
for classification.

## Models

Primary modern neural families:

- MLP;
- ResNet;
- FT-Transformer using `rtdl_revisiting_models`;
- official TabM using the installed `tabm` package.

Strong classical completion:

- native CatBoost;
- sklearn HistGradientBoosting;
- XGBoost;
- LightGBM;
- one-hot logistic/ridge invariant control.

The full classical panel may use the first split where three-split evidence
already exists for CatBoost/HistGB; the four neural families use all three.

TabPFN v2.5 uses the six classification datasets, all three splits, and reports
inference members separately from fitted models.

## Representations and stochastic variables

The exact modern-neural subset uses the declared product:

- four feature-block orders;
- four within-field category-ID permutations;
- two target-label IDs for classification, one for regression;
- two initialization seeds;
- two dataloader-order seeds.

This gives 128 classification and 64 regression cells. Feature blocks contain
standardized numerical scalars or one-hot categorical fields. The one-hot
coordinate order follows the transformed category IDs; all mappings are fit on
training data and reused for validation/test. Predictions are aligned to
canonical targets. The menu is finite and not the full natural symmetry group.

For the broad three-split panel, use a deterministic 64-action reference drawn
uniformly from the same product, plus equal-budget randomized methods at
`B=1,2,4,8,16,32,64`. The exact subset comprises Australian Credit, Bank
Marketing, FREMtpl, and KDD17 Stock Return on the first split for all four
neural families.

Initialization and dataloader seeds are distinct factors. Dataloader order is
changed without changing initialization. A separate row-order semantic control
permutes training rows while holding minibatch grouping rules fixed.

## Training

- fixed model defaults across every representative;
- 20 epochs, AdamW, batch 256, no outcome-adaptive early stopping;
- widths chosen once: MLP/ResNet 128, FT token size 32, TabM latent 64 with
  four internal members;
- primary learning rate `1e-3`, weight decay `1e-4`;
- record wall time, peak GPU memory, device, package versions, and fit count;
- two GPUs may execute independent cells, but each fit is one-threaded on CPU.

## Matched-function controls

For every primary neural family, permute the dense input stem columns and
classification output coordinates so the transformed model is the exact same
aligned initial function. Verify maximum initial gap below `1e-6`. Compare
ordinary and matched schema variance on the four-dataset exact subset. If a
family's internal output cannot be transformed safely, freeze target IDs to
canonical for that matched arm and report the reduced scope.

## TabPFN

On the six classification sources compare, at equal external action budgets:

1. canonical/default inference;
2. one estimator with internal feature/class shifts disabled;
3. default internal ensemble;
4. IID external nuisance averaging;
5. SRSWOR external averaging;
6. strength-1 and strength-2 external covers.

Report total TabPFN forward ensemble members and calls separately. Do not call
an inference call a fitted model.

## Larger-menu experiment

For MLP, ResNet, FT-Transformer, and TabM on Australian Credit and FREMtpl,
sample nested valid transformation menus of sizes 4, 8, 16, 32, and 64 per
schema generator where feasible. Use a 256-action Monte Carlo reference to
separate menu-target movement from within-menu estimator error.

## Metrics and gates

Primary: Brier for classification and standardized MSE for regression.
Secondary: log loss, accuracy, AUROC, ECE, MAE, RMSE, and R2 as appropriate.

At budget 16 report quotient error, ranking fidelity, selected validation
regret, and held-out test regret for IID, SRSWOR, seed-only, schema-only,
strength-1, LHS, Sobol, and strength-2. Strength-3 is evaluated on exact tensors
at budget 64. Cross-score and packing are evaluated at 32/64 fits.

The completion is descriptive/falsificatory: no universal positive gate is
imposed. A requirement is complete only when all configured cells either have
valid outputs or a recorded reproducible failure. No failed dataset/model may
be replaced.

## Required outputs

- configs and protocol hashes;
- per-fit manifests and compact aligned prediction tensors;
- dataset, model, nuisance-risk, estimator, ranking/selection, and
  matched-function tables;
- the ten figure concepts specified by the Day-5 program;
- updated completion matrix, integrity audit, experiment ledger, and
  `results.md`.
