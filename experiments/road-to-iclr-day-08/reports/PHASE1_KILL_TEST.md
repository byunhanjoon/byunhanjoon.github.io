# Phase I kill test: matched task-isomorphic reparameterization

Date: 2026-08-31

## Decision

**Gate G1 passes narrowly through Route B (posterior instability), with a strong Route C heterogeneity signal. Route A does not pass.**

The defensible Phase I claim is not that every modern tabular foundation model fails, nor that matched transformations cause large universal accuracy collapse. It is:

> Under fully matched, invertible, ordinary nonlinear numerical reparameterizations, current TabICLv2 changes its posterior predictions reproducibly beyond identity noise and substantially more than classification tree controls. Recommended/default inference reduces but does not eliminate the effect. Official Mitra is much less sensitive and is mostly at or near its independent-refit noise floor.

This is enough to continue to the full audit and explanation stages. It is not enough to implement RSPF or begin expensive pretraining. Gate G2 remains open.

## Question

Does a current TFM assign materially different predictions to two fully matched, information-equivalent coordinate representations of the same supervised task, after separating matched transforms from context/query mismatch, seed/refit noise, numerical information loss, and ordinary control-model sensitivity?

## Exact frozen protocol

- Config: `configs/audit/pilot.yaml`, SHA-256 `5c75900a6a3bfb607a24a679b9f8a8545d4f7126cea391e0ef301b85e994720f`.
- Prediction-producing source digest: `595cc6ae397a9a551882104812ecaf6e41b0e49fc87833685b747c467d009227`.
- Grid: 12 OpenML tasks × 7 models × 13 transform settings × 3 frozen model/warp seeds = **3,276 jobs**.
- Tasks: four regression, four binary, and four multiclass datasets; one official OpenML repeat/fold and one frozen split per task; context/query caps 2,048/1,024.
- Models: TabICLv2 single/default, official Mitra default fine-tuning, TabPFN-v2.5 single/default, XGBoost, and CatBoost.
- Transform settings: identity plus positive affine, centered signed power, random monotone piecewise-linear, and held-out monotone spline, each at three frozen values/severities.
- Four conditions: clean, matched whole-task, context-only, and query-only.
- Fit pairing: one original-context fit serves clean/query-only; one transformed-context fit serves matched/context-only. This removes fitted-model randomness from within-context comparisons.
- Stored outputs: raw probabilities or regression predictions for every condition, row IDs, targets, transform state, transform audit, telemetry, checkpoints, package versions, and checksums.
- Primary performance quantity: matched minus clean loss, normalized by a train-only target scale for regression.
- Primary disagreement: classification Jensen–Shannon divergence; regression normalized absolute prediction difference. Mitra disagreement is reported after subtracting its paired identity-refit baseline.
- Statistical unit: dataset. Confidence intervals use 10,000 paired dataset bootstrap draws. The generated table separates binary, multiclass, and regression (four datasets each); supplementary classification summaries pool the eight compatible classification datasets.

TabPFN-3 could not be evaluated: the official v3 route available here requires authenticated client/API access and no local open checkpoint was present. TabPFN-v2.5 is labeled historical evidence, never renamed as v3. TabM was not added after the pilot freeze. These limitations prevent Gate G4 from being considered passed.

## Integrity and coverage

- Coverage: **3,276/3,276 complete; 0 missing; 0 failed; 0 unavailable**.
- Every selected NPZ passed SHA-256 verification and contained finite clean/matched/context-only/query-only predictions with aligned row IDs.
- Every record contained the required shared-fit pairing identifiers.
- Missingness was preserved for every transform. No numerical round trip exceeded relative error `1e-6`; the observed maximum was `3.51e-7` for an ill-conditioned spline inverse.
- Forty-nine model records repeat seven positive-affine transform instances on `students_dropout_and_academic_success` with 2–6 strict-order ties. Direct reconstruction showed that every tie joined adjacent one-ULP encodings of the same decimal grade (for example, `12.514285714285714` and `12.514285714285716`); round-trip relative error was at most `2.9e-15`. The validator permits order ties only below `1e-12` round-trip error and rejects material collisions.
- `seismic-bumps` contains a frozen all-missing numerical feature, `ghazard` (`finite_count=0`, `missing_rate=1`). It caused NumPy descriptor warnings but remained all-missing in all representations. Excluding the entire dataset strengthens rather than removes the combined TabICLv2 classification result.
- Validator/analyzer tests: **56 passed**. The post-run validator source has digest `cf5a3c432d5adcbc7650a1165154048a4dc9c235ff80fa68d2d5d1422c8a5916`; analysis explicitly selects the immutable prediction digest above.

## Main results

The table below averages transform/seed effects within dataset, then bootstraps datasets. “Excess disagreement” subtracts identity-refit disagreement. Classification has eight datasets; regression has four.

| Model | Task group | Normalized loss gap, 95% CI | Excess disagreement, 95% CI | Mean matched | Mean identity noise | Mean mismatch |
|---|---|---:|---:|---:|---:|---:|
| TabICLv2 default | classification | 0.003106 [0.000806, 0.006579] | 0.000752 JS [0.000363, 0.001350] | 0.000752 | 0 | 0.111738 |
| TabICLv2 single | classification | 0.022095 [0.002390, 0.057680] | 0.004292 JS [0.000827, 0.010477] | 0.004292 | 0 | 0.113557 |
| Mitra default | classification | -0.000102 [-0.000290, 0.000045] | 0.000045 JS [0.000004, 0.000099] | 0.000108 | 0.000063 | 0.138048 |
| TabPFN-v2.5 default | classification | 0.003440 [0.000607, 0.007574] | 0.000928 JS [0.000366, 0.001699] | 0.000928 | 0 | 0.125340 |
| XGBoost | classification | 0.000051 [-0.000027, 0.000139] | 0.000065 JS [0.000021, 0.000115] | 0.000065 | 0 | 0.164368 |
| CatBoost | classification | -0.000071 [-0.000232, 0.000086] | 0.000067 JS [0.000030, 0.000108] | 0.000067 | 0 | 0.109655 |
| TabICLv2 default | regression | 0.000707 [0.000462, 0.000929] | 0.027238 normalized absolute [0.014053, 0.043840] | 0.027238 | 0 | 1.57858 |
| TabICLv2 single | regression | -0.000048 [-0.001108, 0.001012] | 0.032142 [0.017561, 0.050037] | 0.032142 | 0 | 1.67258 |
| Mitra default | regression | -0.000047 [-0.000152, 0.000046] | 0.002909 [0.000284, 0.005122] | 0.011003 | 0.008094 | 1.52167 |
| TabPFN-v2.5 default | regression | 0.001825 [-0.000515, 0.004600] | 0.021974 [0.012711, 0.032066] | 0.021974 | 0 | 1.87483 |
| XGBoost | regression | 0.000143 [-0.000079, 0.000365] | 0.006422 [0.001810, 0.011996] | 0.006422 | 0 | 1.11641 |
| CatBoost | regression | 0.000546 [0.000024, 0.001069] | 0.015449 [0.004405, 0.026494] | 0.015449 | 0 | 0.94859 |

The generated task-stratified table gives the sharper TabICLv2-default performance pattern:

| Problem type | Datasets | Normalized loss gap, 95% CI | Excess disagreement, 95% CI | Dataset W/T/L |
|---|---:|---:|---:|---:|
| Binary | 4 | 0.003967 [-0.000036, 0.010879] | 0.000899 [0.000249, 0.002076] | 1/0/3 |
| Multiclass | 4 | 0.002246 [0.000864, 0.003222] | 0.000606 [0.000391, 0.000788] | 0/0/4 |
| Regression | 4 | 0.000707 [0.000462, 0.000929] | 0.027238 [0.014053, 0.043840] | 0/0/4 |

Here a loss is a positive loss gap; W/T/L uses a `1e-4` loss-gap tolerance.

## Interpretable classification changes

Across the eight classification datasets, the same dataset-level bootstrap gives:

| Model | Excess total variation | Excess argmax flips | Matched-clean accuracy change |
|---|---:|---:|---:|
| TabICLv2 default | 1.026% [0.725%, 1.297%] | 0.804% [0.321%, 1.297%] | -0.210 pp [-0.353, -0.084] |
| TabICLv2 single | 2.054% [1.224%, 3.186%] | 1.751% [0.865%, 2.787%] | -0.887 pp [-1.942, -0.252] |
| Mitra default | 0.095% [0.016%, 0.192%] | 0.117% [0.013%, 0.260%] | +0.012 pp [-0.005, 0.030] |
| TabPFN-v2.5 default | 1.326% [0.705%, 2.105%] | 1.300% [0.394%, 2.352%] | -0.397 pp [-0.926, -0.041] |
| XGBoost | 0.135% [0.036%, 0.246%] | 0.139% [0.031%, 0.267%] | +0.005 pp [-0.004, 0.016] |
| CatBoost | 0.171% [0.075%, 0.275%] | 0.149% [0.038%, 0.279%] | +0.017 pp [-0.006, 0.045] |

TabICLv2-default therefore changes posterior mass by about six to eight times the classification tree controls and produces a small but reproducible accuracy reduction. This is a reliability effect, not a catastrophic performance failure.

## Matched versus mismatch and default inference

- Context/query mismatch is much larger than matched reparameterization for nearly every aggregate. For TabICLv2-default it is about 149× the matched JS in classification and 58× the matched normalized regression disagreement. The implementation therefore clearly separates the two phenomena.
- The matched result is nevertheless nonzero, dataset-reproducible, and confidence-separated from identity for TabICLv2-default.
- Default inference reduces TabICLv2 classification excess JS by about 82% relative to the single configuration (`0.000752` versus `0.004292`) and reduces classification loss gap by about 86%, but does not eliminate either.
- TabPFN-v2.5 default similarly reduces disagreement relative to single inference by about 50% in classification and 46% in regression. This is corroborating historical evidence only.

## Transform and severity controls

- Positive affine changes are close to invariant for TabICLv2 classification and small for its regression predictions. The effect is not an ordinary unit-scale bug.
- Nonlinear monotone spline, PWL, and signed-power effects generally increase with declared severity. Signed power and high-severity spline/PWL are strongest, but the effect remains under severities at or below 1.
- Restricting to declared severities `<=1` leaves TabICLv2-default classification loss gap `0.002115` [0.000515, 0.004556], excess JS `0.000486` [0.000229, 0.000897], and regression excess disagreement `0.022417` [0.011500, 0.036479]. The result is not confined to extreme transforms.
- Removing positive affine entirely strengthens TabICLv2-default classification excess JS to `0.001003` [0.000484, 0.001800] and regression excess disagreement to `0.036095` [0.018576, 0.058096]. The one-ULP affine ties cannot explain the result.
- Removing `seismic-bumps` leaves TabICLv2-single classification loss gap `0.005009` [0.001374, 0.009331] and excess JS `0.001340` [0.000667, 0.002201]. Its all-missing feature does not create the broad finding.

## Interpretation

1. **Route A — fail.** Only one evaluated current family, TabICLv2, shows systematic matched performance loss. Mitra does not. TabPFN-v2.5 is not current enough to satisfy the two-current-family condition.
2. **Route B — pass for TabICLv2.** Default TabICLv2 exhibits dataset-reproducible posterior changes, label flips, and a small classification accuracy loss under matched nonlinear transforms, even at ordinary severities and after default ensembling.
3. **Route C — strong signal, not yet a mechanism.** TabICLv2 and Mitra differ sharply: Mitra's excess effect is roughly an order of magnitude smaller in classification and below both tree controls in regression. This is scientifically useful heterogeneity, but connecting it to readout/embedding mechanisms is Gate G2 work.
4. **Overall G1 — narrow pass.** Continue with the full audit, marginal-descriptor analysis, controlled synthetic tasks, and architecture-aware mechanism tests. Do not claim universal TFM non-invariance and do not start RSPF/pretraining until an explanation survives G2.

## Alternative explanations and required next checks

- **Model preprocessing/quantization:** tree controls are not perfectly invariant, especially in regression. Phase II must inspect model-native preprocessing and add rank/quantile baselines before attributing all sensitivity to learned priors.
- **Optimization randomness:** Mitra's independent default fine-tunes have visible identity noise. The reported excess subtracts it, but more seeds or an ICL-only Mitra control are needed for precise small effects.
- **Dataset/split uncertainty:** this pilot has 12 development datasets and one split each. Confidence intervals resample datasets, not splits. A larger development suite and later frozen confirmation are required.
- **Metric magnitude:** classification posterior changes are around one percentage point in total variation for TabICLv2-default. This is reliability-relevant but not a broad collapse.
- **Current-model coverage:** unavailable TabPFN-3 means Gate G4 is open. A credentialed run or another genuinely current, strong family is required before a broad current-TFM claim.
- **Causal mechanism:** architecture heterogeneity alone is correlational. Marginal randomization, synthetic prior-conflict tasks, internal geometry, and causal restoration are required for Gate G2.

## Plots

- `results/analysis/phase1/595cc6ae397a9a55/model_landscape.png`
- `results/analysis/phase1/595cc6ae397a9a55/matched_vs_mismatch.png`
- `results/analysis/phase1/595cc6ae397a9a55/severity_curves.png`

## Authoritative artifacts

- Coverage and completion: `results/analysis/phase1/595cc6ae397a9a55/coverage.json`, `DONE.json`.
- Run-level metrics and paths: `results/analysis/phase1/595cc6ae397a9a55/run_metrics.csv`.
- Dataset/model effects: `results/analysis/phase1/595cc6ae397a9a55/dataset_effects.csv`.
- Dataset-bootstrap model table: `results/analysis/phase1/595cc6ae397a9a55/model_summary.csv`.
- Transform/severity table: `results/analysis/phase1/595cc6ae397a9a55/transform_summary.csv`.
- Raw immutable predictions: paths enumerated in `run_metrics.csv` and append-only `results/MANIFEST.jsonl`.
- Frozen panel audit: `results/panel/pilot_panel__5c75900a6a3bfb60.json`.

## Next decision

Proceed to Phase II and Phase III with the claim scope fixed to **TabICLv2 posterior instability and cross-family heterogeneity**. Prioritize:

1. a larger development audit with current models and rank/CDF/atomic/categorical controls;
2. marginal-descriptor correlations and feature-count/context-size controls;
3. S1–S4 controlled synthetic tasks to distinguish nuisance from useful marginal metadata;
4. TabICLv2 readout/embedding and causal restoration tests versus Mitra;
5. Gate G2 before any method ladder or pretraining.
