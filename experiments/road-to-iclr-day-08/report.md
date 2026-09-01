# Task-Isomorphic Reparameterization in Tabular Foundation Models

## Consolidated research report

Date: 2026-08-31  
Program status: **partially completed; scientifically promising, not submission-ready**

## Executive summary

This program asked whether a tabular model changes its predictions when every occurrence of a numerical feature is transformed by the same information-preserving bijection in both the training/context and test/query sets. Such a matched transformation changes coordinates but not the supervised-learning task.

The clearest result is a reproducible **posterior-stability problem in TabICLv2**. On the fully completed 12-dataset Phase I panel, default TabICLv2 changed its classification posterior by about 1.03% total variation, flipped 0.80% of predictions, and lost 0.21 percentage points of accuracy under matched monotone transformations. Its regression predictions changed by 0.027 normalized absolute units. These effects were much smaller than context/query mismatch attacks, but remained above identity noise and were not eliminated by default inference ensembling.

Official Mitra was much less sensitive and generally remained near its independent-refit noise floor. Historical TabPFN-v2.5 showed instability similar in direction to TabICL, but TabPFN-3 could not be evaluated because no authenticated API access or local checkpoint was available. Therefore, the evidence supports a scoped TabICLv2 reliability claim, not a universal claim about all modern tabular foundation models.

Phase I completed all **3,276/3,276 jobs** and passed Gate G1 narrowly through posterior instability. Phase II was stopped after **8,338/13,440 jobs (62.0%)**. Of those, 8,240 jobs had the identity baselines required for inferential analysis. The partial Phase II snapshot corroborates meaningful prediction disagreement in TabICL and TabPFN-v2.5 with relatively small average loss changes, but its model-dependent missingness prevents unbiased cross-model ranking.

An exploratory Phase III descriptor screen found six numerical threshold passes, all for TabICLv2. The strongest result predicted single-estimator TabICL's monotone-spline loss gap with grouped held-out-dataset `R² = 0.662` and a 46.0% MAE improvement over a fold-mean baseline. Random-PWL disagreement was also predicted for both single and default TabICL with `R² ≈ 0.20`. Atom mass, skewness, kurtosis, and robust scale emerged as candidate correlates. However, dataset counts were small, multiple comparisons were not corrected, and the required permutation and split-stability controls were not run. Gate G2 therefore remains unresolved.

No synthetic S1–S6 study, mechanistic intervention, remedy ladder, RSPF pretraining, method freeze, confirmatory benchmark, or final ICLR readiness evaluation was completed. The defensible outcome is a promising TabICLv2-focused reliability and mechanism direction—not a validated remedy or finished ICLR submission.

## 1. Research question and claim boundary

For a context/query episode `(D_c, X_q)` and a featurewise bijection `h`, the primary comparison was:

```text
original:  p(y_q | D_c, X_q)
matched:   p(y_q | h(D_c), h(X_q))
```

The transformation is applied consistently to the whole task. Labels, feature identities, missingness masks, row membership, and train/test membership remain unchanged. This differs from context-only or query-only attacks, which intentionally create a representation mismatch.

The strongest target thesis was that modern TFMs can entangle coordinate representation with useful marginal-distribution metadata. The experiments were designed to falsify that thesis rather than guarantee a positive result.

Claims deliberately excluded as insufficiently novel include:

- monotone transforms can change predictions;
- quantile transforms can improve tabular learning;
- TFMs use preprocessing ensembles;
- transformed views can be ensembled;
- an input adapter can improve a frozen TFM; and
- permutation symmetry matters in tabular models.

The potentially novel boundary is the combination of matched whole-task bijections, posterior disagreement, the useful-versus-nuisance role of marginal shape, and a principled prior or representation intervention. The literature audit did not find prior work combining all four pieces, but novelty must be searched again before any paper claim.

## 2. Experimental protocol

### Four-way evaluation

Every selected transformation was evaluated in four conditions:

| Context/train | Query/test | Meaning |
|---|---|---|
| Original | Original | Clean baseline |
| Transformed | Transformed | Matched task isomorphism |
| Transformed | Original | Context-only mismatch |
| Original | Transformed | Query-only mismatch |

For refitted models, the original-context fit was shared by clean/query-only and the transformed-context fit was shared by matched/context-only. This prevents fitted-model randomness from contaminating within-context comparisons.

### Transformations

The implemented library includes:

- increasing and decreasing affine maps;
- centered signed powers;
- numerically safe asinh maps;
- random monotone piecewise-linear bijections;
- held-out monotone splines with linear tails;
- empirical-CDF and Gaussian-rank transforms;
- atomic/discrete spacing remaps;
- categorical bijections; and
- compositions of monotone maps.

Data-dependent transforms were fit on context/training rows only. Each run retained transform state and audits for inverse reconstruction, monotonicity, missingness, finite values, and equality-class preservation.

### Metrics

The audit retained both predictive quality and prediction disagreement:

- classification: loss, accuracy, Jensen–Shannon divergence, total variation, argmax flips, Brier score, and calibration;
- regression: normalized loss gap and normalized absolute prediction disagreement;
- matched effects were compared with identity-refit noise;
- uncertainty used paired dataset-level or hierarchical dataset/split bootstraps with 10,000 draws.

### Models

Evaluated models included:

- TabICLv2 single and default inference;
- official Mitra default;
- historical TabPFN-v2.5 single and default;
- XGBoost, CatBoost, LightGBM, and Random Forest;
- linear/logistic models;
- RealMLP and TabM.

TabPFN-3 was unavailable locally and required an authenticated client/API route. It was never silently replaced or relabeled; TabPFN-v2.5 is reported only as historical evidence.

## 3. Reproducibility and implementation

The program implemented:

- deterministic dataset and split selection;
- a frozen pilot panel and 20-task development suite;
- composable transformation APIs and serialized transform state;
- train-only dataset descriptors;
- four-way prediction and fit pairing;
- immutable compressed prediction bundles;
- per-run JSON metadata, SHA-256 checksums, checkpoint/package telemetry, timing, and GPU memory;
- an append-only `results/MANIFEST.jsonl`;
- resumable model/dataset/transform jobs;
- isolated Mitra and pytabkit workers; and
- Phase I, Phase II, and Phase III analysis pipelines.

Final verification after the exploratory partial-analysis changes: **73 tests passed**. All workers were stopped after completion of the requested wrap-up, and both GPUs were idle at 0 MiB allocated.

## 4. Phase I: completed kill test

### Coverage and integrity

- Config: `configs/audit/pilot.yaml`
- Config SHA-256: `5c75900a6a3bfb607a24a679b9f8a8545d4f7126cea391e0ef301b85e994720f`
- Prediction source digest: `595cc6ae397a9a551882104812ecaf6e41b0e49fc87833685b747c467d009227`
- Grid: 12 datasets × 7 models × 13 transform settings × 3 seeds
- Coverage: **3,276/3,276 complete**
- Failed, missing, or unavailable jobs: **0**

All retained prediction bundles passed checksum, finite-value, alignment, shared-fit, and transformation-integrity checks. The maximum inverse relative error was `3.51e-7`, below the audited `1e-6` float64 tolerance. A small number of affine ties involved adjacent one-ULP encodings of the same decimal values and had reconstruction error below `2.9e-15`; no material collision explained the findings.

### Main Phase I results

The table averages transform/seed effects within datasets and then bootstraps datasets. Classification contains eight datasets and regression four.

| Model | Task | Normalized loss gap [95% CI] | Excess disagreement [95% CI] | Mean mismatch disagreement |
|---|---|---:|---:|---:|
| TabICLv2 default | Classification | 0.003106 [0.000806, 0.006579] | 0.000752 JS [0.000363, 0.001350] | 0.111738 |
| TabICLv2 single | Classification | 0.022095 [0.002390, 0.057680] | 0.004292 JS [0.000827, 0.010477] | 0.113557 |
| Mitra default | Classification | -0.000102 [-0.000290, 0.000045] | 0.000045 JS [0.000004, 0.000099] | 0.138048 |
| TabPFN-v2.5 default | Classification | 0.003440 [0.000607, 0.007574] | 0.000928 JS [0.000366, 0.001699] | 0.125340 |
| XGBoost | Classification | 0.000051 [-0.000027, 0.000139] | 0.000065 JS [0.000021, 0.000115] | 0.164368 |
| CatBoost | Classification | -0.000071 [-0.000232, 0.000086] | 0.000067 JS [0.000030, 0.000108] | 0.109655 |
| TabICLv2 default | Regression | 0.000707 [0.000462, 0.000929] | 0.027238 [0.014053, 0.043840] | 1.57858 |
| TabICLv2 single | Regression | -0.000048 [-0.001108, 0.001012] | 0.032142 [0.017561, 0.050037] | 1.67258 |
| Mitra default | Regression | -0.000047 [-0.000152, 0.000046] | 0.002909 [0.000284, 0.005122] | 1.52167 |
| TabPFN-v2.5 default | Regression | 0.001825 [-0.000515, 0.004600] | 0.021974 [0.012711, 0.032066] | 1.87483 |
| XGBoost | Regression | 0.000143 [-0.000079, 0.000365] | 0.006422 [0.001810, 0.011996] | 1.11641 |
| CatBoost | Regression | 0.000546 [0.000024, 0.001069] | 0.015449 [0.004405, 0.026494] | 0.94859 |

### Interpretable classification effects

| Model | Excess total variation | Excess argmax flips | Matched-clean accuracy change |
|---|---:|---:|---:|
| TabICLv2 default | 1.026% [0.725%, 1.297%] | 0.804% [0.321%, 1.297%] | -0.210 pp [-0.353, -0.084] |
| TabICLv2 single | 2.054% [1.224%, 3.186%] | 1.751% [0.865%, 2.787%] | -0.887 pp [-1.942, -0.252] |
| Mitra default | 0.095% [0.016%, 0.192%] | 0.117% [0.013%, 0.260%] | +0.012 pp [-0.005, 0.030] |
| TabPFN-v2.5 default | 1.326% [0.705%, 2.105%] | 1.300% [0.394%, 2.352%] | -0.397 pp [-0.926, -0.041] |
| XGBoost | 0.135% [0.036%, 0.246%] | 0.139% [0.031%, 0.267%] | +0.005 pp [-0.004, 0.016] |
| CatBoost | 0.171% [0.075%, 0.275%] | 0.149% [0.038%, 0.279%] | +0.017 pp [-0.006, 0.045] |

### Controls and interpretation

- TabICLv2-default classification disagreement was about six to eight times the tree controls.
- Context/query mismatch was roughly 149× its matched classification JS and 58× its matched regression disagreement. The matched effect is therefore distinct from the much larger mismatch attack.
- Default TabICL inference reduced classification excess JS by about 82% and loss gap by about 86% versus single inference, but did not eliminate the effect.
- The effect persisted after excluding affine maps, restricting severity to at most 1, and excluding the dataset with an all-missing numerical feature.
- Nonlinear signed-power, PWL, and spline transformations were stronger than ordinary positive-affine unit changes.
- Mitra showed little excess sensitivity, creating a strong cross-family heterogeneity signal.

### Gate G1 decision

| Route | Decision | Reason |
|---|---|---|
| A: broad performance failure | Fail | Two current strong TFM families did not show systematic degradation. |
| B: posterior instability | Pass, scoped | TabICLv2 showed reproducible posterior changes, flips, and small accuracy loss. |
| C: mechanistic heterogeneity | Strong signal only | TabICLv2 and Mitra differed sharply, but no causal explanation was established. |

**Overall Gate G1: narrow pass.** This justified a larger audit and explanation work, but not method pretraining.

## 5. Phase II: partial development-suite audit

### Frozen design

The development suite was frozen before Phase II outcomes:

- 20 datasets: 8 regression, 7 binary, and 5 multiclass;
- 12 model configurations;
- 28 transformation settings;
- 2 independent context/query split seeds;
- 13,440 expected jobs.

Source digest: `02882b44384093b972708fe6edfa54a1340d8599988a5d35f9e851a042053b5e`  
Config digest: `cf0e75e34416d324e9e37dd5da1c62a3845bd5cfe0bb5bf8f8b7ef5682a6aa07`

### Final coverage

| Model | Complete | Latest failed | Missing from complete set |
|---|---:|---:|---:|
| CatBoost | 983 | 0 | 137 |
| LightGBM | 1,120 | 0 | 0 |
| Linear/logistic | 1,120 | 0 | 0 |
| Mitra default | 423 | 0 | 697 |
| Random Forest | 1,037 | 0 | 83 |
| RealMLP default | 474 | 5 | 646 |
| TabICLv2 default | 324 | 1 | 796 |
| TabICLv2 single | 408 | 1 | 712 |
| TabM default | 586 | 29 | 534 |
| TabPFN-v2.5 default | 308 | 0 | 812 |
| TabPFN-v2.5 single | 435 | 0 | 685 |
| XGBoost | 1,120 | 0 | 0 |
| **Total** | **8,338** | **36** | **5,102** |

The missing count is expected minus complete and therefore includes the 36 latest failed jobs. Failures were GPU out-of-memory events and isolated pytabkit subprocess crashes/segmentation faults: 29 TabM, 5 RealMLP, and one each for TabICL default and single.

Because coverage depended strongly on model and execution order, the stopped grid is not missing at random. A dated amendment authorized exploratory use of completed jobs while preserving the original confirmatory rule. The partial analyzer retained **8,240 identity-paired jobs** and excluded 98 completed transformed jobs lacking a completed identity baseline. No missing outcome was imputed.

### Selected partial Phase II results

The best-covered current-TFM subset was regression. Values below are observed-case within-model summaries, not unbiased cross-model rankings.

| Model | Datasets | Loss gap [95% CI] | Excess disagreement [95% CI] |
|---|---:|---:|---:|
| TabICLv2 single | 8 | 0.00175 [-0.00164, 0.00566] | 0.04241 [0.03095, 0.05416] |
| TabICLv2 default | 7 | 0.00215 [-0.00072, 0.00527] | 0.03330 [0.02322, 0.04336] |
| TabPFN-v2.5 single | 8 | 0.00438 [0.00022, 0.00860] | 0.05580 [0.03975, 0.07361] |
| TabPFN-v2.5 default | 6 | 0.00005 [-0.00325, 0.00222] | 0.02483 [0.01684, 0.03314] |
| Mitra default | 8 | -0.00040 [-0.00093, 0.00012] | 0.00319 [-0.00653, 0.01037] |
| Random Forest | 8 | -0.00019 [-0.00061, 0.00009] | 0.00537 [0.00387, 0.00678] |
| XGBoost | 8 | 0.00012 [-0.00051, 0.00071] | 0.01327 [0.00811, 0.01905] |
| CatBoost | 8 | 0.00021 [-0.00037, 0.00092] | 0.01441 [0.00704, 0.02316] |
| LightGBM | 8 | 0.00064 [-0.00124, 0.00225] | 0.04839 [0.02777, 0.07065] |

These results corroborate the Phase I reliability story: TabICL and historical TabPFN can change predictions under matched transformations even when average predictive loss changes little. They also reject an overly simple neural-versus-tree interpretation because LightGBM showed considerable regression disagreement despite a near-zero mean loss gap.

Six prespecified exploratory figures were produced:

1. matched versus mismatch curves;
2. model-family robustness;
3. disagreement versus loss gap;
4. dataset sensitivity map;
5. ensemble versus single estimator; and
6. tree versus neural controls.

### Phase II status

Phase II is **not complete**. The partial snapshot is marked `PARTIAL.json`, not `DONE.json`. It supports within-model exploratory interpretation but cannot provide the frozen complete-grid comparison or a confirmatory broad model-family conclusion.

## 6. Phase III: exploratory marginal-descriptor screen

### Protocol

Train-only numerical-feature descriptors included missingness, unique fraction, largest atom mass, binned entropy, zero mass, skewness, excess kurtosis, robust scale, tail heaviness, and spacing irregularity.

The screen used TabICLv2 single/default, TabPFN-v2.5 single/default, and Mitra across signed power, random PWL, monotone spline, asinh, and composition. Ridge and random-forest meta-models were evaluated with grouped dataset holdout so both splits of a dataset remained in the same fold. Each model/transform/target cell required at least five observed datasets and was compared with a training-fold mean baseline.

There were 50 cells and 100 model summaries. Six fits met both prospective numerical thresholds: `R² >= 0.10` and at least 10% lower MAE than the baseline.

| Model | Transform | Target | Meta-model | Datasets / rows | R² | MAE improvement |
|---|---|---|---|---:|---:|---:|
| TabICLv2 single | Monotone spline | Loss gap | Ridge | 8 / 15 | 0.662 | 46.0% |
| TabICLv2 default | Random PWL | Disagreement | Random forest | 7 / 11 | 0.201 | 20.5% |
| TabICLv2 single | Random PWL | Disagreement | Random forest | 8 / 15 | 0.200 | 22.5% |
| TabICLv2 single | Monotone spline | Disagreement | Random forest | 8 / 15 | 0.179 | 20.3% |
| TabICLv2 default | Monotone spline | Loss gap | Random forest | 7 / 11 | 0.119 | 14.8% |
| TabICLv2 default | Signed power | Disagreement | Random forest | 7 / 11 | 0.116 | 17.6% |

All six passes involved TabICLv2. Neither TabPFN-v2.5 nor Mitra met both thresholds in this screen.

Repeated random-forest signals emphasized largest atom mass, skewness, and excess kurtosis for disagreement. The strongest ridge model emphasized kurtosis, robust scale, and skewness. This suggests that distribution shape may help explain where TabICL is unstable, but correlated feature importance is not a causal mechanism.

### Limitations and Gate G2

The descriptor screen cannot pass Gate G2 because:

- each cell contains only 5–8 datasets and 10–16 dataset/split rows;
- Phase II missingness is model-dependent;
- 100 fitted summaries create multiple-comparison risk;
- target-permutation controls were not run;
- direction stability across the two splits was not established;
- current-TFM coverage is mostly regression; and
- no internal representation or causal-restoration experiment was performed.

**Gate G2 remains open.** Route D is promising but unresolved.

## 7. Theory result and interpretation

The prospective theory target was fixed before Phase II outcomes. It is a formal direction, not a completed theorem.

For a finite transformation group `G` acting on whole tasks and a task prior `P`, define the symmetrized prior:

```text
P_sym = (1 / |G|) Σ_g g#P
```

Left multiplication permutes the group, making `P_sym` invariant. Under appropriate measurable-space and likelihood assumptions, its Bayes posterior predictive can therefore be chosen invariant to the corresponding matched whole-task transformation.

The theory also identifies why blind rank canonicalization can be costly. If `S` is retained canonical task information, `M` is discarded marginal metadata, and `Y*` is the query label, then under optimal log loss:

```text
R_log(S) - R_log(S, M) = I(Y*; M | S) >= 0
```

Canonicalization is free only when marginal metadata contains no conditional predictive information beyond the retained context. This motivates separating a coordinate-stable channel from an explicit marginal-metadata channel rather than deleting marginal shape indiscriminately.

The empirical S2/S3/S4 experiments needed to validate this tradeoff were not implemented, so the theory must not be presented as experimentally confirmed.

## 8. Decision gates and final status

| Gate | Requirement | Status |
|---|---|---|
| G1: phenomenon | Clear matched-transform effect | **Narrow pass** for TabICLv2 posterior instability |
| G2: explanation | Descriptor, internal, or controlled-prior explanation | **Open**; descriptor evidence is exploratory |
| G3: remedy | Robustness improvement without material clean loss | Not evaluated |
| G4: current-model relevance | Two current strong TFMs or explanatory new model | Not passed |
| G5: benchmark relevance | Competitive frozen clean and robustness benchmark | Not evaluated |

The program therefore cannot claim:

- universal modern-TFM reparameterization failure;
- a causal marginal-shape mechanism;
- useful-versus-harmful marginal metadata under controlled priors;
- a successful RSPF or factorized representation remedy;
- preserved clean benchmark performance; or
- ICLR submission readiness.

## 9. What was not completed

- The remaining 5,102 Phase II jobs.
- Confirmatory Phase II analysis on a complete grid.
- Target-permutation and split-direction checks for Phase III.
- Synthetic task families S1–S6.
- TabICL embedding, attention, neighborhood, or causal-restoration studies.
- Method ladder M1–M7.
- RSPF matched-compute pretraining.
- `METHOD_FREEZE.md`.
- Frozen TabArena/BeyondArena confirmation.
- Gates G3–G5 and final ICLR readiness scoring.

These omissions are substantive. They are not replaced by implementation tests or partial-data analysis.

## 10. Overall conclusion

The research direction survived its first falsification gate but did not complete the full program.

The strongest defensible finding is:

> TabICLv2 exhibits reproducible posterior instability under fully matched, information-preserving monotone reparameterizations. Default inference reduces but does not remove the effect. Mitra is substantially more stable, suggesting meaningful family-level heterogeneity. Preliminary held-out descriptor models indicate that atom mass and distribution shape may predict TabICL sensitivity, but the explanation is not yet confirmatory or causal.

This is potentially paper-worthy as a focused reliability/mechanism result if a follow-up completes the TabICL descriptor controls and independently validates the explanation through controlled S2/S3/S4 tasks or causal internal restoration. The present evidence is insufficient for a remedy or benchmark paper.

## 11. Artifact index

### Primary reports

- `reports/PHASE1_KILL_TEST.md`
- `reports/REPARAM_AUDIT_PARTIAL.md`
- `reports/PHASE3_DESCRIPTOR_SCREEN.md`
- `reports/PARTIAL_PHASE_AMENDMENT.md`
- `reports/CLAIMS_EVIDENCE_MATRIX.md`
- `reports/NOVELTY_LEDGER.md`
- `reports/THEORY_TARGET.md`
- `reports/EXPERIMENT_LOG.md`

### Phase I outputs

Directory: `results/analysis/phase1/595cc6ae397a9a55/`

- `DONE.json`
- `coverage.json`
- `run_metrics.csv`
- `dataset_effects.csv`
- `model_summary.csv`
- `transform_summary.csv`
- `model_landscape.png`
- `matched_vs_mismatch.png`
- `severity_curves.png`

### Partial Phase II outputs

Directory: `results/analysis/phase2_partial/02882b44384093b9-n8338/`

- `PARTIAL.json`
- `run_metrics.csv`
- `model_summary.csv`
- `dataset_transform_effects.csv`
- `feature_descriptors.csv`
- `matched_vs_mismatch.png`
- `model_family_robustness.png`
- `disagreement_vs_loss.png`
- `dataset_sensitivity_map.png`
- `ensemble_vs_single.png`
- `tree_vs_neural.png`

### Exploratory Phase III outputs

Directory: `results/analysis/phase2_partial/02882b44384093b9-n8338/phase3_descriptors/`

- `PARTIAL.json`
- `descriptor_effects.csv`
- `cross_dataset_metrics.csv`
- `cross_dataset_predictions_and_importances.csv`

### Raw results and provenance

- Append-only manifest: `results/MANIFEST.jsonl`
- Frozen pilot config: `configs/audit/pilot.yaml`
- Frozen development config: `configs/audit/main.yaml`
- Raw immutable prediction paths are enumerated in the run-level CSV files and manifest.

## 12. Reproduction commands

Use the Python 3.10 environment at `/home/byunhanjoon/miniconda3/bin/python`; the system `python` is Python 2.7.

```bash
cd /home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-08
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python -m pytest -q
/home/byunhanjoon/miniconda3/bin/python scripts/analyze_phase2.py \
  --code-sha256 02882b44384093b972708fe6edfa54a1340d8599988a5d35f9e851a042053b5e \
  --allow-partial
/home/byunhanjoon/miniconda3/bin/python scripts/analyze_phase3.py \
  --phase2-dir results/analysis/phase2_partial/02882b44384093b9-n8338 \
  --allow-partial \
  --models tabicl_v2_single tabicl_v2_default tabpfn_v25_single tabpfn_v25_default mitra_default \
  --transforms signed_power random_monotone_pwl monotone_spline asinh composition
```

The partial analyzers deliberately emit `PARTIAL.json`; they do not certify completion or create an authoritative Phase II `DONE.json`.
