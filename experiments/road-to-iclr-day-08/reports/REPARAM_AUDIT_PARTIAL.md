# Exploratory partial reparameterization audit

Date: 2026-08-31

Status: **exploratory partial; not a completed Phase II confirmatory report**

## Question

What can be learned from the checksum-valid Phase II jobs completed before the run was stopped, while preserving the distinction between observed evidence and the frozen complete-grid analysis?

## Protocol and coverage

The analysis uses source digest `02882b44384093b972708fe6edfa54a1340d8599988a5d35f9e851a042053b5e` and config digest `cf0e75e34416d324e9e37dd5da1c62a3845bd5cfe0bb5bf8f8b7ef5682a6aa07`.

- Frozen grid: 13,440 jobs.
- Checksum-valid complete jobs: 8,338 (62.0%).
- Identity-paired jobs used for inference: 8,240.
- Completed transformed jobs excluded for lacking a completed identity baseline: 98.
- Latest failed jobs: 36; unavailable jobs: 0.
- Coverage is model-dependent. All summaries below are observed-case, within-model results and are not unbiased cross-model rankings.

The four-way task protocol, transform audits, stored predictions, and hierarchical dataset/split bootstrap are unchanged. No failed or missing cell was imputed. The dated interpretation change is recorded in `reports/PARTIAL_PHASE_AMENDMENT.md`.

## Selected observed results

The most complete current-TFM subset is regression. Values are mean matched normalized loss gap and excess normalized prediction disagreement, with dataset-level 95% bootstrap intervals.

| Model | Datasets | Loss gap [95% CI] | Excess disagreement [95% CI] |
|---|---:|---:|---:|
| TabICLv2 single | 8 | 0.00175 [-0.00164, 0.00566] | 0.04241 [0.03095, 0.05416] |
| TabICLv2 default | 7 | 0.00215 [-0.00072, 0.00527] | 0.03330 [0.02322, 0.04336] |
| TabPFN-v2.5 single | 8 | 0.00438 [0.00022, 0.00860] | 0.05580 [0.03975, 0.07361] |
| TabPFN-v2.5 default | 6 | 0.00005 [-0.00325, 0.00222] | 0.02483 [0.01684, 0.03314] |
| Mitra default | 8 | -0.00040 [-0.00093, 0.00012] | 0.00319 [-0.00653, 0.01037] |
| Random forest | 8 | -0.00019 [-0.00061, 0.00009] | 0.00537 [0.00387, 0.00678] |
| XGBoost | 8 | 0.00012 [-0.00051, 0.00071] | 0.01327 [0.00811, 0.01905] |
| CatBoost | 8 | 0.00021 [-0.00037, 0.00092] | 0.01441 [0.00704, 0.02316] |
| LightGBM | 8 | 0.00064 [-0.00124, 0.00225] | 0.04839 [0.02777, 0.07065] |

The partial audit corroborates the Phase I reliability story: TabICL and historical TabPFN can change predictions under matched transformations even where average predictive loss moves little. It also prevents an overly simple “neural versus tree” claim: LightGBM has substantial regression disagreement in this snapshot despite a near-zero loss gap.

## Figures and raw outputs

The six prespecified figures were generated under `results/analysis/phase2_partial/02882b44384093b9-n8338/`:

- `matched_vs_mismatch.png`
- `model_family_robustness.png`
- `disagreement_vs_loss.png`
- `dataset_sensitivity_map.png`
- `ensemble_vs_single.png`
- `tree_vs_neural.png`

The same directory contains `run_metrics.csv`, `model_summary.csv`, `dataset_transform_effects.csv`, `feature_descriptors.csv`, and `PARTIAL.json`, including the full missing-job list.

## Interpretation and decision

The evidence supports continuing explanation work, particularly for TabICLv2. It does not complete Phase II, establish a universal current-TFM effect, or authorize a confirmatory Gate G2 pass. Phase III may proceed as exploratory screening; expensive remedy/pretraining claims remain deferred.
