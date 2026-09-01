# Guarded Basis Control — Metric Audit

## Executive Verdict
AUDIT-PASS-WITH-METRIC-CAVEAT

## One-Paragraph Summary

All 3,744 reported validation/test rows (1,872 test-level dataset × model × method × seed cells) were reconstructed from saved predictions, and every raw loss, method loss, training-only trivial loss, disagreement reduction, and per-seed C matches the stored value within the required tolerance. There are 0 per-seed sign violations and the largest loss mismatch is 0.000e+00. The apparent OnlineNewsPopularity/CatBoost contradiction is aggregation-explained: the report independently takes medians of seed losses while its C is the median of per-seed ratios. However, 4 method cells (0.21%) use the 1e-8 denominator because Raw does not beat the trivial predictor, so C is pathological in those cells. The corrected ranking and `FINAL-METHOD-SIGNAL` verdict are unchanged, and the RBF-k16 guarded result is unaffected.

## 1. Files and Code Audited

The complete path/hash inventory contains 7,741 exact source files in `results/audit/artifact_inventory.csv`. Each row records the absolute path, SHA256, Git blob SHA1, experiment protocol-freeze commit `0c456660ae9a87aab7932b569e1954b0ee1d25fe`, and audit-time repository commit `f7c9357e48b1ca8c1ecd9c320bb17fffd0893472`. Inputs include `results.md`, all requested prospective CSVs, all 288 atomic unit JSON files, the exact saved prediction bundles used for loss/control reconstruction, the locked panel/protocol/finalists, final provenance, the target/split loader, metric implementation, aggregation code, ranking code, and verdict code.

## 2. Exact Metric Definitions

Classification uses unweighted multiclass/binary log loss. Probabilities are clipped elementwise to `[1e-8, 1]`, renormalized by row, and indexed by the training-only `LabelEncoder` order. Regression uses RMSE. The trivial classifier is the training class-frequency vector; the trivial regressor is the training target mean. No validation/test labels and no sample weights construct the trivial predictor. Per seed, `C = (L_method - L_raw) / max(L_trivial - L_raw, 1e-8)`. Sensitivity metrics are `C_stable = (L_method - L_raw) / max(L_trivial, 1e-8)`, absolute `delta_L`, and regression `delta_sigma = delta_RMSE / std(y_test)`.

## 3. Raw Prediction -> Loss Verification

| rows checked | mismatches | maximum mismatch |
| --- | --- | --- |
| 3744 | 0 | 0.000e+00 |

The test-only required table has 1,872 rows. Class-column counts match the training-only encoding in all 1,872 classification checks. Probability normalization/clipping diagnostics are in `classification_probability_checks.csv`; the loss routine intentionally reproduces the stored clipping epsilon and renormalization. All raw/method orbit disagreements and disagreement reductions also match; their maximum mismatch is 0.000e+00.

## 4. Trivial Predictor Verification

All 288 unique scope × dataset × model × seed test baselines match their stored values; maximum absolute mismatch is 0.000e+00. `target_split_inventory.csv` and the target snapshots record training/validation/test indices and target hashes. The audited loader fits class encoders, class frequencies, regression means, imputers, and subsampling from training rows only.

## 5. Per-Seed C Verification

| cells | exact matches | mismatches | sign violations |
| --- | --- | --- | --- |
| 1872 | 1872 | 0 | 0 |

The maximum absolute C difference is 0.000e+00. The mandatory sign invariant holds at every seed whose `L_trivial > L_raw`.

## 6. Aggregation Semantics

Seed losses are never pooled at the prediction-row level. For each dataset/model/method, the original code independently takes the median across three seeds of raw loss, method loss, disagreement reduction, selected alpha, and per-seed C. Thus displayed raw and method losses can come from different seeds, while displayed C is `C_B`, the median of the three already-normalized per-seed values. Global medians/p90/p95/max are then computed over the 60 general dataset/model units or 36 embedding dataset/model units. W/T/L and predictive ranks use the independently median-aggregated task losses. `aggregation_diagnostics.csv` records mean C (`C_A`), median C (`C_B`), ratio of mean losses (`C_C`), and ratio of median losses (`C_D`) for every unit. Every existing reported unit C matches `C_B`; none matches a different hidden aggregation systematically.

## 7. OnlineNewsPopularity / CatBoost Forensic Example

| record_type | seed | L_raw | L_method | L_trivial | numerator | raw_headroom | C_recomputed | reported_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregated_report | — | 5254.3235 | 5250.5999 | 5264.7516 | -3.7236316 | 10.428101 | 0.24694695 | 0.24694695 |
| seed | 0 | 5254.3235 | 5256.8987 | 5264.7516 | 2.5751878 | 10.428101 | 0.24694695 | 0.24694695 |
| seed | 1 | 5244.7014 | 5250.5999 | 5264.7516 | 5.8984599 | 20.050193 | 0.2941847 | 0.2941847 |
| seed | 2 | 5258.7515 | 5248.3757 | 5264.7516 | -10.375743 | 6.0001278 | -1.7292536 | -1.7292536 |

The aggregated report displays raw=5254.32351, method=5250.599878, and C=0.2469469522. The lower displayed method loss and positive C are **AGGREGATION-EXPLAINED**, not a per-seed sign bug: raw loss, method loss, and C were independently median-aggregated, and their middle values come from different seeds.

## 8. Other Suspicious Cells

There are 44 aggregate rows where the sign of the independently displayed median-loss difference differs from the median per-seed C. All have correct per-seed mathematics and are classified `AGGREGATION-EXPLAINED`.

| dataset | model | method | reported_raw_task_error | reported_method_task_error | reported_unit_C | diagnosis |
| --- | --- | --- | --- | --- | --- | --- |
| Brazilian_houses | tabpfn_2_6 | SafeGram-t01 | 746.81604 | 743.69919 | 0.0029107511 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabm_d | GuardedGram-G2-g0p0-t01 | 0.17521592 | 0.20322597 | -0.0050421709 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabm_d | PureGram | 0.17521592 | 0.21895457 | -0.003070266 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabm_d | Raw+Gram@0.5 | 0.17521592 | 0.19073085 | -0.0047614047 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabm_d | Raw+Gram@0.75 | 0.17521592 | 0.20322597 | -0.0050421709 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabm_d | SafeGram-t01 | 0.17521592 | 0.19073085 | -0.003070266 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | BlockGuard-Greedy-t01 | 0.10969186 | 0.11992281 | -0.0011232214 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.10969186 | 0.11327712 | -0.0043674186 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | PureGram | 0.10969186 | 0.11992281 | -0.0011232214 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | Raw+Gram@0.5 | 0.10969186 | 0.11119234 | -0.0043123897 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | Raw+Gram@0.75 | 0.10969186 | 0.11327712 | -0.0043674186 | AGGREGATION-EXPLAINED |
| JapaneseVowels | tabpfn_2_6 | SafeGram-t01 | 0.10969186 | 0.11236707 | -0.0011232214 | AGGREGATION-EXPLAINED |
| OnlineNewsPopularity | catboost | GuardedGram-G2-g0p0-t01 | 5254.3235 | 5250.5999 | 0.24694695 | AGGREGATION-EXPLAINED |
| OnlineNewsPopularity | catboost | PureGram | 5254.3235 | 5253.8532 | 0.45266523 | AGGREGATION-EXPLAINED |
| OnlineNewsPopularity | catboost | Raw+Gram@0.5 | 5254.3235 | 5250.85 | 0.1028985 | AGGREGATION-EXPLAINED |
| OnlineNewsPopularity | catboost | Raw+Gram@0.75 | 5254.3235 | 5250.5999 | 0.24694695 | AGGREGATION-EXPLAINED |
| SoilKsatDB | tabm_d | Raw+Gram@0.75 | 935.80673 | 934.38222 | 0.0085103002 | AGGREGATION-EXPLAINED |
| ada_prior | catboost | GuardedGram-G2-g0p0-t01 | 0.37837531 | 0.37908686 | -0.0071147652 | AGGREGATION-EXPLAINED |
| ada_prior | catboost | Raw+Gram@0.5 | 0.37837531 | 0.37908686 | -0.0099125466 | AGGREGATION-EXPLAINED |
| ada_prior | catboost | Raw+Gram@0.75 | 0.37837531 | 0.38127404 | -0.0071147652 | AGGREGATION-EXPLAINED |
| ada_prior | resnet_tabular | Raw+Gram@0.75 RBF-k16 embedding | 0.37473072 | 0.37330769 | 0.016588758 | AGGREGATION-EXPLAINED |
| eye_movements | resnet_tabular | Gram-after-RBF-k16 embedding | 1.0066121 | 0.99826151 | 0.042126401 | AGGREGATION-EXPLAINED |
| eye_movements | tabm_d | Raw+Gram@0.5 | 0.94616423 | 0.95455023 | -0.090122952 | AGGREGATION-EXPLAINED |
| eye_movements | tabm_d | Raw+Gram@0.75 | 0.94616423 | 0.97720165 | -0.012287315 | AGGREGATION-EXPLAINED |
| eye_movements | tabpfn_2_6 | PureGram | 0.89475997 | 0.89322375 | 0.006839641 | AGGREGATION-EXPLAINED |
| mfeat-fourier | controlled_mlp | BlockGuard-Greedy-t01 | 0.62476181 | 0.62578119 | -0.0061915856 | AGGREGATION-EXPLAINED |
| mfeat-fourier | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.62476181 | 0.61684543 | 0.0016003666 | AGGREGATION-EXPLAINED |
| mfeat-fourier | controlled_mlp | Raw+Gram@0.75 | 0.62476181 | 0.61684543 | 0.0016003666 | AGGREGATION-EXPLAINED |
| mfeat-fourier | tabm_d | PureGram | 0.50367737 | 0.50250744 | 0.0070202861 | AGGREGATION-EXPLAINED |
| mozilla4 | tabpfn_2_6 | SafeRankGram-t01 | 0.17979111 | 0.18184019 | -0.01197528 | AGGREGATION-EXPLAINED |

The full seed-level evidence for OnlineNewsPopularity/controlled_mlp/TabICLv2 and all models on Brazilian_houses, SoilKsatDB, and 2dplanes is in `forensic_examples.csv`.

## 9. Denominator Pathology

| headroom_condition | cells | percentage | base_dataset_model_seed_cells | datasets_affected | models_affected |
| --- | --- | --- | --- | --- | --- |
| headroom <= 0 | 4 | 0.21367521 | 1 | eye_movements | tabm_d |
| headroom < 1e-8 | 4 | 0.21367521 | 1 | eye_movements | tabm_d |
| headroom < 1e-6 | 4 | 0.21367521 | 1 | eye_movements | tabm_d |
| headroom < 1e-4 * L_trivial | 4 | 0.21367521 | 1 | eye_movements | tabm_d |

The following 1 unique scope/dataset/model/seed base cells use the epsilon denominator (expanded to 4 method cells):

| scope | dataset | model | seed |
| --- | --- | --- | --- |
| embedding | eye_movements | tabm_d | 2 |

The largest clipped values are retained in the primary audit and listed in `clipped_denominator_cells.csv`. Removing them only for sensitivity gives paper verdict `FINAL-METHOD-SIGNAL`; the primary corrected verdict remains `FINAL-METHOD-SIGNAL`. The RBF-k16 guarded finalist contains one clipped per-seed improvement (`eye_movements`/TabM-D/seed 2), but removing it leaves the finalist's median, p95, and maximum exactly unchanged.

## 10. Corrected General Prospective Summary

| method | median_disagreement_reduction | median_C | p90_C | p95_C | max_C | fraction_C_gt_0p01 | fraction_C_gt_0p05 | wins | ties | losses | fallback_rate | mean_predictive_rank | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BlockGuard-Greedy-t01 | 0.72190261 | 0 | 0.033491877 | 0.047566983 | 2.1635454 | 0.18333333 | 0.05 | 27 | 6 | 27 | 0.26666667 | 5.2583333 | -3.6178891 |
| GuardedGram-G2-g0p0-t01 | 0.75 | -0.0025697509 | 0.010668569 | 0.01750183 | 0.24694695 | 0.11666667 | 0.016666667 | 33 | 3 | 24 | 0.13333333 | 3.725 | 0.28360061 |
| PureGram | 1 | 0.0069299635 | 0.12266681 | 0.18083417 | 4.5036341 | 0.4 | 0.21666667 | 24 | 0 | 36 | 0 | 6.2416667 | -8.4405606 |
| Raw | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 0 | 1 | 4.95 | 0 |
| Raw+Gram@0.5 | 0.5 | -0.0033156375 | 0.009201307 | 0.041348048 | 1.0075553 | 0.1 | 0.05 | 36 | 0 | 24 | 0 | 3.6833333 | -1.5591548 |
| Raw+Gram@0.75 | 0.75 | -0.0022726814 | 0.027928198 | 0.087223707 | 2.4582636 | 0.23333333 | 0.1 | 30 | 0 | 30 | 0 | 4.4583333 | -4.3481983 |
| SafeGram-t01 | 0.5 | -0.00048338847 | 0.0026454422 | 0.0077109005 | 0.03538086 | 0.033333333 | 0 | 32 | 15 | 13 | 0.33333333 | 4.2 | 0.5 |
| SafeRankGram-t01 | 0.5 | -0.00050960363 | 0.0012571316 | 0.0024879489 | 0.058273829 | 0.033333333 | 0.016666667 | 35 | 16 | 9 | 0.4 | 3.4833333 | 0.48345234 |

## 11. Corrected Embedding Prospective Summary

| method | median_disagreement_reduction | median_C | p90_C | p95_C | max_C | fraction_C_gt_0p01 | fraction_C_gt_0p05 | wins | ties | losses | fallback_rate | mean_predictive_rank | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gram-after-RBF-k16 embedding | 1 | -0.013801384 | 0.092203102 | 0.10872586 | 3.3319076 | 0.36111111 | 0.27777778 | 22 | 0 | 14 | 0 | 3 | -5.8599928 |
| GuardedGram-G2-after-RBF-k16 | 0.75 | -0.030651315 | 0.0017955199 | 0.0026466789 | 0.0064371084 | 0 | 0 | 29 | 1 | 6 | 0.13888889 | 1.7638889 | 0.7 |
| Raw RBF-k16 embedding | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 | 1 | 3.1527778 | 0 |
| Raw+Gram@0.75 RBF-k16 embedding | 0.75 | -0.036620638 | 0.037444195 | 0.047367132 | 1.6850066 | 0.22222222 | 0.055555556 | 26 | 0 | 10 | 0 | 2.0833333 | -2.6821145 |

## 12. Stable-Metric Sensitivity Analysis

| scope | method | median_C_stable | p95_C_stable | max_C_stable |
| --- | --- | --- | --- | --- |
| general | BlockGuard-Greedy-t01 | 0 | 0.020325157 | 0.073753096 |
| general | GuardedGram-G2-g0p0-t01 | -0.0015429733 | 0.0062409835 | 0.0077766852 |
| general | PureGram | 0.0028057865 | 0.041998932 | 0.10236521 |
| general | Raw+Gram@0.75 | -0.0012476969 | 0.018973966 | 0.062833058 |
| general | SafeGram-t01 | -0.00033673409 | 0.0055688712 | 0.010580758 |
| general | SafeRankGram-t01 | -0.0003694717 | 0.0018456646 | 0.020811527 |
| embedding | GuardedGram-G2-after-RBF-k16 | -0.01487957 | 0.0017642761 | 0.0035632697 |

`C_stable` and `delta_L` are sensitivity diagnostics, not silently substituted acceptance metrics. The complete per-seed stable values and regression `delta_sigma`/classification `delta_logloss` are in `per_seed_C.csv`; stable and denominator-clipped-removed unit/summary tables are separate files. Sensitivity paper scores and all six corresponding rankings are recorded in `stable_metric_rankings.csv` and `unclipped_metric_rankings.csv`; the original C thresholds are not reinterpreted as calibrated stable-metric thresholds.

## 13. Ranking Before vs After Audit

| ranking | method | old_rank | new_rank | stable_metric_rank | changed |
| --- | --- | --- | --- | --- | --- |
| A — Paper Safety | GuardedGram-G2-after-RBF-k16 | 1 | 1 | 1 | False |
| A — Paper Safety | GuardedGram-G2-g0p0-t01 | — | — | 2 | False |
| A — Paper Safety | Raw | 4 | 4 | 7 | False |
| A — Paper Safety | Raw+Gram@0.5 | — | — | 4 | False |
| A — Paper Safety | Raw+Gram@0.75 | — | — | 3 | False |
| A — Paper Safety | SafeGram-t01 | 3 | 3 | 6 | False |
| A — Paper Safety | SafeRankGram-t01 | 2 | 2 | 5 | False |
| F — Overall Paper Candidate | BlockGuard-Greedy-t01 | 7 | 7 | 5 | False |
| F — Overall Paper Candidate | GuardedGram-G2-after-RBF-k16 | 1 | 1 | 2 | False |
| F — Overall Paper Candidate | GuardedGram-G2-g0p0-t01 | 4 | 4 | 3 | False |
| F — Overall Paper Candidate | PureGram | 9 | 9 | 1 | False |
| F — Overall Paper Candidate | Raw | 5 | 5 | 9 | False |
| F — Overall Paper Candidate | Raw+Gram@0.5 | 6 | 6 | 8 | False |
| F — Overall Paper Candidate | Raw+Gram@0.75 | 8 | 8 | 4 | False |
| F — Overall Paper Candidate | SafeGram-t01 | 2 | 2 | 7 | False |
| F — Overall Paper Candidate | SafeRankGram-t01 | 3 | 3 | 6 | False |

All primary old/new ranks are identical. Stable-metric ranks are shown only as sensitivity because the original C thresholds were not calibrated for `C_stable`.

## 14. Does GuardedGram-G2-after-RBF-k16 Still Hold?

YES

Corrected exact metrics are median control=0.75, median C=-0.03065131464, p90=0.001795519879, p95=0.002646678853, max=0.006437108393, W/T/L=29/1/6, and fraction `C > .01`=0. These reproduce the reported approximately 75% control, p95 0.0026, and max 0.0064.

## 15. Does FINAL-METHOD-SIGNAL Still Hold?

YES

Applying the original prespecified aggregation and verdict code to recomputed losses returns `FINAL-METHOD-SIGNAL`. The result also remains `FINAL-METHOD-SIGNAL` in the clipped-cell-removal sensitivity analysis.

## 16. Recommended Metric for the Paper

Keep canonical per-seed C as the primary protocol metric only when reporting its denominator context. Report `delta_L` (and regression `delta_sigma`) plus `C_stable` as secondary sensitivities. Explicitly mark every case with `L_trivial - L_raw <= 0`, report the clipped-cell count and affected datasets/models, and never interpret epsilon-amplified magnitudes as comparable effect sizes. To avoid apparent sign contradictions, paper tables should display seed-level rows or compute displayed loss deltas from the same seed-level C aggregation; if independent medians are retained, label that fact beside the table.

## 17. Bugs Fixed

No Day-9 experiment or method code was modified. No per-seed loss/C bug was found. The apparent sign contradictions are a presentation-level aggregation artifact, and the denominator caveat is a metric-domain pathology rather than a reproduction failure. This standalone Day-10 audit adds diagnostics only.

## 18. Files Produced

- `metric_audit_results.md`
- `results/audit/audit_manifest.json`
- `results/audit/artifact_inventory.csv`
- `results/audit/target_split_inventory.csv` and `results/audit/targets/*.npz`
- `results/audit/per_seed_losses.csv`, `per_seed_trivial.csv`, `per_seed_C.csv`, and `sign_violations.csv`
- `results/audit/aggregation_diagnostics.csv`, `aggregation_sign_mismatches.csv`, and `forensic_examples.csv`
- `results/audit/denominator_pathology.csv`, its summary, and clipped-cell listing
- corrected, stable-metric, and clipped-cell-removed unit/summary CSVs
- corrected/stable rankings, before/after ranks, and summary-reproduction diagnostics
