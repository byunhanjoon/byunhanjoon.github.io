# Safe Basis Control — Tail-Robust Method Round

## Executive Verdict
STRONG-METHOD-TAIL-UNSOLVED

## One-Paragraph Summary

On the untouched panel, SafeGram-t01 achieved 37.5% median control with median/p95/max C=0.0000/0.0159/0.0276 and a 35.0% raw fallback rate; SafeRankGram reached 50.0% control with p95/max C=0.0051/0.0096. Pure Gram remained exactly invariant but had p95/max C=0.1377/0.1904; fixed alpha=.75 reached 75.0% control with p95/max C=0.0498/0.1278. Rank adaptation cut coordinate count while preserving the training blocks (worst diagnostic reconstruction error below 1e-4), but did not itself solve tail risk. Every worst prior Gram cell was classified Type C: altered generalization despite reconstructible coordinates. PLE/RBF embeddings showed material basis sensitivity across MLP, TabM-D, and ResNet backbones, and SafeGram-after-embedding retained 56.3% median control with p95 C=0.0006. The descriptor gate was discarded after leave-one-dataset-out validation. No final paper method is selected automatically.

## Frozen Protocol

- git commit: `0c456660ae9a87aab7932b569e1954b0ee1d25fe`
- hardware: `NVIDIA H100 NVL`; protocol records two NVIDIA H100 NVL GPUs
- versions: Python 3.10.16; numpy=1.26.4, pandas=2.3.3, scipy=1.11.4, scikit-learn=1.4.2, torch=2.7.0, tabpfn=8.5.0, tabicl=2.0.3, catboost=1.2.10, pytabkit=1.7.3
- seeds: `[0, 1]`; split seed `20260901`
- development datasets: `steel-plates-fault, wilt, eeg-eye-state, satimage, space-ga, california_housing, house_16H, phoneme`
- NEW prospective datasets: `ozone-level-8hr, wall-robot-navigation, cardiotocography, first-order-theorem-proving, sylvine, delta_elevators, quake, visualizing_soil, debutanizer, CPMP-2015-regression`
- TAIL_FINALISTS SHA256: `b7b075362f5c7295fdb55f6d6140e5cf27499fcc1dbac0363f16e7aca9c14d29`

## 1. Previous Result Being Addressed

The previous round found 100% median orthogonal disagreement reduction for GramAnchor at about +0.90% median task cost and 75% reduction with approximately -0.10% task change for Raw+GramAnchor@0.75. Those medians concealed catastrophic dataset/model tails, especially Steel Plates. Relative percentage loss is unstable near zero raw loss, so this round uses normalized excess risk C against the training-prior/mean trivial predictor and judges both median and tail safety.

## 2. SafeGram Development Results

| method | median_alpha | reduction | median_C | p95_C | max_C | raw fallback rate |
| --- | --- | --- | --- | --- | --- | --- |
| SafeGram-t0 | 0 | 0 | 0 | 3.365e-05 | 0.003117 | 0.825 |
| SafeGram-t005 | 0.125 | 0.125 | 0 | 0.00372 | 0.005988 | 0.55 |
| SafeGram-t01 | 0.25 | 0.25 | 0 | 0.004352 | 0.007605 | 0.475 |
| SafeGram-t02 | 0.4375 | 0.4375 | -0.00138 | 0.00808 | 0.01811 | 0.325 |

## 3. Gate Ablations

| method | median_alpha | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate |
| --- | --- | --- | --- | --- | --- | --- |
| G1-point-t01 | 0.875 | 0.875 | 0 | 0.02214 | 0.03694 | 0.225 |
| G2-oneSE-t01 | 0.375 | 0.375 | -0.0005245 | 0.00761 | 0.01202 | 0.375 |
| G3-validation-min | 0.3125 | 0.3125 | -0.0008555 | 0.005785 | 0.02292 | 0.45 |
| G4-constrained-0.001 | 0.375 | 0.375 | -0.001841 | 0.005785 | 0.02292 | 0.375 |
| G4-constrained-0.01 | 0.625 | 0.625 | -0.0008155 | 0.01397 | 0.03118 | 0.275 |
| G4-constrained-0.05 | 1 | 1 | 0 | 0.03756 | 0.0453 | 0.125 |
| SafeGram-t0 | 0 | 0 | 0 | 3.365e-05 | 0.003117 | 0.825 |
| SafeGram-t005 | 0.125 | 0.125 | 0 | 0.00372 | 0.005988 | 0.55 |
| SafeGram-t01 | 0.25 | 0.25 | 0 | 0.004352 | 0.007605 | 0.475 |
| SafeGram-t02 | 0.4375 | 0.4375 | -0.00138 | 0.00808 | 0.01811 | 0.325 |

The optional descriptor gate was **DISCARDED_AFTER_DEVELOPMENT_CV**: LODO descriptor gate did not improve control by >=10 points while satisfying p95<=0.05 and max<=0.20.

## 4. RankAdaptiveGram

| relative_threshold | anchor_rule | normalization | median_total_coordinate_dimension | median_C | p95_C | maximum_C | maximum_reconstruction_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | rank | N1_anchor_norm | 90.5 | -0.1564 | 0.462 | 0.5522 | 8.258e-05 |
| 1e-06 | rank | N1_anchor_norm | 90.5 | -0.1564 | 0.4989 | 0.5956 | 3.384e-12 |
| 0.001 | rank | N1_anchor_norm | 89.5 | -0.1517 | 0.5099 | 0.6069 | 0.0008849 |
| 0.0001 | double_rank_capped_16 | N1_anchor_norm | 181 | -0.1211 | 0.6033 | 0.7159 | 2.874e-12 |
| 1e-06 | double_rank_capped_16 | N1_anchor_norm | 181 | -0.1211 | 0.6311 | 0.7486 | 1.827e-12 |
| 0.0001 | fixed_16 | N1_anchor_norm | 184 | -0.1172 | 0.4964 | 0.5888 | 1.827e-12 |
| 0.001 | fixed_16 | N1_anchor_norm | 184 | -0.1172 | 0.4964 | 0.5888 | 1.827e-12 |
| 1e-06 | fixed_16 | N1_anchor_norm | 184 | -0.1172 | 0.4964 | 0.5888 | 1.827e-12 |
| 0.001 | double_rank_capped_16 | N1_anchor_norm | 179 | -0.1143 | 0.4807 | 0.5692 | 2.874e-12 |
| 0.001 | rank_plus_one | N1_anchor_norm | 101 | -0.1097 | 0.492 | 0.5854 | 3.384e-12 |
| 0.0001 | rank_plus_one | N1_anchor_norm | 102 | -0.1018 | 0.4905 | 0.5807 | 3.384e-12 |
| 1e-06 | rank_plus_one | N1_anchor_norm | 102 | -0.1018 | 0.4897 | 0.5799 | 2.728e-12 |
| 0.0001 | rank_plus_one | N0_raw_inner_product | 102 | -0.1006 | 0.1155 | 0.1325 | 4.981e-12 |
| 0.0001 | rank_plus_one | N1_anchor_norm | 102 | -0.0983 | 0.01924 | 0.02055 | 3.384e-12 |
| 0.0001 | rank_plus_one | N2_cosine | 102 | -0.08986 | 0.005299 | 0.009214 | 0.1493 |
| 0.0001 | rank_plus_one | N3_block_rms | 102 | -0.01142 | 0.03117 | 0.03665 | 2.745e-12 |

Selected on development validation only: `epsilon_r=1e-4`, `m_j=r_j`, N1 anchor normalization, coordinate standardization.

## 5. Normalization Ablations

| normalization | median_total_coordinate_dimension | median_C | p95_C | maximum_C | maximum_reconstruction_error |
| --- | --- | --- | --- | --- | --- |
| N0_raw_inner_product | 102 | -0.1006 | 0.1155 | 0.1325 | 4.981e-12 |
| N1_anchor_norm | 102 | -0.0983 | 0.01924 | 0.02055 | 3.384e-12 |
| N2_cosine | 102 | -0.08986 | 0.005299 | 0.009214 | 0.1493 |
| N3_block_rms | 102 | -0.01142 | 0.03117 | 0.03665 | 2.745e-12 |

## 6. Catastrophic Failure Diagnosis

| dataset | model | seed | method | train_error | validation_error | test_error | disagreement | reconstruction_error | empirical_rank | feature_dimension | anchor_condition | failure_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steel-plates-fault | controlled_mlp | 0 | Raw | 2.629e-05 | 0.0009343 | 0.0002624 | 0.01453 | 0 | NA | 211 | NA | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 0 | GramAnchor | 0.003151 | 0.3803 | 0.3808 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 0 | RankAdaptiveGram | 0.03846 | 0.3567 | 0.2661 | 0 | 8.258e-05 | 8 | 202 | 2.654e+06 | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 0 | SafeGram-t01 | 2.629e-05 | 0.0009343 | 0.0002624 | 0.01453 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 1 | Raw | 2.711e-06 | 0.0003822 | 0.003071 | 0.02738 | 0 | NA | 211 | NA | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 1 | GramAnchor | 0.01574 | 0.422 | 0.4378 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 1 | RankAdaptiveGram | 0.07515 | 0.4019 | 0.3238 | 0 | 8.258e-05 | 8 | 202 | 2.654e+06 | Type C — altered generalization/inductive bias |
| steel-plates-fault | controlled_mlp | 1 | SafeGram-t01 | 2.711e-06 | 0.0003822 | 0.003071 | 0.02738 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 1 | Raw | 0.005068 | 0.04958 | 0.04264 | 0.007253 | 0 | NA | 211 | NA | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 1 | GramAnchor | 0.007528 | 0.05239 | 0.0536 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 1 | RankAdaptiveGram | 0.03704 | 0.07472 | 0.07609 | 0 | 8.258e-05 | 8 | 202 | 2.654e+06 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 1 | SafeGram-t01 | 0.005068 | 0.04958 | 0.04264 | 0.007253 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 0 | Raw | 0.06666 | 0.1073 | 0.1094 | 0.009149 | 0 | NA | 211 | NA | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 0 | GramAnchor | 0.006289 | 0.04684 | 0.04972 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 0 | RankAdaptiveGram | 0.0008792 | 0.02152 | 0.01926 | 0 | 8.258e-05 | 8 | 202 | 2.654e+06 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabm_d | 0 | SafeGram-t01 | 0.06666 | 0.1073 | 0.1094 | 0.009149 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabicl_v2 | 1 | Raw | 1.182e-06 | 0.0007318 | 0.0002686 | 0.005358 | 0 | NA | 211 | NA | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabicl_v2 | 1 | GramAnchor | 3.281e-06 | 0.002314 | 0.0005798 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabicl_v2 | 1 | RankAdaptiveGram | 1.637e-06 | 0.00114 | 0.0002959 | 0 | 8.258e-05 | 8 | 202 | 2.654e+06 | Type C — altered generalization/inductive bias |
| steel-plates-fault | tabicl_v2 | 1 | SafeGram-t01 | 3.281e-06 | 0.002314 | 0.0005798 | 0 | 1.827e-12 | 8 | 403 | 2.396e+09 | Type C — altered generalization/inductive bias |

Optimization and confidence diagnostics:

| dataset | model | seed | method | optimization_convergence | fit_seconds | best_epoch | test_ece_10bin | test_brier_multiclass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steel-plates-fault | controlled_mlp | 0 | Raw | early-stopped checkpoint | 0.7845 | 99 | 0.0002587 | 1.396e-05 |
| steel-plates-fault | controlled_mlp | 0 | GramAnchor | early-stopped checkpoint | 0.2003 | 24 | 0.0512 | 0.1262 |
| steel-plates-fault | controlled_mlp | 0 | RankAdaptiveGram | early-stopped checkpoint | 0.1055 | 10 | 0.03481 | 0.1236 |
| steel-plates-fault | controlled_mlp | 0 | SafeGram-t01 | native converged fit | 0.9848 | NA | 0.0002587 | 1.396e-05 |
| steel-plates-fault | controlled_mlp | 1 | Raw | early-stopped checkpoint | 0.2073 | 33 | 0.001801 | 0.002488 |
| steel-plates-fault | controlled_mlp | 1 | GramAnchor | early-stopped checkpoint | 0.1366 | 17 | 0.05547 | 0.1387 |
| steel-plates-fault | controlled_mlp | 1 | RankAdaptiveGram | early-stopped checkpoint | 0.1038 | 9 | 0.04171 | 0.1633 |
| steel-plates-fault | controlled_mlp | 1 | SafeGram-t01 | native converged fit | 0.3439 | NA | 0.001801 | 0.002488 |
| steel-plates-fault | tabm_d | 1 | Raw | native converged fit | 1.305 | NA | 0.01141 | 0.02069 |
| steel-plates-fault | tabm_d | 1 | GramAnchor | native converged fit | 0.4351 | NA | 0.02199 | 0.0272 |
| steel-plates-fault | tabm_d | 1 | RankAdaptiveGram | native converged fit | 0.254 | NA | 0.03738 | 0.03476 |
| steel-plates-fault | tabm_d | 1 | SafeGram-t01 | native converged fit | 1.74 | NA | 0.01141 | 0.02069 |
| steel-plates-fault | tabm_d | 0 | Raw | native converged fit | 0.2598 | NA | 0.05299 | 0.04966 |
| steel-plates-fault | tabm_d | 0 | GramAnchor | native converged fit | 0.3933 | NA | 0.01249 | 0.02663 |
| steel-plates-fault | tabm_d | 0 | RankAdaptiveGram | native converged fit | 0.3121 | NA | 0.01078 | 0.01201 |
| steel-plates-fault | tabm_d | 0 | SafeGram-t01 | native converged fit | 0.6531 | NA | 0.05299 | 0.04966 |
| steel-plates-fault | tabicl_v2 | 1 | Raw | native converged fit | 0.4927 | NA | 0.0002651 | 1.375e-05 |
| steel-plates-fault | tabicl_v2 | 1 | GramAnchor | native converged fit | 0.3604 | NA | 0.0005674 | 4.775e-05 |
| steel-plates-fault | tabicl_v2 | 1 | RankAdaptiveGram | native converged fit | 0.2886 | NA | 0.0002908 | 1.962e-05 |
| steel-plates-fault | tabicl_v2 | 1 | SafeGram-t01 | native converged fit | 0.8531 | NA | 0.0005674 | 4.775e-05 |

All five automatically selected worst cells were Type C. Optimization rescue status: `NOT_TRIGGERED` (No Type B failure was diagnosed.).

## 7. Steel Plates Deep Dive

| model | seed | method | test_absolute_difference | test_relative_difference | test_C | train_C | reconstruction_error | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled_mlp | 0 | Raw | 0 | 0 | 0 | 0 | 0 | 0 |
| controlled_mlp | 0 | GramAnchor | 0.3805 | 1450 | 0.5896 | 0.004843 | 1.827e-12 | 1 |
| controlled_mlp | 0 | RankAdaptiveGram | 0.2659 | 1013 | 0.412 | 0.05956 | 8.258e-05 | 1 |
| controlled_mlp | 0 | SafeGram-t01 | 0 | 0 | 0 | 0 | 1.827e-12 | 0 |
| controlled_mlp | 1 | Raw | 0 | 0 | 0 | 0 | 0 | 0 |
| controlled_mlp | 1 | GramAnchor | 0.4347 | 141.6 | 0.6766 | 0.02439 | 1.827e-12 | 1 |
| controlled_mlp | 1 | RankAdaptiveGram | 0.3207 | 104.4 | 0.4991 | 0.1164 | 8.258e-05 | 1 |
| controlled_mlp | 1 | SafeGram-t01 | 0 | 0 | 0 | 0 | 1.827e-12 | 0 |
| tabm_d | 1 | Raw | 0 | 0 | 0 | 0 | 0 | 0 |
| tabm_d | 1 | GramAnchor | 0.01096 | 0.2571 | 0.01818 | 0.003842 | 1.827e-12 | 1 |
| tabm_d | 1 | RankAdaptiveGram | 0.03345 | 0.7846 | 0.05548 | 0.04993 | 8.258e-05 | 1 |
| tabm_d | 1 | SafeGram-t01 | 0 | 0 | 0 | 0 | 1.827e-12 | 0 |
| tabm_d | 0 | Raw | 0 | 0 | 0 | 0 | 0 | 0 |
| tabm_d | 0 | GramAnchor | -0.05967 | -0.5455 | -0.1113 | -0.1043 | 1.827e-12 | 1 |
| tabm_d | 0 | RankAdaptiveGram | -0.09012 | -0.8239 | -0.1681 | -0.1137 | 8.258e-05 | 1 |
| tabm_d | 0 | SafeGram-t01 | 0 | 0 | 0 | 0 | 1.827e-12 | 0 |
| tabicl_v2 | 1 | Raw | 0 | 0 | 0 | 0 | 0 | 0 |
| tabicl_v2 | 1 | GramAnchor | 0.0003112 | 1.158 | 0.0004822 | 3.252e-06 | 1.827e-12 | 1 |
| tabicl_v2 | 1 | RankAdaptiveGram | 2.725e-05 | 0.1014 | 4.222e-05 | 7.042e-07 | 8.258e-05 | 1 |
| tabicl_v2 | 1 | SafeGram-t01 | 0.0003112 | 1.158 | 0.0004822 | 3.252e-06 | 1.827e-12 | 1 |

Steel Plates establishes altered inductive bias rather than information loss: fixed-Gram reconstruction is at floating-point scale and training C stays near raw, while validation/test C can become catastrophic. SafeGram observes the validation warning and falls back to raw in the damaging MLP cells.

## 8. Numerical Embedding Basis Test

| dataset | model | embedding | k | original_task | rotated_task | disagreement | best_rotated_task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| california_housing | controlled_mlp | PLE | 8 | 6.123e+04 | 6.234e+04 | 0.1215 | 6.16e+04 |
| california_housing | controlled_mlp | RBF | 8 | 6.49e+04 | 6.557e+04 | 0.1154 | 6.421e+04 |
| california_housing | resnet_tabular | PLE | 8 | 6.361e+04 | 6.514e+04 | 0.2282 | 6.238e+04 |
| california_housing | resnet_tabular | RBF | 8 | 6.488e+04 | 6.662e+04 | 0.2222 | 6.475e+04 |
| california_housing | tabm_d | PLE | 8 | 6.3e+04 | 5.918e+04 | 0.2206 | 5.813e+04 |
| california_housing | tabm_d | RBF | 8 | 6.099e+04 | 6.281e+04 | 0.1741 | 6.088e+04 |
| phoneme | controlled_mlp | PLE | 8 | 0.3275 | 0.3233 | 0.04728 | 0.3147 |
| phoneme | controlled_mlp | RBF | 8 | 0.3546 | 0.3491 | 0.04326 | 0.3418 |
| phoneme | resnet_tabular | PLE | 8 | 0.3292 | 0.3357 | 0.1045 | 0.3197 |
| phoneme | resnet_tabular | RBF | 8 | 0.3574 | 0.3542 | 0.1154 | 0.3297 |
| phoneme | tabm_d | PLE | 8 | 0.3293 | 0.3202 | 0.1147 | 0.3114 |
| phoneme | tabm_d | RBF | 8 | 0.3333 | 0.3361 | 0.09229 | 0.3234 |
| steel-plates-fault | controlled_mlp | PLE | 8 | 0.006919 | 0.009806 | 0.04346 | 0.00129 |
| steel-plates-fault | controlled_mlp | RBF | 8 | 0.001667 | 0.001374 | 0.01024 | 0.0001401 |
| steel-plates-fault | resnet_tabular | PLE | 8 | 0.0342 | 0.05639 | 0.09792 | 0.0003481 |
| steel-plates-fault | resnet_tabular | RBF | 8 | 0.03563 | 0.02868 | 0.08081 | 0.01358 |
| steel-plates-fault | tabm_d | PLE | 8 | 0.06996 | 0.03975 | 0.07697 | 0.01653 |
| steel-plates-fault | tabm_d | RBF | 8 | 0.07601 | 0.04395 | 0.04308 | 0.02341 |

## 9. Gram Inside Numerical Embeddings

| method | disagreement_reduction | median_C | p95_C | max_C | model_families |
| --- | --- | --- | --- | --- | --- |
| Gram-after-embedding | 1 | -0.01095 | 0.132 | 0.145 | 3 |
| RankAdaptiveGram-after-embedding | 1 | 0.02956 | 0.112 | 0.1518 | 3 |
| Raw embedding | 0 | 0 | 0 | 0 | 3 |
| SafeGram-after-embedding | 0.5625 | -0.02182 | 0.0005638 | 0.003759 | 3 |
| SafeRankGram prediction hybrid | 0.375 | -0.0165 | 0.0002375 | 0.001584 | 3 |

The invariant interface is placed explicitly between numerical embedding and backbone. PLE and RBF both show basis sensitivity; Gram removes it, and SafeGram retains useful basis-dependent inductive bias when validation supports it.

## 10. Embedding Dimension Ablation

| embedding | k | disagreement | task_effect | best_basis_task_effect |
| --- | --- | --- | --- | --- |
| PLE | 4 | 0.04171 | -0.0005492 | -0.01033 |
| PLE | 8 | 0.04894 | 0.00525 | -0.004559 |
| PLE | 16 | 0.06382 | 0.01811 | -0.01309 |
| PLE | 32 | 0.1231 | -0.001825 | -0.1044 |
| RBF | 4 | 0.0406 | 0.00139 | -6.144e-06 |
| RBF | 8 | 0.04681 | 0.001457 | -0.01672 |
| RBF | 16 | 0.1109 | 0.003308 | -0.04014 |
| RBF | 32 | 0.1696 | -0.147 | -0.1893 |

Sensitivity generally grows with k. The best rotated basis sometimes beats the default, confirming that arbitrary coordinates can supply useful as well as harmful inductive bias.

## 11. Development Finalist Ranking

| method | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | safety_eligible | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw+GramAnchor@0.75 | 0.75 | -0.002849 | 0.03942 | 0.1598 | 0 | True | 0.75 |
| SafeRankGram-t01 | 0.3125 | -0.01596 | 0 | 0.002757 | 0.3958 | True | 0.3125 |
| SafeGram-t01 | 0.25 | 0 | 0.004352 | 0.007605 | 0.475 | True | 0.25 |
| GramAnchor-m16 | 1 | 0.00108 | 0.06974 | 0.6336 | 0 | False | 0.09002 |
| RankAdaptiveGram | 1 | -0.01685 | 0.4498 | 1.644 | 0 | False | -2.688 |

## 12. Frozen Finalists

1. `GramAnchor-m16` — `{"development_evidence": {"max_C": 0.6336289967048432, "median_C": 0.0010804345566199, "median_disagreement_reduction": 1.0, "p95_C": 0.0697427381636023, "raw_fallback_rate": 0.0}, "interface": "gram_anchor", "interface_parameters": {"anchors": 16, "coordinate_standardization": true, "normalize": true, "selection": "gram_pivot"}, "method_id": "GramAnchor-m16", "type": "invariant_interface"}`
2. `RankAdaptiveGram` — `{"development_evidence": {"max_C": 1.6440246675813928, "median_C": -0.0168455890905317, "median_disagreement_reduction": 1.0, "p95_C": 0.4497934311118689, "raw_fallback_rate": 0.0}, "embedding_evidence": {"median_C": 0.029556398149462502, "median_disagreement_reduction": 1.0, "model_families": 3.0}, "interface": "rank_adaptive_gram", "method_id": "RankAdaptiveGram", "rank_config": {"anchor_rule": "rank", "normalization": "N1_anchor_norm", "relative_threshold": 0.0001, "standardize": true}, "type": "rank_adaptive_invariant_interface"}`
3. `SafeGram-t01` — `{"alpha_rule": {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "bootstrap_resamples": 500, "criterion": "largest alpha with row-bootstrap UCB95(C_alpha) <= tau", "epsilon": 1e-08, "fallback": 0.0, "selection_split": "validation_only", "tau": 0.01}, "development_evidence": {"max_C": 0.0076048343480971, "median_C": 0.0, "median_disagreement_reduction": 0.2500000001881521, "p95_C": 0.0043518505826225, "raw_fallback_rate": 0.475}, "embedding_evidence": {"median_C": -0.0218180831695986, "median_disagreement_reduction": 0.5625000050137565, "model_families": 3.0}, "invariant_branch": "GramAnchor-m16", "method_id": "SafeGram-t01", "type": "validation_controlled_prediction_hybrid"}`
4. `SafeRankGram-t01` — `{"alpha_rule": {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "bootstrap_resamples": 500, "criterion": "largest alpha with row-bootstrap UCB95(C_alpha) <= tau", "epsilon": 1e-08, "fallback": 0.0, "selection_split": "validation_only", "tau": 0.01}, "development_evidence": {"max_C": 0.0027567321987702, "median_C": -0.0159609967514684, "median_disagreement_reduction": 0.3125000035744253, "p95_C": 0.0, "raw_fallback_rate": 0.3958333333333333}, "embedding_evidence": {"median_C": -0.016503432251227752, "median_disagreement_reduction": 0.3749999987286957, "model_families": 3.0}, "invariant_branch": "RankAdaptiveGram", "method_id": "SafeRankGram-t01", "rank_config": {"anchor_rule": "rank", "normalization": "N1_anchor_norm", "relative_threshold": 0.0001, "standardize": true}, "type": "validation_controlled_rank_adaptive_prediction_hybrid"}`

Frozen SHA256: `b7b075362f5c7295fdb55f6d6140e5cf27499fcc1dbac0363f16e7aca9c14d29`. The lock predates every prospective raw artifact.

## 13. NEW Prospective Results

| dataset | model | method | alpha | disagreement reduction | raw task | method task | C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPMP-2015-regression | catboost | GramAnchor-m16 | 1 | 1 | 469.7 | 465.4 | -0.003828 |
| CPMP-2015-regression | catboost | PCA-canonicalization | 1 | 1 | 469.7 | 476.9 | 0.006494 |
| CPMP-2015-regression | catboost | RankAdaptiveGram | 1 | 1 | 469.7 | 466.9 | -0.002469 |
| CPMP-2015-regression | catboost | Raw | 0 | 0 | 469.7 | 469.7 | 0 |
| CPMP-2015-regression | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 469.7 | 465.9 | -0.003362 |
| CPMP-2015-regression | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 469.7 | 465.3 | -0.003959 |
| CPMP-2015-regression | catboost | SafeGram-t01 | 0.75 | 0.75 | 469.7 | 465.2 | -0.003974 |
| CPMP-2015-regression | catboost | SafeRankGram-t01 | 0.875 | 0.875 | 469.7 | 466.8 | -0.002537 |
| CPMP-2015-regression | controlled_mlp | GramAnchor-m16 | 1 | 1 | 469.7 | 474.7 | 0.004458 |
| CPMP-2015-regression | controlled_mlp | PCA-canonicalization | 1 | 1 | 469.7 | 462.4 | -0.006541 |
| CPMP-2015-regression | controlled_mlp | RankAdaptiveGram | 1 | 1 | 469.7 | 614.9 | 0.1301 |
| CPMP-2015-regression | controlled_mlp | Raw | 0 | 0 | 469.7 | 469.7 | 0 |
| CPMP-2015-regression | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 469.7 | 452.1 | -0.01578 |
| CPMP-2015-regression | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 469.7 | 458.5 | -0.01005 |
| CPMP-2015-regression | controlled_mlp | SafeGram-t01 | 0.875 | 0.875 | 469.7 | 466.1 | -0.003265 |
| CPMP-2015-regression | controlled_mlp | SafeRankGram-t01 | 0 | 0 | 469.7 | 469.7 | 0 |
| CPMP-2015-regression | tabm_d | GramAnchor-m16 | 1 | 1 | 452.2 | 472.6 | 0.01795 |
| CPMP-2015-regression | tabm_d | PCA-canonicalization | 1 | 1 | 452.2 | 495.5 | 0.03819 |
| CPMP-2015-regression | tabm_d | RankAdaptiveGram | 1 | 1 | 452.2 | 453.7 | 0.001278 |
| CPMP-2015-regression | tabm_d | Raw | 0 | 0 | 452.2 | 452.2 | 0 |
| CPMP-2015-regression | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 452.2 | 457.8 | 0.004957 |
| CPMP-2015-regression | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 452.2 | 464.1 | 0.01049 |
| CPMP-2015-regression | tabm_d | SafeGram-t01 | 1 | 1 | 452.2 | 472.6 | 0.01795 |
| CPMP-2015-regression | tabm_d | SafeRankGram-t01 | 0.875 | 0.875 | 452.2 | 452.7 | 0.0003909 |
| cardiotocography | catboost | GramAnchor-m16 | 1 | 1 | 0.01819 | 0.02747 | 0.00464 |
| cardiotocography | catboost | PCA-canonicalization | 1 | 1 | 0.01819 | 0.01559 | -0.001304 |
| cardiotocography | catboost | RankAdaptiveGram | 1 | 1 | 0.01819 | 0.0189 | 0.0003538 |
| cardiotocography | catboost | Raw | 0 | 0 | 0.01819 | 0.01819 | 0 |
| cardiotocography | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.01819 | 0.02271 | 0.002256 |
| cardiotocography | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.01819 | 0.02505 | 0.00343 |
| cardiotocography | catboost | SafeGram-t01 | 1 | 1 | 0.01819 | 0.02747 | 0.00464 |
| cardiotocography | catboost | SafeRankGram-t01 | 1 | 1 | 0.01819 | 0.0189 | 0.0003538 |
| cardiotocography | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.01317 | 0.3488 | 0.1673 |
| cardiotocography | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.01317 | 0.03753 | 0.01213 |
| cardiotocography | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.01317 | 0.1851 | 0.08571 |
| cardiotocography | controlled_mlp | Raw | 0 | 0 | 0.01317 | 0.01317 | 0 |
| cardiotocography | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.01317 | 0.05789 | 0.02232 |
| cardiotocography | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.01317 | 0.09624 | 0.04145 |
| cardiotocography | controlled_mlp | SafeGram-t01 | 0 | 0 | 0.01317 | 0.01317 | 0 |
| cardiotocography | controlled_mlp | SafeRankGram-t01 | 0.25 | 0.25 | 0.01317 | 0.02296 | 0.004885 |
| cardiotocography | tabm_d | GramAnchor-m16 | 1 | 1 | 0.01571 | 0.0218 | 0.00304 |
| cardiotocography | tabm_d | PCA-canonicalization | 1 | 1 | 0.01571 | 0.04539 | 0.01482 |
| cardiotocography | tabm_d | RankAdaptiveGram | 1 | 1 | 0.01571 | 0.005925 | -0.004886 |
| cardiotocography | tabm_d | Raw | 0 | 0 | 0.01571 | 0.01571 | 0 |
| cardiotocography | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.01571 | 0.01843 | 0.00136 |
| cardiotocography | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.01571 | 0.02002 | 0.002154 |
| cardiotocography | tabm_d | SafeGram-t01 | 1 | 1 | 0.01571 | 0.0218 | 0.00304 |
| cardiotocography | tabm_d | SafeRankGram-t01 | 1 | 1 | 0.01571 | 0.005925 | -0.004886 |
| debutanizer | catboost | GramAnchor-m16 | 1 | 1 | 0.08762 | 0.09349 | 0.08462 |
| debutanizer | catboost | PCA-canonicalization | 1 | 0.5241 | 0.08762 | 0.1038 | 0.2334 |
| debutanizer | catboost | RankAdaptiveGram | 1 | 1 | 0.08762 | 0.08989 | 0.03264 |
| debutanizer | catboost | Raw | 0 | 0 | 0.08762 | 0.08762 | 0 |
| debutanizer | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.08762 | 0.08991 | 0.03292 |
| debutanizer | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.08762 | 0.09154 | 0.05654 |
| debutanizer | catboost | SafeGram-t01 | 0 | 0 | 0.08762 | 0.08762 | 0 |
| debutanizer | catboost | SafeRankGram-t01 | 0.125 | 0.125 | 0.08762 | 0.08774 | 0.001688 |
| debutanizer | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.1151 | 0.09487 | -0.4857 |
| debutanizer | controlled_mlp | PCA-canonicalization | 1 | 0.9934 | 0.1151 | 0.106 | -0.2201 |
| debutanizer | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.1151 | 0.09379 | -0.5113 |
| debutanizer | controlled_mlp | Raw | 0 | 0 | 0.1151 | 0.1151 | 0 |
| debutanizer | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1151 | 0.09767 | -0.4171 |
| debutanizer | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1151 | 0.0942 | -0.5007 |
| debutanizer | controlled_mlp | SafeGram-t01 | 1 | 1 | 0.1151 | 0.09487 | -0.4857 |
| debutanizer | controlled_mlp | SafeRankGram-t01 | 1 | 1 | 0.1151 | 0.09379 | -0.5113 |
| debutanizer | tabm_d | GramAnchor-m16 | 1 | 1 | 0.08336 | 0.08728 | 0.05318 |
| debutanizer | tabm_d | PCA-canonicalization | 1 | 0.9686 | 0.08336 | 0.09526 | 0.1617 |
| debutanizer | tabm_d | RankAdaptiveGram | 1 | 1 | 0.08336 | 0.0877 | 0.05885 |
| debutanizer | tabm_d | Raw | 0 | 0 | 0.08336 | 0.08336 | 0 |
| debutanizer | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.08336 | 0.08393 | 0.007629 |
| debutanizer | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.08336 | 0.08527 | 0.02586 |
| debutanizer | tabm_d | SafeGram-t01 | 0 | 0 | 0.08336 | 0.08336 | 0 |
| debutanizer | tabm_d | SafeRankGram-t01 | 0 | 0 | 0.08336 | 0.08336 | 0 |
| delta_elevators | catboost | GramAnchor-m16 | 1 | 1 | 0.001669 | 0.001681 | 0.01805 |
| delta_elevators | catboost | PCA-canonicalization | 1 | 1 | 0.001669 | 0.001677 | 0.01319 |
| delta_elevators | catboost | RankAdaptiveGram | 1 | 1 | 0.001669 | 0.001673 | 0.00678 |
| delta_elevators | catboost | Raw | 0 | 0 | 0.001669 | 0.001669 | 0 |
| delta_elevators | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.001669 | 0.001671 | 0.00394 |
| delta_elevators | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.001669 | 0.001675 | 0.009733 |
| delta_elevators | catboost | SafeGram-t01 | 0.75 | 0.75 | 0.001669 | 0.001678 | 0.01335 |
| delta_elevators | catboost | SafeRankGram-t01 | 0.625 | 0.625 | 0.001669 | 0.001672 | 0.005054 |
| delta_elevators | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.001685 | 0.001749 | 0.1015 |
| delta_elevators | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.001685 | 0.001682 | -0.004625 |
| delta_elevators | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.001685 | 0.001708 | 0.03628 |
| delta_elevators | controlled_mlp | Raw | 0 | 0 | 0.001685 | 0.001685 | 0 |
| delta_elevators | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.001685 | 0.001685 | 1.104e-05 |
| delta_elevators | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.001685 | 0.001709 | 0.03852 |
| delta_elevators | controlled_mlp | SafeGram-t01 | 0.25 | 0.25 | 0.001685 | 0.001677 | -0.01295 |
| delta_elevators | controlled_mlp | SafeRankGram-t01 | 0.5 | 0.5 | 0.001685 | 0.001673 | -0.01864 |
| delta_elevators | tabm_d | GramAnchor-m16 | 1 | 1 | 0.001659 | 0.001665 | 0.008483 |
| delta_elevators | tabm_d | PCA-canonicalization | 1 | 1 | 0.001659 | 0.001669 | 0.01551 |
| delta_elevators | tabm_d | RankAdaptiveGram | 1 | 1 | 0.001659 | 0.001665 | 0.009578 |
| delta_elevators | tabm_d | Raw | 0 | 0 | 0.001659 | 0.001659 | 0 |
| delta_elevators | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.001659 | 0.001658 | -0.001552 |
| delta_elevators | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.001659 | 0.00166 | 0.002022 |
| delta_elevators | tabm_d | SafeGram-t01 | 0.625 | 0.625 | 0.001659 | 0.001659 | -0.0005488 |
| delta_elevators | tabm_d | SafeRankGram-t01 | 1 | 1 | 0.001659 | 0.001665 | 0.009578 |
| first-order-theorem-proving | catboost | GramAnchor-m16 | 1 | 1 | 1.282 | 1.292 | 0.03129 |
| first-order-theorem-proving | catboost | PCA-canonicalization | 1 | 1 | 1.282 | 1.3 | 0.05656 |
| first-order-theorem-proving | catboost | RankAdaptiveGram | 1 | 1 | 1.282 | 1.284 | 0.005639 |
| first-order-theorem-proving | catboost | Raw | 0 | 0 | 1.282 | 1.282 | 0 |
| first-order-theorem-proving | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 1.282 | 1.285 | 0.006549 |
| first-order-theorem-proving | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 1.282 | 1.288 | 0.01654 |
| first-order-theorem-proving | catboost | SafeGram-t01 | 0.625 | 0.625 | 1.282 | 1.291 | 0.02764 |
| first-order-theorem-proving | catboost | SafeRankGram-t01 | 0.5 | 0.5 | 1.282 | 1.281 | -0.005995 |
| first-order-theorem-proving | controlled_mlp | GramAnchor-m16 | 1 | 1 | 1.366 | 1.343 | -0.1022 |
| first-order-theorem-proving | controlled_mlp | PCA-canonicalization | 1 | 1 | 1.366 | 1.36 | -0.02759 |
| first-order-theorem-proving | controlled_mlp | RankAdaptiveGram | 1 | 1 | 1.366 | 1.421 | 0.242 |
| first-order-theorem-proving | controlled_mlp | Raw | 0 | 0 | 1.366 | 1.366 | 0 |
| first-order-theorem-proving | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 1.366 | 1.323 | -0.1909 |
| first-order-theorem-proving | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 1.366 | 1.321 | -0.1972 |
| first-order-theorem-proving | controlled_mlp | SafeGram-t01 | 0.375 | 0.375 | 1.366 | 1.329 | -0.1626 |
| first-order-theorem-proving | controlled_mlp | SafeRankGram-t01 | 0.25 | 0.25 | 1.366 | 1.353 | -0.05599 |
| first-order-theorem-proving | tabm_d | GramAnchor-m16 | 1 | 1 | 1.336 | 1.362 | 0.08877 |
| first-order-theorem-proving | tabm_d | PCA-canonicalization | 1 | 1 | 1.336 | 1.309 | -0.1218 |
| first-order-theorem-proving | tabm_d | RankAdaptiveGram | 1 | 1 | 1.336 | 1.3 | -0.1569 |
| first-order-theorem-proving | tabm_d | Raw | 0 | 0 | 1.336 | 1.336 | 0 |
| first-order-theorem-proving | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 1.336 | 1.299 | -0.1626 |
| first-order-theorem-proving | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 1.336 | 1.318 | -0.08836 |
| first-order-theorem-proving | tabm_d | SafeGram-t01 | 0.375 | 0.375 | 1.336 | 1.308 | -0.1217 |
| first-order-theorem-proving | tabm_d | SafeRankGram-t01 | 0.875 | 0.875 | 1.336 | 1.296 | -0.1704 |
| ozone-level-8hr | catboost | GramAnchor-m16 | 1 | 1 | 0.1717 | 0.175 | 0.05225 |
| ozone-level-8hr | catboost | PCA-canonicalization | 1 | 1 | 0.1717 | 0.1977 | 0.4068 |
| ozone-level-8hr | catboost | RankAdaptiveGram | 1 | 1 | 0.1717 | 0.1741 | 0.03795 |
| ozone-level-8hr | catboost | Raw | 0 | 0 | 0.1717 | 0.1717 | 0 |
| ozone-level-8hr | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1717 | 0.1712 | -0.007391 |
| ozone-level-8hr | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1717 | 0.1725 | 0.01299 |
| ozone-level-8hr | catboost | SafeGram-t01 | 0 | 0 | 0.1717 | 0.1717 | 0 |
| ozone-level-8hr | catboost | SafeRankGram-t01 | 0.5 | 0.5 | 0.1717 | 0.172 | 0.005146 |
| ozone-level-8hr | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.1602 | 0.1385 | -0.2902 |
| ozone-level-8hr | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.1602 | 0.168 | 0.1034 |
| ozone-level-8hr | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.1602 | 0.1379 | -0.2976 |
| ozone-level-8hr | controlled_mlp | Raw | 0 | 0 | 0.1602 | 0.1602 | 0 |
| ozone-level-8hr | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1602 | 0.1443 | -0.2127 |
| ozone-level-8hr | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1602 | 0.1405 | -0.2639 |
| ozone-level-8hr | controlled_mlp | SafeGram-t01 | 0 | 0 | 0.1602 | 0.1602 | 0 |
| ozone-level-8hr | controlled_mlp | SafeRankGram-t01 | 0.25 | 0.25 | 0.1602 | 0.1497 | -0.1408 |
| ozone-level-8hr | tabm_d | GramAnchor-m16 | 1 | 1 | 0.3247 | 0.3132 | -1.148e+06 |
| ozone-level-8hr | tabm_d | PCA-canonicalization | 1 | 1 | 0.3247 | 0.3564 | 3.17e+06 |
| ozone-level-8hr | tabm_d | RankAdaptiveGram | 1 | 1 | 0.3247 | 0.3332 | 8.562e+05 |
| ozone-level-8hr | tabm_d | Raw | 0 | 0 | 0.3247 | 0.3247 | 0 |
| ozone-level-8hr | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.3247 | 0.315 | -9.619e+05 |
| ozone-level-8hr | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.3247 | 0.313 | -1.162e+06 |
| ozone-level-8hr | tabm_d | SafeGram-t01 | 1 | 1 | 0.3247 | 0.3132 | -1.148e+06 |
| ozone-level-8hr | tabm_d | SafeRankGram-t01 | 0 | 0 | 0.3247 | 0.3247 | 0 |
| quake | catboost | GramAnchor-m16 | 1 | 1 | 0.1742 | 0.1733 | -23.3 |
| quake | catboost | PCA-canonicalization | 1 | 1 | 0.1742 | 0.1737 | -12.79 |
| quake | catboost | RankAdaptiveGram | 1 | 1 | 0.1742 | 0.1734 | -13.79 |
| quake | catboost | Raw | 0 | 0 | 0.1742 | 0.1742 | 0 |
| quake | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1742 | 0.1737 | -12.94 |
| quake | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1742 | 0.1735 | -18.44 |
| quake | catboost | SafeGram-t01 | 0 | 0 | 0.1742 | 0.1742 | 0 |
| quake | catboost | SafeRankGram-t01 | 0 | 0 | 0.1742 | 0.1742 | 0 |
| quake | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.1785 | 0.1751 | -3.352e+05 |
| quake | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.1785 | 0.1778 | -6.156e+04 |
| quake | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.1785 | 0.1751 | -3.368e+05 |
| quake | controlled_mlp | Raw | 0 | 0 | 0.1785 | 0.1785 | 0 |
| quake | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1785 | 0.1763 | -2.211e+05 |
| quake | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1785 | 0.1755 | -2.917e+05 |
| quake | controlled_mlp | SafeGram-t01 | 0 | 0 | 0.1785 | 0.1785 | 0 |
| quake | controlled_mlp | SafeRankGram-t01 | 0 | 0 | 0.1785 | 0.1785 | 0 |
| quake | tabm_d | GramAnchor-m16 | 1 | 1 | 0.1778 | 0.176 | -1.75e+05 |
| quake | tabm_d | PCA-canonicalization | 1 | 1 | 0.1778 | 0.1782 | 4.87e+04 |
| quake | tabm_d | RankAdaptiveGram | 1 | 1 | 0.1778 | 0.1762 | -1.55e+05 |
| quake | tabm_d | Raw | 0 | 0 | 0.1778 | 0.1778 | 0 |
| quake | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1778 | 0.1766 | -1.134e+05 |
| quake | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1778 | 0.1763 | -1.507e+05 |
| quake | tabm_d | SafeGram-t01 | 0 | 0 | 0.1778 | 0.1778 | 0 |
| quake | tabm_d | SafeRankGram-t01 | 0 | 0 | 0.1778 | 0.1778 | 0 |
| sylvine | catboost | GramAnchor-m16 | 1 | 1 | 0.2091 | 0.2163 | 0.01481 |
| sylvine | catboost | PCA-canonicalization | 1 | 1 | 0.2091 | 0.2208 | 0.02407 |
| sylvine | catboost | RankAdaptiveGram | 1 | 1 | 0.2091 | 0.2127 | 0.007233 |
| sylvine | catboost | Raw | 0 | 0 | 0.2091 | 0.2091 | 0 |
| sylvine | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.2091 | 0.2114 | 0.004669 |
| sylvine | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.2091 | 0.2135 | 0.009068 |
| sylvine | catboost | SafeGram-t01 | 0.375 | 0.375 | 0.2091 | 0.2117 | 0.00535 |
| sylvine | catboost | SafeRankGram-t01 | 0.5 | 0.5 | 0.2091 | 0.2072 | -0.004019 |
| sylvine | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.2526 | 0.2968 | 0.1005 |
| sylvine | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.2526 | 0.2679 | 0.03475 |
| sylvine | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.2526 | 0.2918 | 0.08908 |
| sylvine | controlled_mlp | Raw | 0 | 0 | 0.2526 | 0.2526 | 0 |
| sylvine | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.2526 | 0.2513 | -0.002854 |
| sylvine | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.2526 | 0.2627 | 0.02302 |
| sylvine | controlled_mlp | SafeGram-t01 | 0.125 | 0.125 | 0.2526 | 0.2497 | -0.006526 |
| sylvine | controlled_mlp | SafeRankGram-t01 | 0.125 | 0.125 | 0.2526 | 0.2493 | -0.007493 |
| sylvine | tabm_d | GramAnchor-m16 | 1 | 1 | 0.2196 | 0.3098 | 0.1904 |
| sylvine | tabm_d | PCA-canonicalization | 1 | 1 | 0.2196 | 0.3961 | 0.3728 |
| sylvine | tabm_d | RankAdaptiveGram | 1 | 1 | 0.2196 | 0.2198 | 0.0003389 |
| sylvine | tabm_d | Raw | 0 | 0 | 0.2196 | 0.2196 | 0 |
| sylvine | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.2196 | 0.2552 | 0.07515 |
| sylvine | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.2196 | 0.2802 | 0.1278 |
| sylvine | tabm_d | SafeGram-t01 | 0 | 0 | 0.2196 | 0.2196 | 0 |
| sylvine | tabm_d | SafeRankGram-t01 | 0.25 | 0.25 | 0.2196 | 0.2185 | -0.002447 |
| visualizing_soil | catboost | GramAnchor-m16 | 1 | 1 | 0.3568 | 0.3713 | 0.001222 |
| visualizing_soil | catboost | PCA-canonicalization | 1 | 1 | 0.3568 | 0.3871 | 0.002545 |
| visualizing_soil | catboost | RankAdaptiveGram | 1 | 1 | 0.3568 | 0.3991 | 0.003553 |
| visualizing_soil | catboost | Raw | 0 | 0 | 0.3568 | 0.3568 | 0 |
| visualizing_soil | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.3568 | 0.3362 | -0.00173 |
| visualizing_soil | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.3568 | 0.3472 | -0.0008031 |
| visualizing_soil | catboost | SafeGram-t01 | 1 | 1 | 0.3568 | 0.3713 | 0.001222 |
| visualizing_soil | catboost | SafeRankGram-t01 | 1 | 1 | 0.3568 | 0.3991 | 0.003553 |
| visualizing_soil | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.3258 | 0.2471 | -0.0066 |
| visualizing_soil | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.3258 | 0.2366 | -0.00748 |
| visualizing_soil | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.3258 | 0.2965 | -0.002462 |
| visualizing_soil | controlled_mlp | Raw | 0 | 0 | 0.3258 | 0.3258 | 0 |
| visualizing_soil | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.3258 | 0.2639 | -0.005187 |
| visualizing_soil | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.3258 | 0.2487 | -0.006464 |
| visualizing_soil | controlled_mlp | SafeGram-t01 | 1 | 1 | 0.3258 | 0.2471 | -0.0066 |
| visualizing_soil | controlled_mlp | SafeRankGram-t01 | 1 | 1 | 0.3258 | 0.2965 | -0.002462 |
| visualizing_soil | tabm_d | GramAnchor-m16 | 1 | 1 | 0.2897 | 0.3559 | 0.005527 |
| visualizing_soil | tabm_d | PCA-canonicalization | 1 | 1 | 0.2897 | 0.5562 | 0.02226 |
| visualizing_soil | tabm_d | RankAdaptiveGram | 1 | 1 | 0.2897 | 0.3539 | 0.005363 |
| visualizing_soil | tabm_d | Raw | 0 | 0 | 0.2897 | 0.2897 | 0 |
| visualizing_soil | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.2897 | 0.2526 | -0.0031 |
| visualizing_soil | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.2897 | 0.2913 | 0.0001332 |
| visualizing_soil | tabm_d | SafeGram-t01 | 1 | 1 | 0.2897 | 0.3559 | 0.005527 |
| visualizing_soil | tabm_d | SafeRankGram-t01 | 0.875 | 0.875 | 0.2897 | 0.3237 | 0.002844 |
| wall-robot-navigation | catboost | GramAnchor-m16 | 1 | 1 | 0.08295 | 0.1006 | 0.016 |
| wall-robot-navigation | catboost | PCA-canonicalization | 1 | 1 | 0.08295 | 0.1455 | 0.05658 |
| wall-robot-navigation | catboost | RankAdaptiveGram | 1 | 1 | 0.08295 | 0.08905 | 0.005521 |
| wall-robot-navigation | catboost | Raw | 0 | 0 | 0.08295 | 0.08295 | 0 |
| wall-robot-navigation | catboost | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.08295 | 0.09057 | 0.006893 |
| wall-robot-navigation | catboost | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.08295 | 0.09512 | 0.01102 |
| wall-robot-navigation | catboost | SafeGram-t01 | 0.25 | 0.25 | 0.08295 | 0.08656 | 0.003266 |
| wall-robot-navigation | catboost | SafeRankGram-t01 | 0.5 | 0.5 | 0.08295 | 0.08516 | 0.002001 |
| wall-robot-navigation | controlled_mlp | GramAnchor-m16 | 1 | 1 | 0.2374 | 0.2911 | 0.05649 |
| wall-robot-navigation | controlled_mlp | PCA-canonicalization | 1 | 1 | 0.2374 | 0.2026 | -0.03676 |
| wall-robot-navigation | controlled_mlp | RankAdaptiveGram | 1 | 1 | 0.2374 | 0.2805 | 0.04524 |
| wall-robot-navigation | controlled_mlp | Raw | 0 | 0 | 0.2374 | 0.2374 | 0 |
| wall-robot-navigation | controlled_mlp | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.2374 | 0.2107 | -0.0281 |
| wall-robot-navigation | controlled_mlp | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.2374 | 0.2246 | -0.01355 |
| wall-robot-navigation | controlled_mlp | SafeGram-t01 | 0.375 | 0.375 | 0.2374 | 0.2101 | -0.02873 |
| wall-robot-navigation | controlled_mlp | SafeRankGram-t01 | 0.25 | 0.25 | 0.2374 | 0.2153 | -0.02334 |
| wall-robot-navigation | tabm_d | GramAnchor-m16 | 1 | 1 | 0.1283 | 0.1204 | -0.00745 |
| wall-robot-navigation | tabm_d | PCA-canonicalization | 1 | 1 | 0.1283 | 0.09834 | -0.02827 |
| wall-robot-navigation | tabm_d | RankAdaptiveGram | 1 | 1 | 0.1283 | 0.1145 | -0.01305 |
| wall-robot-navigation | tabm_d | Raw | 0 | 0 | 0.1283 | 0.1283 | 0 |
| wall-robot-navigation | tabm_d | Raw+GramAnchor@0.5 | 0.5 | 0.5 | 0.1283 | 0.1165 | -0.01115 |
| wall-robot-navigation | tabm_d | Raw+GramAnchor@0.75 | 0.75 | 0.75 | 0.1283 | 0.1159 | -0.01167 |
| wall-robot-navigation | tabm_d | SafeGram-t01 | 1 | 1 | 0.1283 | 0.1204 | -0.00745 |
| wall-robot-navigation | tabm_d | SafeRankGram-t01 | 1 | 1 | 0.1283 | 0.1145 | -0.01305 |

## 14. Prospective Aggregate Results

| method | median_disagreement_reduction | median_C | p90_C | p95_C | max_C | task W/T/L | raw_fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GramAnchor-m16 | 1 | 0.007005 | 0.1006 | 0.1377 | 0.1904 | 10/0/20 | 0 |
| PCA-canonicalization | 1 | 0.01401 | 0.3762 | 2.679e+04 | 3.17e+06 | 11/0/19 | 0 |
| RankAdaptiveGram | 1 | 0.005442 | 0.09319 | 0.1916 | 8.562e+05 | 10/0/20 | 0 |
| Raw | 0 | 0 | 0 | 0 | 0 | 0/30/0 | 1 |
| Raw+GramAnchor@0.5 | 0.5 | -0.002977 | 0.009098 | 0.02815 | 0.07515 | 18/1/11 | 0 |
| Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.03881 | 0.04975 | 0.1278 | 14/0/16 | 0 |
| SafeGram-t01 | 0.375 | 0 | 0.006309 | 0.01588 | 0.02764 | 11/10/9 | 0.35 |
| SafeRankGram-t01 | 0.5 | 0 | 0.004902 | 0.005105 | 0.009578 | 14/6/10 | 0.2667 |

## 15. Safety-First Ranking

| method | eligibility_note | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | mean_predictive_rank | paper_candidate_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw+GramAnchor@0.75 | ELIGIBLE | 0.75 | 0.001078 | 0.04975 | 0.1278 | 0 | 4.233 | 0.7468 |
| Raw+GramAnchor@0.5 | ELIGIBLE | 0.5 | -0.002977 | 0.02815 | 0.07515 | 0 | 3.533 | 0.5 |
| SafeRankGram-t01 | ELIGIBLE | 0.5 | 0 | 0.005105 | 0.009578 | 0.2667 | 3.617 | 0.5 |
| SafeGram-t01 | ELIGIBLE | 0.375 | 0 | 0.01588 | 0.02764 | 0.35 | 4.133 | 0.375 |
| Raw | ELIGIBLE | 0 | 0 | 0 | 0 | 1 | 4.217 | 0 |
| GramAnchor-m16 | EXCLUDED_BY_TAIL_GATE | 1 | 0.007005 | 0.1377 | 0.1904 | 0 | 5.617 | 0.8036 |
| PCA-canonicalization | EXCLUDED_BY_TAIL_GATE | 1 | 0.01401 | 2.679e+04 | 3.17e+06 | 0 | 6 | -6.393e+06 |
| RankAdaptiveGram | EXCLUDED_BY_TAIL_GATE | 1 | 0.005442 | 0.1916 | 8.562e+05 | 0 | 4.65 | -1.712e+06 |

## 16. Invariance Ranking

| method | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | mean_predictive_rank | paper_candidate_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GramAnchor-m16 | 1 | 0.007005 | 0.1377 | 0.1904 | 0 | 5.617 | 0.8036 |
| PCA-canonicalization | 1 | 0.01401 | 2.679e+04 | 3.17e+06 | 0 | 6 | -6.393e+06 |
| RankAdaptiveGram | 1 | 0.005442 | 0.1916 | 8.562e+05 | 0 | 4.65 | -1.712e+06 |
| Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.04975 | 0.1278 | 0 | 4.233 | 0.7468 |
| SafeRankGram-t01 | 0.5 | 0 | 0.005105 | 0.009578 | 0.2667 | 3.617 | 0.5 |
| Raw+GramAnchor@0.5 | 0.5 | -0.002977 | 0.02815 | 0.07515 | 0 | 3.533 | 0.5 |
| SafeGram-t01 | 0.375 | 0 | 0.01588 | 0.02764 | 0.35 | 4.133 | 0.375 |
| Raw | 0 | 0 | 0 | 0 | 1 | 4.217 | 0 |

## 17. Predictive Ranking

| method | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | mean_predictive_rank | paper_candidate_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw+GramAnchor@0.5 | 0.5 | -0.002977 | 0.02815 | 0.07515 | 0 | 3.533 | 0.5 |
| SafeRankGram-t01 | 0.5 | 0 | 0.005105 | 0.009578 | 0.2667 | 3.617 | 0.5 |
| SafeGram-t01 | 0.375 | 0 | 0.01588 | 0.02764 | 0.35 | 4.133 | 0.375 |
| Raw | 0 | 0 | 0 | 0 | 1 | 4.217 | 0 |
| Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.04975 | 0.1278 | 0 | 4.233 | 0.7468 |
| RankAdaptiveGram | 1 | 0.005442 | 0.1916 | 8.562e+05 | 0 | 4.65 | -1.712e+06 |
| GramAnchor-m16 | 1 | 0.007005 | 0.1377 | 0.1904 | 0 | 5.617 | 0.8036 |
| PCA-canonicalization | 1 | 0.01401 | 2.679e+04 | 3.17e+06 | 0 | 6 | -6.393e+06 |

## 18. Tail-Robustness Ranking

| method | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | mean_predictive_rank | paper_candidate_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw | 0 | 0 | 0 | 0 | 1 | 4.217 | 0 |
| SafeRankGram-t01 | 0.5 | 0 | 0.005105 | 0.009578 | 0.2667 | 3.617 | 0.5 |
| SafeGram-t01 | 0.375 | 0 | 0.01588 | 0.02764 | 0.35 | 4.133 | 0.375 |
| Raw+GramAnchor@0.5 | 0.5 | -0.002977 | 0.02815 | 0.07515 | 0 | 3.533 | 0.5 |
| Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.04975 | 0.1278 | 0 | 4.233 | 0.7468 |
| GramAnchor-m16 | 1 | 0.007005 | 0.1377 | 0.1904 | 0 | 5.617 | 0.8036 |
| RankAdaptiveGram | 1 | 0.005442 | 0.1916 | 8.562e+05 | 0 | 4.65 | -1.712e+06 |
| PCA-canonicalization | 1 | 0.01401 | 2.679e+04 | 3.17e+06 | 0 | 6 | -6.393e+06 |

## 19. Paper-Candidate Ranking

| method | median_disagreement_reduction | median_C | p95_C | max_C | raw_fallback_rate | mean_predictive_rank | paper_candidate_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GramAnchor-m16 | 1 | 0.007005 | 0.1377 | 0.1904 | 0 | 5.617 | 0.8036 |
| Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.04975 | 0.1278 | 0 | 4.233 | 0.7468 |
| Raw+GramAnchor@0.5 | 0.5 | -0.002977 | 0.02815 | 0.07515 | 0 | 3.533 | 0.5 |
| SafeRankGram-t01 | 0.5 | 0 | 0.005105 | 0.009578 | 0.2667 | 3.617 | 0.5 |
| SafeGram-t01 | 0.375 | 0 | 0.01588 | 0.02764 | 0.35 | 4.133 | 0.375 |
| Raw | 0 | 0 | 0 | 0 | 1 | 4.217 | 0 |
| RankAdaptiveGram | 1 | 0.005442 | 0.1916 | 8.562e+05 | 0 | 4.65 | -1.712e+06 |
| PCA-canonicalization | 1 | 0.01401 | 2.679e+04 | 3.17e+06 | 0 | 6 | -6.393e+06 |

## 20. Natural-Basis Validation

| natural_pair | method | median_disagreement_reduction | median_C | p95_C | max_C | max_equivalence_error | max_coordinate_error | model_families |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_hat_to_spectral_hat | GramAnchor-m16 | 1 | 0.001041 | 0.07178 | 0.1051 | 2.247e-16 | 1.722e-16 | 3 |
| local_hat_to_spectral_hat | RankAdaptiveGram | 1 | 0.0003461 | 3.044e+05 | 1.015e+06 | 2.247e-16 | 1.722e-16 | 3 |
| local_hat_to_spectral_hat | Raw | 0 | 0 | 0 | 0 | 2.247e-16 | 1.722e-16 | 3 |
| local_hat_to_spectral_hat | Raw+GramAnchor@0.75 | 0.75 | -0.005655 | 0.02542 | 0.05542 | 2.247e-16 | 1.722e-16 | 3 |
| local_hat_to_spectral_hat | SafeGram-t01 | 0.75 | 0 | 0.02876 | 0.05542 | 2.247e-16 | 1.722e-16 | 3 |
| local_hat_to_spectral_hat | SafeRankGram-t01 | 0.25 | 0 | 0.002316 | 0.003819 | 2.247e-16 | 1.722e-16 | 3 |
| one_hot_to_helmert | GramAnchor-m16 | 1 | 0.01009 | 0.01923 | 0.02025 | 2.847e-16 | 8.209e-17 | 3 |
| one_hot_to_helmert | RankAdaptiveGram | 1 | 0.0044 | 0.1335 | 0.1479 | 2.847e-16 | 8.209e-17 | 3 |
| one_hot_to_helmert | Raw | 0 | 0 | 0 | 0 | 2.847e-16 | 8.209e-17 | 3 |
| one_hot_to_helmert | Raw+GramAnchor@0.75 | 0.75 | -0.005355 | 0.01039 | 0.01214 | 2.847e-16 | 8.209e-17 | 3 |
| one_hot_to_helmert | SafeGram-t01 | 1 | -0.005355 | 0.01769 | 0.02025 | 2.847e-16 | 8.209e-17 | 3 |
| one_hot_to_helmert | SafeRankGram-t01 | 0.75 | 0 | 0.002364 | 0.002627 | 2.847e-16 | 8.209e-17 | 3 |

Unavailable pairs were not fabricated:

| dataset | pair | reason |
| --- | --- | --- |
| ozone-level-8hr | one_hot_to_helmert | no categorical block with at least three levels |
| ozone-level-8hr | fourier_origin_shift | no frozen cyclic metadata with an eight-coordinate block |
| wall-robot-navigation | one_hot_to_helmert | no categorical block with at least three levels |
| wall-robot-navigation | fourier_origin_shift | no frozen cyclic metadata with an eight-coordinate block |
| quake | one_hot_to_helmert | no categorical block with at least three levels |
| quake | fourier_origin_shift | no frozen cyclic metadata with an eight-coordinate block |
| visualizing_soil | one_hot_to_helmert | no categorical block with at least three levels |
| visualizing_soil | fourier_origin_shift | no frozen cyclic metadata with an eight-coordinate block |
| CPMP-2015-regression | fourier_origin_shift | no frozen cyclic metadata with an eight-coordinate block |

## 21. Strongest Positive Result

On the untouched panel, SafeGram-t01 achieved 37.5% median control with median/p95/max C=0.0000/0.0159/0.0276 and a 35.0% raw fallback rate; SafeRankGram reached 50.0% control with p95/max C=0.0051/0.0096. Inside numerical embeddings, SafeGram attained 56.3% median control with p95 C=0.0006 across three backbones. Natural-pair coordinate checks remain below 1e-6.

## 22. Strongest Negative Result

Pure Gram and pure RankAdaptiveGram remain tail-unsafe: prospective p95/max C are 0.1377/0.1904 and 0.1916/856237.6444, respectively. The strongest safely gated method may still fall below the desired 70% median control target. The descriptor gate reproduced a catastrophic max C and was discarded.

## 23. Does Adaptive Gating Actually Solve Tail Risk?

PARTLY

SafeGram p95/max C=0.0159/0.0276; SafeRank p95/max C=0.0051/0.0096. Their fallback rates are 35.0% and 26.7%. Gating prevents the fixed-interface catastrophes, but its median control must still be judged against the 70% target and its raw fallback burden.

## 24. Is RankAdaptiveGram Better Than Fixed m=16?

PARTLY

Rank adaptation materially reduces coordinate count and preserves empirical information, but pure Rank has p95/max C=0.1916/856237.6444 versus fixed Gram 0.1377/0.1904. Its value is interface minimality, not a standalone tail-risk cure.

## 25. Does the Phenomenon Exist Inside Standard Numerical Embeddings?

YES

PLE/RBF original-versus-rotated models have median disagreement 0.0994 across MLP, TabM-D, and ResNet. Gram control removes this basis dependence; safe hybrids preserve some helpful raw-view bias.

## 26. Recommended Paper Method Candidates

| rank | method | median invariance | median_C | p95_C | model breadth | embedding success | complexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GramAnchor-m16 | 1 | 0.007005 | 0.1377 | 0 | YES | 1 invariant fit |
| 2 | Raw+GramAnchor@0.75 | 0.75 | 0.001078 | 0.04975 | 2 | not gated | 2 fits |
| 3 | SafeRankGram-t01 | 0.5 | 0 | 0.005105 | 1 | YES | 2 fits + rank diagnostics + gate |

These are ranked recommendations, not an automatic final method selection.

## 27. Reviewer Attack Audit

### "Median performance hides catastrophic failures."

The primary table includes p90, p95, maximum C, every cell, and a denominator-sensitive reanalysis. Safety-First eligibility explicitly uses all three tail gates.

### "The gate is just validation overfitting."

Alpha uses only validation rows, is frozen before test evaluation, applies one fixed monotone rule, and is tested on ten untouched datasets. The more flexible descriptor gate failed leave-one-dataset-out validation and was discarded.

### "The invariant representation throws away information."

Diagnostic least-squares reconstruction is below 1e-4 for selected rank coordinates and near machine precision for fixed m=16 in the worst failures. Those failures are Type C, not Type A.

### "Why not simply use the original scalar feature?"

The experiment targets standard multidimensional numerical embeddings. Returning to a scalar discards the embedding architecture and does not address categorical or cyclic basis choices.

### "Random rotations are artificial."

Local/spectral hats and one-hot/Helmert are natural exact-equivalence pairs; Fourier-origin is reported only when cyclic metadata is available. The phenomenon also appears inside PLE/RBF pipelines.

### "The method doubles inference cost."

Prediction hybrids require raw and invariant branches. The report exposes fit-time overhead; a later shared-backbone/distillation test is required before an efficiency claim.

### "The method just falls back to raw everywhere."

All five alpha frequencies and exact alpha=0 rates are reported. Fallback is substantial where necessary, and this burden directly determines the SAFE-BUT-TOO-CONSERVATIVE verdict when median control is below 40%.

### "This is only relevant to handcrafted preprocessing."

PLE and RBF embeddings were rotated before three standard backbones, and invariant interfaces were inserted at the embedding-to-backbone boundary.

## 28. Recommended Next Experiment for Top-3

1. **GramAnchor-m16:** Test architecture-specific regularization that targets the Type-C generalization shift without sacrificing exact invariance.
2. **Raw+GramAnchor@0.75:** Run a larger tail-focused external panel; one catastrophic C>0.20 should kill the fixed-alpha paper method.
3. **SafeRankGram-t01:** Test whether blockwise, feature-specific alpha can raise median control above 40% without violating the current p95/max gates.

## 29. Files Produced

- `results.md` — this report
- `configs/NEW_TAIL_PROSPECTIVE_PANEL.json` and SHA256 — untouched-panel lock
- `configs/SAFE_BASIS_PROTOCOL.json` and SHA256 — frozen protocol
- `configs/TAIL_FINALISTS.json` and SHA256 — finalist freeze
- `results/raw/` — prediction bundles, telemetry, gate evidence, and coordinate audits
- `results/processed/` — development, diagnostic, embedding, prospective, ranking, and audit tables
- `figures/` — eight critical figures in PNG and PDF
