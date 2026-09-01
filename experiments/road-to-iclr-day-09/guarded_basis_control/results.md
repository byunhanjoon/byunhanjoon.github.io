# Guarded Basis Control — Final Method Search

## Executive Verdict
FINAL-METHOD-SIGNAL

## One-Paragraph Summary

The locked prospective panel supports the verdict **FINAL-METHOD-SIGNAL**. The strongest frozen candidate by prospective median control was `GuardedGram-G2-g0p0-t01` (0.7500 control; median/p95/max C = -0.0026/0.0175/0.2469), while the most adverse finalist p95 was `Raw+Gram@0.75` at 0.0872. These are method rankings and diagnostic classifications, not an automatic choice of the paper's final method.

## Frozen Protocol

- commit: `0c456660ae9a87aab7932b569e1954b0ee1d25fe`
- hardware: NVIDIA H100 NVL, NVIDIA H100 NVL, 95830 MiB each
- packages: numpy=1.26.4, pandas=2.3.3, scikit-learn=1.4.2, torch=2.7.0, catboost=1.2.10, matplotlib=3.10.9
- seeds: development [0, 1]; prospective [0, 1, 2]
- development datasets: steel-plates-fault, cardiotocography, sylvine, debutanizer, wall-robot-navigation, california_housing, phoneme, space-ga
- untouched prospective datasets: mfeat-fourier, waveform-5000, JapaneseVowels, ada_prior, eye_movements, mozilla4, 2dplanes, puma8NH, bank32nh, OnlineNewsPopularity, SoilKsatDB, Brazilian_houses
- finalist config SHA256: `6c01ba331417454a3907707d19abf937d593b6e96fa9de909de2cf01bab144dc`

Two target-independent loader adapters were recorded after the freeze without changing the panel or finalist configuration: SoilKsatDB retains its 8703 rows with observed `ksat_lab` and drops 4369 missing-target rows before the frozen split; 2dplanes treats its ten 2--3-level numeric inputs as low-rank RBF blocks because the global minimum-unique rule would otherwise yield no transformable block. The exact audit is saved in `prospective_loader_audit.json`; neither adapter used validation/test outcomes.

## 1. Prior Evidence Treated as Fixed

Pure Gram exact invariance, the fixed `.75` control–task tradeoff, SafeRank safety behavior, prior Type-C failures, and numerical-embedding sensitivity were treated as frozen evidence. Natural-pair reuse passed the exact-equivalence threshold: reconstruction 2.847e-16, Gram 1.611e-16, Rank 1.722e-16; Fourier origin shift was unavailable under frozen metadata.

## 2. GuardedGram Development

| method | median_alpha | median_disagreement_reduction | median_C | p95_C | max_C | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- |
| GuardedGram-G1-t0 | 0.7500 | 0.7500 | 0.0000 | 0.0257 | 0.0280 | 0.2750 |
| GuardedGram-G2-g0p0-t01 | 0.7500 | 0.7500 | -0.0001 | 0.0107 | 0.0280 | 0.1000 |
| GuardedGram-G3 | 0.6250 | 0.6250 | -0.0001 | 0.0151 | 0.0217 | 0.0750 |

### G1 Harm Detection

G1 tests one-sided bootstrap evidence of harm and falls back recursively; its frozen Stage-1 choice was evaluated without retuning on all eight development datasets.

### G2 Confidence Guard

G2 chooses the largest descending alpha satisfying `C_hat + gamma SE <= tau`; the frozen setting is `gamma=0`, `tau=.01`.

### G3 Two-Stage Guard

G3 uses 80% upper/lower bounds around the `.75` candidate and an ambiguous `.5` branch. It was retained as an ablation, not a finalist.

## 3. GuardedGram Ablations

| method | median_disagreement_reduction | p95_C | max_C | mean_predictive_rank |
| --- | --- | --- | --- | --- |
| GuardedGram-G1-t0 | 0.7500 | 0.0114 | 0.0280 | 10.8500 |
| GuardedGram-G1-t005 | 0.7500 | 0.0114 | 0.0280 | 12.4500 |
| GuardedGram-G1-t01 | 0.7500 | 0.0114 | 0.0280 | 13.5000 |
| GuardedGram-G2-g0p0-t005 | 0.7500 | 0.0107 | 0.0150 | 10.8750 |
| GuardedGram-G2-g0p0-t01 | 0.7500 | 0.0114 | 0.0280 | 11.7500 |
| GuardedGram-G2-g0p0-t02 | 0.7500 | 0.0114 | 0.0280 | 12.9500 |
| GuardedGram-G2-g0p5-t005 | 0.6250 | 0.0044 | 0.0058 | 9.8250 |
| GuardedGram-G2-g0p5-t01 | 0.7500 | 0.0107 | 0.0150 | 10.1750 |
| GuardedGram-G2-g0p5-t02 | 0.7500 | 0.0114 | 0.0280 | 12.7500 |
| GuardedGram-G2-g1p0-t005 | 0.4375 | 0.0014 | 0.0034 | 9.7250 |
| GuardedGram-G2-g1p0-t01 | 0.6250 | 0.0044 | 0.0058 | 9.3750 |
| GuardedGram-G2-g1p0-t02 | 0.7500 | 0.0107 | 0.0150 | 11.2500 |
| GuardedGram-G2-g1p64-t005 | 0.2500 | 0.0014 | 0.0034 | 11.6000 |
| GuardedGram-G2-g1p64-t01 | 0.4375 | 0.0035 | 0.0058 | 11.2000 |
| GuardedGram-G2-g1p64-t02 | 0.6250 | 0.0060 | 0.0097 | 9.9250 |
| GuardedGram-G3 | 0.6250 | 0.0100 | 0.0150 | 10.2500 |

## 4. BlockGuard

| method | median_invariant_feature_fraction | median_disagreement_reduction | median_C | p95_C | max_C | inference_multiplier |
| --- | --- | --- | --- | --- | --- | --- |
| BlockGuard-Greedy-t01 | 0.7009 | 0.5330 | 0.0003 | 0.0332 | 0.0702 | 1.0000 |
| BlockGuard-Grouped-t01 | 0.9643 | 0.8961 | 0.0004 | 0.0247 | 0.1307 | 1.0000 |

Grouped failed Stage 1 and is reported as a pruned ablation. Greedy uses exact one-block retraining, benefit/cost ordering, at most eight cumulative stages, and one model at inference.

## 5. Which Features Stay Raw?

| status | features | median_empirical_rank | median_block_dimension | median_spectrum_entropy | median_condition_proxy | median_one_block_C | median_orbit_benefit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gram-selected | 755 | 8.0000 | 8.0000 | 1.0240 | 126.0489 | -0.0002 | 0.0107 |
| raw-retained | 375 | 8.0000 | 8.0000 | 1.0084 | 141.6042 | 0.0110 | 0.0252 |

The table contrasts Gram-selected and raw-retained blocks using empirical rank, spectrum entropy, dimension, condition proxy, one-block validation C, and measured orbit benefit; it is descriptive and not an extra selection rule.

## 6. DualViewGram

| method | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | median_parameter_count | median_inference_seconds | max_peak_gpu_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DualViewGram-D1-a75 | 0.3821 | 0.0423 | 0.1782 | 0.2034 | 1.0000 | 468597.5000 | 0.0024 | 312541184.0000 |

DualView fixed `.75` was pruned at Stage 1, so D2–D4 were not promoted to the full panel.

## 7. Efficiency Comparison

| method | scope | median_disagreement_reduction | inference_multiplier | training_multiplier | parameter_multiplier | parameter_count |
| --- | --- | --- | --- | --- | --- | --- |
| BlockGuard-Greedy-t01 | general | 0.7219 | 1.0000 | 23.5000 | 1.0000 | — |
| GuardedGram-G2-g0p0-t01 | general | 0.7500 | 2.0000 | 2.0000 | 2.0000 | — |
| Raw | general | 0.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| Raw+Gram@0.75 | general | 0.7500 | 2.0000 | 2.0000 | 2.0000 | — |
| DualViewGram-D1-a75 | development (pruned) | 0.3821 | 1.0000 | 1.0000 | — | 468597.5000 |

Raw and single-representation BlockGuard use one inference model. Fractional Raw+Gram, GuardedGram, and Safe references require two prediction branches; pure endpoint selections require one. BlockGuard's training multiplier is the median count of exact one-block interventions plus cumulative candidates per prospective unit. It is a lower-bound full-fit accounting measure—not a wall-clock claim—because orbit/reference fits and cached representation work add further cost.

## 8. Numerical Embedding Confirmation

| dataset | model | embedding | k | default_vs_rotated_disagreement | task_span |
| --- | --- | --- | --- | --- | --- |
| california_housing | controlled_mlp | PLE | 4 | 0.1197 | 354.3331 |
| california_housing | controlled_mlp | PLE | 8 | 0.1192 | 564.8224 |
| california_housing | controlled_mlp | PLE | 16 | 0.1272 | 200.1743 |
| california_housing | controlled_mlp | PLE | 32 | 0.1406 | 12.2257 |
| california_housing | controlled_mlp | RBF | 4 | 0.0701 | 9.7841 |
| california_housing | controlled_mlp | RBF | 8 | 0.1162 | 261.8659 |
| california_housing | controlled_mlp | RBF | 16 | 0.1540 | 757.9826 |
| california_housing | controlled_mlp | RBF | 32 | 0.1946 | 1526.4601 |
| california_housing | resnet_tabular | PLE | 4 | 0.2025 | 1161.1988 |
| california_housing | resnet_tabular | PLE | 8 | 0.2241 | 2062.1801 |
| california_housing | resnet_tabular | PLE | 16 | 0.2043 | 100.5523 |
| california_housing | resnet_tabular | PLE | 32 | 0.2059 | 1179.9725 |
| california_housing | resnet_tabular | RBF | 4 | 0.1800 | 883.8163 |
| california_housing | resnet_tabular | RBF | 8 | 0.2251 | 1156.8338 |
| california_housing | resnet_tabular | RBF | 16 | 0.2764 | 1804.3583 |
| california_housing | resnet_tabular | RBF | 32 | 0.2693 | 810.8693 |
| california_housing | tabm_d | PLE | 4 | 0.2015 | 741.4158 |
| california_housing | tabm_d | PLE | 8 | 0.2185 | 149.8384 |
| california_housing | tabm_d | PLE | 16 | 0.1907 | 63.3601 |
| california_housing | tabm_d | PLE | 32 | 0.1772 | 195.2218 |
| california_housing | tabm_d | RBF | 4 | 0.1424 | 194.3317 |
| california_housing | tabm_d | RBF | 8 | 0.1730 | 297.8027 |
| california_housing | tabm_d | RBF | 16 | 0.2236 | 217.0098 |
| california_housing | tabm_d | RBF | 32 | 0.2427 | 48.7224 |
| cardiotocography | controlled_mlp | PLE | 4 | 0.0047 | 0.0016 |
| cardiotocography | controlled_mlp | PLE | 8 | 0.0095 | 0.0059 |
| cardiotocography | controlled_mlp | PLE | 16 | 0.0220 | 0.0085 |
| cardiotocography | controlled_mlp | PLE | 32 | 0.0237 | 0.0022 |
| cardiotocography | controlled_mlp | RBF | 4 | 0.0045 | 0.0020 |
| cardiotocography | controlled_mlp | RBF | 8 | 0.0133 | 0.0168 |
| cardiotocography | controlled_mlp | RBF | 16 | 0.0166 | 0.0036 |
| cardiotocography | controlled_mlp | RBF | 32 | 0.0256 | 0.0297 |
| cardiotocography | resnet_tabular | PLE | 4 | 0.0003 | 0.0003 |
| cardiotocography | resnet_tabular | PLE | 8 | 0.0108 | 0.0026 |
| cardiotocography | resnet_tabular | PLE | 16 | 0.0338 | 0.0083 |
| cardiotocography | resnet_tabular | PLE | 32 | 0.0478 | 0.0135 |
| cardiotocography | resnet_tabular | RBF | 4 | 0.0006 | 0.0001 |
| cardiotocography | resnet_tabular | RBF | 8 | 0.0167 | 0.0073 |
| cardiotocography | resnet_tabular | RBF | 16 | 0.0286 | 0.0183 |
| cardiotocography | resnet_tabular | RBF | 32 | 0.0468 | 0.0465 |
| cardiotocography | tabm_d | PLE | 4 | 0.0096 | 0.0025 |
| cardiotocography | tabm_d | PLE | 8 | 0.0097 | 0.0043 |
| cardiotocography | tabm_d | PLE | 16 | 0.0182 | 0.0064 |
| cardiotocography | tabm_d | PLE | 32 | 0.0193 | 0.0080 |
| cardiotocography | tabm_d | RBF | 4 | 0.0051 | 0.0004 |
| cardiotocography | tabm_d | RBF | 8 | 0.0124 | 0.0061 |
| cardiotocography | tabm_d | RBF | 16 | 0.0154 | 0.0063 |
| cardiotocography | tabm_d | RBF | 32 | 0.0197 | 0.0124 |

## 9. Embedding Dimension Scaling

| embedding | model | k | median_disagreement |
| --- | --- | --- | --- |
| PLE | controlled_mlp | 4 | 0.0271 |
| PLE | controlled_mlp | 8 | 0.0418 |
| PLE | controlled_mlp | 16 | 0.0574 |
| PLE | controlled_mlp | 32 | 0.0964 |
| PLE | resnet_tabular | 4 | 0.0853 |
| PLE | resnet_tabular | 8 | 0.1086 |
| PLE | resnet_tabular | 16 | 0.1250 |
| PLE | resnet_tabular | 32 | 0.1231 |
| PLE | tabm_d | 4 | 0.0729 |
| PLE | tabm_d | 8 | 0.0963 |
| PLE | tabm_d | 16 | 0.0768 |
| PLE | tabm_d | 32 | 0.0910 |
| RBF | controlled_mlp | 4 | 0.0240 |
| RBF | controlled_mlp | 8 | 0.0310 |
| RBF | controlled_mlp | 16 | 0.0853 |
| RBF | controlled_mlp | 32 | 0.1494 |
| RBF | resnet_tabular | 4 | 0.0510 |
| RBF | resnet_tabular | 8 | 0.0916 |
| RBF | resnet_tabular | 16 | 0.1245 |
| RBF | resnet_tabular | 32 | 0.1888 |
| RBF | tabm_d | 4 | 0.0772 |
| RBF | tabm_d | 8 | 0.0859 |
| RBF | tabm_d | 16 | 0.0773 |
| RBF | tabm_d | 32 | 0.1187 |

The fitted median log2-dimension slopes are:

| embedding | model | median_log2_dimension_slope |
| --- | --- | --- |
| PLE | controlled_mlp | 0.0078 |
| PLE | resnet_tabular | 0.0093 |
| PLE | tabm_d | -0.0011 |
| RBF | controlled_mlp | 0.0384 |
| RBF | resnet_tabular | 0.0235 |
| RBF | tabm_d | 0.0166 |

## 10. Gram/Guard Methods Inside Embeddings

| method | median_disagreement_reduction | median_C | p95_C | max_C |
| --- | --- | --- | --- | --- |
| Gram-after-embedding | 1.0000 | 0.0010 | 0.1333 | 0.1904 |
| GuardedGram-G2-after-embedding | 0.7500 | -0.0151 | 0.0217 | 0.0443 |
| Raw embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BlockGuard-Greedy-t01-transferred | 0.3798 | 0.0031 | 0.1148 | 0.1870 |

Transferred BlockGuard is included where applicable and is explicitly labeled as a portability test, not embedding-specific retuning.

## 11. Basis Portfolio / Basis Search

The optional portfolio was not run because core confirmation and the prospective freeze took priority. Default-basis best-test frequency was 0.2708; validation selection beat default in 0.4167 of cells, versus oracle headroom in 0.7292.

## 12. Stage-1 Pruning

| method | median_disagreement_reduction | median_C | p95_C | max_C | fallback_rate |
| --- | --- | --- | --- | --- | --- |
| BlockGuard-Greedy-t0 | 0.0665 | 0.0000 | 0.0210 | 0.0322 | 0.4000 |
| BlockGuard-Greedy-t005 | 0.3790 | 0.0005 | 0.0234 | 0.0322 | 0.1000 |
| BlockGuard-Greedy-t01 | 0.6417 | 0.0005 | 0.0235 | 0.0324 | 0.1000 |
| BlockGuard-Greedy-t02 | 1.0000 | 0.0002 | 0.0332 | 0.0489 | 0.1000 |
| BlockGuard-Grouped-t0 | 0.4848 | 4.143e-07 | 0.0262 | 0.1307 | 0.2500 |
| BlockGuard-Grouped-t005 | 0.6884 | 0.0002 | 0.0262 | 0.1307 | 0.1000 |
| BlockGuard-Grouped-t01 | 1.0000 | 0.0005 | 0.0471 | 0.1307 | 0.1000 |
| BlockGuard-Grouped-t02 | 1.0000 | 0.0005 | 0.0496 | 0.1307 | 0.1000 |
| DualViewGram-D1-a75 | 0.3821 | 0.0423 | 0.1782 | 0.2034 | — |

GuardedGram G1/G2/G3 and BlockGuard-Greedy survived. BlockGuard-Grouped and DualView fixed `.75` failed the prescribed gates; their negative results remain visible.

## 13. Full Development Ranking

| method | experiment | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank |
| --- | --- | --- | --- | --- | --- | --- |
| PureGram | general | 1.0000 | 0.0045 | 0.1686 | 0.6336 | 8.1000 |
| BlockGuard-Grouped-t01 | BlockGuard | 0.8961 | 0.0004 | 0.0247 | 0.1307 | 1.4500 |
| Raw+Gram@0.75 | general | 0.7500 | 0.0003 | 0.0873 | 0.1598 | 5.9750 |
| GuardedGram-G1-t0 | general | 0.7500 | 0.0000 | 0.0257 | 0.0280 | 4.5250 |
| GuardedGram-G2-g0p0-t01 | general | 0.7500 | -0.0001 | 0.0107 | 0.0280 | 4.5625 |
| GuardedGram-G3 | general | 0.6250 | -0.0001 | 0.0151 | 0.0217 | 4.5000 |
| BlockGuard-Greedy-t01 | BlockGuard | 0.5330 | 0.0003 | 0.0332 | 0.0702 | 1.5500 |
| Raw+Gram@0.5 | general | 0.5000 | -7.174e-05 | 0.0524 | 0.0861 | 4.4125 |
| SafeRankGram-t01 | general | 0.5000 | -0.0016 | 0.0050 | 0.0064 | 3.3625 |
| DualViewGram-D1-a75 | DualView-stage1 | 0.3821 | 0.0423 | 0.1782 | 0.2034 | 1.0000 |
| SafeGram-t01 | general | 0.3750 | 0.0000 | 0.0047 | 0.0058 | 4.5875 |
| Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9750 |

## 14. Frozen Finalists

| method | scope | threshold | confidence_level | embedding_setting | architecture |
| --- | --- | --- | --- | --- | --- |
| Raw+Gram@0.75 | general | none | none | general RBF feature blocks, k=8, frozen split and scaling | shared_training_settings.architecture across five general model families |
| GuardedGram-G2-g0p0-t01 | general | 0.0100 | gamma=0.0 bootstrap-SE margin; 1000 paired row-bootstrap resamples | general RBF feature blocks, k=8, frozen split and scaling | shared_training_settings.architecture across five general model families |
| BlockGuard-Greedy-t01 | general | 0.0100 | none; exact validation normalized excess risk | general RBF feature blocks, k=8; selected blocks replaced by GramAnchor-16 coordinates | shared_training_settings.architecture across five general model families; one selected representation and one model at inference |
| GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.0100 | gamma=0.0 bootstrap-SE margin; 1000 paired row-bootstrap resamples | RBF numerical embedding, k=16, Gram interface between numerical embedding and backbone | controlled MLP, TabM-D, and ResNet-style tabular backbone |

Exactly 4 configurations were frozen before any prospective outcomes, under SHA `6c01ba331417454a3907707d19abf937d593b6e96fa9de909de2cf01bab144dc`.

## 15. NEW Untouched Prospective Results

All rows below use the locked datasets, seeds, model families, and finalists. The target-independent SoilKsatDB/2dplanes loader adaptations disclosed under Frozen Protocol are the only runtime data adapters.

| dataset | model | method | selected_alpha | invariant_feature_fraction | control | raw_task_error | method_task_error | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2dplanes | catboost | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 1.0528 | 1.0448 | -0.0024 |
| 2dplanes | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 1.0528 | 1.0446 | -0.0025 |
| 2dplanes | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 1.0528 | 1.0446 | -0.0025 |
| 2dplanes | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.7000 | 0.1429 | 0.9826 | 1.0042 | 0.0083 |
| 2dplanes | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.9826 | 1.0118 | 0.0096 |
| 2dplanes | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9826 | 1.0118 | 0.0096 |
| 2dplanes | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.9847 | 0.9896 | 0.0014 |
| 2dplanes | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.9847 | 0.9868 | 0.0006 |
| 2dplanes | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9847 | 0.9868 | 0.0006 |
| 2dplanes | tabm_d | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 1.0378 | 1.0183 | -0.0022 |
| 2dplanes | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 1.0378 | 1.0194 | -0.0029 |
| 2dplanes | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 1.0378 | 1.0194 | -0.0029 |
| 2dplanes | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.9646 | 0.9730 | 0.0025 |
| 2dplanes | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.9646 | 0.9678 | 0.0009 |
| 2dplanes | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9646 | 0.9678 | 0.0009 |
| Brazilian_houses | catboost | BlockGuard-Greedy-t01 | — | 0.2000 | 0.0000 | 959.1540 | 1045.9244 | 0.0049 |
| Brazilian_houses | catboost | GuardedGram-G2-g0p0-t01 | 0.0000 | — | 0.0000 | 959.1540 | 959.1540 | 0.0000 |
| Brazilian_houses | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 959.1540 | 1256.3260 | 0.0788 |
| Brazilian_houses | controlled_mlp | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 2954.6903 | 1140.2453 | -0.9722 |
| Brazilian_houses | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 2954.6903 | 1294.1192 | -0.9144 |
| Brazilian_houses | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 2954.6903 | 1294.1192 | -0.9144 |
| Brazilian_houses | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 967.3993 | 960.4297 | -0.0019 |
| Brazilian_houses | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 967.3993 | 956.9510 | -0.0028 |
| Brazilian_houses | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 967.3993 | 956.9510 | -0.0028 |
| Brazilian_houses | tabm_d | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 2320.5164 | 2294.3195 | -0.0109 |
| Brazilian_houses | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 2320.5164 | 2226.9891 | -0.0539 |
| Brazilian_houses | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 2320.5164 | 2226.9891 | -0.0539 |
| Brazilian_houses | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 746.8160 | 743.6992 | -0.0053 |
| Brazilian_houses | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 746.8160 | 756.1259 | 0.0029 |
| Brazilian_houses | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 746.8160 | 756.1259 | 0.0113 |
| JapaneseVowels | catboost | BlockGuard-Greedy-t01 | — | 0.8571 | 0.6111 | 0.3938 | 0.4122 | 0.0106 |
| JapaneseVowels | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3938 | 0.4062 | 0.0069 |
| JapaneseVowels | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3938 | 0.4062 | 0.0069 |
| JapaneseVowels | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.2857 | 0.1924 | 0.3848 | 0.3749 | -0.0055 |
| JapaneseVowels | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3848 | 0.3109 | -0.0418 |
| JapaneseVowels | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3848 | 0.3109 | -0.0418 |
| JapaneseVowels | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.1260 | 0.1198 | -0.0030 |
| JapaneseVowels | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.1260 | 0.1161 | -0.0048 |
| JapaneseVowels | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.1260 | 0.1161 | -0.0048 |
| JapaneseVowels | tabm_d | BlockGuard-Greedy-t01 | — | 0.1429 | 0.2040 | 0.1752 | 0.2006 | 0.0016 |
| JapaneseVowels | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.1752 | 0.2032 | -0.0050 |
| JapaneseVowels | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.1752 | 0.2032 | -0.0050 |
| JapaneseVowels | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.1097 | 0.1199 | -0.0011 |
| JapaneseVowels | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.1097 | 0.1133 | -0.0044 |
| JapaneseVowels | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.1097 | 0.1133 | -0.0044 |
| OnlineNewsPopularity | catboost | BlockGuard-Greedy-t01 | — | 0.1463 | -0.0469 | 5254.3235 | 5254.0735 | 0.0000 |
| OnlineNewsPopularity | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 5254.3235 | 5250.5999 | 0.2469 |
| OnlineNewsPopularity | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 5254.3235 | 5250.5999 | 0.2469 |
| OnlineNewsPopularity | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.7561 | 0.7201 | 5161.0207 | 5396.2246 | 2.1635 |
| OnlineNewsPopularity | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.2500 | — | 0.2500 | 5161.0207 | 5177.1024 | 0.0000 |
| OnlineNewsPopularity | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 5161.0207 | 5247.4939 | 0.8473 |
| OnlineNewsPopularity | tabicl_v2 | BlockGuard-Greedy-t01 | — | 0.2683 | 0.2681 | 5203.2604 | 5194.3081 | -0.1456 |
| OnlineNewsPopularity | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.0000 | — | 0.0000 | 5203.2604 | 5203.2604 | 0.0000 |
| OnlineNewsPopularity | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 5203.2604 | 5354.4220 | 2.4583 |
| OnlineNewsPopularity | tabm_d | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 5184.1102 | 5165.6616 | -0.2136 |
| OnlineNewsPopularity | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 5184.1102 | 5167.2108 | -0.2152 |
| OnlineNewsPopularity | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 5184.1102 | 5167.2108 | -0.2152 |
| OnlineNewsPopularity | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 0.1463 | 0.0711 | 5141.9363 | 5149.7540 | 0.0000 |
| OnlineNewsPopularity | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 5141.9363 | 5127.7464 | -0.1155 |
| OnlineNewsPopularity | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 5141.9363 | 5131.7657 | -0.1155 |
| SoilKsatDB | catboost | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 971.4365 | 971.4365 | 0.0000 |
| SoilKsatDB | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 971.4365 | 963.9668 | -0.0421 |
| SoilKsatDB | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 971.4365 | 963.9668 | -0.0421 |
| SoilKsatDB | controlled_mlp | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 992.1005 | 961.6919 | -0.1941 |
| SoilKsatDB | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 992.1005 | 961.5470 | -0.1950 |
| SoilKsatDB | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 992.1005 | 961.5470 | -0.1950 |
| SoilKsatDB | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 1029.1521 | 955.8114 | -0.6132 |
| SoilKsatDB | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 1029.1521 | 967.3468 | -0.5167 |
| SoilKsatDB | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 1029.1521 | 967.3468 | -0.5167 |
| SoilKsatDB | tabm_d | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 935.8067 | 935.8067 | 0.0000 |
| SoilKsatDB | tabm_d | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 935.8067 | 929.3271 | -0.0185 |
| SoilKsatDB | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 935.8067 | 934.3822 | 0.0085 |
| SoilKsatDB | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 0.1333 | 0.0523 | 992.3641 | 944.5787 | -0.1218 |
| SoilKsatDB | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 992.3641 | 954.9397 | -0.3134 |
| SoilKsatDB | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 992.3641 | 954.9397 | -0.3134 |
| ada_prior | catboost | BlockGuard-Greedy-t01 | — | 0.5000 | 0.0978 | 0.3784 | 0.3832 | 0.0033 |
| ada_prior | catboost | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.3784 | 0.3791 | -0.0071 |
| ada_prior | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3784 | 0.3813 | -0.0071 |
| ada_prior | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.8333 | 0.3661 | 0.3682 | 0.4085 | 0.2152 |
| ada_prior | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.3682 | 0.3694 | 0.0188 |
| ada_prior | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3682 | 0.3767 | 0.0562 |
| ada_prior | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.3601 | 0.3663 | 0.0309 |
| ada_prior | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3601 | 0.3636 | 0.0174 |
| ada_prior | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3601 | 0.3636 | 0.0174 |
| ada_prior | tabm_d | BlockGuard-Greedy-t01 | — | 0.1667 | 0.0581 | 0.3649 | 0.3615 | -0.0183 |
| ada_prior | tabm_d | GuardedGram-G2-g0p0-t01 | 0.0000 | — | 0.0000 | 0.3649 | 0.3674 | 0.0000 |
| ada_prior | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3649 | 0.3755 | 0.0543 |
| ada_prior | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 0.5000 | 0.4283 | 0.3682 | 0.3646 | -0.0015 |
| ada_prior | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.2500 | — | 0.2500 | 0.3682 | 0.3664 | -0.0047 |
| ada_prior | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3682 | 0.3684 | 0.0024 |
| bank32nh | catboost | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 0.0862 | 0.0862 | 0.0000 |
| bank32nh | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.0862 | 0.0870 | 0.0200 |
| bank32nh | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.0862 | 0.0870 | 0.0200 |
| bank32nh | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.1429 | 0.0000 | 0.0932 | 0.0996 | 0.1766 |
| bank32nh | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.0932 | 0.0914 | -0.0681 |
| bank32nh | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.0932 | 0.0928 | -0.0237 |
| bank32nh | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.0844 | 0.0857 | 0.0354 |
| bank32nh | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.0844 | 0.0849 | 0.0144 |
| bank32nh | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.0844 | 0.0849 | 0.0144 |
| bank32nh | tabm_d | BlockGuard-Greedy-t01 | — | 0.8929 | 0.7237 | 0.0891 | 0.0902 | 0.0408 |
| bank32nh | tabm_d | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.0891 | 0.0892 | 0.0036 |
| bank32nh | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.0891 | 0.0897 | 0.0237 |
| bank32nh | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.0809 | 0.0799 | -0.0242 |
| bank32nh | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.0809 | 0.0800 | -0.0266 |
| bank32nh | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.0809 | 0.0800 | -0.0266 |
| eye_movements | catboost | BlockGuard-Greedy-t01 | — | 0.4444 | 0.0091 | 0.9492 | 0.9507 | 0.0333 |
| eye_movements | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.9492 | 0.9514 | 0.0125 |
| eye_movements | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9492 | 0.9514 | 0.0125 |
| eye_movements | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 0.9996 | 0.9902 | 0.0000 |
| eye_movements | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.9996 | 0.9583 | -0.3346 |
| eye_movements | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9996 | 0.9721 | -0.2007 |
| eye_movements | tabicl_v2 | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 0.9544 | 0.9544 | 0.0000 |
| eye_movements | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.9544 | 0.9406 | -0.1052 |
| eye_movements | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9544 | 0.9406 | -0.1052 |
| eye_movements | tabm_d | BlockGuard-Greedy-t01 | — | 0.1667 | 0.0975 | 0.9462 | 0.9462 | -0.0275 |
| eye_movements | tabm_d | GuardedGram-G2-g0p0-t01 | 0.0000 | — | 0.0000 | 0.9462 | 0.9462 | 0.0000 |
| eye_movements | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.9462 | 0.9772 | -0.0123 |
| eye_movements | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 0.1667 | 0.0000 | 0.8948 | 0.8868 | -0.0217 |
| eye_movements | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.8948 | 0.8893 | -0.0284 |
| eye_movements | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.8948 | 0.8899 | -0.0188 |
| mfeat-fourier | catboost | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.5763 | 0.5849 | 0.0049 |
| mfeat-fourier | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.5763 | 0.5808 | 0.0026 |
| mfeat-fourier | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.5763 | 0.5808 | 0.0026 |
| mfeat-fourier | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.6447 | 0.8138 | 0.6248 | 0.6258 | -0.0062 |
| mfeat-fourier | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.6248 | 0.6168 | 0.0016 |
| mfeat-fourier | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.6248 | 0.6168 | 0.0016 |
| mfeat-fourier | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.3329 | 0.3274 | -0.0028 |
| mfeat-fourier | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3329 | 0.3208 | -0.0062 |
| mfeat-fourier | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3329 | 0.3208 | -0.0062 |
| mfeat-fourier | tabm_d | BlockGuard-Greedy-t01 | — | 0.5263 | 0.5974 | 0.5037 | 0.4889 | -0.0082 |
| mfeat-fourier | tabm_d | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 0.5037 | 0.4866 | -0.0050 |
| mfeat-fourier | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.5037 | 0.4876 | -0.0026 |
| mfeat-fourier | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.3617 | 0.3530 | -0.0034 |
| mfeat-fourier | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3617 | 0.3500 | -0.0051 |
| mfeat-fourier | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3617 | 0.3500 | -0.0051 |
| mozilla4 | catboost | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.1895 | 0.1931 | 0.0083 |
| mozilla4 | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.1895 | 0.1912 | 0.0041 |
| mozilla4 | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.1895 | 0.1912 | 0.0041 |
| mozilla4 | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.7500 | 0.5840 | 0.3007 | 0.3132 | 0.0371 |
| mozilla4 | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3007 | 0.3011 | 0.0012 |
| mozilla4 | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3007 | 0.3011 | 0.0012 |
| mozilla4 | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.2036 | 0.2096 | 0.0140 |
| mozilla4 | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.2036 | 0.2053 | 0.0039 |
| mozilla4 | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.2036 | 0.2053 | 0.0039 |
| mozilla4 | tabm_d | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.2639 | 0.2503 | -0.0270 |
| mozilla4 | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.2639 | 0.2489 | -0.0399 |
| mozilla4 | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.2639 | 0.2489 | -0.0399 |
| mozilla4 | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.1798 | 0.1832 | 0.0076 |
| mozilla4 | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.1798 | 0.1811 | 0.0029 |
| mozilla4 | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.1798 | 0.1811 | 0.0029 |
| puma8NH | catboost | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 3.1169 | 3.1130 | 0.0004 |
| puma8NH | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 3.1169 | 3.1025 | -0.0032 |
| puma8NH | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 3.1169 | 3.1025 | -0.0032 |
| puma8NH | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.8750 | 0.7396 | 3.2276 | 3.2365 | -0.0081 |
| puma8NH | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 3.2276 | 3.2476 | 0.0076 |
| puma8NH | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 3.2276 | 3.2930 | 0.0250 |
| puma8NH | tabicl_v2 | BlockGuard-Greedy-t01 | — | 0.8750 | 0.3909 | 2.9752 | 2.9755 | 0.0001 |
| puma8NH | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 2.9752 | 2.9692 | -0.0021 |
| puma8NH | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 2.9752 | 2.9692 | -0.0021 |
| puma8NH | tabm_d | BlockGuard-Greedy-t01 | — | 0.5000 | 0.5067 | 3.0457 | 3.0490 | 0.0012 |
| puma8NH | tabm_d | GuardedGram-G2-g0p0-t01 | 0.5000 | — | 0.5000 | 3.0457 | 3.0409 | -0.0003 |
| puma8NH | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 3.0457 | 3.0501 | 0.0030 |
| puma8NH | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 2.9643 | 2.9664 | 0.0083 |
| puma8NH | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 2.9643 | 2.9596 | 0.0055 |
| puma8NH | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 2.9643 | 2.9596 | 0.0055 |
| waveform-5000 | catboost | BlockGuard-Greedy-t01 | — | 0.8750 | 0.5389 | 0.3642 | 0.3621 | -0.0008 |
| waveform-5000 | catboost | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3642 | 0.3645 | -0.0029 |
| waveform-5000 | catboost | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3642 | 0.3645 | -0.0029 |
| waveform-5000 | controlled_mlp | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 0.4138 | 0.3877 | 0.0000 |
| waveform-5000 | controlled_mlp | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.4138 | 0.3924 | -0.0442 |
| waveform-5000 | controlled_mlp | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.4138 | 0.3924 | -0.0442 |
| waveform-5000 | tabicl_v2 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.2820 | 0.2988 | 0.0206 |
| waveform-5000 | tabicl_v2 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.2820 | 0.2906 | 0.0105 |
| waveform-5000 | tabicl_v2 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.2820 | 0.2906 | 0.0105 |
| waveform-5000 | tabm_d | BlockGuard-Greedy-t01 | — | 0.0000 | 0.0000 | 0.3292 | 0.3292 | 0.0000 |
| waveform-5000 | tabm_d | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3292 | 0.3291 | -0.0002 |
| waveform-5000 | tabm_d | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3292 | 0.3349 | 0.0032 |
| waveform-5000 | tabpfn_2_6 | BlockGuard-Greedy-t01 | — | 1.0000 | 1.0000 | 0.3073 | 0.3064 | -0.0013 |
| waveform-5000 | tabpfn_2_6 | GuardedGram-G2-g0p0-t01 | 0.7500 | — | 0.7500 | 0.3073 | 0.3053 | -0.0027 |
| waveform-5000 | tabpfn_2_6 | Raw+Gram@0.75 | 0.7500 | — | 0.7500 | 0.3073 | 0.3053 | -0.0027 |
| 2dplanes | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.0000 | 1.0288 | 1.0317 | 0.0019 |
| 2dplanes | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.0000 | 1.1142 | 1.0812 | -0.0080 |
| 2dplanes | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.0000 | 0.9802 | 0.9919 | 0.0035 |
| Brazilian_houses | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 3132.9684 | 1634.4607 | -0.7552 |
| Brazilian_houses | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.2500 | — | 0.2500 | 1447.7044 | 1392.8069 | -0.0114 |
| Brazilian_houses | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 2776.6041 | 2491.9577 | -0.1426 |
| JapaneseVowels | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.5478 | 0.4178 | -0.0656 |
| JapaneseVowels | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.4918 | 0.3499 | -0.0826 |
| JapaneseVowels | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.2620 | 0.2056 | -0.0296 |
| OnlineNewsPopularity | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.2500 | — | 0.2500 | 5185.3473 | 5145.3888 | -0.6109 |
| OnlineNewsPopularity | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.0000 | — | 0.0000 | 5130.3582 | 5149.7283 | 0.0000 |
| OnlineNewsPopularity | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 5158.4280 | 5147.4094 | -0.0247 |
| SoilKsatDB | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 1000.0366 | 994.8324 | -0.0487 |
| SoilKsatDB | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 1011.5905 | 999.7303 | -0.1799 |
| SoilKsatDB | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 978.0778 | 943.5767 | -0.2318 |
| ada_prior | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.5000 | — | 0.5000 | 0.3786 | 0.3775 | -0.0059 |
| ada_prior | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.3747 | 0.3733 | -0.0117 |
| ada_prior | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.0000 | — | 0.0000 | 0.3638 | 0.3638 | 0.0000 |
| bank32nh | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.1076 | 0.1051 | -0.1923 |
| bank32nh | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.1073 | 0.1039 | -0.2218 |
| bank32nh | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.0963 | 0.0937 | -0.1059 |
| eye_movements | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 1.0052 | 0.9861 | -0.1805 |
| eye_movements | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 1.0066 | 0.9775 | -0.2235 |
| eye_movements | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 1.0363 | 0.9960 | -0.3492 |
| mfeat-fourier | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.7840 | 0.7144 | -0.0317 |
| mfeat-fourier | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.7418 | 0.6474 | -0.0608 |
| mfeat-fourier | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.5835 | 0.5441 | -0.0229 |
| mozilla4 | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.2500 | — | 0.2500 | 0.3163 | 0.3173 | 0.0017 |
| mozilla4 | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.0000 | — | 0.0000 | 0.3321 | 0.3297 | 0.0000 |
| mozilla4 | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.2487 | 0.2367 | -0.0462 |
| puma8NH | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 3.3824 | 3.4227 | 0.0024 |
| puma8NH | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 3.6207 | 3.4969 | -0.0862 |
| puma8NH | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 3.1745 | 3.1423 | -0.0121 |
| waveform-5000 | controlled_mlp | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.4905 | 0.4908 | 0.0064 |
| waveform-5000 | resnet_tabular | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.4893 | 0.4510 | -0.0184 |
| waveform-5000 | tabm_d | GuardedGram-G2-after-RBF-k16 | 0.7500 | — | 0.7500 | 0.3747 | 0.3552 | -0.0276 |

## 16. Prospective Aggregate Table

| method | scope | median_disagreement_reduction | p25_disagreement_reduction | p75_disagreement_reduction | median_C | p90_C | p95_C | max_C | wins | ties | losses | fraction_C_lt_0 | fraction_C_gt_0p01 | fraction_C_gt_0p05 | fallback_rate | median_invariant_feature_fraction | mean_predictive_rank | inference_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BlockGuard-Greedy-t01 | general | 0.7219 | 0.0909 | 1.0000 | 0.0000 | 0.0335 | 0.0476 | 2.1635 | 27 | 6 | 27 | 0.4500 | 0.1833 | 0.0500 | 0.2667 | 0.8750 | 5.2583 | 1.0000 |
| GuardedGram-G2-g0p0-t01 | general | 0.7500 | 0.5000 | 0.7500 | -0.0026 | 0.0107 | 0.0175 | 0.2469 | 33 | 3 | 24 | 0.5667 | 0.1167 | 0.0167 | 0.1333 | — | 3.7250 | 2.0000 |
| PureGram | general | 1.0000 | 1.0000 | 1.0000 | 0.0069 | 0.1227 | 0.1808 | 4.5036 | 24 | 0 | 36 | 0.4000 | 0.4000 | 0.2167 | 0.0000 | — | 6.2417 | 1.0000 |
| Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 60 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | — | 4.9500 | 1.0000 |
| Raw+Gram@0.5 | general | 0.5000 | 0.5000 | 0.5000 | -0.0033 | 0.0092 | 0.0413 | 1.0076 | 36 | 0 | 24 | 0.6500 | 0.1000 | 0.0500 | 0.0000 | — | 3.6833 | 2.0000 |
| Raw+Gram@0.75 | general | 0.7500 | 0.7500 | 0.7500 | -0.0023 | 0.0279 | 0.0872 | 2.4583 | 30 | 0 | 30 | 0.5167 | 0.2333 | 0.1000 | 0.0000 | — | 4.4583 | 2.0000 |
| SafeGram-t01 | general | 0.5000 | 0.0000 | 1.0000 | -0.0005 | 0.0026 | 0.0077 | 0.0354 | 32 | 15 | 13 | 0.5167 | 0.0333 | 0.0000 | 0.3333 | — | 4.2000 | 1.0000 |
| SafeRankGram-t01 | general | 0.5000 | 0.0000 | 1.0000 | -0.0005 | 0.0013 | 0.0025 | 0.0583 | 35 | 16 | 9 | 0.5333 | 0.0333 | 0.0167 | 0.4000 | — | 3.4833 | 1.0000 |
| GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | 0.4375 | 0.7500 | -0.0307 | 0.0018 | 0.0026 | 0.0064 | 29 | 1 | 6 | 0.7778 | 0.0000 | 0.0000 | 0.1389 | — | 1.7639 | 2.0000 |

## 17. Worst 10 Tail Cells

| dataset | model | problem_type | method | raw_task_error | method_task_error | validation_C | test C | selected_alpha | invariant_feature_fraction | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OnlineNewsPopularity | tabicl_v2 | regression | PureGram | 5203.2604 | 5480.1943 | 2.3200 | 4.5036 | 1.0000 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | tabicl_v2 | regression | Raw+Gram@0.75 | 5203.2604 | 5354.4220 | 1.4251 | 2.4583 | 0.7500 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | controlled_mlp | regression | BlockGuard-Greedy-t01 | 5161.0207 | 5396.2246 | -0.6430 | 2.1635 | — | 0.7561 | validation-safe but test-harmful: validation miss/noise or distribution shift |
| OnlineNewsPopularity | controlled_mlp | regression | PureGram | 5161.0207 | 5325.1910 | 1.7549 | 1.9250 | 1.0000 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | tabicl_v2 | regression | Raw+Gram@0.5 | 5203.2604 | 5265.2162 | 0.7363 | 1.0076 | 0.5000 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | controlled_mlp | regression | Raw+Gram@0.75 | 5161.0207 | 5247.4939 | 0.6547 | 0.8473 | 0.7500 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | controlled_mlp | regression | Raw+Gram@0.5 | 5161.0207 | 5205.2567 | 0.2956 | 0.4735 | 0.5000 | — | harm visible on both validation and test; fixed/reference method is unconstrained |
| OnlineNewsPopularity | catboost | regression | PureGram | 5254.3235 | 5253.8532 | -2.8676 | 0.4527 | 1.0000 | — | validation-safe but test-harmful: validation miss/noise or distribution shift |
| OnlineNewsPopularity | catboost | regression | Raw+Gram@0.75 | 5254.3235 | 5250.5999 | -2.2287 | 0.2469 | 0.7500 | — | validation-safe but test-harmful: validation miss/noise or distribution shift |
| OnlineNewsPopularity | catboost | regression | GuardedGram-G2-g0p0-t01 | 5254.3235 | 5250.5999 | -2.2287 | 0.2469 | 0.7500 | — | validation-safe but test-harmful: validation miss/noise or distribution shift |

## 18. Paper-Safety Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 2 | SafeRankGram-t01 | general | 0.5000 | -0.0005 | 0.0025 | 0.0583 | 3.4833 | 1.0000 | 2.0000 | 2.0000 | 0.4835 |
| 3 | SafeGram-t01 | general | 0.5000 | -0.0005 | 0.0077 | 0.0354 | 4.2000 | 1.0000 | 2.0000 | 2.0000 | 0.5000 |
| 4 | Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## 19. Strict-Safety Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 2 | SafeGram-t01 | general | 0.5000 | -0.0005 | 0.0077 | 0.0354 | 4.2000 | 1.0000 | 2.0000 | 2.0000 | 0.5000 |
| 3 | Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## 20. Basis-Control Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PureGram | general | 1.0000 | 0.0069 | 0.1808 | 4.5036 | 6.2417 | 1.0000 | 1.0000 | 1.0000 | -8.4406 |
| 2 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 3 | GuardedGram-G2-g0p0-t01 | general | 0.7500 | -0.0026 | 0.0175 | 0.2469 | 3.7250 | 2.0000 | 2.0000 | 2.0000 | 0.2836 |
| 4 | Raw+Gram@0.75 | general | 0.7500 | -0.0023 | 0.0872 | 2.4583 | 4.4583 | 2.0000 | 2.0000 | 2.0000 | -4.3482 |
| 5 | BlockGuard-Greedy-t01 | general | 0.7219 | 0.0000 | 0.0476 | 2.1635 | 5.2583 | 1.0000 | 23.5000 | 1.0000 | -3.6179 |
| 6 | Raw+Gram@0.5 | general | 0.5000 | -0.0033 | 0.0413 | 1.0076 | 3.6833 | 2.0000 | 2.0000 | 2.0000 | -1.5592 |
| 7 | SafeRankGram-t01 | general | 0.5000 | -0.0005 | 0.0025 | 0.0583 | 3.4833 | 1.0000 | 2.0000 | 2.0000 | 0.4835 |
| 8 | SafeGram-t01 | general | 0.5000 | -0.0005 | 0.0077 | 0.0354 | 4.2000 | 1.0000 | 2.0000 | 2.0000 | 0.5000 |
| 9 | Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## 21. Predictive Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 2 | SafeRankGram-t01 | general | 0.5000 | -0.0005 | 0.0025 | 0.0583 | 3.4833 | 1.0000 | 2.0000 | 2.0000 | 0.4835 |
| 3 | Raw+Gram@0.5 | general | 0.5000 | -0.0033 | 0.0413 | 1.0076 | 3.6833 | 2.0000 | 2.0000 | 2.0000 | -1.5592 |
| 4 | GuardedGram-G2-g0p0-t01 | general | 0.7500 | -0.0026 | 0.0175 | 0.2469 | 3.7250 | 2.0000 | 2.0000 | 2.0000 | 0.2836 |
| 5 | SafeGram-t01 | general | 0.5000 | -0.0005 | 0.0077 | 0.0354 | 4.2000 | 1.0000 | 2.0000 | 2.0000 | 0.5000 |
| 6 | Raw+Gram@0.75 | general | 0.7500 | -0.0023 | 0.0872 | 2.4583 | 4.4583 | 2.0000 | 2.0000 | 2.0000 | -4.3482 |
| 7 | Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 8 | BlockGuard-Greedy-t01 | general | 0.7219 | 0.0000 | 0.0476 | 2.1635 | 5.2583 | 1.0000 | 23.5000 | 1.0000 | -3.6179 |
| 9 | PureGram | general | 1.0000 | 0.0069 | 0.1808 | 4.5036 | 6.2417 | 1.0000 | 1.0000 | 1.0000 | -8.4406 |

## 22. Efficiency Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PureGram | general | 1.0000 | 0.0069 | 0.1808 | 4.5036 | 6.2417 | 1.0000 | 1.0000 | 1.0000 | -8.4406 |
| 2 | BlockGuard-Greedy-t01 | general | 0.7219 | 0.0000 | 0.0476 | 2.1635 | 5.2583 | 1.0000 | 23.5000 | 1.0000 | -3.6179 |
| 3 | GuardedGram-G2-g0p0-t01 | general | 0.7500 | -0.0026 | 0.0175 | 0.2469 | 3.7250 | 2.0000 | 2.0000 | 2.0000 | 0.2836 |
| 4 | Raw+Gram@0.75 | general | 0.7500 | -0.0023 | 0.0872 | 2.4583 | 4.4583 | 2.0000 | 2.0000 | 2.0000 | -4.3482 |
| 5 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 6 | Raw+Gram@0.5 | general | 0.5000 | -0.0033 | 0.0413 | 1.0076 | 3.6833 | 2.0000 | 2.0000 | 2.0000 | -1.5592 |

## 23. Overall Paper-Candidate Ranking

| rank | method | scope | median_disagreement_reduction | median_C | p95_C | max_C | mean_predictive_rank | inference_multiplier | training_multiplier | parameter_multiplier | paper_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GuardedGram-G2-after-RBF-k16 | RBF-k16 embedding | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 1.7639 | 2.0000 | 2.0000 | 2.0000 | 0.7000 |
| 2 | SafeGram-t01 | general | 0.5000 | -0.0005 | 0.0077 | 0.0354 | 4.2000 | 1.0000 | 2.0000 | 2.0000 | 0.5000 |
| 3 | SafeRankGram-t01 | general | 0.5000 | -0.0005 | 0.0025 | 0.0583 | 3.4833 | 1.0000 | 2.0000 | 2.0000 | 0.4835 |
| 4 | GuardedGram-G2-g0p0-t01 | general | 0.7500 | -0.0026 | 0.0175 | 0.2469 | 3.7250 | 2.0000 | 2.0000 | 2.0000 | 0.2836 |
| 5 | Raw | general | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 6 | Raw+Gram@0.5 | general | 0.5000 | -0.0033 | 0.0413 | 1.0076 | 3.6833 | 2.0000 | 2.0000 | 2.0000 | -1.5592 |
| 7 | BlockGuard-Greedy-t01 | general | 0.7219 | 0.0000 | 0.0476 | 2.1635 | 5.2583 | 1.0000 | 23.5000 | 1.0000 | -3.6179 |
| 8 | Raw+Gram@0.75 | general | 0.7500 | -0.0023 | 0.0872 | 2.4583 | 4.4583 | 2.0000 | 2.0000 | 2.0000 | -4.3482 |
| 9 | PureGram | general | 1.0000 | 0.0069 | 0.1808 | 4.5036 | 6.2417 | 1.0000 | 1.0000 | 1.0000 | -8.4406 |

The score is `R - 3 max(median C,0) - 3 max(p95 C-.01,0) - 2 max(max C-.05,0) - .05 log2(inference multiplier)`; every raw component is shown and saved in `prospective_six_rankings.csv`.

## 24. Does GuardedGram Beat SafeGram?

**PARTLY.** G2 control/p95 C = 0.7500/0.0175; SafeGram = 0.5000/0.0077.

## 25. Does Feature-Level Selection Beat Global Gating?

**NO.** This comparison uses the same untouched panel and counts BlockGuard as a single-representation inference method.

## 26. Can We Avoid Two Full Models?

**PARTLY.** BlockGuard's prospective control/tail metrics determine this answer; DualView did not survive Stage 1.

## 27. Does Basis Sensitivity Grow With Embedding Dimension?

**YES.** 0.8750 of dataset/model/embedding slopes are positive across `k=4,8,16,32`.

## 28. Is the Default Numerical-Embedding Basis Usually Optimal?

**NO.** It is oracle-best in 0.2708 of full-development embedding cells.

## 29. Strongest Positive Finding

`GuardedGram-G2-g0p0-t01` produced the largest prospective median control among frozen candidates (0.7500) with median/p95/max C -0.0026/0.0175/0.2469.

## 30. Strongest Negative Finding

The hardest finalist tail belongs to `Raw+Gram@0.75` (p95/max C 0.0872/2.4583). DualView also failed its early gate, and transferred BlockGuard inside embeddings showed that feature selections do not automatically port across embedding families.

## 31. Reviewer Attack Audit

### "The adaptive rule is just validation overfitting."

All thresholds were chosen on development only and hashed before the untouched panel. Section 17 exposes validation-safe/test-harmful cells rather than hiding them.

### "The method still has catastrophic tails."

The report gives median, p90, p95, max, fractions above `.01`/`.05`, a tail CDF, and the ten worst cells. Paper- and strict-safety rankings exclude violations mechanically.

### "The method requires twice the compute."

Fractional prediction mixtures do. BlockGuard uses one inference model; DualView was benchmarked as one model but pruned. Training overhead is reported separately.

### "The phenomenon is caused by artificial preprocessing."

Frozen natural equivalences reconstruct to below `3e-16`, well under `1e-6`, for local/spectral hats and one-hot/Helmert pairs. Fourier was marked unavailable rather than fabricated.

### "Why not just use scalar features?"

Scalarization discards the representational capacity being audited. The experiment instead measures whether equivalent multi-coordinate blocks can be made stable with bounded task cost.

### "Why should basis dependence be removed if some bases are better?"

It should not be removed blindly. Sections 6 and 11 quantify validation selection and oracle headroom; guarded methods trade reproducibility against genuine predictive gains.

### "The method is too conservative."

Fallback rates and achieved control are reported jointly. Safe references establish the conservative floor; G2 and BlockGuard test whether control can rise without exceeding tail gates.

### "The method is architecture-specific."

The general panel spans five model families and the embedding panel spans MLP, TabM-D, and ResNet. Model-family counts are an explicit success requirement.

## 32. Ranked Final Candidates for Human Decision

| rank | method | control | median C | p95 | max | breadth | single-model? | embedding evidence | complexity | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GuardedGram-G2-after-RBF-k16 | 0.7500 | -0.0307 | 0.0026 | 0.0064 | 12 datasets / 3 model families | NO | direct prospective RBF-k16 evidence | embedding gate plus two prediction branches | repeat unchanged on a second independently locked panel |
| 4 | GuardedGram-G2-g0p0-t01 | 0.7500 | -0.0026 | 0.0175 | 0.2469 | 12 datasets / 5 model families | NO | general-coordinate evidence; embedding analogue reported separately | validation gate plus two prediction branches | diagnose the failed control/tail gate before promotion |
| 7 | BlockGuard-Greedy-t01 | 0.7219 | 0.0000 | 0.0476 | 2.1635 | 12 datasets / 5 model families | YES | transferred embedding variant was negative | one inference model; exact intervention search during training | diagnose the failed control/tail gate before promotion |
| 8 | Raw+Gram@0.75 | 0.7500 | -0.0023 | 0.0872 | 2.4583 | 12 datasets / 5 model families | NO | general-coordinate evidence; embedding analogue reported separately | fixed two-branch prediction mixture | diagnose the failed control/tail gate before promotion |

This is a ranked evidence table for human decision; no final paper method is automatically selected.

## 33. Best Next Step for Each Top-3 Candidate

1. `GuardedGram-G2-after-RBF-k16` — repeat the exact frozen configuration on a second independently locked panel.
2. `GuardedGram-G2-g0p0-t01` — run a seed-expansion audit focused on its worst prospective cell.
3. `BlockGuard-Greedy-t01` — measure end-to-end wall-clock and memory under deployment-sized batches.

## 34. Files Produced

- `configs/GUARDED_FINALISTS.json` and SHA256 sidecar
- `results/processed/prospective_general_cells.csv`, units, summary, and manifest
- `results/processed/prospective_embedding_cells.csv`, units, summary, and manifest
- `results/processed/prospective_six_rankings.csv`
- `results/processed/prospective_worst_10_tail_cells.csv`
- `results/processed/prospective_loader_audit.json`
- `results/processed/blockguard_feature_descriptor_summary.csv`
- `results/processed/natural_basis_reuse_manifest.json`
- `results/processed/final_provenance.json`
- `results/processed/final_audit.json`
- eight PNG figures in `figures/`
- this `results.md`
