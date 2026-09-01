# Semantic Symmetries / Representation Orbits — Kill Experiment

## Executive Verdict
GO

## One-Paragraph Summary

Across the frozen six-dataset, three-seed panel, the strongest observed cell was tabpfn_2_6 on wine-quality-red under mixed_monotone (disagreement 0.1451); the automated kill-rule audit returned **GO**. The decision uses repeated effects across datasets and model families plus repair behavior, not the single maximum cell.

## Experimental Setup
- hardware: NVIDIA H100 NVL; CatBoost on CPU, frozen TFMs on a dedicated H100 NVL
- runtime: 1.21 aggregate primary bundle-hours across 54 immutable bundles; bounded TabM 2.1 bundle-minutes and training ablations 1.0 bundle-minutes
- packages: {"catboost": "1.2.10", "numpy": "1.26.4", "pandas": "2.3.3", "scikit-learn": "1.4.2", "scipy": "1.11.4", "tabicl": "2.0.3", "tabpfn": "8.5.0", "torch": "2.7.0"}
- model versions: TabICLv2 official 2026-02-12 checkpoints; TabPFN-2.6 exact v2.6 checkpoints; CatBoost 1.2.10
- datasets / OpenML IDs: adult (OpenML 1590 v2), bank-marketing (OpenML 1461 v1), diamonds (OpenML 42225 v1), bike-sharing (OpenML 42712 v2), california_housing (OpenML 43939 v1), wine-quality-red (OpenML 40691 v1)
- seeds: [0, 1, 2]; fixed row split seed 20260901
- transformations: T0–T6, eight members each; positive-affine, strictly order-preserving, cyclic-metadata, and condition-number constraints are serialized per row

## Main Schema-Sensitivity Table
dataset | model | transformation | pred disagreement | task metric original | orbit mean | orbit worst | orbit span

dataset | model | variant | pred_disagreement | task_metric_original | orbit_mean | orbit_worst | orbit_span
--- | --- | --- | --- | --- | --- | --- | ---
wine-quality-red | tabpfn_2_6 | mixed_monotone | 0.14514 | 0.63299 | 0.63852 | 0.64322 | 0.0088129
california_housing | tabicl_v2 | cond_le_3 | 0.13721 | 55704 | 55788 | 57022 | 2169.8
california_housing | tabicl_v2 | orthogonal | 0.13057 | 55704 | 56044 | 56844 | 1768.2
wine-quality-red | tabpfn_2_6 | column_permutation | 0.12829 | 0.63299 | 0.64403 | 0.65429 | 0.021308
california_housing | tabicl_v2 | cond_le_10 | 0.12396 | 55704 | 55197 | 55693 | 1518.5
california_housing | tabpfn_2_6 | column_permutation | 0.11569 | 50731 | 50507 | 51318 | 1394.8
california_housing | tabpfn_2_6 | mixed_monotone | 0.10289 | 50731 | 51132 | 51317 | 618.66
california_housing | tabpfn_2_6 | orthogonal | 0.1023 | 50365 | 50940 | 51802 | 1688.3
california_housing | tabpfn_2_6 | cond_le_3 | 0.10184 | 50365 | 50826 | 51702 | 1447.8
california_housing | tabpfn_2_6 | cond_le_10 | 0.10092 | 50365 | 51020 | 52045 | 1835
wine-quality-red | tabpfn_2_6 | cond_le_10 | 0.10049 | 0.64109 | 0.64049 | 0.64615 | 0.010714
diamonds | catboost | nominal_relabeling | 0.099674 | 735.34 | 833.4 | 886.44 | 113.82
wine-quality-red | tabpfn_2_6 | orthogonal | 0.098496 | 0.64109 | 0.64095 | 0.64798 | 0.011418
wine-quality-red | tabpfn_2_6 | cond_le_3 | 0.096578 | 0.64109 | 0.6402 | 0.64446 | 0.0082973
wine-quality-red | catboost | cond_le_3 | 0.096308 | 0.65383 | 0.65023 | 0.65724 | 0.013789
wine-quality-red | catboost | orthogonal | 0.095902 | 0.65383 | 0.65277 | 0.66363 | 0.018106
bike-sharing | catboost | cyclic_shift | 0.094564 | 55.386 | 55.79 | 57.24 | 2.9437
wine-quality-red | catboost | cond_le_10 | 0.094489 | 0.65383 | 0.65017 | 0.65815 | 0.014603
wine-quality-red | catboost | column_permutation | 0.088573 | 0.65531 | 0.65714 | 0.66092 | 0.0077484
california_housing | catboost | cond_le_3 | 0.086254 | 60088 | 61007 | 61758 | 1474
bike-sharing | tabpfn_2_6 | column_permutation | 0.085034 | 45.593 | 45.914 | 48.911 | 5.1831
california_housing | tabicl_v2 | mixed_monotone | 0.084291 | 50120 | 51594 | 52640 | 1487.3
california_housing | catboost | orthogonal | 0.083446 | 60088 | 60939 | 61340 | 788.73
california_housing | tabpfn_2_6 | nominal_relabeling | 0.083243 | 51583 | 51497 | 52304 | 1415.5
wine-quality-red | tabpfn_2_6 | affine | 0.083225 | 0.63299 | 0.63847 | 0.64539 | 0.014099
california_housing | catboost | nominal_relabeling | 0.082878 | 59154 | 59069 | 59474 | 856.78
california_housing | catboost | cond_le_10 | 0.08224 | 60088 | 60783 | 61225 | 839.5
bike-sharing | tabpfn_2_6 | cyclic_shift | 0.082066 | 45.593 | 47.896 | 48.905 | 3.6415
wine-quality-red | tabicl_v2 | mixed_monotone | 0.082056 | 0.6363 | 0.63444 | 0.6381 | 0.0084904
california_housing | tabpfn_2_6 | nominal_relabeling | 0.081302 | 51583 | 51381 | 51987 | 1087.3
bike-sharing | tabicl_v2 | cond_le_3 | 0.080725 | 48.802 | 49.369 | 51.252 | 3.5151
california_housing | catboost | nominal_relabeling | 0.079742 | 59154 | 59115 | 59578 | 863.68
bike-sharing | tabicl_v2 | orthogonal | 0.078914 | 48.802 | 48.592 | 50.278 | 4.0237
bike-sharing | tabicl_v2 | cond_le_10 | 0.078546 | 48.802 | 49.235 | 51.299 | 3.9678
california_housing | tabpfn_2_6 | nominal_relabeling | 0.078522 | 50731 | 51168 | 51753 | 1485.7
bike-sharing | tabicl_v2 | cyclic_shift | 0.076919 | 45.8 | 46.017 | 47.463 | 2.9557
bike-sharing | tabicl_v2 | cyclic_rotation | 0.075794 | 46.47 | 45.508 | 46.781 | 2.7375
california_housing | tabpfn_2_6 | nominal_relabeling | 0.07478 | 50731 | 50943 | 51725 | 1456.5
california_housing | catboost | column_permutation | 0.072555 | 59154 | 59231 | 59561 | 839.1
bike-sharing | catboost | cyclic_rotation | 0.070208 | 53.409 | 52.892 | 53.72 | 1.7908
california_housing | tabicl_v2 | column_permutation | 0.069943 | 50120 | 50977 | 51281 | 814.77
bike-sharing | tabpfn_2_6 | cyclic_rotation | 0.064945 | 46.453 | 46.656 | 47.903 | 2.6909
bike-sharing | tabpfn_2_6 | cond_le_10 | 0.064729 | 44.776 | 45.535 | 46.609 | 2.1702
california_housing | tabpfn_2_6 | affine | 0.064687 | 50731 | 51181 | 51911 | 1370.8
bike-sharing | catboost | cyclic_shift | 0.063411 | 55.386 | 55.613 | 56.403 | 1.3278
wine-quality-red | tabicl_v2 | orthogonal | 0.062442 | 0.6386 | 0.63226 | 0.63521 | 0.0061693
wine-quality-red | tabicl_v2 | cond_le_3 | 0.062047 | 0.6386 | 0.63187 | 0.63736 | 0.010489
bike-sharing | tabicl_v2 | column_permutation | 0.061828 | 45.8 | 46.502 | 47.485 | 2.2519
wine-quality-red | tabicl_v2 | cond_le_10 | 0.061082 | 0.6386 | 0.63254 | 0.63691 | 0.0078256
bike-sharing | tabpfn_2_6 | orthogonal | 0.05988 | 44.776 | 45.1 | 45.955 | 1.4229
bike-sharing | tabpfn_2_6 | cond_le_3 | 0.059772 | 44.776 | 45.099 | 45.819 | 1.4824
wine-quality-red | tabicl_v2 | column_permutation | 0.058233 | 0.6363 | 0.63752 | 0.63909 | 0.0028555
bike-sharing | tabicl_v2 | cyclic_shift | 0.05816 | 45.8 | 46.432 | 47.414 | 2.1747
bike-sharing | catboost | nominal_relabeling | 0.057127 | 55.386 | 55.161 | 55.973 | 1.5316
diamonds | tabpfn_2_6 | nominal_relabeling | 0.056983 | 608.56 | 629.65 | 704.18 | 106.13
diamonds | tabpfn_2_6 | nominal_relabeling | 0.056689 | 603.75 | 615.72 | 647.96 | 57.028
bike-sharing | tabpfn_2_6 | cyclic_shift | 0.050354 | 45.593 | 45.932 | 46.967 | 2.0064
bike-sharing | catboost | cond_le_10 | 0.050277 | 55.868 | 56.25 | 56.937 | 1.3241
bike-sharing | catboost | orthogonal | 0.049896 | 55.868 | 56.196 | 56.766 | 1.2367
bike-sharing | tabpfn_2_6 | nominal_relabeling | 0.049625 | 45.677 | 46.139 | 47.139 | 2.145
bike-sharing | catboost | column_permutation | 0.049579 | 55.386 | 55.327 | 55.745 | 1.0015
diamonds | catboost | nominal_relabeling | 0.049107 | 735.34 | 753.1 | 792.04 | 75.276
bike-sharing | catboost | cond_le_3 | 0.0483 | 55.868 | 56.503 | 57.252 | 1.2112
bike-sharing | tabpfn_2_6 | nominal_relabeling | 0.046377 | 45.593 | 46.109 | 47.049 | 1.9927
diamonds | tabpfn_2_6 | column_permutation | 0.04615 | 608.56 | 611.34 | 655.2 | 68.832
bike-sharing | tabpfn_2_6 | mixed_monotone | 0.044798 | 45.593 | 45.691 | 46.152 | 1.0625
california_housing | tabicl_v2 | nominal_relabeling | 0.044794 | 50120 | 50615 | 51063 | 1005.4
bike-sharing | tabicl_v2 | nominal_relabeling | 0.042995 | 45.8 | 45.729 | 46.318 | 1.3965
california_housing | tabicl_v2 | nominal_relabeling | 0.041715 | 50120 | 50406 | 50907 | 863.28
diamonds | catboost | column_permutation | 0.039311 | 735.34 | 733.43 | 751.03 | 29.977
diamonds | tabpfn_2_6 | cond_le_10 | 0.038965 | 579.84 | 622.17 | 660.92 | 64.179
diamonds | tabpfn_2_6 | orthogonal | 0.03838 | 579.84 | 621.12 | 672.69 | 78.233
diamonds | tabicl_v2 | nominal_relabeling | 0.038172 | 535.7 | 543.89 | 591.79 | 72.768
diamonds | tabpfn_2_6 | mixed_monotone | 0.038151 | 608.56 | 606.22 | 618.99 | 21.13
adult | catboost | nominal_relabeling | 0.037508 | 0.33127 | 0.331 | 0.34175 | 0.020278
diamonds | tabicl_v2 | cond_le_10 | 0.03748 | 555.3 | 563.61 | 599.2 | 55.356
bike-sharing | tabpfn_2_6 | affine | 0.036879 | 45.593 | 45.682 | 46.803 | 1.9432
diamonds | catboost | cond_le_10 | 0.036599 | 756.08 | 757.19 | 771.53 | 33.132
diamonds | catboost | orthogonal | 0.03643 | 756.08 | 756.96 | 769.53 | 25.097
diamonds | tabicl_v2 | column_permutation | 0.036366 | 535.7 | 562.18 | 573.59 | 27.66

## Results by Transformation
### Column permutation control

Strongest cell: tabpfn_2_6 on wine-quality-red (column_permutation, all), disagreement 0.1283, orbit span 0.02131.
### Nominal relabeling

Strongest cell: catboost on diamonds (nominal_relabeling, all), disagreement 0.09967, orbit span 113.8.
### Numerical affine/unit transforms

Strongest cell: tabpfn_2_6 on wine-quality-red (affine, all), disagreement 0.08323, orbit span 0.0141.
### Monotone transforms

Strongest cell: tabpfn_2_6 on wine-quality-red (mixed_monotone, all), disagreement 0.1451, orbit span 0.008813.
### Ordinal spacing

Strongest cell: tabicl_v2 on diamonds (ordinal_integer, all), disagreement 0.03143, orbit span 35.8.
### Cyclic recoding

Strongest cell: catboost on bike-sharing (cyclic_shift, all), disagreement 0.09456, orbit span 2.944.
### Equivalent basis changes

Strongest cell: tabicl_v2 on california_housing (cond_le_3, one), disagreement 0.1372, orbit span 2170.

## Model Comparison
TabICLv2 vs TabPFN vs CatBoost vs TabM

model | pred_disagreement | relative_orbit_span
--- | --- | ---
catboost | 0.023859 | 0.0014985
tabicl_v2 | 0.029573 | 0.016256
tabpfn_2_6 | 0.037146 | 0.02871
tabm_d | 0.099203 | 0.02399

TabM was run as a predeclared bounded follow-up on the three datasets and three T6 condition bands that triggered the primary-grid decision: nine immutable bundles, 225 separately trained representations, 8 members per orbit. It is not presented as a full-grid comparison. Every TabM dataset/variant/seed cell exceeded 0.05 normalized disagreement:

dataset | variant | pred_disagreement | min_across_seeds | orbit_span
--- | --- | --- | --- | ---
wine-quality-red | cond_le_10 | 0.14202 | 0.12704 | 0.019669
wine-quality-red | cond_le_3 | 0.12771 | 0.12415 | 0.01605
wine-quality-red | orthogonal | 0.11945 | 0.10101 | 0.015751
california_housing | cond_le_10 | 0.096036 | 0.094242 | 1199.6
california_housing | cond_le_3 | 0.095984 | 0.094217 | 1440.6
california_housing | orthogonal | 0.092788 | 0.092513 | 1128.6
bike-sharing | cond_le_10 | 0.07335 | 0.072163 | 2.4525
bike-sharing | cond_le_3 | 0.072783 | 0.069738 | 2.262
bike-sharing | orthogonal | 0.072711 | 0.07204 | 2.1939

## Repair Results
standardization | quantile | nominal canonicalization | ordinal canonicalization | cyclic frontend | orbit ensemble

repair | median_reduction | median_task_change | cells
--- | --- | --- | ---
cyclic_frontend | 1 | -0.71156 | 6
nominal_canonicalization | 1 | 0.0036143 | 25
ordinal_canonicalization | 1 | -0.35374 | 6
standardization | 1 | 0 | 72
quantile_rank | 0.97151 | -0.00016499 | 18
orbit_ensemble_8 | 0.25052 | -0.0039134 | 231

Orbit and ordinary ensemble measurements are in `results/semantic_orbits/processed/orbit_ensembles.csv` and `ordinary_seed_ensembles.csv`; Figure 5 uses equal three-prediction budgets. Orbit diversity beat ordinary seed diversity in 68/231 cells; the median relative task-error difference was +0.400%. This is descriptive, not a universal orbit-ensemble win.

## Training-Time Ablations
orbit augmentation / consistency / dual-view if run

Run conditionally on California Housing and Wine Quality after the primary grid identified them as strong T6 cells. A three-layer, width-256 MLP was trained for each method and seed on the eight orthogonal (condition-one) basis views; the generic control adds Gaussian noise with standard deviation 0.1 after reference-coordinate standardization. Canonical-only and dual-view use the serialized basis metadata, with inverse reconstruction audited to <2e-5. These six follow-up bundles remain separate from the frozen kill verdict.

dataset | method | prediction_disagreement | disagreement_reduction_vs_raw | reference_rmse | orbit_mean_rmse | orbit_worst_rmse | orbit_worst_change_vs_raw
--- | --- | --- | --- | --- | --- | --- | ---
california_housing | canonical_only | 0 | 1 | 63271 | 63271 | 63271 | -4.6149e+05
california_housing | dual_view | 0.13627 | 0.93837 | 63635 | 63226 | 63876 | -4.6088e+05
california_housing | consistency_1.0 | 0.1388 | 0.93723 | 60554 | 62292 | 63090 | -4.6167e+05
california_housing | consistency_0.1 | 0.31849 | 0.85596 | 60866 | 66391 | 68008 | -4.5675e+05
california_housing | orbit_augmentation | 0.32781 | 0.85175 | 73770 | 66853 | 67834 | -4.5692e+05
california_housing | raw | 2.2112 | 0 | 63271 | 2.5728e+05 | 5.2476e+05 | 0
california_housing | generic_noise | 2.2828 | -0.032407 | 66281 | 2.6303e+05 | 5.2307e+05 | -1688.7
wine-quality-red | canonical_only | 0 | 1 | 0.69376 | 0.69376 | 0.69376 | -0.1356
wine-quality-red | consistency_1.0 | 0.1207 | 0.77352 | 0.67022 | 0.67281 | 0.67688 | -0.15248
wine-quality-red | dual_view | 0.12129 | 0.77242 | 0.69069 | 0.68364 | 0.69066 | -0.1387
wine-quality-red | orbit_augmentation | 0.17999 | 0.66228 | 0.67922 | 0.67814 | 0.68525 | -0.14411
wine-quality-red | consistency_0.1 | 0.2533 | 0.52473 | 0.67372 | 0.68406 | 0.69274 | -0.13662
wine-quality-red | generic_noise | 0.46631 | 0.12505 | 0.68549 | 0.74206 | 0.77767 | -0.051689
wine-quality-red | raw | 0.53296 | 0 | 0.69376 | 0.77099 | 0.82936 | 0

Semantic consistency at lambda=1 reduced disagreement by 93.7% on California and 77.4% on wine, versus -3.2% and 12.5% for generic Gaussian augmentation. It also improved mean and worst-orbit RMSE relative to raw training on both datasets. Canonical-only gives exact invariance because the transformation metadata permits exact inversion; dual-view retains the raw branch while reducing disagreement by 93.8% and 77.2%.

## Strongest Positive Finding

Well-conditioned basis changes are the strongest repeated finding: orthogonal, condition≤3, and condition≤10 changes each crossed the 0.05 threshold on all three headline datasets and all three required model families in aggregate. For every condition band, 8 of the 9 dataset/model cells passed in all three seeds; only Bike Sharing/CatBoost was just below threshold. The bounded TabM panel then passed all 9 corresponding cells. The largest primary-grid T6 cell was tabicl_v2 / california_housing / cond_le_3 at 0.13721 disagreement.

## Strongest Negative Finding

Ordinal spacing was materially weaker than the basis result and did not establish the repeated three-dataset signal (it is available only on diamonds). Nominal relabeling also stayed below 3% flips for TabPFN-2.6 on adult and for every model on bank marketing. These limits prevent a universal “all semantic recodings break all models” claim.

## Information-Equivalence Sanity Checks

Nominal mappings are bijections; positive-affine maps have positive scale; monotone PWL maps have positive slopes and linear tails; known ordinal orders are retained; cyclic shifts serialize periods and origins; and every 8×8 basis matrix is full rank with measured condition number at most 10. The synthetic sanity dataset reconstructs the structural target function after inverse semantic decoding to numerical tolerance. Targets and test-row hashes are identical across every model and seed bundle.

## Failures / Unexpected Results
Report all.

- TabPFN package 6.3.0 did not contain the requested v2.6 architecture, so the environment was upgraded to 8.5.0 and exact v2.6 checkpoints were pinned by path and SHA-256.
- The account-level convenience-constructor license token was absent; the already downloaded official checkpoints were supplied through the documented explicit `model_path` interface.
- TabM initially rejected California Housing's missing continuous values. A training-median fill was applied in original coordinates before constructing the RBF orbit, retaining exact synchronization across basis views.
- Diagnostics: `{"effect_cells": 49, "qualifying_repeated_effects": [{"cells": 8, "datasets": 4, "family": "T1", "models": 3, "scope": "all", "variant": "nominal_relabeling"}, {"cells": 8, "datasets": 3, "family": "T6", "models": 3, "scope": "one", "variant": "cond_le_10"}, {"cells": 8, "datasets": 3, "family": "T6", "models": 3, "scope": "one", "variant": "cond_le_3"}, {"cells": 8, "datasets": 3, "family": "T6", "models": 3, "scope": "one", "variant": "orthogonal"}], "raw_cells": 213, "strong_ensemble_cells": 158, "strong_repair_cells": 77}`.

## Does This Look Like an ICLR/ICML/NeurIPS-Level Direction?
Give evidence-based YES / MAYBE / NO and why.

**YES.** The case rests on an information-identical, well-conditioned basis result repeated on three real datasets, all three required model families, every seed, and the bounded TabM follow-up. It does not rest on column permutation or ill-conditioning. A condition-one training ablation then shows semantic consistency decisively beating generic noise while improving worst-orbit error, although it still falls short of metadata-based exact canonicalization and needs validation beyond two regression datasets.

## Best Next Method
If signal exists, propose the simplest model suggested by the results.

Use a metadata-aware canonical branch plus a raw residual branch, trained with lambda≈1 orbit consistency. The dual-view result preserves clean-coordinate performance while sharply reducing instability, and the canonical-only result provides an exact-invariance ceiling. Next, extend this minimal design to nominal and cyclic features and test it on classification rather than adding architectural complexity.

## Files Produced
- `configs/semantic_orbits.yaml`
- `configs/semantic_orbits_tabm_basis.yaml`
- `SEMANTIC_ORBITS_PROTOCOL.md`
- `src/semantic_orbits.py`
- `scripts/run_semantic_orbits.py`
- `scripts/run_semantic_orbits_tabm.py`
- `scripts/analyze_semantic_orbits.py`
- `results/semantic_orbits/raw/`
- `results/semantic_orbits/processed/all_metrics.csv`
- `results/semantic_orbits/processed/orbit_summary.csv`
- `results/semantic_orbits/processed/orbit_ensembles.csv`
- `results/semantic_orbits/processed/dataset_bootstrap_10000.csv`
- `results/semantic_orbits/processed/ensemble_comparison.csv`
- `results/semantic_orbits/tabm_basis/`
- `results/semantic_orbits/processed/tabm_basis_summary.csv`
- `configs/semantic_orbit_training.yaml`
- `scripts/run_semantic_orbit_training.py`
- `results/semantic_orbits/training_ablations/`
- `results/semantic_orbits/processed/training_ablation_summary.csv`
- `results/semantic_orbits/manifest.json`
- `results/semantic_orbits/synthetic_sanity.json`
- `figures/semantic_orbits/01_disagreement_heatmap.png through 08_worst_vs_average_performance.png`
- `environment/semantic_orbits_lockfile`
