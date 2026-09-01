# Interaction-Aware Context Selection — Kill Experiment

## Executive Verdict
METHOD-FAILS-BUT-SIGNAL

## One-Paragraph Summary
Across 24 dataset/budget cells, the best ID-FM exceeded additive held-out R2 by at least 0.10 in 0 cells. The best interaction-aware selector beat the strongest equal-budget non-interaction selector in 10/24 cells (41.7%). The resulting preregistered kill decision is **METHOD-FAILS-BUT-SIGNAL**.

## Experimental Setup
- Hardware: 2 × NVIDIA H100 NVL (one isolated GPU per concurrent worker).
- Runtime: 1.01 summed TabICLv2 GPU-hours across random surfaces, selected-context tests, direct diagnostics/search, and failure fallbacks; per-stage runtimes are preserved in CSV/JSON audits.
- Packages: TabICL 2.0.3, PyTorch 2.7.0, scikit-learn 1.4.2, OpenML 0.15.1.
- Exact datasets / OpenML IDs: adult (OpenML:1590 (adult), n=48842, p=14); bank-marketing (OpenML:1461 (bank-marketing), n=45211, p=16); credit-g (OpenML:31 (credit-g), n=1000, p=20); electricity (OpenML:151 (electricity), n=45312, p=8); california_housing (sklearn:california_housing, n=20640, p=8); diamonds (OpenML:42225 (diamonds), n=53940, p=9); churn (OpenML:40701 (churn), n=5000, p=20); house_16H (OpenML:574 (house_16H), n=22784, p=16).
- Splits: one fixed seed-0 stratified split per dataset: 256 candidate rows, 128 selector/meta-validation queries, and 256 untouched final-test queries. Regression stratification uses target quantile bins.
- TFM versions: official frozen TabICLv2 checkpoints `tabicl-classifier-v2-20260212.ckpt` and `tabicl-regressor-v2-20260212.ckpt` (`tabicl` 2.0.3), one deterministic estimator, no fine-tuning.
- Context budgets: K=16, 32, 64; random-context seeds 0, 1, 2.
- Number of context evaluations: 114,124 total: 36,864 random surface evaluations (512 per dataset/K/seed), 768 selected-context final tests, 74,440 direct diagnostic/search evaluations, and 2,052 failure-fallback evaluations.

## Main Result Table
| dataset | task | K | random | additive | CRUMB-like | LUCoS-like | DPP | best pairwise | direct-search oracle |
|---|---|---|---|---|---|---|---|---|---|
| adult | classification | 16 | -0.6008 | -0.4958 | -0.5078 | -0.4811 | -0.5757 | -0.4677 | not run |
| adult | classification | 32 | -0.5251 | -0.7125 | -0.4993 | -0.5086 | -0.4809 | -0.5961 | not run |
| adult | classification | 64 | -0.4763 | -0.7665 | -0.4350 | -0.4254 | -0.4277 | -0.5535 | not run |
| bank-marketing | classification | 16 | -0.4820 | -0.3021 | -0.4751 | -0.4743 | -0.3996 | -0.3021 | not run |
| bank-marketing | classification | 32 | -0.3687 | -0.3559 | -0.3436 | -0.3312 | -0.3844 | -0.3559 | not run |
| bank-marketing | classification | 64 | -0.3069 | -0.3432 | -0.3023 | -0.3052 | -0.3112 | -0.3227 | not run |
| california_housing | regression | 16 | -0.8288 | -0.8083 | -0.6465 | -0.9305 | -1.1113 | -0.6533 | not run |
| california_housing | regression | 32 | -0.6385 | -0.5851 | -0.5790 | -0.6660 | -0.6421 | -0.5638 | not run |
| california_housing | regression | 64 | -0.5487 | -0.5158 | -0.5697 | -0.6104 | -0.5650 | -0.5102 | not run |
| churn | classification | 16 | -0.5847 | -0.5775 | -0.5155 | -0.4914 | -0.5533 | -0.4429 | not run |
| churn | classification | 32 | -0.4782 | -0.5153 | -0.4784 | -0.4583 | -0.6844 | -0.4383 | not run |
| churn | classification | 64 | -0.4174 | -0.6423 | -0.5559 | -0.3976 | -0.3673 | -0.3852 | not run |
| credit-g | classification | 16 | -0.8103 | -0.6327 | -0.7710 | -0.6489 | -0.9363 | -0.6095 | not run |
| credit-g | classification | 32 | -0.7576 | -0.7119 | -0.7696 | -0.7486 | -0.8982 | -0.6648 | -0.5958 |
| credit-g | classification | 64 | -0.6351 | -0.5754 | -0.6361 | -0.6189 | -0.6553 | -0.5407 | not run |
| diamonds | regression | 16 | -0.4543 | -0.3802 | -0.3549 | -0.3780 | -0.4733 | -0.3501 | not run |
| diamonds | regression | 32 | -0.3618 | -0.2899 | -0.3159 | -0.3979 | -0.2810 | -0.2897 | -0.2595 |
| diamonds | regression | 64 | -0.2949 | -0.2678 | -0.3079 | -0.3239 | -0.2743 | -0.2572 | not run |
| electricity | classification | 16 | -0.8174 | -0.6800 | -0.6795 | -1.2031 | -0.8052 | -0.6535 | not run |
| electricity | classification | 32 | -0.7621 | -0.8336 | -0.8300 | -0.7164 | -0.7666 | -0.6835 | not run |
| electricity | classification | 64 | -0.6055 | -0.6880 | -0.6589 | -0.5521 | -0.6344 | -0.5818 | not run |
| house_16H | regression | 16 | -1.3128 | -1.1436 | -1.1275 | -1.4104 | -1.5631 | -1.1039 | not run |
| house_16H | regression | 32 | -1.1671 | -1.0493 | -1.0835 | -1.1067 | -1.4283 | -1.0493 | not run |
| house_16H | regression | 64 | -1.0755 | -0.9633 | -1.0305 | -1.0462 | -1.1638 | -0.9362 | not run |

All entries are final-test primary utility (higher is better): negative log loss for classification and negative candidate-normalized RMSE for regression. Random is the mean of 20 frozen random contexts. "Best pairwise" is descriptive best-of-methods and is not a separately tuned baseline.

## Utility Prediction
| dataset | K | additive R2 | FM R2 | feature-FM R2 | DeepSets R2 | ΔR2 |
|---|---|---|---|---|---|---|
| adult | 16 | 0.3072 | 0.1701 | 0.1741 | 0.2163 | -0.1371 |
| adult | 32 | 0.2285 | 0.1630 | 0.1491 | 0.1843 | -0.0655 |
| adult | 64 | 0.3521 | 0.2954 | 0.2645 | 0.3093 | -0.0567 |
| bank-marketing | 16 | 0.1563 | 0.0648 | 0.0071 | 0.1332 | -0.0915 |
| bank-marketing | 32 | 0.1503 | 0.0711 | -0.0207 | 0.1218 | -0.0792 |
| bank-marketing | 64 | 0.2778 | 0.2214 | 0.1138 | 0.2646 | -0.0565 |
| credit-g | 16 | 0.2236 | 0.2165 | 0.2329 | 0.3089 | -0.0071 |
| credit-g | 32 | 0.1663 | 0.1623 | 0.1388 | 0.0671 | -0.0041 |
| credit-g | 64 | 0.1312 | 0.0829 | -0.0493 | 0.1176 | -0.0483 |
| electricity | 16 | 0.1013 | 0.0845 | 0.0904 | 0.0514 | -0.0168 |
| electricity | 32 | 0.0989 | 0.0917 | 0.0449 | 0.0316 | -0.0073 |
| electricity | 64 | 0.1317 | 0.0953 | 0.0033 | 0.0495 | -0.0364 |
| california_housing | 16 | 0.1711 | 0.1307 | 0.1369 | 0.2573 | -0.0404 |
| california_housing | 32 | 0.2129 | 0.1501 | 0.1417 | 0.1935 | -0.0628 |
| california_housing | 64 | 0.1810 | 0.1095 | 0.0297 | 0.1675 | -0.0715 |
| diamonds | 16 | 0.1466 | 0.0710 | 0.0577 | 0.4862 | -0.0756 |
| diamonds | 32 | 0.0924 | 0.0528 | -0.0394 | 0.0852 | -0.0397 |
| diamonds | 64 | 0.2491 | 0.1538 | 0.0739 | 0.2089 | -0.0953 |
| churn | 16 | 0.2018 | 0.1390 | 0.1941 | 0.2737 | -0.0628 |
| churn | 32 | 0.1704 | 0.0980 | 0.0917 | 0.1280 | -0.0724 |
| churn | 64 | 0.2628 | 0.1860 | 0.1530 | 0.2395 | -0.0768 |
| house_16H | 16 | 0.4386 | 0.1113 | 0.3635 | 0.5086 | -0.3274 |
| house_16H | 32 | 0.4485 | 0.2021 | 0.3142 | 0.3996 | -0.2464 |
| house_16H | 64 | 0.4072 | 0.3382 | 0.2070 | 0.4012 | -0.0690 |

Each row uses a 70/30 split over context sets, stratified by random-context seed. Ridge tuning and neural early stopping use surrogate-train data only.

## Direct Interaction Diagnostic
- california_housing: n=300, positive=51.7%, negative=48.3%, median |I|=0.009099, base-context correlation=-0.025; Spearman with cosine=-0.106, distance=0.079, label/bin agreement=-0.019.
- credit-g: n=300, positive=56.0%, negative=44.0%, median |I|=0.013418, base-context correlation=0.067; Spearman with cosine=-0.021, distance=-0.020, label/bin agreement=-0.061.
- diamonds: n=300, positive=48.0%, negative=52.0%, median |I|=0.003414, base-context correlation=0.104; Spearman with cosine=0.032, distance=-0.014, label/bin agreement=0.020.
- house_16H: n=300, positive=49.7%, negative=50.3%, median |I|=0.008408, base-context correlation=-0.026; Spearman with cosine=-0.027, distance=-0.048, label/bin agreement=-0.067.

The raw table includes similarity, distance, and label/target-bin agreement for every pair/base-context evaluation. The finite difference uses selector labels only and is independent of surrogate fitting.

California housing and house_16H were the two fastest datasets by recorded random-context evaluation time and are the prespecified timing-based panels. Credit-g and diamonds are additional panels retained from the initial smoke-test ranking; they are reported rather than discarded.

## b_ij Ablations
| parameterization | best rank | regularization | best held-out R2 |
|---|---|---|---|
| DeepSets | not run | 0.0100 | 0.5086 |
| ID-factor residual | 8 | 0.0200 | 0.4494 |
| label_target_complementarity | not run | 1.0000 | 0.4485 |
| geometry + complementarity | not run | 1.0000 | 0.4485 |
| rbf_diversity | not run | 1.0000 | 0.4485 |
| cosine_diversity | not run | 1.0000 | 0.4485 |
| euclidean_neighbor_diversity | not run | 1.0000 | 0.4484 |
| feature-bilinear MLP | 4 | 0.0100 | 0.3635 |
| ID-factor joint | 2 | 0.0100 | 0.3382 |
| signed bilinear | 16 | 0.0200 | 0.2356 |

All requested ranks were attempted: ID-FM 2/4/8/16; feature and signed bilinear 4/8/16; joint and additive-residual fits; cosine, RBF, Euclidean-neighbor diversity; label/target complementarity; combined geometry+complementarity; DPP; and DeepSets.

Post-failure 128-candidate/1024-context controls (including stronger L2, query-cluster conditioning, and pairwise+DeepSets correction):

| dataset | model | rank | weight decay | held-out R2 |
|---|---|---|---|---|
| credit-g | additive_ridge | not run | not run | 0.1093 |
| credit-g | id_fm | 4 | 0.1000 | 0.0700 |
| credit-g | pairwise_plus_deepsets_correction | 4 | 0.0500 | 0.0714 |
| credit-g | query_cluster_0_additive | not run | not run | 0.2768 |
| credit-g | query_cluster_0_fm | 4 | 0.0500 | 0.1675 |
| credit-g | query_cluster_1_additive | not run | not run | 0.2054 |
| credit-g | query_cluster_1_fm | 4 | 0.0500 | 0.1255 |
| credit-g | query_cluster_2_additive | not run | not run | 0.3271 |
| credit-g | query_cluster_2_fm | 4 | 0.0500 | 0.2612 |
| credit-g | query_cluster_3_additive | not run | not run | 0.3343 |
| credit-g | query_cluster_3_fm | 4 | 0.0500 | 0.2589 |
| credit-g | residual_fm | 8 | 0.0500 | 0.1111 |
| diamonds | additive_ridge | not run | not run | 0.1818 |
| diamonds | id_fm | 4 | 0.0100 | 0.1217 |
| diamonds | pairwise_plus_deepsets_correction | 4 | 0.1000 | 0.1234 |
| diamonds | query_cluster_0_additive | not run | not run | 0.2441 |
| diamonds | query_cluster_0_fm | 4 | 0.0500 | 0.1503 |
| diamonds | query_cluster_1_additive | not run | not run | 0.1921 |
| diamonds | query_cluster_1_fm | 4 | 0.0500 | 0.1658 |
| diamonds | query_cluster_2_additive | not run | not run | 0.2178 |
| diamonds | query_cluster_2_fm | 4 | 0.0500 | 0.0886 |
| diamonds | query_cluster_3_additive | not run | not run | 0.1089 |
| diamonds | query_cluster_3_fm | 4 | 0.0500 | 0.0514 |
| diamonds | residual_fm | 16 | 0.0500 | 0.1824 |

## Selector Results
Interaction-aware versus strongest non-interaction win/tie/loss: 10/2/12. Mean normalized difference: -0.0566; median: -0.0052. Average ranks (lower is better): oracle_best_of_random=2.50, k_medoids=5.38, complementarity:geometry_plus_complementarity=5.50, pairwise_FM_greedy=5.81, pairwise_FM_swap=5.90, complementarity:cosine_diversity=6.00, feature_FM_greedy=6.33, CRUMB-like=6.42.

Query-row bootstrap comparisons for the prespecified FM+swap selector against the mean of 20 random contexts are in `results/processed/bootstrap_comparisons.csv`; 12/24 cells have a strictly positive 95% interval.

## Cross-Model Check
An official TabPFN-2.6 classifier `.fit()` was attempted after v2.6 Hugging Face blobs appeared in the cache, but it raised `TabPFNLicenseError`: the one-time license was not accepted and the package had no API key in this non-interactive environment. No resolved v2.6 checkpoint was usable. Cached TabPFN-2.5 weights were not silently substituted.

## Failures and Negative Results
- The official CRUMB, LUCoS, and VIP-COP repositories were not integrated; the reported methods are faithful lightweight `-like` controls and are never labeled official reproductions.
- TabPFN-2.6 runtime validation was attempted but blocked by the official one-time license gate; unresolved cache blobs were not treated as authorization, and v2.5 was not substituted.
- Direct TFM local search was intentionally restricted to the diagnostic datasets at K=32 and at most five exhaustive improving rounds; the audit file records whether it converged.

## Strongest Evidence FOR the Hypothesis
Static pairwise prediction did not provide positive evidence (the largest held-out ΔR2 was -0.0041). The evidence for exploitable set dependence instead comes from direct TFM search: after five selector-only swaps, final-test utility improved by 0.0304–0.1161 on both diagnostics. DeepSets also reached R2=0.5086, showing higher-order set structure on the strongest cell.

## Strongest Evidence AGAINST the Hypothesis
The median held-out ΔR2 was -0.0641; the interaction-aware selector lost to the strongest non-interaction control in 12/24 final-test cells. Selector gains selected on meta-validation are therefore not assumed to transfer unless the untouched test confirms them.

## Recommended Next Research Direction
Pivot from static pair factors to query-conditioned or higher-order set utility modeling; direct TFM search shows headroom, but current learned interactions do not reliably capture it.

## Files Produced
- `results/raw/context_evaluations/*.csv`: every membership set, utility, auxiliary metric, seed, and runtime.
- `results/raw/predictions/*.npz`: cached selector-query predictions for every sampled context.
- `results/raw/test_predictions.csv`: tidy per-query final-test predictions for bootstrap recomputation.
- `results/processed/utility_prediction.csv`, `utility_prediction_summary.csv`, and `b_ij_ablations.csv`: surrogate metrics and every failed/successful ablation.
- `results/processed/selector_results.csv`, `selector_comparisons.csv`, and `interaction_vs_noninteraction.csv`: all equal-budget selector results and aggregate comparisons.
- `results/processed/direct_interactions.csv`, `direct_search/*.csv`, and `bootstrap_comparisons.csv`: independent diagnostics and uncertainty.
- `results/processed/failure_fallbacks.csv`: 128-candidate, 1024-context, stronger-L2, query-conditioned, and pairwise+higher-order failure controls.
- `results/processed/cross_model_availability.json` and `official_selector_availability.json`: explicit integration/runtime availability audits.
- `plots/*.png`: six required result plots plus the strongest-dataset interaction heatmap.
- `experiments/run_pipeline.py`, `src/core.py`, `src/selectors.py`, and `tests/`: reproducible code and unit tests.
