# Exploratory Phase III descriptor screen

Date: 2026-08-31

Status: **hypothesis-generating; Gate G2 ineligible because Phase II is incomplete**

## Question

Can train-only marginal descriptors predict the observed sensitivity of current TFMs well enough to prioritize a focused mechanism or synthetic follow-up?

## Protocol

The screen uses the identity-paired partial Phase II snapshot. It is restricted to TabICLv2 single/default, TabPFN-v2.5 single/default, and Mitra on five non-affine families: signed power, random monotone piecewise-linear, monotone spline, asinh, and composition.

For each model/transform/target cell with at least five datasets, ridge and random-forest regressors were evaluated with grouped dataset holdout. Both split seeds for a dataset remain in the same fold. The comparison is the training-fold mean predictor. There are 50 cells and 100 fitted model summaries; no multiple-testing or target-permutation claim is made.

## Screening results

Six fits met both prospective numerical thresholds (`R² >= 0.10` and at least 10% lower MAE than the fold-mean baseline). All six involve TabICLv2.

| Model | Transform | Target | Meta-model | Datasets / rows | R² | MAE improvement |
|---|---|---|---|---:|---:|---:|
| TabICLv2 single | monotone spline | loss gap | ridge | 8 / 15 | 0.662 | 46.0% |
| TabICLv2 default | random PWL | disagreement | random forest | 7 / 11 | 0.201 | 20.5% |
| TabICLv2 single | random PWL | disagreement | random forest | 8 / 15 | 0.200 | 22.5% |
| TabICLv2 single | monotone spline | disagreement | random forest | 8 / 15 | 0.179 | 20.3% |
| TabICLv2 default | monotone spline | loss gap | random forest | 7 / 11 | 0.119 | 14.8% |
| TabICLv2 default | signed power | disagreement | random forest | 7 / 11 | 0.116 | 17.6% |

No TabPFN-v2.5 or Mitra fit met both thresholds in this screen.

The repeated random-forest signals emphasize largest atom mass, skewness, and excess kurtosis for TabICL disagreement. The strongest ridge result emphasizes kurtosis, robust scale, and skewness. These are associations among correlated dataset summaries, not identified causal factors.

## Alternative explanations and limitations

- Only 5–8 datasets and 10–16 dataset/split rows contribute to each cell.
- Phase II missingness is model-dependent and may select easier or earlier-scheduled cells.
- The 100 fitted summaries create a multiple-comparisons risk.
- The frozen target-permutation control and directional stability analysis have not been run.
- Random-forest importance is not a mechanism.
- The screen is mostly regression for current TFMs and does not establish classification generality.

## Decision

Descriptor Route D is **promising but unresolved**. The best next bounded experiment is a preregistered TabICLv2-focused replication of the spline/random-PWL signals with complete identity-paired coverage, target permutations, and split-direction checks. A controlled S2/S3/S4 synthetic experiment is the strongest orthogonal route. Gate G2 remains open; pretraining and broad remedy claims should not be presented as validated.

Raw outputs are in `results/analysis/phase2_partial/02882b44384093b9-n8338/phase3_descriptors/`.
