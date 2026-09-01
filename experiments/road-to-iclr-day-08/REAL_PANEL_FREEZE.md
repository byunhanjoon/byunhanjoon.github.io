# REAL-DATA SCREENING PANEL — FROZEN BEFORE MODEL RESULTS

Freeze time: 2026-08-31, before any Day-8 model fit or outcome inspection.

Source: cached public TabPack-format arrays already present under
`experiments/road-to-iclr-day-01/data`. No Day-8 dataset may be replaced based
on performance.

| Dataset | Task | Rows | Numerical | Categorical/binary | Day-8 train cap | Reason for inclusion |
|---|---|---:|---:|---:|---:|---|
| Adult | binary classification | 48,842 | 6 | 8 | 4,096 | mixed types, moderate size |
| Churn | binary classification | 10,000 | 7 | 4 | 4,096 | mixed types, lower-dimensional |
| HIGGS-small | binary classification | 98,049 | 28 | 0 | 4,096 | numerical, noisier classification |
| Otto | 9-class classification | 61,878 | 93 | 0 | 4,096 | high-dimensional multiclass |
| California Housing | regression | 20,640 | 8 | 0 | 4,096 | low-dimensional spatial/nonlinear target |
| Diamond | regression | 53,940 | 6 | 3 | 4,096 | mixed-type regression |
| House | regression | 22,784 | 16 | 0 | 4,096 | medium-dimensional numerical regression |
| Black Friday | regression | 166,821 | 4 | 5 | 4,096 | larger mixed-type regression |

## Frozen protocol

- Split: deterministic 60/20/20 train/validation/test using seed 20260831,
  stratified for classification. The train cap is applied after splitting and
  is stratified for classification.
- Seeds: 20260831, 20260832, 20260833 for the core first-pass cells.
- Models: MLP, TabR-like retrieval/correction model, and ModernNCA.
- Core representations: raw standardized inputs, quantile PLE, periodic/PLR,
  a trainable monotone quantile-initialized warp (LocalWarp), and deliberately wrong inverse
  warp. Learned-knot geometry is represented by a compact trainable monotone
  piecewise-linear warp in the branch ablation; no novelty is assigned to it.
- Main branch ablation: raw/raw, nonlinear-prediction/raw-retrieval,
  raw-prediction/nonlinear-retrieval, nonlinear/nonlinear.
- Core training uses at most 18 epochs with validation early stopping; this is
  a direction screen, not SOTA tuning.
- Retrieval candidate pool: capped training set; fixed diagnostic query subset
  of 128 test rows; top-k = 16.
- Primary metrics: accuracy for classification and standardized RMSE for
  regression, converted to a unified higher-is-better score only for
  dataset-balanced summaries.
- Categorical preprocessing is frozen to train-fitted ordinal codes followed
  by one-hot columns for all three core models. The same processed matrix is
  used across representation comparisons.
- Retrieval risk analysis uses five-fold out-of-fold ExtraTrees conditional
  means/probabilities. Regression candidate-noise estimates use a second
  five-fold out-of-fold smoother over squared OOF residuals; no test target
  enters the proxy used to rank candidate risk.

The panel intentionally favors local availability and breadth over matching a
particular benchmark leaderboard. Any runtime reduction must apply uniformly
by dataset size and be documented, never selected from observed performance.
