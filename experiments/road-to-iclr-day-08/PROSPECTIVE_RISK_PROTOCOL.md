# PROSPECTIVE PILOT FREEZE — COMPATIBILITY × CANDIDATE RELIABILITY

Freeze date: 2026-08-31 (Asia/Seoul), before downloading outcome arrays or
fitting any model in this pilot.

## Question and hypotheses

The Day-8 retrieval-risk identity separates query–candidate conditional-mean
compatibility from candidate-only outcome noise.  A symmetric learned distance
can represent the first term, but not a directed candidate penalty in general.

For a learned squared key distance `d_theta(q, i)`, define the frozen
training-free reranking score

```text
s_lambda(q, i)
  = d_theta(q, i)
  + lambda * median_j[d_theta(q, j)] * rank01(u_hat_i),
```

where `u_hat_i` is a strictly out-of-fold candidate unreliability estimate.
For regression it is a cross-fitted estimate of conditional residual variance;
for classification it is the cross-fitted Gini impurity
`1 - sum_c p_hat_i(c)^2`.  `rank01` is computed over training candidates only.
The query-dependent median makes `lambda` dimensionless without using labels.

- **H1 (mechanism):** the two-factor score selects lower proxy-risk
  neighborhoods than learned distance alone.
- **H2 (prediction):** lower-risk neighborhoods improve held-out prediction
  for both a TabR-like correction model and ModernNCA.
- **H3 (directionality):** a candidate-wise reliability term beats a
  distribution-matched random permutation of the same values.
- **H4 (boundary):** gains concentrate in datasets/splits with heterogeneous
  OOF uncertainty; near-homoscedastic tasks choose `lambda=0` or show no gain.

This is a mechanism pilot using the compact Day-8 implementations.  It is not
a published-implementation leaderboard or an ICLR method claim.

## Prospective public panel

These OpenML dataset IDs and versions are frozen before outcome inspection.
They do not overlap the original eight-dataset Day-8 screen.

| Dataset | OpenML ID | Task |
|---|---:|---|
| bank-marketing v1 | 1461 | binary classification |
| credit-g v1 | 31 | binary classification |
| electricity v1 | 151 | binary classification |
| jannis v1 | 41168 | multiclass classification |
| covertype v4 | 1596 | multiclass classification |
| MagicTelescope v1 | 1120 | binary classification |
| abalone v5 | 42726 | regression |
| cpu_act v2 | 573 | regression |
| elevators v1 | 216 | regression |
| Bike_Sharing_Demand v2 | 42712 | regression |
| sulfur v1 | 23515 | regression |
| superconduct v1 | 43174 | regression |

OpenML metadata reported every selected dataset as public; superconduct is
explicitly CC BY 4.0.  Features are downloaded from the immutable data-ID URL.

## Splits, preprocessing, and budgets

- Split seeds: `20260901`, `20260902`, `20260903`.
- Model seeds within every split: `20260911`, `20260912`, `20260913`.
- Each split is 60/20/20 and stratified for classification.
- Post-split caps: 8,192 train, 2,048 validation, 2,048 test.  Caps depend only
  on size, never outcomes; small datasets retain all available rows.
- Numerical values: train-median imputation and train-fitted standardization.
- Categorical values: train-fitted one-hot encoding with unknown-value support.
- Regression targets: train-mean/SD standardization.
- Models: compact Day-8 TabR and ModernNCA, raw representation, standard/deep
  key encoders, maximum 48 epochs with validation early stopping.
- Context size: 16 for TabR; ModernNCA retains its full-candidate softmax.
- `lambda` grid: `[0, 0.01, 0.03, 0.1, 0.3, 1, 3]`, selected on validation
  loss independently for the true and permuted candidate costs.  Ties choose
  the smaller value.  Test labels never select `lambda`.
- OOF proxy: five folds.  Conditional means/probabilities and the regression
  residual-variance model are cross-fitted on training rows.  Validation/test
  labels never enter either candidate scores or hyperparameter features.
- Primary metrics: accuracy and log loss for classification; standardized RMSE
  for regression.  Dataset-balanced signed score uses accuracy or negative
  RMSE, matching the original Day-8 convention.

## Frozen comparisons

For every dataset × split × model seed × retrieval model:

1. learned distance (`lambda=0`);
2. learned distance + true OOF candidate unreliability;
3. learned distance + a model-seed permutation of candidate unreliability.

Synthetic controls use the four Day-8 generators, eight fresh seeds, both
retrieval models, exact candidate variance, estimated OOF variance, and the
permutation control.  The high-noise S3 task is primary; the globally linear
S2 task is a boundary control.

## Success and kill gates

Promote to a published-implementation study only if all primary gates pass:

1. true reliability improves the dataset-balanced score on at least 8/12
   datasets for both TabR and ModernNCA;
2. it lowers cross-fitted top-16 proxy risk on at least 8/12 for both models;
3. its dataset-balanced gain exceeds the permuted-cost gain for both models;
4. the prediction-gain versus proxy-risk-reduction association is positive;
5. S3 shows a clear exact-variance advantage and the estimated proxy recovers
   at least 25% of that improvement without a comparable permutation effect.

Kill the method direction if either retrieval model has a non-positive
dataset-balanced gain, if the permutation control is comparable, or if lower
proxy risk does not accompany prediction gains.  In that case retain the risk
identity as an explanatory boundary only, not as an ICLR method.

## Reporting

All cells, including failures and `lambda=0` selections, must be retained.
Seeds are not independent datasets.  Report per-dataset split/seed means,
within-dataset uncertainty, W/L/ties, task-family results, mechanism metrics,
runtime, and exact deviations.  No dataset or lambda grid may be changed after
seeing outcomes.
