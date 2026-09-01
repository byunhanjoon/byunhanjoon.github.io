# General Numeric Tabular Pilot

Status: **modest gate passed; no broad superiority or heterogeneous-data claim.**

## Setup

Five random seeds and 60/20/20 splits are used on scikit-learn's Diabetes
regression data and two binary classification tasks derived from Wisconsin
Diagnostic Breast Cancer and Wine Recognition. Classification splits are
stratified. TabALU here is deliberately restricted to a sparse affine model
over exact raw, squared, and pairwise-product numeric terms. Controls are
linear/logistic regression, Random Forest, XGBoost, MLP, and CatBoost. All 90
planned records are finite.

## Result

| Dataset | Metric ↓ | Best baseline | Sparse exact | Ratio |
|---|---|---:|---:|---:|
| Diabetes | NRMSE | 0.733 (linear) | 0.742 | 1.011 |
| Breast cancer | log loss | 0.0735 (logistic) | 0.0851 | 1.158 |
| Wine binary | log loss | 0.0510 (logistic) | 0.0746 | 1.462 |

The model is within the frozen 1.25× competitiveness threshold on two of three
datasets and remains below the 2× catastrophic-failure ceiling on all three.
Plain linear/logistic models are best on every dataset, so this result does not
show value from the extra exact interaction library.

## Interpretation

The restricted exact model can behave as a compact conventional tabular model
(9–28 selected terms) without catastrophic failure on this tiny numeric-only
subset. It does not establish broad usefulness, average-rank superiority, or
benefit from program induction. Larger heterogeneous datasets with missingness,
categorical variables, and paired statistical comparisons remain mandatory.
