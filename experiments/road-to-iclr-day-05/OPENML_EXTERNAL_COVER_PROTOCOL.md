# External OpenML exact nuisance-cover panel

Status: frozen before cover outcomes on 2026-08-28.

These eight OpenML datasets were used earlier in Day 5 only for the failed
HeteroBag semantic-placebo test. No schema-cover or exact-factorial result has
been computed on them. They now provide eight new dataset sources for a
prospective transport check of the paper's primary method.

## Frozen panel

- datasets: Breast-W (15), Sonar (40), PC4 (1049), KC1 (1067), Blood
  Transfusion (1464), ILPD (1480), WDBC (1510), Mammography (310);
- deterministic 60/20/20 train/validation/test splits with seed 2026082814;
- algorithms: logistic regression, random forest, histogram gradient
  boosting, and Adam MLP;
- exact product: four feature orders, binary target numbering, four model
  seeds, and four category-numbering maps when categorical fields are present;
- retain all aligned validation/test predictions for the exact 32 cells on
  numeric-only datasets and 64 cells on ILPD, whose one binary categorical
  field admits exactly two category-numbering maps.

The primary frozen analysis is the same validation-material screen and
strength-2 OA-16 comparison against IID-16, four strength-1 blocks, and four
seed blocks. Each dataset is its own source group. The gate requires at least
6/8 dataset-source means with material cells to beat all controls. Strength-3
and downstream four-candidate model selection are predeclared secondary
analyses. Reuse of dataset identities is explicit; this is independent-source
transport for the cover hypothesis, not a second untouched HeteroBag test.
