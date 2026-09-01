# Repeated held-out partition audit

Status: protocol frozen before inspecting repeated-partition outcomes; analysis
complete.

## Question

The external model-selection panel showed that a lower-variance estimate of the
validation nuisance quotient need not improve the selected model on the test
split. Is that observation exceptional for the original validation/test
partition, or is candidate ranking itself unstable across exchangeable
held-out partitions?

## Scope and estimand

Use the eight OpenML external cells and the eight task-balanced OpenML cells.
For every trained candidate, average all 128 stored nuisance predictions to
obtain the exact finite-orbit quotient prediction for each validation and test
row. The models were fitted using the training split only; their validation and
test rows are therefore pooled for this conditional audit.

For each dataset, generate 4,096 deterministic repartitions of the pooled
evaluation rows, always preserving the original validation-set size. Preserve
the original validation class counts for classification. Use simple random
partitions for regression. No model is retrained.

For each repartition report:

1. whether the exact validation-quotient winner is the exact test-quotient
   winner;
2. the test regret of the validation winner (the target-shift floor faced by a
   perfect validation-score estimator);
3. Spearman candidate-rank correlation between validation and test;
4. the best-minus-second-best validation margin.

Also report the same quantities for the original partition and its empirical
percentile in the repartition distribution.

## Frozen interpretation

- **Finite-partition explanation supported:** the original target-shift floor
  lies inside the central 95% repeated-partition interval and winner agreement
  is materially below 95% for at least half of the datasets in that panel.
- **Original split exceptional:** its target-shift floor exceeds the 97.5th
  repeated-partition percentile in at least half of the datasets in that panel.
- Otherwise the result is mixed.

These labels are descriptive conditional randomization diagnostics. Pooling
assumes exchangeability of the two held-out samples, so the audit cannot rule
out genuine distribution shift or replace evaluation on new sources.

## Outcome

The frozen interpretation is **finite-partition explanation supported** for
both panels.

- External OpenML: mean repeated-partition winner agreement is 60.4% versus
  25.0% on the original splits. Seven of eight original target-shift floors
  lie inside their conditional 95% intervals. Sonar alone is above its 97.5%
  quantile; its floor is 0.290 and accounts for 77% of the total original
  external floor. Mean repeated-partition floor is 0.00754 versus original
  0.0470.
- Task-balanced OpenML: mean repeated-partition agreement is 77.8% versus
  100% originally. All four classification datasets have only 49.5--66.6%
  agreement, whereas three large regression datasets have 100% and Abalone
  has 99.9%. The perfect original task-balanced transfer was therefore a
  favorable partition outcome, not evidence that classification ranks are
  intrinsically stable.

This refines the earlier language: the observed floor is primarily consistent
with finite held-out sampling instability, with an exceptional Sonar split.
It should not be labeled population or dataset shift without new data.
