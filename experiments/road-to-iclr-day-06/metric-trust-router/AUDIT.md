# Metric Trust Router — integrity audit

Status: **PASS** (14/14 checks passed)

| Check | Status | Detail |
|---|:---:|---|
| protocol hash | PASS | 626c536c8dd637b6bbaafeb420ec382023faf2a999a318c0acbb30cade0aa4d8 |
| router completeness | PASS | 45/45 |
| unique router cells | PASS | 45 unique |
| artifact seals | PASS | protocol/status/test seal |
| declared task menu | PASS | 45 cells |
| five-fold state partition | PASS | all stored folds reproduce |
| fold-train-only landmarks | PASS | all landmark sets checked |
| fixed representations and alpha grid | PASS | 45 x 5 folds |
| zero-threshold decisions | PASS | all 45 decisions recomputed |
| no partial writes | PASS | none |
| router/outer-join code isolation | PASS | runner cannot load outer result folders |
| analysis regeneration | PASS | status=complete cells=45/45 |
| frozen feasibility gate | PASS | {'broad_ridge_source_balanced_no_worse': True, 'neural_source_balanced_improvement_at_least_5_percent': True, 'no_broad_ridge_source_degradation_above_2_percent': True, 'no_neural_source_degradation_above_1_percent': True, 'rejects_raw_in_both_medical_neural_partitions': True} |
| unit tests | PASS | 2 passed in 1.49s |
