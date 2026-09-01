# Metric Trust Router — exploratory results

Status: **COMPLETE** (45/45 router cells)

These results are post-outcome feasibility evidence only. Original test targets remain sealed.

## Broad nine-task Ridge join

| Source | Weight MSE | Always raw MSE | Routed MSE | Routed improvement | Raw selections |
|---|---:|---:|---:|---:|---:|
| ACS | 0.448118 | 0.446091 | 0.447319 | +0.18% | 5/10 |
| BTS | 1.138846 | 1.137426 | 1.137546 | +0.11% | 9/10 |
| CITI_BIKE | 0.993526 | 0.988435 | 0.988435 | +0.51% | 5/5 |
| NYC_TLC | 0.765858 | 0.758657 | 0.759973 | +0.77% | 9/10 |
| STRING_BENCHMARK | 0.623569 | 0.721320 | 0.629109 | -0.89% | 1/10 |

Source-balanced routed improvement: +0.19%. Wins/ties/losses: 24/16/5.

## Four-source neural join

| Source | Weight MSE | Always raw MSE | Routed MSE | Routed improvement | Raw selections |
|---|---:|---:|---:|---:|---:|
| ACS | 0.379062 | 0.374571 | 0.375333 | +0.98% | 3/6 |
| CITI_BIKE | 0.647234 | 0.467964 | 0.467964 | +27.70% | 6/6 |
| NYC_TLC | 0.294347 | 0.254222 | 0.254222 | +13.63% | 6/6 |
| STRING_BENCHMARK | 0.848950 | 1.177341 | 0.848950 | +0.00% | 0/6 |

Source-balanced routed improvement: +10.28%. Wins/ties/losses: 15/9/0.

## Frozen feasibility gate

- PASS — `neural_source_balanced_improvement_at_least_5_percent`
- PASS — `no_neural_source_degradation_above_1_percent`
- PASS — `rejects_raw_in_both_medical_neural_partitions`
- PASS — `broad_ridge_source_balanced_no_worse`
- PASS — `no_broad_ridge_source_degradation_above_2_percent`

Decision: **RECOMMEND NEW-DATA CONFIRMATION**.
