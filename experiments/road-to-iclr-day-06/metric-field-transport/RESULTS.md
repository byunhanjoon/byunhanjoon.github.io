# Metric-Field Transport — development results

Status: **COMPLETE**

These are post-outcome development results. The original MPE test targets remain sealed; no result here is confirmatory evidence.

## Integrity and completeness

| Stage | Complete cells | Expected | Complete |
|---|---:|---:|:---:|
| E0 | 90 | 90 | True |
| E1a | 90 | 90 | True |
| E1b | 144 | 144 | True |
| E2 | 0 | 0 | True |

Protocol SHA-256: `b56e60589222c9da6f5786b5d53fa9c6eb7e6fd7314e80427d6918f8100f4132`. Every loaded artifact records `sealed_original_test=true` and `test_target_evaluations=0`.

## E0 — factorization control

| Condition vs direct weights | Wins | Mean relative change | Median relative change |
|---|---:|---:|---:|
| factor_random_learned | 8/18 | -2.86% | -0.25% |
| factor_identity_learned | 9/18 | -1.12% | -0.09% |
| factor_orthogonal_frozen | 3/18 | -2.87% | -2.64% |
| factor_rezero | 10/18 | +0.24% | +0.05% |

Identity/ReZero exact paired initial-score check: **True** (36 comparisons).

## E1a — metric-coordinate Ridge screen

| Representation | Full-table source-balanced MSE | Isolated-field source-balanced MSE |
|---|---:|---:|
| weights_m32 | 0.793984 | 1.041371 |
| affinity_m32 | 0.799009 | 1.080084 |
| distance_m32 | 0.810386 | 1.078353 |
| distance_m64 | 0.822933 | 1.009790 |
| distance_m128 | 0.817729 | 1.008724 |
| distance_all | 0.828702 | 1.042306 |
| distance_plus_weights_m128 | 0.821298 | 1.011939 |

Predeclared raw-distance selection: **distance_m32**.

## E1b — neural promotion gate

Selected candidate: **distance_m32**.

| Source | Weight MSE | Candidate MSE | Relative improvement |
|---|---:|---:|---:|
| ACS | 0.379062 | 0.374571 | +1.18% |
| CITI_BIKE | 0.647234 | 0.467964 | +27.70% |
| NYC_TLC | 0.294347 | 0.254222 | +13.63% |
| STRING_BENCHMARK | 0.848950 | 1.177341 | -38.68% |

Paired wins: 17/24 (70.8%). Source-balanced improvement: -4.82%.

- PASS — `beats_at_least_3_of_4_sources`
- PASS — `wins_at_least_60_percent_cells`
- FAIL — `source_balanced_improvement_at_least_1_percent`
- FAIL — `no_source_degradation_above_5_percent`

Decision: **REJECT E2 PROMOTION**.

## E2 — whole-state task transport

E2 was not authorized because E1 did not pass its frozen promotion gate.
