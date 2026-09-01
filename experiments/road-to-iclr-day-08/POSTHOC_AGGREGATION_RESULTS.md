# POST-HOC RESULT — AGGREGATION-AWARE RETRIEVAL RISK

## Verdict

**STOP the retrieval-risk method direction.**
This corrective experiment was designed after the frozen candidate-wise screen failed;
its status cannot be upgraded retroactively. The frozen stop rule is applied below.

## Integrity

- Real panel: 216 cells and 1080 method rows across 12 datasets.
- Synthetic panel: 64 cells and 256 method rows across four mechanisms.
- Minimum objective-nonincreasing fraction: 1.000000.
- Maximum simplex error: 2.384e-07; minimum weight: 0.000e+00.
- Independent SLSQP audit: 21 systems; maximum objective gap 2.085e-08.
- Shortlist size was selected on validation only; no test target entered the QP or selection.

## Frozen real-data gates

| Model | gain vs model [dataset bootstrap 95% CI] | W/L/T vs model | W/L vs direct proxy | real gate |
|---|---:|---:|---:|---:|
| TabR | 0.01829 [0.00862, 0.02846] | 10/2/0 | 9/3 | PASS |
| ModernNCA | 0.01826 [0.00611, 0.03268] | 9/3/0 | 8/4 | PASS |

The gain is positive in both task families: TabR classification/regression
`+0.01278` / `+0.02381`, and ModernNCA
`+0.01304` / `+0.02348`.
Validation selected the maximum tested shortlist `k=64` in 67/108 TabR and
82/108 ModernNCA cells. This boundary preference leaves a wider-neighborhood
alternative unresolved; the frozen protocol forbids expanding the grid after outcomes.

## Per-dataset full-QP result

| Dataset | TabR vs model | TabR vs proxy | ModernNCA vs model | ModernNCA vs proxy |
|---|---:|---:|---:|---:|
| Bike_Sharing_Demand | 0.04863 | 0.04842 | 0.01941 | 0.05168 |
| MagicTelescope | 0.00439 | 0.00304 | 0.00168 | 0.00288 |
| abalone | 0.00118 | 0.01485 | -0.00161 | 0.01261 |
| bank-marketing | 0.00374 | -0.00125 | -0.00114 | -0.00629 |
| covertype | -0.00206 | 0.01644 | 0.04823 | 0.00298 |
| cpu_act | 0.01530 | 0.00225 | 0.02525 | 0.00917 |
| credit-g | 0.02389 | 0.02500 | -0.01000 | 0.00611 |
| electricity | 0.01774 | 0.00423 | 0.02485 | -0.00212 |
| elevators | -0.00373 | 0.13683 | 0.00091 | 0.16022 |
| jannis | 0.02897 | -0.01166 | 0.01459 | -0.02083 |
| sulfur | 0.03712 | 0.04917 | 0.01857 | 0.03837 |
| superconduct | 0.04432 | -0.00085 | 0.07836 | -0.00281 |

## Ablations

Dataset-balanced score gain relative to the original neural model:

| Model | full | mismatch-only | reliability-only |
|---|---:|---:|---:|
| TabR | 0.01829 | 0.01878 | -0.01183 |
| ModernNCA | 0.01826 | 0.01802 | 0.00099 |

## Synthetic S3 gate

A clear gain is operationalized before aggregation as negative mean RMSE change with
at least 6/8 seed wins. This makes the protocol's qualitative word `clear` auditable.

| Model | estimator | RMSE change | wins/8 | clear |
|---|---|---:|---:|---:|
| TabR | aggregate_exact | -0.00967 | 8/8 | YES |
| TabR | aggregate_estimated | -0.00914 | 8/8 | YES |
| ModernNCA | aggregate_exact | -0.00112 | 5/8 | NO |
| ModernNCA | aggregate_estimated | 0.00003 | 5/8 | NO |

## All synthetic mechanisms

| Task | Model | exact RMSE change (wins) | estimated RMSE change (wins) |
|---|---|---:|---:|
| S1_rotating | TabR | -0.00826 (8/8) | -0.00604 (8/8) |
| S1_rotating | ModernNCA | -0.00301 (7/8) | -0.00120 (5/8) |
| S2_global | TabR | -0.02624 (8/8) | -0.02231 (8/8) |
| S2_global | ModernNCA | -0.00116 (6/8) | 0.00038 (3/8) |
| S3_noise | TabR | -0.00967 (8/8) | -0.00914 (8/8) |
| S3_noise | ModernNCA | -0.00112 (5/8) | 0.00003 (5/8) |
| S4_warp | TabR | -0.00557 (8/8) | -0.00296 (8/8) |
| S4_warp | ModernNCA | -0.00401 (8/8) | -0.00149 (4/8) |

## Decision

Machine-readable decision: `stop_retrieval_risk_method_direction`.

Even a positive post-hoc result would require a newly frozen prospective replication.
Per protocol, this Day-8 run launches no larger benchmark.
