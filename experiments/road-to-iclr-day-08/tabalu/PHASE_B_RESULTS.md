# Phase B — Structural Recovery Stress Result

Status: **partial pass; measurement-noise gate failed and triggers Phase C.**

## Setup

Twenty independent depth-1–3 programs were evaluated under clean data, 10%
target noise, 10% feature measurement noise, four independent nuisance
features, and four near-duplicate correlated features. Corruption conditions use
five seeds; clean search is deterministic. Evaluation is on the disjoint 8×
magnitude shell. The 420 planned records are present and finite.

## Result

| Condition | 8× NRMSE | Feature F1 | Operator accuracy | Exact graph |
|---|---:|---:|---:|---:|
| Clean | 0.000 | 1.000 | 0.967 | 0.900 |
| Target noise | 0.008 | 0.907 | 0.625 | 0.330 |
| Irrelevant features | 0.000 | 1.000 | 0.917 | 0.870 |
| Correlated alternatives | 3.24e-8 | 0.913 | 0.683 | 0.550 |
| Measurement noise | 0.697 | 0.952 | 0.618 | 0.250 |

All preregistered checks except measurement-noise NRMSE passed. The failure is
heavy-tailed: median measurement-noise NRMSE is 0.061, but the maximum is 34.60.
Protected-division programs dominate the tail. In the worst task, the ground
truth is `safe_divide(safe_divide(x3, x1), x0)`; noisy training operands lead to
wrong but superficially predictive structures and catastrophic OOD execution.

## Interpretation

The constrained search is stable to target noise, independent nuisance
features, and non-identifiable correlated representations. Literal graph
recovery falls while functional recovery remains high, confirming that exact
syntax is too strict when equivalent expressions exist.

Raw operands are not robust enough under measurement error. This is not a
reason to relax the failed threshold: it is the preregistered trigger for Phase
C's required raw/bounded/confidence-gated/unrestricted operand comparison. The
next experiment must determine whether a conservative operand estimator reduces
the division-driven tail without becoming an unrestricted predictor.
