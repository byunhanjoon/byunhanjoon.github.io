# Phase E — Temporal Structure/Parameter Separation Result

Status: **partial; retain discrete regime coefficients, reject unconstrained
context coefficients, and do not claim a sample-efficiency win.**

## Setup

Sixteen tasks, five seeds, one invariant executable graph, an abrupt coefficient
change at time 0.70, about 64 post-change training rows, 5% target noise, and a
future-only 8× test period. Known synthetic regime labels are supplied to the
independent-program and shared-structure controls to isolate structure versus
parameters. Context coefficients, MLP, and neural MoE receive time but not the
labels. All 960 planned records are finite.

## Result

| Model | IID NRMSE | Future 8× NRMSE |
|---|---:|---:|
| Shared graph + regime coefficients | 0.00447 | 0.00639 |
| Independent program per regime | 0.00453 | 0.00591 |
| Context-conditioned coefficients | 0.0911 | 0.1385 |
| Global program | 0.478 | 0.761 |
| MLP | 0.352 | 2.449 |
| Neural MoE | 0.389 | 4.248 |

Shared structure beats the global program and neural MoE and remains within 8%
of independent experts. It does not improve on independent programs in this
panel. Operator multiset recovery is 0.65, below the 0.70 gate, despite excellent
functional prediction; equivalent short expressions again weaken literal graph
metrics. The context-conditioned variant is 21.7× worse than discrete regime
coefficients and fails its gate.

## Interpretation

Stable executable structure plus discrete coefficients is a strong predictive
bias under this controlled temporal shift, but the experiment does not show a
sample-efficiency advantage over independent programs. Continuous
context-conditioned coefficients are too flexible and extrapolate poorly, so
Variant C is rejected. The surviving method is the simpler known-regime shared
graph with discrete constants; temporal regime discovery remains unproven.
