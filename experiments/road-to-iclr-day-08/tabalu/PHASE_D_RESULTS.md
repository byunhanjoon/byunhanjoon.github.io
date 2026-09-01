# Phase D — Categorical Regime Routing Result

Status: **GO for temporal/structure–parameter experiments; categorical result
only.**

## Setup

Twelve two-regime tasks and five seeds. Each regime has a distinct executable
program. A noisy categorical context is clustered without target labels, a
sparse router learns those pseudo-labels, and each cluster receives a compiled
program. Training is balanced; the 8× test shell shifts regime-1 prevalence to
80%. Controls are one global program, MLP, random forest, and a conventional
neural MoE. All 720 planned records are finite.

## Result

| Model | IID NRMSE | 8× + frequency-shift NRMSE | OOD regime accuracy |
|---|---:|---:|---:|
| Program MoE, hard | 4.05e-10 | 4.01e-12 | 1.000 |
| Program MoE, soft | 2.86e-4 | 3.32e-4 | 1.000 |
| Neural MoE | 0.141 | 0.954 | 0.542 |
| Single MLP | 0.125 | 0.949 | — |
| Random forest | 0.262 | 1.174 | — |
| Single program | 0.786 | 1.700 | — |

Executable experts recover operator multisets with 0.933 accuracy. Router
entropy is 0.0016 and permutation-aligned regime accuracy is 1.0. All five
preregistered prediction, routing, and recovery checks pass.

## Interpretation

Sparse exact experts decisively outperform both a single exact program and a
conventional neural MoE when a nearly explicit categorical regime indicator is
available. This supports the narrow H3 mechanism: exact experts retain
arithmetic extrapolation after correct routing.

This is not evidence for difficult latent-regime discovery. The two context
clusters are almost perfectly separated, and the program search shares the
generator's short-chain family. The next gate must move to time, scarce
per-regime samples, and shared structure with changing coefficients.
