# Direct Exact-Execution Ablation

Status: **GO; exact arithmetic execution is retained.**

## Setup

The true graph is held fixed on 30 independently generated short arithmetic
programs and five training seeds. Each graph is evaluated in three ways:

- its protected operators execute exactly;
- each operator is replaced by a separately trained two-layer, 64-unit neural
  approximator receiving the same oracle operands;
- a whole-function MLP is reused from the Phase-A panel.

All models see the same 1,024 training rows. Evaluation spans 1×, 2×, 4×, and
8× input magnitudes. This intervention changes execution while removing program
discovery as a confound. All 1,800 planned records are present and finite.

## Result

| Model | 1× NRMSE | 2× NRMSE | 4× NRMSE | 8× NRMSE |
|---|---:|---:|---:|---:|
| Exact primitives | 0 | 0 | 0 | 0 |
| Neural primitives, oracle graph | 0.130 | 1.041 | 2.889 | 6.475 |
| Whole-function MLP | 0.158 | 0.581 | 1.512 | 3.461 |

The neural-primitive executor's mean error grows 49.8× from interpolation to
8× extrapolation. The exact executor remains exact. All three preregistered
checks pass. Task-cluster bootstrap intervals and the full records are stored in
`results/exact_execution_ablation/`.

## Interpretation

This is direct causal evidence for H1 within the synthetic operator family:
even with the correct graph and operands, learned local arithmetic does not
preserve the computation outside the training range. Exact execution does.
The result does not establish that the graph can be discovered in larger or
heterogeneous spaces, nor that every protected operator is the right semantic
choice for noisy real data.
