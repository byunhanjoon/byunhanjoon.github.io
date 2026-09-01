# Phase A — Arithmetic Extrapolation Result

Status: **GO for structural-recovery stress tests, with a narrowed induction
claim.** The exact-execution result passes; randomly initialized general program
discovery does not.

## Hypothesis

When the data-generating computation lies in a short protected-arithmetic chain
family, selecting that computation and executing it exactly should retain
accuracy under magnitude shift better than end-to-end approximation.

## Setup

- 100 independently generated programs; depths 1/2/3 occur 37/25/38 times.
- 88 unique expressions and 24–43 occurrences of each primitive.
- Five training seeds and independent 1,024/512/1,024-row train/validation/test
  samples per task.
- IID and disjoint magnitude shells at 2×, 4×, and 8×.
- Baselines: linear regression, random forest, sparse degree-3 polynomial
  regression, a three-layer MLP, and a two-layer EQL-style arithmetic network.
- TabALU uses exhaustive search over the complete depth-three chain family as a
  disclosed selector warm start, then straight-through selectors, compilation,
  pruning, and scalar-only output fitting.
- Confidence intervals resample independent tasks after averaging seeds within
  each task.

## Result

The completeness audit passes: 16,000/16,000 unique model/seed/task/shift cells,
all metrics finite, no failed fits, 100 task specifications, 500 compiled
programs, and 500 optimization histories.

| Model | IID NRMSE | 2× | 4× | 8× |
|---|---:|---:|---:|---:|
| TabALU compiled | 1.67e-19 | 6.66e-20 | 3.77e-20 | 2.06e-20 |
| MLP | 0.124 | 0.515 | 1.111 | 2.301 |
| Random forest | 0.172 | 1.389 | 1.499 | 1.577 |
| Linear | 0.726 | 1.445 | 1.297 | 1.558 |
| PolynomialSR | 0.310 | 5.088 | 23.397 | 151.664 |
| EQL | 0.280 | 13.995 | 129.265 | 1,276.966 |

Soft and hard TabALU predictions are numerically identical to ground truth in
all 4,000 cells each. Compiled predictions have maximum NRMSE 1.67e-17.
Therefore compilation does not explain away the result.

Structural recovery is strong but not perfect under syntactic matching:

- feature F1: 1.000;
- operator multiset accuracy: 0.948;
- exact ground-truth graph: 0.790;
- functional recovery on all evaluated samples: 1.000.

The exact-graph gap is primarily algebraic non-identifiability: an equivalent
shorter/sign-flipped/square-via-multiply expression can execute the same
function without matching the sampled syntax.

## Interpretation

This is clean evidence that deterministic execution preserves arithmetic
extrapolation once the correct short computation is selected. It is not yet
evidence that unconstrained differentiable program induction works: the random
straight-through variant failed even simple fixed-program diagnostics, and the
successful panel relies on a complete search over the known chain family.

The result clears the development-order gate for Phase B (structural recovery
under nuisance features, noise, and correlated alternatives). It does not clear
paper Claim 1 until a competitive symbolic-regression control and broader
arithmetic benchmark are run, and it does not clear the general-induction claim
until the warm-start restriction is ablated.

## Artifacts

- `results/phase_a_pilot/audit.json`: completeness and go/no-go decision.
- `results/phase_a_pilot/records.csv`: all per-cell prediction metrics.
- `results/phase_a_pilot/program_recovery.csv`: structural recovery metrics.
- `results/phase_a_pilot/summary.csv`: task-clustered means and bootstrap CIs.
- `results/phase_a_pilot/extrapolation_curve.{png,pdf}`: main curve.
- `results/phase_a_pilot/tasks/`, `programs/`, `histories/`: exact regeneration
  and compilation records.
