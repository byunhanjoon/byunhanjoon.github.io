# Day 3 experiments: from a useful trick to a falsifiable method

This directory asks whether the Day 2 exact-state results support a broad
tabular-learning method. The answer is deliberately split in two:

1. **Mechanism:** yes. A smooth numerical basis can leave stable state-local or
   joint-state residuals, and a small exact-state shortcut can make them easier
   for several neural backbones to learn.
2. **Breadth:** not yet. The frozen untouched-dataset gate selected a structure
   on only 1/6 datasets, below the predeclared 2/6 threshold. The exact-state
   method should therefore remain a mechanistic case study rather than the
   headline ICLR contribution.

The stronger paper direction is to use this result as a concrete mechanism and
intervention inside the already-confirmed **information-equivalent schema
sensitivity** project in `/home/byunhanjoon/2027ICLR/projects/multi_ple/schema_fragmentation`.

## Fixed method

The starting representation is schema information plus 128-bin PLE. Discovery
uses training rows only:

- fit a smooth Ridge or logistic probe;
- test support-shrunk singleton residual maps for numerical columns with at
  most 128 states;
- for pairs with at most 512 joint states, backfit both singleton effects and
  project their weighted marginals out of the joint residual map;
- accept a term only when it improves all five discovery folds and clears a
  0.05% relative-loss threshold;
- greedily keep at most four singleton and four pair terms under a 512-state
  representation budget.

The deployed neural view contains target-free train-vocabulary one-hot states.
Each active coordinate is multiplied by `count / (count + 20)`; an unseen state
maps to zero. Neural widths are adjusted to the PLE parameter budget.

The immutable protocol is in `configs/frozen_protocol.json`.

## What survived

### Ground-truth and null checks

- On smooth synthetic data, the selector abstained in all 80 unpermuted runs.
- It recovered the true pure interaction in 30%, 75%, 100%, and 100% of runs
  at 500, 1,000, 3,000, and 10,000 training rows.
- Mean downstream MSE reduction for that interaction rose from 0.60% to 58.83%
  over the same sample-size sequence and reached the oracle result by 3,000
  rows.
- On 20 shuffled-label repetitions each for Adult and Black Friday, it made
  zero discoveries.

Singleton recovery is intentionally not presented as a monotone consistency
result. With enough PLE knots, PLE and singleton identity span the same
finite-support function class, so the exact singleton effect becomes
non-identifiable as PLE itself improves. That observation motivates the
geometry analysis rather than an expressivity claim.

### Nested development evidence

| Dataset | Full-data selection | Mean outer loss reduction | Outer wins | Non-nested estimate |
| --- | --- | ---: | ---: | ---: |
| Adult | capital gain; capital loss | 3.86% | 15/15 | 4.17% |
| Black Friday | occupation + three product-category pairs | 3.29% | 15/15 | 3.29% |

The full selector is repeated for three split seeds. Every outer fold reruns
five-fold discovery using only the outer-training partition. The small Adult
gap between 4.17% and 3.86% quantifies selection optimism; Black Friday shows
almost none.

On the downstream Adult benchmark, the support-gated encoder improves test AUC
by 1.18, 1.16, and 1.13 percentage points for MLP, ResNet, and TabM,
respectively. It wins all 9/9 paired runs and stays within 0.19% of the baseline
parameter count. Its mean gain over cardinality-matched random views is 1.18
percentage points. Adding every eligible singleton also gains 1.05 points on
average, so Adult supports the value of identity coordinates more strongly than
the value of sparse selection.

On Black Friday, the selected singleton and pure pairs reduce RMSE by 0.05%,
0.14%, and 0.20% for MLP, ResNet, and TabM. The mean is 0.13%, with 7/9 paired
wins. This is directionally consistent but much smaller and noisier than the
nested linear correction, so it should not be a headline performance result.

### Prospective untouched block

| Dataset | Full-data selection | Mean nested loss reduction | Outer wins |
| --- | --- | ---: | ---: |
| Wine Quality | abstain | 0.00% | 0/5 |
| Miami Housing | pair 11 + 13 | 1.64% | 5/5 |
| Food Delivery Time | abstain | 0.01% | 1/5 |
| Seismic Bumps | abstain | 0.00% | 0/5 |
| HELOC | abstain | -0.56% | 0/5 |
| Credit Card Default | abstain | 0.00% | 0/5 |

Miami was the only full-data nonempty selection. On its previously untouched
official test split, the frozen support-gated pair reduces RMSE by 1.76% for an
MLP and 1.63% for a ResNet. It beats the matched-cardinality random pair on both
backbones. Adding every eligible singleton instead changes RMSE by -0.26% and
-1.62%, showing that the gain is not explained by indiscriminate width.

The frozen gate required at least two nonempty datasets. Because only one was
found, no broad confirmation sweep is authorized.

## Mechanistic interpretation

For a column with `K` supported values, cumulative PLE evaluated on those
values has centered rank `K - 1`, exactly like centered one-hot identity. The
two bases carry the same information and span the same singleton functions,
but they induce different conditioning and L2 geometry.

For Adult capital gain, both bases have centered rank 115. The support-weighted
condition number is 10,738 for PLE versus 46 for identity, and the minimum
coefficient norm for the observed residual state effect is 23.05 versus 3.01.
Capital loss is also better conditioned: 474 versus 35. This gives a concrete
explanation for why identity can help without adding representational power.

Pure pairs are different. A crossed identity basis adds joint-state functions
outside an additive collection of univariate PLE blocks. Removing both
singleton marginals makes that incremental expressivity testable rather than
crediting a pair for two main effects.

## Reproduction map

- `hierarchical_residual.py`: selector, pure interaction projection, nested audit
- `adaptive_atomic_benchmark.py`: parameter-matched neural comparisons
- `synthetic_recovery.py`: recovery, code permutation, and sample efficiency
- `permutation_null.py`: shuffled-label false-discovery check
- `basis_geometry.py`: rank, conditioning, and coefficient-norm analysis
- `tabarena_nested_screen.py`: frozen untouched discovery gate
- `tabarena_adaptive_screen.py`: prospective selected-dataset neural test
- `tree_references.py`: CatBoost, LightGBM, and XGBoost references
- `THEORY.md`: limited formal claims
- `test_hierarchical_residual.py`: invariants and regression tests

Use the Python environment at
`/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python`.

## ICLR decision

Do not pitch this as “a universally better numerical encoder.” The untouched
breadth gate falsified that story. The publishable insight is narrower and more
interesting: **information-equivalent bases can have radically different
optimization geometry, and residual structure can identify where a local
basis shortcut is useful.** Adult supplies an additive change-of-basis example;
Miami and Black Friday supply pure-interaction examples.

The ICLR-scale paper should lead with information-equivalent schema sensitivity
across many datasets and models, then use adaptive exact-state views as one
mechanistically motivated robustness intervention. The next decisive experiment
is the frozen 51-dataset prevalence and compute-matched intervention study in
the schema-fragmentation project—not further tuning on these three successes.
