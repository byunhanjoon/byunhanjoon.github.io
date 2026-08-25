# Day 3 extension — what actually causes equivalent-basis sensitivity?

## Executive result

The function-matched decomposition changes the Day 3 interpretation.

Across 600 paired cells (1,200 trained models), three datasets, MLP and ResNet,
five seeds, three controlled condition numbers, five initialization/update
arms, and one exact natural encoding pair:

1. **Initialization is the dominant cause of the large controlled endpoint
   losses.** At κ=3000, ordinary AdamW had mean normalized harm `0.53995`.
   Starting from the same function reduced it to `0.02886`, a **94.66%
   reduction**. The reduction was 76.6%--98.3% in every dataset/model group.
2. **AdamW is still not basis equivariant.** Function-matched models began at
   mean prediction drift `5.8e-6`, then reached `0.0710` after one update. All
   90 controlled matched-AdamW cells moved apart; the dataset-cluster bootstrap
   interval for the mean one-step increase was `[0.0360, 0.0951]`.
3. **Early drift is not a universal performance predictor.** In
   leave-one-dataset-out evaluation, log condition number had R² `-0.201` and
   one-step drift had R² `-0.160`. The `+0.041` improvement missed the frozen
   `+0.10` gate, and both absolute predictions were poor.
4. **The ideal input-natural closure works on the full-rank controlled
   coordinates.** Its mean step-200 drift was `7.57e-5`, its maximum was
   `6.21e-4`, and mean endpoint harm was `1.64e-7`.
5. **That closure can fail numerically on a natural exact pair.** Adult's
   123-column natural representation had rank 116. The undamped inverse amplified
   its seven null directions, producing mean final drift `0.374` despite a
   basis-map condition number of one and representation error below `5e-13`.
   California and Diamond were full rank and retained numerical closure.

There were no failed cells. Every requested cell has all 12 trajectory rows
(six steps × two probe splits), and the maximum basis-relation error was
`7.64e-13`.

## 1. What was separated

For reference coordinates `X` and exactly equivalent changed coordinates

`X_changed = X B`,

the changed first-layer weight was initialized as

`W_changed = W_reference B^{-T}`.

Thus both networks produced the same deterministic function before training.
They then received the same rows, labels, minibatch order, and dropout masks.
Any later difference under AdamW is an optimizer-trajectory effect.

The study compared:

| Arm | Initial function paired? | First-layer update | Interpretation |
| --- | --- | --- | --- |
| Ordinary AdamW | No | AdamW | total effect |
| Matched AdamW | Yes | AdamW | optimizer-only residual |
| Covariance initialization + AdamW | In distribution | AdamW | deployable initialization control |
| Matched input-natural | Yes | inverse input moment | ideal closure |
| Ordinary input-natural | No | inverse input moment | update invariance without initialization invariance |

Training used exactly 200 updates, with no validation selection. Prediction
drift was recorded at steps 0, 1, 5, 20, 100, and 200.

## 2. Initialization explains most of the endpoint damage

Mean normalized harm at κ=3000 was:

| Dataset | Model | Ordinary AdamW | Function-matched AdamW | Reduction |
| --- | --- | ---: | ---: | ---: |
| Adult | MLP | 0.0244 | 0.0053 | 78.1% |
| Adult | ResNet | 0.0241 | 0.0057 | 76.6% |
| California | MLP | 0.3838 | 0.0343 | 91.1% |
| California | ResNet | 0.3128 | 0.0052 | 98.3% |
| Diamond | MLP | 1.2262 | 0.0772 | 93.7% |
| Diamond | ResNet | 1.2684 | 0.0454 | 96.4% |

This result rules out the simplest version of the original story. The model
does not merely receive a harder optimization surface after the basis change;
ordinary parameter initialization also samples a different initial function
prior. Under a fixed update budget, that prior accounts for most of the damage.

Covariance-metric initialization reached a similar endpoint to explicit
function matching on most controlled groups without knowing the hidden basis.
It did not make paired predictions identical, so it should be described as an
in-distribution intervention rather than exact canonicalization.

## 3. Optimizer non-equivariance is real but smaller

Function matching does not make AdamW invariant. Mean matched-AdamW drift after
one update increased monotonically with κ within every dataset/model group,
and final drift remained substantial. For example:

| Dataset/model | κ=1 step-1 drift | κ=30 | κ=3000 |
| --- | ---: | ---: | ---: |
| Adult / MLP | 0.0208 | 0.0445 | 0.0757 |
| California / MLP | 0.0502 | 0.1099 | 0.1700 |
| Diamond / MLP | 0.0517 | 0.1321 | 0.1976 |
| Adult / ResNet | 0.0123 | 0.0234 | 0.0395 |
| California / ResNet | 0.0257 | 0.0545 | 0.0819 |
| Diamond / ResNet | 0.0279 | 0.0664 | 0.0950 |

This is direct causal evidence about the update rule: the models begin from the
same function, see paired stochasticity, and separate after one AdamW step.

However, similar drift magnitudes implied very different performance costs on
different datasets. Orbit drift therefore detects non-equivariance but does
not by itself measure how damaging that divergence will be.

## 4. The preregistered predictive hypothesis failed

The primary novelty gate required one-step orbit drift to improve
leave-one-dataset-out R² over log condition number by at least `0.10`.

| Predictor | Held-out-dataset R² | MAE | Sign agreement |
| --- | ---: | ---: | ---: |
| Log condition number | -0.201 | 0.0207 | 70.0% |
| One-step prediction drift | -0.160 | 0.0215 | 73.3% |
| Early drift AUC | -0.233 | 0.0222 | 65.6% |

One-step drift improved R² by only `0.041`; early AUC was worse. Neither
predictor generalized in absolute terms. The natural-pair transfer test reached
exactly 60% sign agreement but badly overpredicted harm (`-0.0211` predicted
mean versus `0.00012` observed mean), so it is not positive evidence.

The honest conclusion is:

> Function-space drift is a clean measure of optimizer non-equivariance, but
> final harm also depends on task curvature, target alignment, and where the
> trajectory moves. A scalar drift magnitude is insufficient.

## 5. Closure and the rank-deficiency counterexample

On the jointly whitened, full-rank controlled coordinates, function-matched
input-natural training nearly closed the entire trajectory. Ordinary
input-natural training did not: an equivariant update cannot repair two models
that started as different functions.

The Adult natural pair exposed the complementary failure. Each semantic block
was whitened and the cumulative/local map was essentially orthogonal, but the
combined 123-column matrix had only rank 116. The implementation followed the
frozen undamped policy, so eigenvalues at the numerical floor were inverted.
Small coordinate-dependent floating-point differences became enormous first
updates. California and Diamond had full rank and closed normally.

This does not refute ideal natural-gradient invariance. It demonstrates that an
exact algebraic guarantee needs an explicit effective-rank/pseudoinverse policy
before it becomes a robust algorithm.

## 6. Revised scientific claim

The strongest supported statement is now:

> Information-equivalent tabular coordinates change both the function prior
> induced by ordinary initialization and the subsequent optimizer trajectory.
> Function-prior mismatch explains about 95% of the high-condition endpoint
> damage in this fixed-budget study. AdamW remains non-equivariant after exact
> function matching, while ideal affine-equivariant updates close full-rank
> trajectories but can become numerically unstable at rank boundaries.

This is a more informative result than “condition number hurts,” but it is
still a three-dataset mechanistic extension. The failed universal-prediction
gate and the Adult rank counterexample must remain central rather than being
hidden as limitations.

## Reproduction

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m pytest -q
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.trajectory_decomposition --device cuda:0
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_trajectory_decomposition
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_trajectory_decomposition
```

Primary outputs are in `results/day3/trajectory_decomposition/`.

