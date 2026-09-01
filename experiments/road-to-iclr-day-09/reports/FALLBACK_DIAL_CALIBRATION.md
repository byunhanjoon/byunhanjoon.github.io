# Fallback benchmark — information calibration and routing alignment

Status: **independent benchmark replication complete; regression performance replicated,
classification performance did not**.

This continuation follows failure branch 16L after the E3 method kill. It strengthens
the theory/benchmark evidence without reopening M6 or relabeling the run as E5 method
confirmation.

## Exact dial calibration

New Proposition T4 gives the population mutual information of the balanced coupling.
For `K` mechanism/warp families,

`I(C;W) = a log(K a) + (K-1)b log(K b)`,

with `a=rho+(1-rho)/K` and `b=(1-rho)/K`. The warp and mechanism marginals stay uniform,
but information grows nonlinearly from zero to `log K`. This makes explicit that rho is
a coupling probability, not an information-linear x-axis. The implementation is unit
tested at both endpoints and over a 101-point monotonicity grid.

## Independent replication

The frozen replication used a new generator seed, 630 tasks per rho and task type,
`n_context=96`, `d=12`, and 256 queries. It retained `prior_dial_v1_1`, the six declared
mechanisms/warps, and all seven rho values. The 8,820 episodes completed in 486.7 seconds.

Marginal-only mechanism identification replicated more strongly than in development:

| Task | rho=0 accuracy [95% CI] | rho=1 accuracy [95% CI] |
|---|---:|---:|
| Classification | 0.127 [0.102, 0.152] | 0.978 [0.965, 0.989] |
| Regression | 0.144 [0.117, 0.171] | 0.987 [0.978, 0.995] |

The finite schedule stayed close to the exact population calibration: maximum absolute
empirical-minus-population MI residual was 0.0198 nats in the replication (0.0487 in the
smaller development run).

## Predictive performance: one replication and one failure

Shape-informed routing is evaluated against the invariant/stable selector using exactly
the same prefit mechanism experts. Positive numbers mean lower loss after adding marginal
shape.

| Task | Development rho=1 gain [95% CI] | Replication rho=1 gain [95% CI] |
|---|---:|---:|
| Classification log loss | +0.00733 [0.00433, 0.01028] | **-0.00402 [-0.00653, -0.00152]** |
| Regression MSE | +0.20523 [0.17594, 0.23603] | **+0.24339 [0.21664, 0.27044]** |

Regression is the real performance result: at rho=1, the same stable mixture fell from
MSE 0.45083 to 0.20743 when label-free marginal shape was added, a 54.0% relative risk
reduction. The rho=1 minus rho=0 utility contrast was +0.25078 with a 95% bootstrap CI of
[0.22277, 0.27916].

Classification does not replicate. Its rho=1 loss rose from 0.62780 to 0.63183. The
endpoint utility contrast was -0.00518 [-0.00781, -0.00263]. Therefore the earlier
classification predictive-utility claim is not stable across the frozen context/feature
regime and must not be used as general evidence that marginal shape improves prediction.

## Routing-alignment diagnosis

The failure is not lack of task-family information. At rho=1, the combined selector
identified the mechanism on 99.2% of classification tasks, yet predictive routing was
worse. Sending each task to its known matched-family expert was worse still: relative to
the stable mixture, the matched-family gain was -0.02301 [-0.02674, -0.01933]. In
regression, the same matched-family gain was +0.24505 [0.21759, 0.27350].

This exposes an important distinction:

> Task-family identification is not sufficient for predictive expert routing when the
> family-labelled experts are misspecified or when soft ensembling supplies calibration
> and variance reduction.

The result also explains why the E3 auxiliary mechanism/warp objective could reach high
accuracy without rescuing the learned gate. Generator metadata is a useful routing target
only when expert specialization is aligned with predictive loss. The raw field named
`oracle_expert_loss` is consequently interpreted only as **matched-family expert loss**,
not as a Bayes or loss oracle.

## Post-hoc regime-axis diagnostic

Because the independent replication changed context size and feature count together, two
fresh 420-task-per-cell controls isolated one axis at a time. These are explicitly
post-hoc and use independent seeds.

| Task | n=64,d=8 | n=96,d=8 | n=64,d=12 | n=96,d=12 |
|---|---:|---:|---:|---:|
| Classification routing gain | +0.00733 [0.00432, 0.01038] | +0.00631 [0.00308, 0.00963] | +0.00201 [-0.00079, 0.00471] | -0.00402 [-0.00660, -0.00148] |
| Regression routing gain | +0.20523 [0.17561, 0.23599] | +0.17423 [0.15226, 0.19725] | +0.21991 [0.19122, 0.24958] | +0.24339 [0.21649, 0.27117] |

At `d=8`, the classification context-size contrast was -0.00104 [-0.00546, 0.00335].
At `n=64`, increasing from 8 to 12 features reduced classification gain by 0.00532
[-0.00937, -0.00129]. The corner interaction was negative but uncertain. Thus larger
context alone does not explain the reversal; higher dimension is the supported weakening
axis, with an additional possible context-by-dimension interaction.

The mismatch is especially revealing because rho=1 classification mechanism-selection
accuracy increases across these cells (95.2%, 98.3%, 96.2%, 99.2%) while routing gain
falls. More recoverable generator metadata is not more useful predictive information for
the misspecified family experts. Regression stays strongly positive in all four cells.

## All-family decomposition

The dimensional classification weakening is concentrated rather than universal. At
`n=64,d=8`, interaction, periodic, and linear tasks contribute positive routing gains
of +0.03950, +0.01894, and +0.00662; additive, threshold, and partition routing is already
negative. Moving to `d=12` changes the gains as follows:

| Mechanism | d=12 minus d=8 routing gain [95% CI] |
|---|---:|
| Linear | +0.00702 [0.00256, 0.01190] |
| Additive | -0.00128 [-0.00657, 0.00401] |
| Threshold | -0.00008 [-0.00539, 0.00525] |
| Interaction | **-0.01985 [-0.03057, -0.00926]** |
| Partition | -0.00119 [-0.01489, 0.01265] |
| Periodic | **-0.01652 [-0.02552, -0.00769]** |

Thus the aggregate classification reversal is driven by loss of the interaction and
periodic advantages while three already-misaligned families remain harmful. Linear
routing actually improves. This supports an expert-specialization/alignment diagnosis,
not a universal claim that dimensionality makes marginal information harmful.

## Scientific decision

- Retain T1–T4 and the fixed-marginal information dial as the candidate theory/benchmark
  contribution.
- Retain regression as replicated evidence that explicit marginal shape can deliver
  large predictive gains in an information-rich prior.
- Reject a task-general performance claim: classification contradicted it.
- Do not reopen gate or dual-channel model development. A future method would first need
  loss-aligned experts/routing targets on a new protocol, not another metadata classifier.

Artifacts:

- `configs/fallback_dial_replication.yaml`
- `results/raw/fallback_dial_replication_dbe28a797c_t630_n96.npz`
- `results/processed/fallback_dial_calibration_v1.csv`
- `results/processed/fallback_dial_replication_contrasts_v1.csv`
- `results/processed/fallback_routing_alignment_v1.csv`
- `results/processed/fallback_routing_axis_v1.csv`
- `results/processed/fallback_routing_mechanisms_v1.csv`
- `figures/fallback_dial_information_performance_v1.png`
- `figures/fallback_routing_alignment_v1.png`
- `figures/fallback_routing_axis_v1.png`
- `figures/fallback_routing_mechanisms_v1.png`
