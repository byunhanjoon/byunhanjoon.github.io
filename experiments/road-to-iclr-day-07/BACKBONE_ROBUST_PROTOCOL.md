# Post-hoc protocol — backbone-robust value of an optional field bias

Status: **EXPLORATORY; FORMULATED AFTER THE ARCHITECTURE GATE FAILED**

## Hypothesis

The value of side information is conditional not only on the table and target,
but on the residual task produced by the learned base. A deployable claim that
"this field geometry helps" should therefore survive ordinary model
specification uncertainty.

For a predeclared base set `B` and operator set `A`, define the
specification-robust value as

`V_robust(a) = inf_{b in B} Delta_b(a)`.

Expose an operator only when a simultaneous lower bound on `V_robust(a)` is
positive; otherwise use the exact zero adapter. Here `B` is MLP, ResNet,
FT-Transformer, and TabM, and `A` is the nine-operator geometry library.

## Development replay

Use the twenty-fold state-level means and fold-cluster standard errors from
`STATE_CERTIFICATE_PROTOCOL.md`. Apply one Gaussian Bonferroni correction over
all `4 × 9` backbone-operator comparisons:

`z = Phi^-1(1 - 0.05 / (2 × 4 × 9))`.

For each task-split and operator, take the minimum backbone LCB. Choose the
operator with the largest minimum LCB, and deploy it for every backbone only
if that value is positive.

Baselines are independent per-backbone family-wise decisions and an averaged
backbone LCB. The latter tests whether average apparent robustness can hide a
single-backbone failure.

## Interpretation constraints

This rule was formulated after observing architecture-specific harm. Its
result is hypothesis-generating only. The four architectures are a convenient
model set, not a validated Rashomon set, Gaussian fold bounds are not finite-
sample guarantees, and reused sources cannot establish generalization. A
confirmatory test must predeclare how the admissible base set is constructed
and evaluate new source families.
