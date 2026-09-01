# Frozen protocol H6 — Semantic Lyapunov Screen

Status: **FROZEN AFTER THREE DEVELOPMENT BUNDLES, BEFORE 33 TEST BUNDLES**  
Freeze time: 2026-08-29 00:12 Asia/Seoul

## Development observation and prospective split

H6 was motivated after observing three completed H3 bundles:

- `bank_marketing_subscription__mlp__seed8101`;
- `bank_marketing_subscription__resnet__seed8101`;
- `credit_card_default__mlp__seed8101`.

Bank/ResNet stayed near the numerical floor through epoch 20 and then became
macroscopically schema-divergent by epoch 50.  All three named bundles are
excluded from every H6 gate.  At freeze time, the other 33 H3 artifacts did
not exist.  H6 does not change or add any training.

## Hypothesis

In the locally linear regime, aligned prediction displacement approximately
obeys `delta_t = M_t delta_0 + ...`.  If one unstable mode dominates over an
early interval, its squared magnitude is approximately exponential, so

`log10 E_g ||delta_t(g)||^2 approximately a + b t`.

The estimated early growth rate `b` may reveal a delayed instability even when
the discrepancy level at epoch 20 is tiny.  H6 tests whether extrapolating the
line through epochs 5, 10, and 20 is a useful screen for material orbit error
at epoch 200.

This is a finite-horizon empirical approximation, not a claim that nonlinear
neural training has a global Lyapunov exponent.

## Frozen score and target

For each prospective H3 bundle, average FP32 validation prediction MSE over
the three matched nonidentity schema views at each checkpoint.  Fit ordinary
least squares to

`(t, log10(mean_mse_t + 1e-30))`, for `t in {5,10,20}`.

The primary score is the extrapolated log MSE at epoch 200.  The binary target
is whether mean FP32 validation orbit MSE at epoch 200 exceeds `1e-5`.
The fixed decision predicts material divergence when the extrapolated score is
greater than `-5`.

The negative comparator is the unextrapolated `log10` epoch-20 MSE.  It tests
whether growth rate adds information beyond simply measuring the early level.

## Frozen gates

H6 passes only if all are true on the 33 untouched bundles:

1. all 33 artifacts are complete and pass the Day-6 integrity audit;
2. pooled AUROC is at least 0.90, with both target classes present;
3. AUROC is at least 0.80 in at least 2/3 datasets, with both classes required
   for a dataset to count;
4. the fixed `-5` decision has sensitivity and specificity both at least 0.75;
5. equal-dataset rank-pooled Spearman between score and final log MSE is at
   least 0.70;
6. pooled AUROC exceeds the raw epoch-20-level AUROC by at least 0.10.

For gate 5, bundle scores and targets are average-ranked within dataset before
concatenation.  Seeds are repeated measurements, and dataset-balanced results
are reported alongside pooled bundle-level metrics.

## Decision and novelty boundary

Failure means a 20-epoch shadow cannot safely screen 200-epoch semantic
instability under this matrix; H6 is discarded rather than rescued with a new
window or nonlinear fit.  Passing retains a cheap numerical-stability screen,
but a new dataset/model panel would still be required before a paper claim.

Early learning-curve prediction, perturbation growth, and finite-time Lyapunov
analysis are established ideas.  The narrow candidate contribution is the
growth rate of an exact function-matched *schema* orbit as a tabular optimizer
stability screen, especially when the orbit level is still numerically tiny.
