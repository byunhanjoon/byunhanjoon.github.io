# HeteroBag second-panel Phase-1 protocol

Status: frozen before any three-member HeteroBag outcome on this panel was
computed.

Freeze time: 2026-08-28T00:04:28+09:00 (Asia/Seoul).

## Scientific status and panel

This is `PROSPECTIVE_CONFIRMATORY_CONDITIONAL`, not an untouched-data claim.
The datasets have appeared in two earlier frozen Day-4 multi-view panels, but
none appeared in the successful four-dataset HeteroBag-3 panel and no
three-member HeteroBag outcome on them has been inspected. To avoid outcome
selection, this screen includes all eight datasets from those two panels:

- classification: OpenML Credit-G 31, Spambase 44, Banknote 1462, and QSAR
  biodegradation 1494;
- regression: OpenML Abalone 183, Kin8nm 189, Pol 201, and Puma32H 308.

The deterministic 60/20/20 splits, sampling limits, preprocessing, model
width/depth, optimizer, epoch budget, and active-parameter matching are copied
unchanged from the earlier frozen protocols. No dataset is replaced after a
failure.

## Frozen first screen

Architectures are MLP, ResNet, and FT-Transformer. TabM is not included in
Phase 1 because this exact HeteroBag implementation has no stable TabM
interface; absence of TabM blocks standalone promotion even if the screen
passes.

The first independent triplet is `20260931, 20261032, 20261133`. For each
dataset and architecture, compare fixed one-third prediction averages:

- classification candidate: `T(A) + T(B) + Q(C)`;
- regression candidate: `T(A) + T(B) + Midrank(C)`;
- equal-compute control: `T(A) + T(B) + T(C)`.

All members use the same architecture, training budget, and matched active
parameter count. The primary loss is log loss for classification and RMSE for
regression.

The Phase-1 screen passes only if all clauses hold on the 24 dataset-model
cells:

1. at least 16/24 candidate test wins;
2. positive unweighted mean relative gain overall;
3. positive mean in classification and regression separately;
4. positive dataset-level mean on at least 6/8 datasets;
5. no architecture has mean relative gain below -0.5%.

Dataset is the generalization unit. Architecture cells are repeated strata.
Test outcomes are opened only after all 24 cells complete.

If this screen fails, stop HeteroBag as a standalone paper and retain it as a
Day-3 consequence. If it passes, freeze and run two further seed triplets,
homogeneous alternate-representation ensembles, a coordinate-only transformed-T
placebo, and diversity/error-correlation measurements before promotion.

