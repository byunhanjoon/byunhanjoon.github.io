# Frozen protocol H4 — Semantic Shadowing Forecast

Status: **FROZEN BEFORE H4 OUTCOMES; PRE-RUN STRATIFICATION CORRECTION**  
Freeze date: 2026-08-28 (Asia/Seoul)

Pre-run correction (2026-08-29, before any H4 artifact): pooled gates rank
scores within dataset before concatenation.  This prevents fixed differences
in dataset MSE scale from creating a spurious pooled association.  Thresholds,
the matrix, and the per-dataset primary gate are unchanged.

Pre-run convention (2026-08-29 ~01:19, still before any H4 artifact): if
either argument of a Spearman comparison is constant, its rank-predictive
association is defined as zero.  In particular, epoch-zero shadows are
necessarily constant across optimizer configurations because no optimizer
choice has acted yet.  Treating this no-information negative control as zero
prevents an undefined `NaN` from mechanically deciding gates 2 in H4 and 3 in
H5.  All nonconstant comparisons use ordinary average-tie Spearman.

## Motivation

H1 causally localized large FT-Transformer schema divergence to interface
roundoff, but generic bit-level training instability is already established.
H4 asks for a useful consequence specific to tabular semantics: can an exact
schema conjugacy act as a natural, label-preserving metamorphic perturbation
whose first two epochs forecast late numerical instability?

## Mathematical hypothesis

For paired schema paths, linearization gives

`delta_(t+1) = J_t delta_t + eta_t + O(||delta_t||^2)`.

The initial orbit error measures only the injection `eta_0`.  After a short
shadow horizon `s`,

`delta_s approximately sum_(r<s) [prod_(q=r+1)^(s-1) J_q] eta_r`,

so `||delta_s||` jointly probes interface injection and the early optimizer
amplification operator.  If amplification regimes persist, the two-epoch
prediction discrepancy should rank final discrepancy better than epoch-zero
roundoff alone.

This is a forecasting hypothesis, not a universal stability certificate.

## Frozen matrix

- same three datasets and MLP/ResNet/FT-Transformer models as H1;
- 2,048/512/512 fixed rows;
- seeds 9101, 9202, 9303;
- canonical plus two matched nonidentity schema views;
- 12 optimizer configurations from learning rate
  `{3e-4, 1e-3, 3e-3}`, weight decay `{0, 1e-4}`, and batch size `{128, 256}`;
- fixed architecture/dropout, fp32 interface, deterministic common random tape;
- checkpoints 0, 1, 2, 5, 10, 20.

This is 324 bundles and 972 trained paths.  Configurations are repeated within
dataset/model; datasets remain the replication unit.

## Frozen targets and gates

The primary early statistic is mean aligned validation prediction MSE across
the two schema views at epoch 2.  The target is the analogous epoch-20 MSE.
A final configuration is material when MSE exceeds `1e-5`.

H4 passes only if all are true:

1. epoch-2 versus epoch-20 Spearman correlation across optimizer configs is at
   least 0.70 in at least 2/3 FT dataset cells;
2. equal-dataset rank-pooled FT epoch-2 Spearman exceeds rank-pooled epoch-0
   Spearman by at least 0.20;
3. rank-pooled FT AUROC for predicting material epoch-20 configurations from
   the within-dataset epoch-2 score rank is at least 0.85, with both classes
   present in the pooled target;
4. at least 90% of MLP/ResNet configurations remain below `1e-8` at epoch 20;
5. all exact initialization gaps remain below `1e-6`.

No threshold is tuned on outcomes: AUROC is threshold-free.  If gates 1–3 fail,
the forecast idea is discarded.  If only stable-control gate 4 fails, scope
expands rather than being silently filtered to FT.

For gates 2–3, the 12 configurations are average-ranked within each dataset
and the three rank vectors are concatenated.  Gate 2 correlates concatenated
early and final ranks.  Gate 3 uses the concatenated early ranks as scores and
the original frozen `1e-5` material labels.  Each dataset therefore contributes
12 observations on the same score scale.

A constant score is assigned Spearman association zero, as frozen above.

## Novelty boundary

Metamorphic testing, numerical neural-training instability, early learning-
curve prediction, and tabular permutation invariance are established
separately.  Candidate novelty is the exact parameter-conjugated schema shadow
as a cheap empirical probe of the optimizer's semantic numerical amplification
operator.
