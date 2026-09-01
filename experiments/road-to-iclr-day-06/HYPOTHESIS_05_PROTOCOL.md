# Frozen protocol H5 — Cross-Perturbation Fragility Transfer

Status: **FROZEN BEFORE H4/H5 OUTCOMES**  
Freeze date: 2026-08-29 (Asia/Seoul)

Pre-run convention (2026-08-29 ~01:19, with zero permanent H4 artifacts): a
constant score is assigned Spearman association zero.  Epoch-zero shadows are
necessarily constant across optimizer configurations because the optimizer
has not acted.  This makes the negative-control improvement gate defined
without changing any nonconstant rank calculation, outcome, threshold, or
matrix.  Ties otherwise use average ranks.

## Motivation

H4 asks whether an early semantic shadow forecasts its own late semantic
divergence.  That target is mechanistically clean but narrow.  H5 asks a more
useful question with the same prospectively generated tensors: can a cheap,
two-epoch exact-schema shadow forecast the prediction instability produced by
ordinary independent training seeds at epoch 20?

The two perturbations are different.  A semantic shadow injects coordinate-
reduction roundoff while holding the initialization and random tape fixed;
changing the seed perturbs initialization, minibatch order, and dropout.  H5
does not assume that their amplitudes or directions match.  It predicts only
that optimizer configurations with persistently large finite-horizon response
operators amplify both sources and therefore have a transferable fragility
ranking.

## Frozen estimands

The H4 matrix and its frozen data/model/configuration menu are reused without
additional fitting.  For dataset `d`, model `m`, optimizer configuration `c`,
and seed `s`, define the early shadow score

`S_2(d,m,c,s) = mean_g ||f_(s,g,2) - f_(s,e,2)||^2`,

where `e` is the canonical schema and the mean is over the two matched
nonidentity views on validation rows.  The configuration score is the mean
over the three frozen seeds.

Define late seed fragility on test covariates by

`V_20(d,m,c) = mean_(s<s') ||f_(s,e,20) - f_(s',e,20)||^2`.

Predictions are aligned to the original class coordinates before either
quantity is computed.  The primary analysis uses `log10(value + 1e-30)`.
No target labels enter either score.

Epoch-zero shadow `S_0` is the negative control: it measures only the initial
arithmetic injection before optimizer amplification.  Epoch-20 shadow is a
descriptive upper-horizon comparator and cannot count as an early forecast.

## Frozen gates

H5 is supported only if all are true:

1. across the 12 optimizer configurations, Spearman correlation between
   `S_2` and `V_20` is at least 0.60 in at least 2/3 FT-Transformer datasets;
2. equal-dataset pooled FT Spearman is at least 0.60;
3. pooled FT epoch-2 Spearman exceeds pooled epoch-zero Spearman by at least
   0.20;
4. using `S_2` to classify the top quartile of `V_20` configurations gives
   equal-dataset mean AUROC at least 0.80, with both classes present in every
   FT dataset;
5. all 324 H4 bundles are complete and their maximum matched initialization
   gap remains below `1e-6`.

The equal-dataset pooled Spearman is computed by ranking the 12 configurations
within each dataset, concatenating the three rank vectors, and correlating the
concatenated ranks.  This prevents dataset scale from becoming an implicit
weight.  Ties use average ranks.  The top quartile is the three largest
`V_20` values within each dataset.

MLP and ResNet results are reported as scope controls but are not gates because
H1 places most of their semantic shadows near a numerical floor.  Test labels
and test loss are excluded from the primary H5 gate.

## Decision rule and claim boundary

If any gate fails, discard cross-perturbation transfer as a Day-6 paper claim.
If all gates pass, retain it as a screening result requiring a new-seed and
new-optimizer confirmation before promotion over OrbitCover.

The proposed novelty is not generic learning-curve extrapolation or the known
fact that seeds affect neural training.  It is a label-free, exactly
function-matched tabular schema perturbation used to probe and forecast a
different source of training fragility.
