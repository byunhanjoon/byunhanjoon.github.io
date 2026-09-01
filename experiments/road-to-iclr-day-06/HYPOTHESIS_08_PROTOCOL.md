# Frozen protocol H8 — Level-or-Acceleration Semantic Screen

Status: **FROZEN AFTER SEVEN DEVELOPMENT BUNDLES, BEFORE 29 TEST BUNDLES**  
Freeze time: 2026-08-29 00:47:10 Asia/Seoul

## Why H6 changed

H6 extrapolates one line through log orbit MSE at epochs 5, 10, and 20.  The
seventh completed H3 bundle, FreMTPL/MLP/8101, is H6's first prospective false
positive: its fitted epoch-200 score is `-4.684`, but its observed final log
MSE is `-8.716`.  The discrepancy grows early without sustaining that rate.
H6 remains frozen and will be adjudicated unchanged.

H8 uses all seven H3 bundles available at 00:47 as development and excludes
them from every gate.  The remaining 29 H3 bundles did not exist at freeze
time.  H8 adds no training and reuses only H3 checkpoints.

## Modal-mixture motivation

Suppose a local squared perturbation energy is approximated by

`V(t) = sum_k a_k exp(lambda_k t)`, with `a_k > 0`.

Then `ell(t)=log V(t)` has

`ell''(t) = Var_{w(t)}(lambda_k) >= 0`,

where `w_k(t)` is proportional to `a_k exp(lambda_k t)`.  Positive log-curve
acceleration therefore signals competition or takeover among modes, while a
small nearly constant slope can extrapolate badly without indicating such a
transition.  This is a diagnostic approximation, not a global model of neural
optimization.

## Frozen decision rule

For each bundle, average FP32 validation prediction-orbit MSE across the three
nonidentity schema views and define

- `L_t = log10(mean_mse_t + 1e-18)`;
- `s_early = (L_10 - L_5) / 5`;
- `s_late = (L_20 - L_10) / 10`;
- `A = s_late - s_early`.

Clarification added 2026-08-29 ~00:58 after the first H8 test bundle, changing
no formula, threshold, or gate: `A` is a finite-difference *slope increase*
with units of log10-MSE per epoch;
it is not divided by the separation between interval midpoints and is not a
dimensionally normalized second derivative.  Proposition 10 motivates its
sign, while the numeric `0.02` threshold is development-frozen.

Predict material epoch-200 divergence if either:

1. the orbit is already material at epoch 20: `L_20 > -5`; or
2. delayed-mode acceleration is present: `A > 0.02`.

The target is mean FP32 validation orbit MSE at epoch 200 greater than `1e-5`.
The H6 comparator uses its unchanged decision: extrapolated epoch-200 log MSE
from epochs 5/10/20 greater than `-5`.

## Frozen gates

H8 is supported only if all are true on the 29 untouched bundles:

1. all 29 bundles exist and the complete H3 family passes integrity;
2. fixed-rule sensitivity and specificity are each at least 0.75;
3. balanced accuracy is at least 0.80;
4. ordinary accuracy is at least 0.75 in at least 2/3 datasets;
5. among delayed positives (`L_20 <= -5`, material at epoch 200), recall is at
   least 0.75 and at least one such positive exists;
6. fixed-rule accuracy exceeds H6's fixed-decision accuracy by at least 0.10.

No threshold, window, logical branch, or metric may change after freeze.
Dataset results are descriptive repeated-bundle evidence, not 29 independent
dataset replications.

## Decision and novelty boundary

Failure discards H8.  Passing keeps a cheap two-branch diagnostic for external
confirmation.  Log-convex exponential mixtures and early instability screens
are established; H8's narrow possible contribution is the prospectively
tested use of curvature in an exact function-matched tabular schema orbit to
distinguish delayed numerical mode takeover from harmless early growth.
