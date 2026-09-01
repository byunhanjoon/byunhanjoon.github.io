# Frozen protocol H9 — Post-Breach Arithmetic Attenuation

Status: **FROZEN AFTER ELEVEN DEVELOPMENT BUNDLES, BEFORE 25 TEST BUNDLES**  
Freeze time: 2026-08-29 01:38 Asia/Seoul

## Why H7 changed

H7 remains frozen and asks whether IEA64 postpones the first material schema
divergence.  In the eleventh observed H3 bundle, Credit/ResNet/seed 8202, all
three IEA64 paths hit the material threshold at epoch 100—the same stored
checkpoint as FP32—yet each has lower final orbit MSE.  Delay and final damage
are therefore distinct intervention targets.

H9 excludes all eleven H3 bundles available at freeze time, named verbatim in
`hypothesis_09_config.json`.  The remaining 25 bundles did not exist.  It adds
no training and does not change H7's verdict.

### Development-calibration disclosure

Disclosure prose added at ~01:50 after the first prospective bundle was known
to be noneligible; it changes no threshold, exclusion, estimand, or gate.

The thresholds are explicitly development-informed, not theory-derived.  In
the 11 excluded bundles, 21/33 pairs are FP32-material: IEA64 wins all 21,
rescues 14/21, worsens none by twofold, and has dataset median final ratios
approximately `2.22e-29`, `0.702`, and `6.08e-30` (floor-dependent for exact
zeros).  Equal-dataset canonical loss change is `-0.245%`.  The frozen gates
relax those observations to 80% wins, 50% rescue, at most 10% twofold
worsening, ratio at most 0.5 in 2/3 datasets, and ±1% loss change.  Only the 25
untouched bundles adjudicate H9.

## Linear-response motivation

For an aligned final prediction perturbation, write the local response to
injections over time as

`delta_T = sum_s M_(T,s) eta_s`.

If the two arithmetic arms share the same response maps, their zero-mean
injections are independent across time, and the block covariance of IEA64
injections is smaller in positive-semidefinite order, then expected final
squared displacement is smaller even when neither arm remains closed.  The
formal statement and its restrictive assumptions appear as Proposition 11 in
`THEORY_FOUNDATIONS.md`.

This motivates attenuation after a breach; it does not imply a later hitting
time or pathwise dominance in nonlinear training.

## Frozen paired estimands

For every remaining H3 bundle and each of its three nonidentity schema views,
compare FP32 and IEA64 validation orbit MSE at epoch 200.  A pair is eligible
when its FP32 MSE exceeds `1e-5`.  Define the final ratio

`R = (MSE_IEA64 + 1e-30) / (MSE_FP32 + 1e-30)`.

A material rescue occurs when the eligible IEA64 path is at or below `1e-5`.
A twofold worsening has `R > 2`.  Canonical test-loss change uses the same
paired IEA64-versus-FP32 definition and equal-dataset averaging as H3.

## Frozen gates

H9 is supported only if all are true on the 25 untouched bundles (75 pairs):

1. all 25 bundles exist and the complete H3 family passes integrity;
2. IEA64 has strictly lower final MSE in at least 80% of eligible pairs;
3. the median final ratio is at most 0.5 in at least 2/3 datasets, with at
   least one eligible pair required for a dataset to count;
4. IEA64 rescues at least 50% of eligible pairs below the material threshold;
5. at most 10% of eligible pairs are worsened by more than twofold;
6. equal-dataset mean canonical relative test-loss change is within ±1%.

Views and seeds are repeated measurements; dataset summaries are primary.
Exact zeros are reported separately because their finite log ratio depends on
the declared floor.

Because bundles completed in runtime order rather than random order, the
11/25 split is prospective but not composition-balanced across datasets,
models, or seeds.  Passing gates cannot remove this schedule-induced scope
limitation; an independently randomized confirmation panel would be required.

## Decision and novelty boundary

Passing changes the long-horizon IEA64 claim from survival extension to
**post-breach attenuation**.  Failure means the interface intervention is a
causal probe but not a reliable long-horizon mitigator.  Even a pass is a
narrow successor within Semantic Arithmetic, not a new standalone ranking
candidate, and canonical gathering remains the preferred exact intervention
when semantic metadata are available.
