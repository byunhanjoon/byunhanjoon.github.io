# Frozen protocol H7 — Rounding-Cell Survival

Status: **FROZEN AFTER FIVE DEVELOPMENT BUNDLES, BEFORE 31 TEST BUNDLES**  
Freeze time: 2026-08-29 00:28:34 Asia/Seoul

## Why H1 changed

The first five completed H3 bundles are development observations and are
excluded from every H7 gate.  In Credit/FT-Transformer/seed 8101, IEA64 aligned
predictions are exactly closed through epoch 20.  One view is material by epoch
50; two other views remain exact through epoch 100 and become material by epoch
200.  Thus “float64 makes schema commutation exact forever” is false.  The
rounding-cell theorem was always conditional; H7 tests its appropriate
finite-horizon consequence.

The excluded stems are recorded verbatim in `hypothesis_07_config.json`.  At
freeze time the other 31 H3 artifacts did not exist.

## Theory

For aligned interface evaluation `e`, let `B_e` be the event that two float64
coordinate reductions land in different float32 rounding cells, conditional on
all previous evaluations being closed.  Over a population of declared seeds,
schema views, and datasets, write

`p_e = P(B_e | B_1^c, ..., B_(e-1)^c)`.

No independence is assumed.  The probability of surviving `N` interface
evaluations is

`P(tau > N) = product_(e=1)^N (1 - p_e)`

and therefore, by the union bound,

`P(tau <= N) <= sum_(e=1)^N p_e`.

Proposition 3 makes `p_e = 0` whenever its rounding-cell margin condition
holds, but does not guarantee that condition along an arbitrarily long learned
trajectory.  Float64 can dramatically lower the boundary-crossing hazard
without eliminating it.  Once a cast mismatch occurs, the perturbation
recurrence can amplify it into material prediction divergence.

H7 does not estimate microscopic `p_e` because H3 stores checkpoint
predictions rather than every interface output.  It tests the observable
survival consequence at the material MSE threshold `1e-5`.

## Frozen paired estimands

For each of the 31 prospective H3 bundles and each of its three nonidentity
views, define FP32 and IEA64 hitting epochs as the first stored checkpoint whose
validation orbit MSE exceeds `1e-5`; paths never crossing by epoch 200 are
censored at 201.

The paired delay is `hit_iea64 - hit_fp32`.  Early exact survival means IEA64
MSE is bitwise zero at every stored checkpoint through epoch 20.  Final
reduction compares the paired epoch-200 MSEs.

## Frozen gates

H7 passes only if all are true on the 31 untouched bundles (93 view pairs):

1. all 31 artifacts are complete and pass the Day-6 integrity audit;
2. among paths whose FP32 arm is material by epoch 200, IEA64 hits strictly
   later in at least 80%;
3. median paired delay among those paths is at least 30 epochs in at least 2/3
   datasets;
4. at least 90% of all IEA64 paths are exactly closed through epoch 20;
5. among paths whose FP32 arm is material by epoch 200, IEA64 has lower final
   MSE in at least 80%;
6. at most 25% of all prospective IEA64 paths are material by epoch 200.

Dataset summaries are primary scope evidence; views/seeds are repeated
measurements.  Ties do not count as “later” or as strict reduction wins.

## Decision and novelty boundary

Passing changes H1 from exact closure to **survival extension**: interface
float64 is a strong finite-horizon hazard suppressor with explicit rare-failure
cases.  Failure discards IEA64 as a reliable long-horizon intervention, though
it remains a causal instrument in H1.

Survival analysis, union bounds, floating-point rounding cells, and numerical
training instability are established.  The narrow contribution is their
combination for exact tabular schema conjugacy and a prospective 31-bundle
paired boundary map.  Canonical gathering remains the stronger intervention
when schema metadata are available.
