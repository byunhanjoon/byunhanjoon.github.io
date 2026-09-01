# H9 final report — Post-Breach Arithmetic Attenuation

Status: **FINAL**

## Verdict

**SUPPORTED, WITH DEVELOPMENT-SELECTION AND SCHEDULE-ORDER CAVEATS.**

The prospective panel contains exactly 25 bundles and 75 paired paths after
the eleven development exclusions.  Among 51 pairs whose FP32 orbit is
material at epoch 200, IEA64 has lower final MSE on 51/51, rescues 36/51 below
the threshold, and worsens 0/51 by more than twofold.  Dataset median ratios
pass in 3/3 datasets; exact final zeros number 34.  The equal-dataset canonical
test-loss change is +0.760%, inside the frozen ±1% safety gate.

## Interpretation

This estimand is deliberately conditional on FP32 material divergence.  It is
not evidence that IEA64 improves predictive accuracy, and exact zeros are
reported separately because finite log ratios depend on the declared floor.
Proposition 11 supplies a sufficient linear covariance-order result under
shared response maps, zero-mean independent injections, and PSD covariance
ordering.  The nonlinear experiments test its attenuation prediction; they do
not verify those assumptions or prove pathwise dominance.

The gates were calibrated from the eleven development bundles, and the
remaining split is prospective but runtime-ordered rather than composition-
randomized.  Any positive verdict therefore requires a new randomized panel.
Seeds/views remain repeated measurements within only three datasets.

## Decision

Keep post-breach attenuation as the strongest long-horizon clause inside
Semantic Arithmetic, subject to a new composition-randomized confirmation.
Canonical gathering remains preferable whenever the schema action is
available; IEA64 is a local arithmetic fallback and causal probe.
