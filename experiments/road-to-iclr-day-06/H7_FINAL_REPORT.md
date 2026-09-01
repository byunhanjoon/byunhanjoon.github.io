# H7 final report — Rounding-Cell Survival

Status: **FINAL**

## Verdict

**SUPPORTED, WITH A FINITE-HORIZON AND DATASET-HETEROGENEITY BOUNDARY.**

After excluding the five development bundles fixed in the protocol, the
prospective panel contains exactly 31 bundles and 93 paired schema paths.  The
frozen gates are: later material hitting 95.2%, dataset median delay at least
30 epochs in 3/3 datasets (medians 151, 101, and 181 epochs), exact closure
through epoch 20 on 96.8% of paths, lower final MSE on 100% of 63 eligible
paths, and final IEA64 material failures on 20.4% of all paths.

## Interpretation

H7 tests the observable consequence of a conditional rounding-boundary hazard,
not microscopic boundary events: H3 stores only declared prediction
checkpoints.  A late first stored hit is interval-censored, and seeds/views are
repeated measurements within three dataset replications.

The dataset summaries must remain visible because pooled paths can hide
heterogeneity, particularly in Credit.  Exact early survival and lower final
error are separate estimands; neither implies perpetual closure.  Canonical
gathering remains the preferred exact intervention when semantic metadata are
available.

## Decision

Keep the survival-extension clause as a narrow prospective result.  It needs a
new randomized, broader, multi-hardware confirmation before promotion to a
standalone paper claim; Credit's 33.3% failure fraction is an explicit boundary.
