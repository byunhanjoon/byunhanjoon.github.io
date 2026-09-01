# H8 final report — Level-or-Acceleration Semantic Screen

Status: **FINAL**

## Verdict

**FALSIFIED AS AN INCREMENTAL SUCCESSOR.**

The 29 prospective bundles exclude all seven development observations.  The
frozen level-or-acceleration rule obtains sensitivity 1.000, specificity .889,
balanced accuracy .944, qualifying dataset accuracy in 3/3 datasets, and
delayed-positive recall 1.000 over six delayed positives.  Its accuracy
improvement over the unchanged H6 fixed rule is exactly zero, below the
required 0.10.

## Interpretation

Proposition 10 correctly gives log-convexity for a positive exponential modal
mixture, but actual optimizer trajectories need not follow that model.  More
importantly, even a high absolute accuracy cannot pass a protocol whose
scientific purpose is incremental discrimination over H6.  The development-
selected `L20 > -5 OR A > .02` rule therefore receives no success credit.

The observed tests are repeated bundles on three datasets, not 29 independent
replications.  The slope increase is a finite difference with units of
log10-MSE per epoch, not a dimensionally normalized second derivative and not
a global Lyapunov quantity.

## Decision

Discard H8.  Do not tune a new curvature threshold on this completed matrix.
