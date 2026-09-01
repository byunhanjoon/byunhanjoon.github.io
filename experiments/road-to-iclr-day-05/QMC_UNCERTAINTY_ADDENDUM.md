# QMC Monte Carlo uncertainty addendum

Status: frozen after the 4,096-design QMC comparison and before these outcomes.

The primary QMC gate remains failed and is not redefined. To determine whether
its positive pooled reductions are stable rather than simulation noise,
generate 16 disjoint deterministic batches of 4,096 independent scrambles for
Sobol and Latin-hypercube sampling (65,536 designs per family total). For each
batch, recompute the exact design covariance and all 25 confirmation-cell
residuals.

Report the distribution of cell win counts and pooled strength-2 reductions.
The uncertainty addendum passes only if the 2.5th percentile of the 16 batch
pooled reductions is positive for both QMC controls. This supports only a
pooled variance claim; it cannot retroactively pass the original 20/25 cell
gate.

