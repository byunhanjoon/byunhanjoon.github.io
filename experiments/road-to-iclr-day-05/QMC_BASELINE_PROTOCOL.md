# Scrambled-QMC and Latin-hypercube baselines

Status: frozen before outcomes on 2026-08-28.

Randomized orthogonal arrays sit inside a broader variance-reduction literature.
On the 25 validation-material confirmation cells, compare the exact expected
strength-2 OA-16 residual with two additional unbiased, marginally balanced
budget-16 design families:

- 4,096 independently scrambled Sobol base-2 nets, mapped to each cell's
  declared four-factor levels (including singleton factors);
- 4,096 independently randomized Latin hypercubes, similarly discretized.

The same design covariance is applied to each saved vector-prediction tensor;
IID-16 remains an exact reference. Report balance diagnostics. The gate
requires strength-2 to have lower pooled test residual and win at least 20/25
cells against both baselines. This does not claim novelty for QMC/OA methods;
it tests whether explicit declared-factor pair balance matters beyond generic
space filling.
