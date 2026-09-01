# Four-pack covariance-operator calibration protocol

Status: frozen after the real-tensor four-pack result and before inspecting
these controlled outcomes.

On the full `4 x 4 x 2 x 4` product, draw 262,144 mutually disjoint four-cover
packs from the fixed equivariant sampler. For one normalized pure fANOVA field
for each of the four triple subsets and the four-way subset, estimate the mean
squared pack-average error and its Monte Carlo standard error. Compare with the
exact covariance coefficient of the average of two independent disjoint pairs,
which is half the Proposition-27 pair coefficient.

Also audit cell-incidence marginals against the required inclusion probability
`64/128`.

The operator gate passes if every one of the five pack coefficients is below
the two-pair coefficient with its 99% normal upper endpoint still below the
control, and the maximum absolute standardized cell-incidence deviation is
below 4. This is controlled design Monte Carlo, separate from source inference.
