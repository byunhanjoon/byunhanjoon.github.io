# Frozen protocol: exhaustive equal-fit control at product closure

For the 23 candidates on the known 128-cell `4 x 4 x 2 x 4` product, compare
pack-cross128 with the exact quotient score obtained by evaluating the eight
cosets of the mixed resolution once each. Both cost 128 fits. The exhaustive
score has no Monte Carlo error or finite-ensemble bias.

The stronger-control gate passes if exhaustive128 has strictly lower score
RMSE on all 23 candidates, no lower exact-winner agreement and no higher
validation regret on every represented panel, and maximum numerical error below
`1e-12`. A pass explicitly demotes pack-cross128 from “optimal at 128” to an
intermediate construction useful when the nuisance product exceeds budget or
an exhaustive resolution is unavailable.
