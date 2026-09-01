# Adaptive strength-2 nuisance-cover protocol

Status: frozen after the budget-four strength-1 result, before strength-2
outcomes were computed.

At budget 16, construct a mixed strength-2 orthogonal array from all pairs in
`GF(4)^2`. Feature, category, and seed use four-level linear columns; the
binary target factor uses a nondegenerate GF(4)-trace column. Singleton factors
are retained as constants. Independently permuting each factor's level names
generates the design family.

Every design balances all one- and two-factor margins, so every main and
pairwise product-fANOVA component is annihilated. Independent random level
permutations make the estimator unbiased for the full schema--seed quotient.
Expected residual risk is computed exactly from the design incidence
covariance and the fitted prediction Gram matrix, not sampled designs.

Equal-budget comparators are:

- 16 iid joint schema--seed draws (`joint risk / 16`);
- four independent budget-four strength-1 covers (`R_strength1 / 4`);
- four independent schema draws, each averaging all four seeds
  (`persistent schema risk / 4`).

The adaptive gate requires lower residual than all three comparators in a
strict majority of material cells and in panel mean. Initial and conditional-
confirmation datasets are reported separately, with the latter grouped by
the six frozen source groups.

This is classical orthogonal-array mathematics applied to the learned
pipeline nuisance tensor; neither GF(4) arrays nor variance reduction by
orthogonal designs is claimed as new.

