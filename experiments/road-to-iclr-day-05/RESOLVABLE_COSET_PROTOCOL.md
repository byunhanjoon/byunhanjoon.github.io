# Resolvable finite-field coset packing protocol

Status: frozen before inspecting controlled outcomes.

On the full `4^4` product, treat the 16-row strength-2 base as a two-dimensional
linear subspace of `GF(4)^4`. Enumerate its 16 affine cosets and verify that
they are disjoint strength-2 covers whose union is the complete 256-cell
product. Randomize a whole resolution by common independent level
permutations.

For each of the four pure triple fields and the pure four-way field, average
over 8,192 randomized resolutions. At pack sizes `K=1,2,4,8,16`, compare the
exact conditional without-replacement risk with `K` independent randomized
covers. The predicted ratio is

`R_pack(K) / R_independent(K) = (16-K)/15`.

The gate passes if all coset/design invariants hold, full `K=16` closure is
numerical, and every nonzero empirical ratio is within `1e-10` of the formula.
This is a controlled construction theorem, not new real-source evidence.
