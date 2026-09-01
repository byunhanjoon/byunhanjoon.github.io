# Frozen protocol: resolvable packing on the mixed nuisance product

## Construction

Treat the 16-row mixed-level strength-2 base for the observed
`4 x 4 x 2 x 4` nuisance product as an additive subgroup. Enumerate all
coordinate shifts, quotient out duplicate row sets, and require exactly eight
disjoint strength-2 cosets whose union is the 128-cell product. Randomize a
resolution with independent level permutations.

## Controlled and real-tensor frontier

For `K in {1,2,4,8}`, sample `K` cosets without replacement from a randomized
resolution. Compare its exact conditional prediction risk with `K` draws with
replacement from that same resolution. Proposition 29 predicts

`R_pack(K)/R_independent(K) = (8-K)/7`.

Evaluate this law on pure triple/four-way controlled fields and on every stored
five-panel candidate having the full 128-cell product, using 4,096 randomized
resolutions. Report candidate and panel mean direct prediction residuals. At
`K=4`, additionally compare the resolvable risk with the previously frozen
sequential mutually-disjoint graph sampler; this is a distributional comparison,
not needed for the construction theorem.

## Frozen gates

The construction gate passes if there are eight pairwise-disjoint strength-2
cosets, they partition the product, every nonzero controlled and real-tensor
ratio matches `(8-K)/7` within `1e-10`, and `K=8` closes numerically.

The stronger empirical gate passes if resolvable `K=4` has lower mean direct
prediction residual than the sequential graph pack on at least 3/5 panels and
at least 16/23 full-product candidates. A failure here only chooses between two
valid pack distributions; it does not invalidate resolvability.
