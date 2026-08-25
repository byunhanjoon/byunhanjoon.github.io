# Canonicalization numerical audit

## Bottom line

The full training-row anchor construction behaves as the exact algebraic
control predicts. The progressively sketched construction does not remain
pointwise invariant at condition number 1,000 in finite precision and was not
faster in the present implementation. It must be described as a useful
approximation/failure-mode experiment, not as an exact scalable method.

## Algebraic expectation

For centered full-rank data `X`, select a row-basis matrix `R` using only
`col(X)` and use coefficients `C` satisfying `X = CR`. Under an invertible
recoding `X' = XB`, the same rows become `R' = RB`, hence

`X' = XB = CRB = CR'`.

The coefficient matrix is therefore unchanged if numerical rank and anchor
selection agree. A deterministic normalization or whitening applied afterward
also remains unchanged.

## Direct Adult audit

The audit compares the same training-fitted information orbit at κ=`1` and
κ=`1000`.

| Quantity | Full anchors | Progressive sketch |
|---|---:|---:|
| Retained anchors | 116 / 116 | 116 / 116 |
| Identical anchor set | Yes | No |
| Anchor symmetric difference | 0 | 2 |
| Train coordinate relative difference | `2.41e-13` | `5.95e-2` |
| Validation coordinate relative difference | `2.41e-13` | `2.70e-1` |
| Test coordinate relative difference | `2.41e-13` | `3.04e-1` |
| Train difference after whitening | `7.12e-12` | `1.35` |
| κ=1 construction time | 1.63 s | 4.97 s |
| κ=1000 construction time | 1.60 s | 4.68 s |

Both sketches eventually used 8,192 deterministic rows and reconstructed the
full training matrix to roughly `2e-13`. Thus the issue is not failure to span
the data. A near-tied pivot changed under the ill-conditioned coordinate system;
the resulting valid row bases describe the same space but yield different
coefficient coordinates. Subsequent whitening can turn that difference into a
large orthogonal-coordinate change.

Across the 25-dataset MLP screen, the sketch still reduced the mean performance
sensitivity almost completely, but its mean transform time was 2.56 s versus
1.85 s for the full construction. This implementation therefore supplies no
scalability claim.

## Claim boundary

- **Defensible:** full anchors are a near-exact empirical closure control under
  the tested full-rank conditions; the sketch is a numerically effective
  approximate control on most tasks.
- **Not defensible:** the sketch is exactly invariant in floating point, is
  uniformly continuous, or is already a faster deployable replacement.
- **Required follow-up for a method paper:** stable subspace/pivot handling,
  explicit tie detection, a coordinate-alignment rule across valid anchor sets,
  and a genuine large-width speed/memory advantage.

The machine-readable source is
`results/day3/broad_benchmark/canonicalization_numerical_audit.json`.
