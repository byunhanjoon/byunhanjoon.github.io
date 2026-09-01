# Native Feature Geometry — theory foundations

Status: **FROZEN BEFORE OUTCOME-BEARING RUNS**

## 1. Semantic domains and schema charts

A typed feature is a finite semantic domain

`M = (V, d, tau)`,

where `V` is a set of values, `d` is a declared semantic metric (or a graph
from which one is induced), and `tau` is a type such as cyclic, ordinal,
hierarchical, or nominal.  A storage schema is an injective chart
`s: V -> {0, ..., |V|-1}`.  Relabeling a schema changes `s`, not `M`.

For a predictor trained under chart `s`, write `f_s(v,z)` for its aligned
prediction on semantic value `v` and ordinary covariates `z`.  Define the
schema quotient and schema risk over a declared chart distribution `S` by

`Q(v,z) = E_s f_s(v,z)`,

`R_schema(v,z) = E_s ||f_s(v,z) - Q(v,z)||^2`.

For squared error, the usual Hilbert decomposition gives

`E_s ||f_s-y||^2 = ||Q-y||^2 + R_schema`.

This identity is not novel.  Its role is to distinguish predictive quality
from chart dependence.

## 2. A typed kernel and its spectral interface

Let `K_M` be a positive-semidefinite kernel determined by the declared metric:

- cycle and path: a heat kernel `exp(-t L)` of the corresponding graph
  Laplacian;
- hierarchy: an exponential kernel of leaf-to-leaf tree distance;
- nominal: the equality kernel, used as a no-neighborhood control.

Center `K_M` with `H = I - 11^T/n`, and let

`H K_M H = U Lambda U^T`, `lambda_1 >= ... >= lambda_n >= 0`.

The rank-`r` native interface is

`Phi_r = U_r Lambda_r^(1/2)`.

It is a finite feature map whose Gram matrix is the best rank-`r`
approximation to the centered semantic kernel in Frobenius norm.  The pilot
uses the same rank for all interfaces.

### Proposition 1 — chart equivariance of the native Gram geometry

Let `P` be the permutation matrix induced by a schema relabeling.  Then

`K_M^P = P K_M P^T`

and the rank-`r` native Gram matrix obeys

`G_r^P = P G_r P^T`, where `G_r = Phi_r Phi_r^T`,

provided `r` does not split an equal-eigenvalue block.  If it does, the claim
holds when the entire tied block is retained.

**Proof.** Centering commutes with permutations because `P1=1`.  Conjugation
preserves eigenvalues and maps each eigenspace by `P`.  Spectral truncation
therefore conjugates, and rotations within a retained degenerate eigenspace
cancel in the Gram matrix.  The split-block caveat is necessary.

This is a statement about Gram geometry, not about a particular eigenvector
sign convention.

### Proposition 2 — optimal truncated semantic kernel

Among positive-semidefinite matrices of rank at most `r`, `G_r` minimizes

`||H K_M H - G||_F`.

**Proof.** This is the symmetric positive-semidefinite specialization of the
Eckart–Young–Mirsky theorem.

The proposition does not say that `G_r` is optimal for an arbitrary target.
That requires a smoothness relation between the target and `K_M` and is tested
with a negative control.

## 3. Native smoothness and approximation

Let a category effect `a: V -> R` have expansion `a = sum_j alpha_j u_j` in
the centered kernel eigenbasis.  A linear head on `Phi_r` represents the first
`r` components exactly.  Its irreducible squared category error under the
uniform measure is

`(1/n) sum_(j>r) alpha_j^2`.

### Proposition 3 — native spectral tail identity

The best linear prediction of `a` from `Phi_r` has uniform squared error equal
to the spectral tail above.

**Proof.** The nonzero columns of `Phi_r` span `u_1,...,u_r`; orthogonal
projection gives the result by Parseval's identity.

This explains a possible benefit only when semantic effects concentrate in
low native frequencies.  A nominal random effect is deliberately outside that
claim.

## 4. Geometry defect and a schema-risk bound

For an aligned learned embedding table `E_s`, define its normalized centered
Gram matrix

`A_s = H E_s E_s^T H / ||H E_s E_s^T H||_F`

and the normalized native Gram `A_* = G_r/||G_r||_F`.  The geometry defect is

`D_s = ||A_s - A_*||_F`.

Suppose a common downstream map `h(e,z)` is `L`-Lipschitz in `e`.  For two
charts, align their embeddings by an orthogonal Procrustes map and denote the
rowwise residual by `Delta_(s,t)(v)`.

### Proposition 4 — prediction-orbit bound after chart alignment

`||h(E_s(v),z) - h(E_t(v)R,z)|| <= L ||Delta_(s,t)(v)||`.

Consequently, mean schema prediction variance is bounded by a constant times
mean squared aligned embedding discrepancy.

**Proof.** The pointwise result is the Lipschitz definition.  Squaring,
averaging, and using the pairwise representation of variance gives the second
statement.

This is intentionally conditional and does **not** imply that `D_s` alone
predicts schema risk: Gram defect can be distributed differently by category,
downstream maps differ after independent training, and optimization can ignore
parts of an embedding.  H3 tests the stronger empirical link rather than
claiming it as a theorem.

## 5. Geometry transport as a causal intervention

Let `O` be categories observed in training and `U` held-out categories.  Given
a trained embedding table `E` and native coordinates `Phi`, fit an affine
transport on observed rows,

`(A,b) = argmin_(A,b) ||Phi_O A + 1b^T - E_O||_F^2 + lambda ||A||_F^2`.

Patch only the unseen rows:

`E'_U = Phi_U A + 1b^T`, `E'_O = E_O`.

### Proposition 5 — specificity of the unseen-row intervention

For a lookup interface, this patch leaves every prediction whose category is
in `O` exactly unchanged, assuming deterministic evaluation.

**Proof.** Lookup returns the same row for every observed-category input, and
all downstream parameters and operations are unchanged.

Thus the intervention has built-in specificity.  Sufficiency is not built in:
if the learned chart does not carry a linearly transported native geometry,
the unseen patch need not work.  Correct transport is compared against mean,
random, and category-shuffled transports.  A performance rescue specific to
correct transport is the causal test.

### Proposition 6 — affine transport is a kernel extension

Augment native coordinates with an intercept, `Psi=[Phi,1]`, and let
`K_tilde=Psi Psi^T`.  Ridge transport from observed embedding rows `E_O`
predicts

`E_hat_U = K_tilde_UO (K_tilde_OO + lambda I)^(-1) E_O`

when the intercept receives the same penalty; the unpenalized-intercept version
has the corresponding centered-kernel expression.

**Proof.** Substitute the primal ridge solution
`B=(Psi_O^T Psi_O + lambda I)^(-1)Psi_O^T E_O` and apply the standard primal–
dual ridge identity.

Therefore the transport intervention depends on intrinsic Gram relations, not
on eigenvector signs or rotations.  This also exposes a boundary: with a full
rank finite kernel, the benefit is kernel interpolation over categories, not a
claim that the network literally preserves the displayed spectral axes.

## 6. Hypotheses

### H1 — Native Gram Equivariance

The implemented compiler obeys Proposition 1 under every frozen schema
permutation, including tied-eigenspace-safe truncation.  This is a construction
audit, not empirical evidence of utility.

### H2 — Native Geometry Emergence

On structured smooth tasks, unconstrained learned category embeddings recover
more native Gram geometry during training than random initialization and more
than a corrupted-metric control.  This is deliberately risky: supervised
networks may encode the target without recovering the declared feature metric.

### H3 — Geometry–Schema Risk Coupling

Across outcome-independent seeds and schema charts, lower native geometry
defect predicts lower held-category error and lower prediction-orbit damage.
This goes beyond Proposition 4 and may fail.

### H4 — Native Geometry Mitigation

A fixed native interface reduces held-category error and schema risk relative
to label-scalar, random fixed, and unconstrained learned interfaces, without a
material in-distribution penalty.  It must not receive credit for a comparable
gain on the nominal/random negative control.

### H5 — Native Chart Transport

Correct native transport of unseen rows causally rescues a trainable embedding
more than mean, random, or category-shuffled transport, while leaving observed-
category predictions exactly unchanged.  Failure of correct transport is
evidence against a reusable native chart even if H4 passes.

### H6 — Metric-Corruption Dose Response (outcome-informed successor)

After the H1–H5 pilot, interpolate the transport kernel between correct native
geometry and the already-frozen category-shuffled geometry while holding the
trained network fixed.  Held-category error should worsen monotonically as the
metric is corrupted.  This successor addresses the fact that H3's pooled CKA
correlation may be driven by interface identity rather than within-interface
geometry.

## 7. Scope and non-claims

- The pilot assumes the metric metadata are correct and available.
- Heat kernels, spectral embeddings, Fourier features, tree embeddings,
  learned categorical embeddings, kernel methods, and schema permutation
  invariance are established ideas.
- Exact compiler equivariance is a design property, not a discovery.
- Synthetic smooth targets can establish mechanism and falsify claims, but
  cannot establish practical breadth.
- The proposed novelty, if any survives, is the composition of chart-level
  schema risk, typed intrinsic value geometry, and a specific causal transport
  consequence.  It is not any component alone.
