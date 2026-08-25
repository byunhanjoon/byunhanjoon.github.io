# Day 3 theory and novelty boundary

## 1. Setup

Use homogeneous coordinates `u = [x; 1]` so the first affine layer is

`h = W u`.

An invertible affine input recoding is `u' = T u`, with the usual homogeneous
constraint on `T`. The same function is represented by

`W' = W T^{-1}`.

Everything below also holds for a purely linear recoding by dropping the final
homogeneous coordinate.

## 2. Gradient transformation

Let `G = dL/dW`. Since `W = W'T`, the chain rule gives

`G' = G T^T`.

Ordinary gradient descent therefore gives `ΔW' = -ηGT^T`, whereas the mapped
reference update is `ΔW T^{-1} = -ηG T^{-1}`. They agree only for special
transformations such as orthogonal `T`. The predictor class is unchanged, but
the optimization trajectory is not.

Diagonal adaptive methods cannot repair this for every `T`: a rotated
non-diagonal covariance cannot be represented by per-coordinate second
moments. A two-dimensional covariance with unequal eigenvalues, followed by a
45-degree rotation, is an immediate counterexample.

## 3. Exact input-side natural update

Let

`A = E[u u^T]`

be the training-input second moment. Under the recoding,

`A' = T A T^T`.

Consider the first-layer update

`ΔW = -η G A^{-1}`.

Then

`ΔW' = -η G T^T (TAT^T)^{-1}`

`     = -η G A^{-1} T^{-1}`

`     = ΔW T^{-1}`.

Thus the discrete first-layer update is exactly equivariant under an arbitrary
invertible affine input recoding, conditional on function-matched parameters,
the same batches, exact matrix inverses, and no parameterization-dependent
clipping or damping.

### Uniqueness among covariance-only right preconditioners

Suppose an update has the form `G P(A)`, is congruence-equivariant for every
invertible `T`, and is normalized by `P(I)=I`. Equivariance requires

`P(TAT^T) = T^{-T} P(A) T^{-1}`.

Choose `T=A^{-1/2}`. The left side becomes `P(I)=I`, implying

`A^{1/2} P(A) A^{1/2}=I`, and therefore

`P(A)=A^{-1}`.

So the inverse input second moment is the unique normalized covariance-only
right preconditioner with full affine equivariance. This proposition explains
why diagonal Adam, diagonal standardization, and inverse-square-root whitening
updates cannot satisfy the same guarantee.

## 4. Initialization

Update equivariance alone is insufficient.

- Exact paired trajectory equivalence requires `W'_0 = W_0 T^{-1}`.
- Sampling `W_0 = Z A^{-1/2}` with isotropic Gaussian `Z` is invariant **in
  distribution**, because alternative symmetric square roots differ by an
  orthogonal factor after mapping into whitened function space. It does not
  generally couple the same seed to the exact same initial function.
- A deterministic canonical coordinate system can give paired equality without
  knowing a hidden reference `T`.

Finite minibatches, eigenvalue floors, early stopping, later-layer AdamW, and
floating-point arithmetic explain why the practical hybrid optimizer is near-
rather than exactly invariant. Decoupled AdamW weight decay commutes with the
linear parameter map as a multiplicative update; the non-invariant behavior is
not correctly attributed to decoupled decay alone. An explicit Euclidean L2
objective `||W||²`, however, is not invariant.

## 5. K-FAC relationship

For a first affine layer, the K-FAC block has the form

`F ≈ S ⊗ A`,

where `S` is the second moment of the preactivation gradient. Its update is

`S^{-1} G A^{-1}`.

The right factor is exactly the input-natural update above; the left factor
also corrects output-coordinate geometry. When the represented function and
batches are matched, `S` is unchanged by an input-only recoding, so K-FAC has
the same input-affine equivariance. Damping and approximate statistics weaken
the finite-step statement.

This is established prior art, not a new theorem of this project. Martens and
Grosse (ICML 2015) explicitly prove that idealized K-FAC is invariant to
arbitrary affine transformations of network inputs and hidden activities:
https://proceedings.mlr.press/v37/martens15.html

The Day 3 empirical contribution must therefore be framed around the systematic
tabular schema phenomenon, its causal evaluation protocol, and the gap between
ideal invariance and practical optimizers—not the invention of natural gradient
or affine-invariant optimization.

## 6. Anchor canonicalization theorem

Let centered training data `X` have rank `r`. Select `r` training rows whose
matrix `R` spans the row space. Every row has unique coefficients `c_i` such
that

`x_i = c_i R`.

After `X' = XB`, the selected rows are `R'=RB`, and

`x'_i = x_iB = c_i RB = c_iR'`.

The coefficients are unchanged. If anchor indices are selected using only the
column space of `X`—for example, pivoted QR on an orthonormal basis of that
space—then the anchor indices are also unchanged because

`col(XB) = col(X)`.

Consequently the complete coefficient matrix is invariant to every invertible
right basis transformation, apart from numerical tie-breaking. A train-fitted
diagonal normalization or whitening of these already-identical coefficients
remains identical.

The sketched variant applies the same construction to deterministic training
row subsets, doubling the subset until the anchors reconstruct the full
training matrix below tolerance. It is exact when the subset spans the retained
row space and fails closed otherwise.

## 7. Limitations of canonicalization

The map is piecewise smooth but can be discontinuous when rank changes or two
candidate pivots exchange order. It also depends on the training sample and can
be expensive for very wide matrices. These are not incidental implementation
details: general canonicalization is known to face continuity limitations. See
Dym, Lawrence, and Siegel (ICML 2024):
https://proceedings.mlr.press/v235/dym24a.html

The robustness benchmark therefore reports rank duplication, covariance ridge,
training-to-test shift, preprocessing time, and a progressively sketched exact
variant rather than presenting canonicalization as universally safe.

## 8. What is potentially new

The following are **not** new: natural gradient, K-FAC affine invariance,
whitening, full-matrix adaptive optimization, or canonicalization in general.

The paper candidate is the conjunction of:

1. a cross-schema definition of information-equivalent tabular representation
   orbits covering numerical spline, nominal contrast, ordinal path, and cyclic
   bases;
2. controlled evidence that commonly used neural tabular learners have large
   finite-training and test-performance variation within those orbits;
3. a prospective multi-dataset invariance audit separating hypothesis-class
   changes from optimizer changes;
4. evidence about which practical approximations recover the known idealized
   affine invariance and at what performance, stability, and compute cost.

Whether that conjunction is sufficiently new and important is an empirical
question answered only after the preregistered broad benchmark, not by the
algebra alone.
