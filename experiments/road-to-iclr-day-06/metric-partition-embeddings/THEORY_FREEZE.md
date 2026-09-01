# Day 6 Metric Partition Embeddings theory freeze

Frozen before outcome-bearing runs on 2026-08-29.

## The residual question

Ordinary numerical PLE assumes that a field lives on an interval in its stored
coordinate.  That assumption is right for many numerical fields and wrong for
cyclic, hierarchical, and nominal fields.  The proposed object is therefore
not a universal replacement for PLE.  It is a typed generalization:

> Given a field value `x`, a declared metric `d`, and landmarks `a_1,...,a_m`,
> form a partition of unity over landmarks and use its weights to mix learned
> landmark vectors.

For bandwidth `tau`, the **metric partition embedding** (MPE) is

```text
k_j(x; tau) = exp(-d(x,a_j)^2 / (2 tau^2))
w_j(x; tau) = k_j(x; tau) / sum_l k_l(x; tau)
e(x) = sum_j w_j(x; tau) v_j.
```

`v_j` are learned with the predictor.  Thus this takes a raw field value, but
its map into the token space is nonlinear and depends on the feature's metric.
The fixed weights can also be viewed as a preprocessing strategy.

The second candidate is a **multiscale MPE** (MMPE):

```text
w_j(x) = sum_s alpha_s w_j(x; tau_s),
alpha_s >= 0, sum_s alpha_s = 1.
```

The pilot uses a frozen equal mixture to avoid target leakage, then selects
single versus multiscale using validation only in its explicitly labelled
selector analysis.  A later end-to-end version could learn `alpha` by softmax.

## Simple theoretical hypotheses

### T1 — chart equivariance

Let `g` be an isometry between two schemas and transport both the metric and
landmarks.  Then every weight is preserved:

```text
w_{g(j)}(g(x); tau) = w_j(x; tau).
```

Transporting landmark vectors by the same permutation makes `e(g(x))=e(x)`.
This is exact, not statistical.  PLE has this property only for the interval
charts it was designed for, not arbitrary category relabellings or a missing
cycle edge.

### T2 — local metric regularity

If the metric space and bandwidth are bounded, Gaussian landmark weights are
Lipschitz in `d(x,y)`.  Therefore a bounded landmark table gives a Lipschitz
token map.  Nearby semantic values cannot receive unrelated tokens unless the
learned table itself has extreme variation.

### T3 — interpolation from fill distance

Suppose `f` is `L`-Lipschitz and every query is within fill distance `h` of a
landmark with nonnegligible kernel weight.  Kernel interpolation with landmark
values has error controlled by a weighted landmark distance and hence scales
as `O(Lh)` as the cover becomes dense.  This predicts an advantage on unseen
cycle and tree nodes.  It predicts no advantage for a nominal/equality metric,
where distinct nodes provide no local information.

### T4 — schema risk follows metric distortion

For a stored-code metric `d_z` and native metric `d`, define the pairwise
distortion on evaluated points.  MPE should be stable when `d` is transported,
whereas code-RBF and PLE risk should rise with distortion of the stored chart.
The causal control is an intentionally permuted metric with the same dimension,
bandwidth rule, landmark count, and predictor.

### T5 — no universal line win

On an ordinary interval, PLE is already a strong local piecewise-linear chart.
MPE/MMPE need only be competitive there.  A claim that they universally beat
PLE on scalar numerical data is rejected unless it transfers broadly to real
tables.  The main hypothesized gain is typed topology plus schema stability.

## What would actually be novel

Gaussian RBFs, Shepard interpolation, partitions of unity, kernel mixtures,
and metric embeddings are classical.  Numerical PLE and trainable periodic
features are established tabular encoders.  Spectral metric encodings were
already tested on Days 4 and 7 in this repository.

The possible residual contribution is narrower:

```text
PLE-style per-field tokenizer interface
+ arbitrary declared feature metric/topology
+ exact chart-equivariance test
+ corrupted-metric causal control
+ safe nominal/interval boundary
+ schema-risk evaluation across equivalent encodings.
```

This is only a differentiated research hypothesis until a broad real-data
benchmark with trustworthy field metadata succeeds.
