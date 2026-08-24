# Statistical notes for hierarchical residual discovery

These statements define the intended mechanism and the limited theory claim.
They are not a substitute for the empirical evaluation.

## Support shrinkage

For an exact state `s`, suppose the residual observations satisfy

`r_i = theta_s + epsilon_i`, with `E[epsilon_i] = 0` and
`Var(epsilon_i) = sigma^2`.

The implemented estimator is

`theta_hat_s = sum_{i: x_i=s} r_i / (n_s + alpha)`.

Conditional on `theta_s`, its mean squared error is

`(alpha^2 theta_s^2 + n_s sigma^2) / (n_s + alpha)^2`.

Thus unsupported states (`n_s = 0`) return zero, weakly supported states shrink
toward the smooth baseline, and the shrinkage vanishes as support grows. Under
the prior `theta_s ~ Normal(0, tau^2)`, the posterior mean has this form with
`alpha = sigma^2 / tau^2`.

## Singleton identity is a change of basis, not extra information

Suppose a numerical column has `K` supported values and PLE places a knot at
each value. Evaluating the cumulative PLE basis on those values gives a
`K x (K - 1)` triangular matrix. After adding the model intercept, this matrix
has rank `K`, exactly like a `K`-state one-hot basis. Therefore every function
of the observed singleton states expressible by identity is already expressible
by sufficiently fine PLE, and conversely. The singleton augmentation changes
neither the data information nor the finite-support function class.

It does change geometry. If `I = P A` maps the centered PLE basis `P` to the
centered identity basis `I`, L2 weight decay on one parameterization becomes
`||A w||^2`, rather than `||w||^2`, on the other. Gradient descent is likewise
preconditioned by a different Gram matrix. Thus an identity shortcut can help
because a state-local residual has a smaller or better-conditioned path in the
augmented basis, even when expressivity is unchanged. `basis_geometry.py`
measures rank, weighted condition number, and minimum coefficient norm for this
claim. Pair identities are different: they add a joint-state function space
that an additive collection of univariate PLE blocks does not contain.

## Why the interaction is pure

Let `H_j` and `H_k` be the spaces of functions of columns `j` and `k`, with the
empirical support-weighted inner product. The pair routine first backfits the
additive projection in `H_j + H_k`, fits a joint-state residual, and iteratively
removes its weighted row and column marginals. Its final prediction therefore
has zero empirical conditional mean for every supported state of either member:

`E_hat[b_jk(X_j, X_k) | X_j=v] = 0` and
`E_hat[b_jk(X_j, X_k) | X_k=w] = 0`.

Consequently an additive exact-value effect cannot be credited to the pair
unless finite-sample held-out noise survives all discovery folds. Requiring an
improvement in all five total-loss and incremental-loss folds is the fixed
finite-sample safeguard used by the experiments.

## What nested cross-fitting guarantees

For a fixed outer fold, every candidate, threshold, and selected structure is
chosen using only the outer training rows. Conditional on that training set,
the mean loss difference on the untouched outer holdout is an unbiased estimate
of the conditional generalization-risk difference of the complete discovery
and correction procedure for that fitted training set.

This statement does not make the selected correction unbiased, guarantee a
positive improvement, or remove benchmark-level dataset-selection bias. It
does prevent candidate search over the inner folds from reusing the outer
evaluation outcomes. The official validation and test labels remain outside
the discovery procedure.

## Capacity accounting

Each accepted singleton or pair contributes one train-vocabulary state basis,
with at most 512 total states under the frozen protocol. Neural comparisons
reduce backbone width to match the parameter count of the PLE baseline. This
does not equalize optimization difficulty, but it rules out a raw parameter-count
explanation. Runtime, memory, and encoded width are reported separately.
