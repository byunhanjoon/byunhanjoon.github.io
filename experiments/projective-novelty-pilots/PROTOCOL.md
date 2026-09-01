# PROJECTIVE NOVELTY PILOTS: FROZEN PROTOCOL

Status: frozen before inspecting any novelty-pilot outcomes.

These studies reuse only models and data from the completed real-data
projective follow-up. Each new training run has a two-hour cap. They test three
components that could make the direction more than a joint-Gaussian baseline.

## Pilot Q: compositional query-complexity law

### Hypothesis

A direct query-conditioned model's probabilistic error grows with the number
of coordinates composed into an unseen signed linear query, whereas a model
that projects one joint distribution remains stable.

### Design

Use the three real datasets, seeds `20261301`--`20261303`, and saved
`QueryNetBroad` and ProjectiveNet checkpoints. For support sizes
`k={1,2,4,8,16,32}`, generate 4,096 random signed queries with exactly `k`
nonzero coefficients and unit L2 norm. Evaluate identical queries and outcomes
for both models using Gaussian NLL, RMSE, and 50%/90% coverage.

### Go criterion

The benchmark component passes if, on at least two datasets:

1. Spearman correlation between support size and the mean direct-minus-
   projective NLL regret is at least `0.7`.
2. Mean regret at `k=16,32` is at least `0.2` NLL and at least twice the
   singleton regret (using a `0.02` floor for the denominator).

This requires a graded compositional law, not merely one favorable query split.

## Pilot M: non-Gaussian projective mixtures

### Hypothesis

A conditional mixture of joint diagonal Gaussians retains exact linear-query
consistency while improving likelihood when the future is non-Gaussian.

### Design

Train on the same three datasets and seeds with identical broad query batches:

1. `joint_gaussian`: one projective diagonal Gaussian.
2. `projective_mixture4`: a four-component joint diagonal Gaussian mixture;
   every scalar query is the analytic projection of that mixture.
3. `direct_mixture4`: a four-component scalar mixture conditioned directly on
   history and query.

Use equal two-layer backbone width, 3,000 steps, Gaussian-mixture NLL, held-out
difference/dense/scaled queries, PIT calibration, and moment-identity tests.

### Go criterion

The non-Gaussian component passes if `projective_mixture4`:

1. improves mean NLL over `joint_gaussian` by at least `0.05` on at least two
   datasets and wins at least six of nine dataset-seed cells;
2. beats `direct_mixture4` NLL in at least six of nine cells;
3. has maximum moment-identity violation below `1e-5`; and
4. has PIT calibration error no more than five percentage points worse than
   the better comparator.

## Pilot R: black-box Gaussian reconciliation

### Representability fact being operationalized

A scalar Gaussian query family comes from one joint Gaussian exactly when its
mean is a linear functional and its variance is a positive-semidefinite
quadratic form. The joint parameters can be recovered from basis queries:

`mu_i = mean(e_i)`, `Sigma_ii = var(e_i)`, and
`Sigma_ij = [var(e_i+e_j)-var(e_i)-var(e_j)]/2`.

### Hypothesis

Projecting a trained direct model's basis-query answers onto this representable
family can repair compositional forecasts without retraining.

### Design

For the first 1,024 test histories in each dataset and seed, query the saved
`QueryNetBroad` at all 32 bases and 496 basis-pair sums. Reconstruct the mean
and covariance, project the covariance to the PSD cone by eigenvalue clipping,
and answer the original held-out queries analytically. Compare the raw direct,
reconciled, and trained ProjectiveNet predictions on identical examples.
Report NLL, coverage, PSD correction size, and inference cost.

### Go criterion

Reconciliation passes if it:

1. improves mean NLL over the raw direct model on at least two datasets;
2. closes at least 50% of the direct-to-trained-projective NLL gap on at least
   two datasets; and
3. has mean coverage error no more than five percentage points worse than the
   raw direct model.

## Shared integrity

- Protocol, seeds, thresholds, and queries are fixed before outcomes.
- Persist per-seed raw metrics and all trained checkpoints.
- Post-hoc analyses cannot change pass/fail gates.
- A passing pilot is still preliminary; a failed component is excluded from
  the proposed paper contribution.
