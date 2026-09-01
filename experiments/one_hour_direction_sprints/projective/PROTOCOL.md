# Static projectivity sprint protocol (frozen before execution)

## Question

Can a static-tabular model that emits one coherent joint law over a query set
match point prediction while generalizing substantially better to unseen linear
aggregate queries than a capacity-matched model trained to answer each query
directly?

The theoretical restriction is testable: for any coefficient vectors `a,b`
and scalar `c`, a projective Gaussian law must satisfy linearity of projected
means, degree-two scaling of projected variances, and the variance
polarization identity.  The direct model is not constrained to satisfy these
identities.

## Data-generating prior

- Static regression episodes have 8 features, 16 labeled context rows, and 12
  query rows.
- A latent task is sampled independently per episode from four families:
  linear, additive sinusoidal, rank-one interaction, and threshold/stump.
- Observation noise is Gaussian and shared task uncertainty induces dependence
  among query targets after conditioning on the context.
- Training covariates are correlated Gaussian.  Evaluation uses fresh latent
  tasks both on that distribution and on four held-out empirical covariate
  distributions from scikit-learn (`diabetes`, `wine`, `breast_cancer`, and
  `digits`).  Natural feature matrices are used, but targets remain sampled
  from the declared prior; this is a semisynthetic mechanism test, not a claim
  of natural-dataset SOTA.

## Models and equal information

- `projective`: a neural-process-style context encoder whose rowwise head emits
  a joint low-rank-plus-diagonal Gaussian.  Any query is answered by analytic
  linear projection.
- `direct`: the same context encoder plus a permutation-invariant set encoder
  over `(query row, coefficient)` pairs, outputting a scalar Gaussian directly.
  Its parameter count is required to be at least that of `projective`.
- Both see the identical deterministic stream of episodes and sparse training
  queries (points, subset means, and pair differences), and are optimized for
  the same number of updates.
- Three independent seeds are run.  No checkpoint or hyperparameter is chosen
  from evaluation outcomes.

## Evaluation

Primary queries are unseen dense signed and scaled-dense linear functionals.
Secondary queries are the three training families.  Metrics are Gaussian NLL,
RMSE, 90% interval coverage, and exact projective-identity residuals.  Point
NLL measures whether coherence sacrifices ordinary prediction.

## Go gate

All conditions must hold:

1. integrity checks pass and the direct model has at least as many parameters;
2. projective mean NLL is at least `0.05` nat lower on OOD dense queries;
3. projective wins at least 70% of paired OOD evaluation cells and wins on at
   least 3 of the 4 empirical covariate domains;
4. projective point-query NLL is no more than `0.02` nat worse;
5. projective maximum mean/scale/polarization identity residual is below
   `1e-5`;
6. the direct model has at least two mean identity residuals above `0.01`, so
   the test actually exposes incoherence rather than an accidentally coherent
   solution.

The run is capped at one wall-clock hour.  A timeout or incomplete three-seed
panel fails the gate.
