# THREE ORAL-CEILING PILOTS: FROZEN PROTOCOL

Status: frozen before pilot outcome inspection.

Each direction receives at most 30 wall-clock minutes for its model run. A
pilot may finish earlier. These are mechanism screens, not paper-level
benchmarks.

## Pilot A: View-consistent longitudinal learning

### Question

Do standard tabular learners produce materially different predictions when an
identical forecasting history is expressed through lossless coordinate views?

### Data and task

Reuse the corrected, training-standardized eight-channel Jena Weather,
Electricity, and Traffic arrays from the PhaseCover confirmation. For each
channel, predict the next value from its previous 32 values. Use chronological
60/20/20 boundaries, at most 4,096 training examples and 2,048 test examples
per dataset, sampled uniformly without outcome-based selection.

### Exactly invertible views

1. `levels`: chronological levels.
2. `differences`: first level followed by 31 first differences.
3. `dct`: orthonormal type-II discrete cosine transform.
4. `reverse`: levels in reverse coordinate order.

All four views are deterministic bijections and preserve the same 32 values.
Verify numerical round-trip error before training.

Evaluate LightGBM, a two-layer sklearn MLP, and TabPFN 6.3 regression with
seeds `20261221`, `20261222`, and `20261223`. Every model is fitted separately
on each view with otherwise fixed hyperparameters.

### Go criterion

The direction passes if at least two model families on at least two datasets
have both (i) worst-to-best RMSE spread of at least 10% of best RMSE and (ii)
RMS prediction dispersion across views of at least 10% of canonical RMSE.
Also report whether model rankings flip across views on at least two datasets.

## Pilot B: Projectively consistent temporal query distributions

### Question

Can an unconstrained query-conditioned probabilistic forecaster give
contradictory distributions for algebraically related horizon/channel queries,
and can a joint PSD parameterization remove those contradictions without
sacrificing held-out query likelihood?

### Process and queries

Generate nonlinear four-channel dynamical histories of length 16 and a
two-step, eight-dimensional Gaussian future with state-dependent covariance.
Training queries are single coordinates, subset means, and pair sums. Testing
also includes held-out dense, difference, and scaled queries.

Compare:

1. `QueryNet`: directly maps history and query vector to scalar Gaussian mean
   and variance.
2. `ProjectiveNet`: maps history to a joint mean and PSD covariance, then
   answers every linear query by analytic projection.

Use seeds `20261231`, `20261232`, and `20261233`, equal hidden width, identical
training samples, and a maximum of 8,000 optimization steps.

Measure held-out-query NLL plus normalized mean additivity, scale equivariance,
and variance polarization violations.

### Go criterion

Pass if QueryNet has at least 5% normalized contradiction on two of three
identities, ProjectiveNet is below `1e-5`, and ProjectiveNet held-out-query NLL
is no worse than QueryNet in at least two of three seeds.

## Pilot C: Interventional pretraining for temporal tables

### Question

Can meta-pretraining across confounded temporal transition tables use a handful
of randomized transitions to estimate a new environment's intervention
response better than observational prediction or per-environment regression?

### Synthetic environments

Each environment has latent Gaussian state `u`, observed proxy `x`, action
`a`, and next outcome `y`. Environment-specific coefficients govern dynamics,
treatment effect, behavior policy, and hidden confounding. Observational
actions depend on `x` and `u`; randomized actions are independent. Evaluation
targets are analytic `E[y | x, do(a)]`.

Train a permutation-invariant context transformer (`CausalPFN`) across
environments with 48 observational and 0--8 randomized context rows. Compare
an otherwise identical observational-context meta-learner, naive per-task
ridge, and intervention-balanced per-task ridge. Evaluate new environments at
`k={0,2,4,8}` randomized rows with seeds `20261241`, `20261242`, and
`20261243`.

### Go criterion

Pass if at `k=4` and `k=8`, CausalPFN reduces interventional RMSE by at least
20% versus both the observational meta-learner and the better ridge baseline,
averaged across seeds. It must improve on at least 80% of test environments,
not only the aggregate.

## Shared integrity and decisions

- Persist per-seed raw metrics, aggregate tables, and run times.
- Stop a pilot at 30 minutes without changing its protocol.
- Treat seeds as repeated measurements, not datasets.
- Report all failed gates and implementation deviations.
- Rank directions as `continue`, `reformulate`, or `kill` only from their own
  frozen gates.
