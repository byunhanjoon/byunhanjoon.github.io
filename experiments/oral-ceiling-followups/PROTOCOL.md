# PROJECTIVE + VIEW FOLLOW-UPS: FROZEN PROTOCOL

Status: frozen before follow-up outcome inspection.

These are falsification studies, not paper-level benchmarks. Each run has a
two-hour wall-clock cap and may finish earlier. All thresholds, seeds, data
splits, and models below are fixed before inspecting outcomes.

## Study P: projectively consistent distributions on real time series

### Question

Does the synthetic projective result transfer to real multivariate futures:
can exact marginal consistency remove contradictions without worsening
held-out-query likelihood or calibration?

### Data and task

Use the corrected training-standardized Jena Weather, Electricity, and Traffic
arrays. A sample contains 32 historical steps from all eight channels and the
next four steps from all eight channels, giving a 256-dimensional history and
a 32-dimensional future. Use chronological train/test boundaries, 16,384
uniformly spaced training windows and 4,096 test windows.

Training queries are single coordinates, channel means across horizons,
horizon means across channels, and sparse sums. Held-out evaluation queries
are differences, normalized dense contrasts, and scaled dense contrasts.

Compare a parameter-matched direct scalar `QueryNet` with a `ProjectiveNet`
that emits a 32-dimensional mean and low-rank-plus-diagonal PSD covariance.
Use seeds `20261301`, `20261302`, and `20261303`, 3,000 optimizer steps per
cell, and identical sampled training queries within each dataset and seed.

Measure Gaussian NLL, empirical 50% and 90% interval coverage, mean
additivity, mean/variance scale equivariance, and variance polarization.

### Go criterion

Pass if all conditions hold:

1. QueryNet exceeds 5% normalized violation on at least two of three
   identities on at least two datasets.
2. ProjectiveNet's maximum identity violation is below `1e-5` on every
   dataset.
3. ProjectiveNet has no worse mean NLL on at least two of three datasets and
   wins at least six of nine dataset-seed cells.
4. Its mean absolute 50%/90% coverage error is no more than five percentage
   points worse than QueryNet's.

## Study V: learned consistency across lossless views

### Question

Can a shared consistency objective reduce representation regret on both seen
and unseen invertible views without sacrificing canonical accuracy?

### Data and task

Use the same three corrected datasets and next-value task as the first view
pilot: for each of eight channels, predict the next value from 32 lags, using
4,096 chronological training examples and 2,048 test examples per dataset.
Compute per-view coordinate normalization using training inputs only.

Training views are levels, reverse order, orthonormal DCT-II, and invertible
first differences. Held-out views are orthonormal DST-II, normalized Hadamard,
and a fixed even-then-odd coordinate permutation. Verify round trips for every
view. Learned models receive transformed values but not the inverse transform
or a view identifier.

Compare equal-width, equal-step shared MLPs:

1. `levels_erm`: canonical levels only.
2. `view_aug`: mean supervised loss across the four training views.
3. `view_consistent`: the same supervised loss plus a fixed `lambda=1`
   cross-view prediction-variance penalty.

Also report `oracle_canonical`, which applies the known inverse before the
levels model. It is a diagnostic ceiling, not a learned competitor. Use seeds
`20261311`, `20261312`, and `20261313`, 3,000 optimizer steps per model.

### Go criterion

The learned consistency direction passes only if, averaged over seeds:

1. `view_consistent` reduces worst seen-view RMSE by at least 10% versus
   `view_aug` on at least two datasets.
2. It reduces mean held-out-view RMSE by at least 5% versus `view_aug` on at
   least two datasets.
3. Its canonical levels RMSE is no more than 2% worse than `levels_erm` on
   every dataset.

The oracle gap is reported explicitly. A failure caused by inability to infer
an unseen coordinate system is a substantive negative result, not an
implementation failure.

## Shared integrity and decisions

- Persist per-seed metrics, trained checkpoints, aggregate tables, and run
  times.
- Treat seeds as repeated measurements, not datasets.
- Do not tune objectives, widths, steps, queries, or gates after outcome
  inspection.
- Report any implementation correction while preserving the invalid run.
- Rank each direction `scale`, `reformulate`, or `stop` strictly from its own
  frozen gate.
