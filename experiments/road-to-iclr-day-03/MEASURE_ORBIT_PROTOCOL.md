# Mixed-measure and selective Measure-Orbit protocol

## Chronology

This experiment chain followed the failed breadth result for exact-state
residual discovery. Each hypothesis was frozen before its corresponding
outcomes were inspected.

1. **Mixed-measure PLE replacement.** Freeze:
   `experiments/day3/configs/mixed_measure_ple_preregistered.json`.
   The direct replacement had to improve at least 60% of eight datasets, clear
   0.5% mean proper-loss reduction with a positive dataset-bootstrap lower
   bound, and retain an Adult gain of at least 0.5 accuracy points.
2. **Measure-Orbit TabM.** Freeze:
   `experiments/day3/configs/measure_orbit_preregistered.json`. The model had
   to pass the same broad loss and win gates while keeping at least +0.25
   Adult accuracy points.
3. **Selective Measure-Orbit.** The development results suggested an explicit
   validation-loss abstention rule. Before running the 21-dataset confirmation,
   the rule, datasets, seeds, endpoint, and gate were frozen in
   `experiments/day3/configs/selective_measure_orbit_preregistered.json`.

The eight development datasets had already appeared elsewhere in Day 2/3.
The 21 confirmation datasets had also been used by the earlier Orbit-TabM
study, but no Measure-Orbit outcomes had been generated on them. Thus this is
prospective method confirmation, not a globally untouched benchmark.

## Representations

Every numerical column receives a budget `B`. The benchmark uses

`B = clip(floor(1024 / number_of_numerical_columns), 4, 128)`.

All three appended blocks have exactly `B` coordinates per column:

- **fixed PLE:** ordinary quantile PLE, padded with zero coordinates when
  repeated quantiles collapse;
- **tail-reallocated PLE:** identify training values whose frequency is at
  least `ceil(n / B)`, remove at most 16 such atoms while fitting conditional
  quantile knots, and evaluate the resulting spline on all values;
- **mixed-measure PLE:** direct sum of atom indicators, a non-atom gate, and
  the conditional spline. Known atoms have zero continuous coordinates;
  unsupported values use the continuous component.

Atom discovery is target-free and uses training features only. At least 32
non-atom training rows must remain. All arms retain the ordinary schema view,
including a one-to-one quantile transform of every numerical scalar, so no arm
receives additional source information.

## Model and selection

The ordinary arm sends fixed-PLE coordinates to all eight members. The
Measure-Orbit arm uses the fixed assignment

`baseline × 4, tail-reallocated × 2, mixed-measure × 2`.

All members share one dense stem and the ordinary TabM backbone. The arms have
identical input width and trainable parameter count. The selective method
trains both arms and chooses the lower-validation-proper-loss arm independently
for each dataset and seed. Test targets are never used for selection.

The proper loss is binary log loss, multiclass log loss, or MSE on the
training-standardized regression target. Seeds are averaged within datasets;
datasets are the inference units.

## Frozen confirmation gate

All clauses were required:

- at least 0.5% mean selected proper-loss reduction;
- positive mean on at least 60% of datasets;
- positive lower endpoint of a 95% dataset bootstrap interval;
- positive improvement in at least 50% of paired dataset-seed cells;
- no excess failures.

## Claim boundary

Selective Measure-Orbit is a two-fit method and has not yet been compared with
a two-seed prediction ensemble under exactly equal wall-clock compute. Raw
Measure-Orbit is parameter matched, but the current baseline implementation
also evaluates its dense stem per member to keep the execution path paired; it
is not an optimized ordinary-TabM runtime control.
