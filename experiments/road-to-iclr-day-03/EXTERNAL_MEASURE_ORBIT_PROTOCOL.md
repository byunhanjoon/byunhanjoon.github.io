# Untouched external Selective Measure-Orbit protocol

Frozen at `2026-08-26T07:10:05+09:00`, before any external method outcome was
generated or inspected.

## Question

Does the locked Selective Measure-Orbit method transfer to datasets absent from
all previous method development and confirmation, and does it beat the obvious
two-fit control: a two-seed ordinary-TabM prediction ensemble?

## External panel

The panel contains seven locally available RTDL-format datasets with zero
overlap with the eight development and 21 confirmation datasets:

- Coil2000 Caravan Insurance;
- Give Me Some Credit;
- OpenML credit risk dataset 43454;
- Taiwanese Bankruptcy Prediction;
- Helena multiclass;
- Year Prediction MSD regression;
- the latest date-purged Sberbank Housing window.

The selection rule and all exclusions are machine-readable in
`experiments/day3/configs/external_measure_orbit_preregistered.json`. Dataset
selection used paths, task metadata, dimensions, and duplication/source checks
only. No method outcome existed at freeze time.

## Locked portfolios

For each dataset and seed, train three fits:

1. `baseline_anchor`: ordinary TabM with fixed PLE sent to all eight members;
2. `measure_orbit`: the locked 4/2/2 fixed/tail/mixed member assignment;
3. `baseline_seedmate_update_matched`: another ordinary TabM initialization.

Selective Measure-Orbit chooses fit 1 or 2 using validation proper loss. The
ordinary seed control averages predictions from fits 1 and 3. Classification
averages probabilities; regression averages standardized predictions.

The seedmate is forced to execute exactly the number of epochs and minibatch
updates executed by the Measure-Orbit fit. Because input width, parameter
count, batch size, and execution path are matched, the two portfolios have the
same fit and gradient-update budget. Wall-clock seconds are recorded as a
diagnostic rather than assumed identical.

## Frozen decision

The primary comparison is Selective Measure-Orbit versus the update-matched
two-seed prediction ensemble. It must obtain at least 0.25% mean relative
proper-loss reduction, positive means on at least 60% of datasets, a positive
dataset-bootstrap lower bound, positive gains in at least half of paired
dataset-seed cells, and no excess failures.

It must also preserve the earlier gate versus a single ordinary baseline: at
least 0.5% mean reduction, at least 60% dataset wins, a positive bootstrap
lower bound, at least 50% paired wins, and no excess failures.

Both gates must pass. A failure is retained; thresholds and dataset membership
will not be changed after outcomes are visible.
