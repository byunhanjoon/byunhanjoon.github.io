# PHASECOVER PRELIMINARY PROTOCOL

Status: frozen before model outcomes.

## Question

Does an arbitrary patch origin materially change forecasts from a patch-based
model, and can a deterministic equally spaced phase design approximate the
all-phase quotient forecast more efficiently than an equal-budget random phase
ensemble?

## Exact-information phase construction

For context length `L=96` and patch length `p=16`, phase `o` left-pads the same
96 timestamped observations by `o` masked positions and right-pads to a patch
multiple. Values, temporal order, targets, and model weights are identical.
Padding masks enter the patch projection. No observation is dropped, repeated,
or circularly wrapped. The model is trained with a uniformly random phase per
batch so that nonzero phases are in distribution for every arm.

The full quotient prediction is

```text
Q(x) = (1/16) sum_{o=0}^{15} f(T_o(x)).
```

## Data and targets

Three public multivariate forecasting datasets:

1. ETTh1: hourly transformer-temperature data; target `OT`.
2. Exchange Rate: daily exchange-rate panel; target channel 0.
3. Solar Energy: 10-minute solar-production panel; target is the training-set
   highest-variance channel, selected without validation or test outcomes.

Each dataset uses a chronological 60/20/20 target split, train-fitted channel
standardization, horizon 24, at most 20,000 training windows, and at most 2,048
uniformly spaced validation/test windows. Context may precede a split boundary;
targets never cross it.

## Model and seeds

- Mask-aware patch Transformer: width 96, two layers, four heads, FFN 192.
- One CLS token and learned patch-position embeddings; scalar 24-step target.
- AdamW, validation early stopping, maximum 30 epochs.
- Model seeds: `20261121`, `20261122`, `20261123`.
- The same trained checkpoint supplies every evaluation arm.

## Three equal-purpose baselines

1. `canonical`: phase `{0}`, one call.
2. `iid4`: expected result over 64 independently frozen uniform four-phase
   subsets sampled without replacement, four calls each.
3. `phasecover4`: equally spaced phases `{0,4,8,12}`, four calls.

`full16` is a 16-call diagnostic reference, not a baseline. IID subsets are
generated from seed `20261201` before predictions are inspected and reused for
every dataset/model seed.

## Metrics

- Forecast standardized RMSE and MAE.
- Phase materiality: RMS prediction deviation across all phases divided by
  canonical RMSE.
- Quotient error: mean squared prediction distance from each finite design to
  `full16`.
- IID mean, standard deviation, and quantiles are retained; model seeds are not
  treated as datasets.

## Frozen preliminary gates

PhaseCover is **preliminarily supported** only if all hold on dataset means:

1. phase materiality is at least 5% on at least two of three datasets;
2. `full16` beats `canonical` RMSE on at least two datasets and on average;
3. `phasecover4` quotient error is below expected `iid4` on at least two
   datasets and its dataset-balanced ratio is at most 0.80;
4. `phasecover4` RMSE is no worse than expected `iid4` on at least two datasets.

Otherwise kill or reformulate the method. Regardless of outcome, this compact
screen is not a leaderboard or ICLR claim: it uses one compact architecture.

## Required controls and integrity

- Verify every phase reconstructs the exact unpadded context bit-for-bit.
- Verify each four-phase design has four unique valid offsets.
- Verify all methods share checkpoint and test examples within a cell.
- Do not tune phase offsets, patch length, or thresholds after results.
- Report failures and maximum-grid/boundary effects.

## Sources

- ETTh1: `zhouhaoyi/ETDataset`.
- Exchange Rate and Solar Energy: `laiguokun/multivariate-time-series-data`.
