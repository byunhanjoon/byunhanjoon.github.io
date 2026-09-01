# PHASECOVER PUBLISHED-BACKBONE CONFIRMATION PROTOCOL

Status: frozen before dataset download, training, or outcome inspection.

## Questions

1. Does patch-origin sensitivity transfer to published fixed-patch models?
2. Does random-phase training reduce that sensitivity?
3. Does equally spaced PhaseCover approximate the all-phase quotient more
   efficiently than the exact uniform four-phase design distribution?
4. Does quotient efficiency translate into forecast accuracy?

The four questions are reported separately. A mechanism result cannot be used
to relabel a failed forecasting result.

## Exact-information construction

Each example contains the same `L=505` standardized observations and a
24-step future. Published models receive a fixed length of 512. For phase
`o in {0,...,7}`, the observations are placed after `o` zero/mean boundary
slots and before `7-o` zero/mean boundary slots. Observed values, order,
targets, weights within an evaluation cell, and total number of boundary slots
are unchanged. No observation is dropped, duplicated, interpolated, or
circularly wrapped. Reconstruction of all 505 observations must be exact.

Native partial-patch masks are not used: MOMENT converts any partially
observed patch into a fully masked token, which would discard observations.
Because every channel is standardized using training data, boundary fill zero
is the training mean. This practical boundary-fill choice is a limitation and
is reported explicitly.

Patch length and stride are 8. The full quotient is the mean prediction over
all eight phases. `PhaseCover4={0,2,4,6}`. The IID4 comparator is the exact
uniform distribution over all `C(8,4)=70` four-phase subsets; it has no Monte
Carlo error.

## Untouched datasets

Three datasets not used in the preliminary PhaseCover screen:

1. Jena Weather, from the TensorFlow/Keras public archive.
2. Electricity, from `laiguokun/multivariate-time-series-data`.
3. Traffic, from `laiguokun/multivariate-time-series-data`.

Use chronological 60/20/20 target splits and training-fitted per-channel
standardization. Select eight channels without outcomes by rounded evenly
spaced indices over the raw column order. Use at most 8,192 uniformly spaced
training windows and 1,024 validation/test windows. Context may precede a split
boundary; targets may not cross it.

## Published backbones and training controls

### PatchTST

Use `transformers==4.49.0` `PatchTSTForPrediction`, patch length/stride 8,
width 128, three encoder layers, four attention heads, FFN 256, and the native
prediction head. Train all parameters from scratch.

### MOMENT

Use official `momentfm==0.1.4` and pretrained `AutonLab/MOMENT-1-small` at
Hugging Face revision `411e288267f82cce86296dbe4d6c8bc533cc162f`.
Freeze the pretrained encoder and fine-tune only its native linear forecasting
head, as a linear-probe confirmation. MOMENT is channel-independent.

For each backbone, compare:

- `canonical_train`: phase 0 for every training batch;
- `phase_aug_train`: a uniformly random phase for each training batch.

Both controls select checkpoints using phase-0 validation loss to keep model
selection identical. AdamW, maximum 12 epochs, patience 3, and seeds
`20261211` and `20261212` are fixed. The same checkpoint supplies all phase
evaluation arms within a cell.

The complete matrix is 3 datasets x 2 backbones x 2 training controls x 2
seeds = 24 cells.

## Metrics

- Standardized RMSE and MAE, micro-averaged across examples, horizons, and the
  eight preselected channels.
- Phase materiality: RMS prediction deviation across phases divided by
  canonical RMSE.
- Quotient error: prediction MSE to the all-eight-phase mean.
- Exact-IID4 mean, dispersion, and PhaseCover percentile across all 70 designs.
- Full8 and PhaseCover4 forecast RMSE relative to canonical and exact IID4.

Model seeds are repeated measurements, not independent datasets.

## Frozen decisions

Report three decisions rather than one omnibus verdict.

### Phase sensitivity transfers

Pass per backbone if canonical-trained phase materiality is at least 5% on at
least two of three datasets. Both backbones must pass for a cross-backbone
claim.

### Phase augmentation is a remedy

Pass per backbone if phase augmentation reduces dataset-balanced phase
materiality by at least 15% relative and worsens dataset-balanced canonical
RMSE by no more than 2% relative.

### PhaseCover is an efficient quotient design

Pass a backbone/training-control group if PhaseCover quotient error is below
exact IID4 on at least two datasets and the dataset-balanced quotient-error
ratio is at most 0.80. At least three of four groups must pass.

### PhaseCover improves forecasting

Pass a backbone/training-control group if PhaseCover RMSE is no worse than
exact IID4 on at least two datasets. At least three of four groups must pass.
This is a separate claim and may fail even if quotient efficiency passes.

## Integrity and stopping

- Verify exact reconstruction for all phases and eight-channel tensors.
- Verify all 70 designs contain four unique valid phases.
- Verify 24 complete cells and finite predictions before aggregation.
- Record official package/model revisions and the protocol SHA-256.
- Do not tune phase set, thresholds, channel selection, or datasets after
  outcomes.
- Stop for implementation invalidity, non-finite training, or a two-hour wall
  clock budget; report incomplete cells rather than changing the design.

## Sources

- PatchTST implementation: Hugging Face Transformers.
- MOMENT implementation and weights: CMU Auton Lab `momentfm` and
  `AutonLab/MOMENT-1-small`.
- Jena Weather archive:
  `storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip`.
- Electricity and Traffic:
  `github.com/laiguokun/multivariate-time-series-data`.
