# PROTOCOL DEVIATION: JENA MISSING-VALUE SENTINEL

The first completed matrix exposed standardized Jena Weather test RMS of 96.1
for the selected wind-speed channel. Source inspection found 18 test-period
values encoded as the finite missing-value sentinel `-9999`. The generic
non-finite-value imputation in the frozen implementation therefore did not
recognize them.

The original eight Jena cells, checkpoints, predictions, and processed array
were declared invalid and moved intact to
`raw/invalid_jena_sentinel_run/`. No Electricity or Traffic artifact was
changed.

Before rerunning Jena, values at or below `-999` are converted to missing and
then handled by the already specified training-mean imputation. This changes
data cleaning only; datasets, selected columns, splits, seeds, models, phase
sets, training controls, metrics, and frozen decision thresholds remain
unchanged. Both the invalid and corrected results remain auditable.
