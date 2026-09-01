# Diagonal-mixture calibration control

## Frozen question

Did the successful Traffic calibration in the rank-4 follow-up require low-rank covariance, or is validation-only global variance calibration sufficient for the original diagonal projective mixture?

## Procedure

- Load the nine existing `projective_mixture4` checkpoints without neural retraining.
- Use exactly the same validation windows, query distribution, scalar-temperature NLL fit, test contexts, query seeds, 256 samples, and metrics as the rank-4 follow-up.
- Multiply every component's diagonal covariance by one fitted positive scalar per dataset and seed. Means and mixture weights remain fixed.
- Compare `diagonal_calibrated` to the original diagonal model and to `lowrank4_calibrated`.

## Frozen component gates

- Calibration is useful if it reduces Traffic coverage error by at least 3 percentage points while average CRPS across datasets is no more than 2% worse than the original diagonal model.
- Low rank adds value beyond calibration only if it improves calibrated Electricity CRPS by at least 2% relative to calibrated diagonal and is no more than 1% worse in mean CRPS across all datasets.

This addendum does not change or rescue the failed Electricity gate in the rank-4 protocol.
