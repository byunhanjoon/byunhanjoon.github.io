# Rank-4 covariance and calibration follow-up

## Frozen question

Can additional within-component dependence close the projective mixture's Electricity gap to TACTiS-2 while preserving exact linear projections, parameter matching, dense-query quality, and fast inference? Can one validation-only global variance temperature repair the Traffic under-coverage without changing means, mixture weights, or projective consistency?

## Model and capacity

- Four Gaussian mixture components.
- Each component has a mean, positive diagonal scale, and rank-4 covariance factor, so its covariance is `F F^T + diag(s^2)`.
- Every scalar linear query remains an exact four-component Gaussian mixture with component variance `||F^T q||^2 + ||s * q||^2`.
- Hidden width is reduced from 192 to 118. The resulting 136,236 trainable parameters must be within 1% of the existing diagonal mixture's 136,580 parameters.
- Training is otherwise unchanged: the same 16,384 windows, broad-query generator, three seeds, 3,000 AdamW steps, batch 512, learning rate 4e-4, weight decay 1e-5, and gradient clipping at 1.0.

## Calibration

- Construct 4,096 deterministic calibration windows exclusively from the validation interval `[train_end, validation_end)`; test windows remain in `[validation_end, end)`.
- Fit one positive variance multiplier per dataset and seed by minimizing scalar-query mixture NLL on the four frozen query families.
- The multiplier scales every component covariance equally. It cannot change means, weights, rankings, or projective consistency.
- Fit only this scalar after model training; no neural parameter is updated.

## Evaluation

Reuse the exact 1,024 test contexts, four query families, 256 samples, query seeds, CRPS estimator, coverage metrics, and TACTiS-2 checkpoints from the frozen closest-baseline protocol. Evaluate both uncalibrated and calibrated rank-4 variants.

## Frozen gates

Call the repair successful only if all hold on seed-averaged results:

1. calibrated rank-4 CRPS is within 2% of TACTiS-2 on at least two of three datasets;
2. calibrated rank-4 improves Electricity CRPS over the diagonal projective mixture by at least 5%;
3. calibrated rank-4 is no more than 2% worse than the diagonal model on the average of dense and scaled-dense CRPS across all datasets;
4. calibration reduces the projective model's Traffic coverage error by at least 3 percentage points;
5. calibrated rank-4 inference remains at least 50x faster than TACTiS-2 for the four-query workload;
6. capacity is within 1%, and every cell is finite.

Report the uncalibrated variant separately so covariance expressivity is not conflated with variance rescaling. No threshold will be changed after observing results.
