# Electricity-first mixture/representation 2x2 screen

## Frozen question

Is the remaining Electricity gap caused by insufficient mixture shape (4 versus 8 components) or by the flattened-history MLP (MLP versus channel-aware temporal attention)?

## Factorial models

1. `mlp_k4`: the existing calibrated diagonal four-component model.
2. `mlp_k8`: an eight-component diagonal projective mixture with hidden width reduced to match capacity.
3. `attention_k4`: four components with a two-layer Transformer encoder over 32 timestep tokens; each token jointly embeds the eight channels and receives a learned temporal embedding.
4. `attention_k8`: the same temporal architecture with eight components and a width selected independently for capacity matching.

Every head defines one diagonal Gaussian mixture over the full 32-dimensional future, so arbitrary linear queries retain exact analytic mixture distributions and projective consistency.

## Capacity and optimization

- Target capacity: the existing `mlp_k4` model's 136,580 parameters.
- Every new configuration must be within 2% of that target.
- Electricity only, with the identical 16,384 deterministic training windows and seeds 20261301–20261303.
- 3,000 AdamW updates, batch 512, learning rate 4e-4, weight decay 1e-5, gradient clipping 1.0.
- Identical broad scalar-query generator and mixture-NLL objective.
- `mlp_k4` reuses the existing checkpoints because their architecture, initialization seeds, data, and optimization budget already match exactly.

## Calibration and evaluation

- Fit one global covariance temperature per model/seed on the same 4,096 validation-only windows used in the calibration control.
- Evaluate the same first 1,024 test contexts, four query families and seeds, 256 samples, ensemble CRPS, coverage, and latency.
- TACTiS-2 remains the frozen reference at mean Electricity CRPS 0.18702643.
- Calibrated `mlp_k4` remains the frozen starting point at 0.21311931.

## Frozen advancement gate

The midpoint between the starting model and TACTiS is 0.20007287. Advance a configuration to the full three-dataset benchmark only if all hold:

1. mean calibrated Electricity CRPS is at most 0.20007287, closing at least half the gap;
2. it beats calibrated `mlp_k4` in at least two of three paired seed cells;
3. its mean coverage error is no more than 3 percentage points worse than TACTiS-2;
4. it remains at least 100x faster than TACTiS-2 for the four-query workload;
5. capacity is within 2% and every result is finite.

The factor-level diagnosis is descriptive: compare K8 against K4 within backbone and attention against MLP within component count. No architecture or threshold will be changed after results are observed.
