# Natural-label anchor (frozen before execution)

This diagnostic was frozen after the semisynthetic primary gate and direct
stress controls, but before inspecting any natural-label outcome.  It cannot
retroactively alter those gates.

## Panel

- Six regression datasets already used by the repository's final closure:
  FreMTPL claim count, KDD17 stock return, Abalone, Kin8nm, Pol, and Puma32H.
- Three existing split seeds per dataset.
- For each split, numeric features are imputed and standardized from the
  capped 2,048-row training pool.  PCA is fit on that pool when more than eight
  numeric features are present; fewer than eight are zero padded.  Categorical
  fields are omitted, a stated limitation.
- Targets are standardized from the training pool.
- Each evaluation episode supplies only 16 randomly sampled labeled context
  rows and asks for linear functionals of 12 held-out test targets.
- The three projective and original direct checkpoints remain exactly as
  trained on the declared synthetic prior.  There is no natural-data tuning.
- A batched ridge predictor fit separately to each 16-row context is an
  ordinary point-performance anchor (`lambda=1`).

## Queries and metrics

Point, dense signed, and scaled-dense queries are evaluated with paired
episodes.  Neural models receive Gaussian NLL, RMSE, and 90% coverage; ridge
receives RMSE.

## Diagnostic bridge signal

The zero-shot bridge is called positive only if:

1. all 486 expected rows are finite where defined;
2. projective point RMSE beats direct on at least 4 of 6 datasets;
3. projective dense-query NLL has positive mean advantage and wins at least
   60% of paired cells;
4. projective point RMSE is within 25% of per-episode ridge on at least 4 of 6
   datasets.

This is deliberately a demanding sanity check, not evidence of competitive
tabular foundation-model performance.
