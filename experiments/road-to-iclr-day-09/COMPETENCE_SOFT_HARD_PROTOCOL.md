# Frozen diagnostic: soft competence mixtures versus hard CV selection

Frozen: 2026-09-01, after the aggregate loss-aligned routing result and before computing
any hard-selection test loss.

## Question

Does context cross-validation help because it reliably identifies one best expert, or
because its noisy loss evidence is useful only after calibrated soft aggregation?

## Immutable data and methods

Use only `fallback_loss_router_2e46ddf857_test.npz`. No expert is refit and no query
outcome is used to choose a method or parameter.

- `fixed`: the task-specific six-way mixture fitted on development episodes.
- `hard_cv`: one-hot selection of the minimum three-fold context-CV-loss expert.
- `soft_cv`: the already frozen task-specific softmax temperature and shrinkage.
- `best_individual_oracle`: minimum query loss among individual experts, diagnostic only.

Tie handling is deterministic expert order. Classification uses log loss and regression
uses standardized MSE, exactly as in the parent protocol.

## Frozen analysis

1. Primary contrast: `hard_cv loss - soft_cv loss`, equal-weighted over the same 20
   regime/rho cells within each task type.
2. Secondary contrast: `fixed loss - hard_cv loss`.
3. Use 10,000 paired episode-bootstrap draws, resampling independently within each cell
   and then averaging cells equally.
4. Report the hard-CV/query-best match rate, mean Spearman correlation between the six
   CV and query expert losses, soft weight entropy/effective expert count, and soft-minus-
   hard gain by deterministic CV-margin quintile.
5. Assert recomputed fixed and soft losses match the already written parent cell table to
   `1e-6`; a mismatch invalidates the diagnostic.

## Interpretation rule

- If soft beats hard with a positive interval, call that evidence that calibrated
  aggregation, rather than accurate hard identification alone, carries performance.
- If hard beats soft, narrow the parent result to expert selection.
- This is a post-result mechanism diagnostic on immutable predictions, not independent
  confirmation and not a novelty claim for softmax weighting.

No method choice from this diagnostic may be evaluated on the same test seeds as a new
confirmatory method. Any follow-up performance claim requires fresh data.
