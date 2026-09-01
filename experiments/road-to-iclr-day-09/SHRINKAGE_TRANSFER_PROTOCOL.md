# Synthetic-to-real shrinkage transfer diagnostic

Status: frozen on 2026-09-01 after the tail and weight-shift diagnostics and before
computing any shrinkage curve. This is post-result and cannot support tuning claims.

## Question

Does the loss-minimizing amount of movement from the fixed mixture to the frozen
competence mixture transfer from PriorDial to real data in classification and regression?

## Frozen path

For every stored episode define prediction
`p(lambda) = (1-lambda) p_fixed + lambda p_competence` on the fixed grid
`lambda = {0.0, 0.1, ..., 1.0}`. This is equivalent to shrinking competence weights
toward fixed weights. No expert is refit and no continuous optimizer is used.

- Synthetic scope: all 9,600 untouched PriorDial test episodes, equal weight per frozen
  `(context size, feature count, rho)` cell.
- Real scope: all 9 classification and 16 regression identities, equal dataset weight.
- Endpoint is log loss or standardized MSE. Gain is loss at lambda zero minus loss at
  each lambda.
- For each lambda, use a 20,000-draw bootstrap over equal-weight cells/datasets, seeds
  215001 onward. These pointwise intervals are descriptive and not multiplicity-adjusted.
- Report the grid minimizer of the aggregate curve and the distribution of cell/dataset
  minimizers. Ties use the smallest lambda.

Agreement supports routing-strength transfer; disagreement identifies a task-specific
synthetic-to-real shift. Because real outcomes motivated this analysis, no lambda chosen
here is a validated method or authorized for performance reporting.
