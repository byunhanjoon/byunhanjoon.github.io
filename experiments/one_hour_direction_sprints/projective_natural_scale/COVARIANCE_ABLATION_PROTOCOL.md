# Frozen covariance-mechanism ablation

Frozen on 2026-08-31 after the primary natural-scale benchmark completed, but
before computing any outcome from this ablation.

## Purpose

The primary benchmark showed that the original projective checkpoint transfers
well relative to its direct neural control. This secondary experiment asks the
strict mechanism question: do its learned off-diagonal predictive covariances
help, holding every rowwise mean and marginal variance exactly fixed?

## Reused design

Reuse without modification all 12 datasets, three split seeds, four 16-row
contexts, validation/test query groups, target scaling, five functional
families, three-checkpoint moment ensemble, Gaussian scores, and validation-only
variance-temperature rule from `PROTOCOL.md` and `config.json`.

## Frozen variants

1. `projective_full`: the checkpoint ensemble's complete covariance.
2. `projective_independent`: identical mean and diagonal, with every
   off-diagonal entry set to zero.
3. `projective_shuffled`: identical mean and diagonal; within each query group,
   the learned correlation matrix is randomly permuted and then rescaled by the
   original marginal standard deviations. This remains PSD and projective but
   assigns dependence to the wrong query pairs.

Each variant gets its own scalar covariance temperature chosen on validation.
This gives the ablations every chance to compensate for global scale while
preventing them from recovering query-specific dependence.

## Precommitted mechanism gate

Across subset, difference, dense, and scaled-dense test queries:

- `projective_full` must improve both mean NLL and mean CRPS over
  `projective_independent`;
- it must win dataset-level NLL on at least 7/12 datasets; and
- it must have lower mean NLL than `projective_shuffled`.

All conditions are required for positive natural-data evidence that learned
cross-row covariance—not only coherent API construction or a strong rowwise
mean—is doing useful work.

## Integrity

The full variant must reproduce the primary benchmark's `neural_projective`
test metrics to 1e-7, all three variants must preserve means and covariance
diagonals to 1e-10, and covariance symmetry/minimum-eigenvalue errors must stay
within 1e-8. No mechanism gate is interpreted unless all 36 cells complete.
