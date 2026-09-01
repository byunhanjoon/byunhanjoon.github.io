# Task-balanced external model-selection protocol

Status: frozen before selection outcomes on 2026-08-28. The underlying exact
tensors have already been used for the cover-risk scope experiment, so this is
a new downstream endpoint rather than a new dataset panel.

On the eight-source task-balanced OpenML panel, select among the frozen
one-hot linear, ordinal forest, and one-hot Adam MLP candidates using 1,024
validation-only actions. Compare strength-2, IID-16, SRSWOR-16, four
strength-1 blocks, four seed blocks, scrambled Sobol-16, and LHS-16 with shared
nuisance coordinates across candidates. Evaluate realized proper loss on the
held-out test split.

The primary core-control gate requires:

- lower equal-source mean test loss than IID, SRSWOR, strength-1 blocks, and
  seed blocks;
- lower test loss than IID on at least 6/8 sources;
- higher mean full-validation-quotient-winner agreement than IID;
- on the population-`>16` subset alone, lower mean test loss than every core
  control and lower loss than IID on at least 4/5 sources.

Report classification/regression strata and QMC controls, but they do not
alter the core gate. Products of exactly 16 nuisance cells are explicitly
marked as exhaustive at the strength-2 budget.
