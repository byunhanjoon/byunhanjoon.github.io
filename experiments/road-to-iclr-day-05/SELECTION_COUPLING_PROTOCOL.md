# Model-selection nuisance-coordinate coupling ablation

Status: frozen before outcomes on 2026-08-28.

The primary model-selection experiment uses common nuisance coordinates across
the five candidate algorithms, analogous to common random numbers. This
ablation repeats all three panels and all four budget-16 methods under:

1. shared coordinates across candidates; and
2. independently randomized coordinates for every candidate.

Each regime uses 1,024 deterministic draws, validation-only selection, and
held-out proper loss. The independent-coordinate robustness gate requires
strength-2 to have lower panel-mean realized test loss than IID-16 in at least
two of three panels and lower dataset-mean loss on at least 60% of datasets in
those passing panels. The blocked controls and selection-only regret remain
fully reported.

Post-gate task-balanced addendum (frozen before its outcomes on 2026-08-28):
repeat the shared/independent comparison on the eight-source task-balanced
selection panel. Require lower mean realized test loss than IID-16 and wins on
at least 5/8 sources for the descriptive extension to pass. This does not alter
the original three-panel gate.
