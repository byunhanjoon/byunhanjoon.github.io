# Untouched classification: semantic HeteroBag versus coordinate placebo

Status: frozen before any model fit or validation/test outcome on this panel.

Freeze context: the conditional eight-dataset Phase-2 panel showed a positive
HeteroBag effect overall, but the semantic `T+T+Q` candidate lost its frozen
panel-mean gate to `T+T+transformed-T` on classification. This experiment tests
that failure boundary; it is not allowed to redefine the claim task after
seeing outcomes.

## Dataset selection

Before freezing, a fixed candidate list was checked only for OpenML
availability, row/feature shape, presence of at least one numeric field, and
binary class count. All eight candidates were compatible and all eight are
retained: Breast-W 15, Sonar 40, PC4 1049, KC1 1067, Blood Transfusion 1464,
ILPD 1480, WDBC 1510, and Mammography 310. Repository search found no earlier
Day-1--5 use of these named datasets. No predictive metric was computed during
feasibility screening.

## Frozen experiment

- deterministic stratified 60/20/20 split with new seed `2026082814`;
- MLP, ResNet, and FT-Transformer, unchanged Phase-2 budgets;
- one new seed triplet `20261840,20261941,20262042`;
- equal three-fit systems: `T+T+T`, semantic `T+T+Q`, coordinate-only
  `T+T+reversed-field-T`, and homogeneous `Q+Q+Q`;
- primary metric: held-out log loss; all predictions and controls retained.

Dataset is the generalization unit. The semantic-specific gate requires both:

1. the mean of per-dataset `gain(TTQ vs TTT) - gain(TT transformed-T vs TTT)`
   is positive; and
2. that difference is positive on at least 6/8 dataset means after averaging
   architectures.

Separately report TTQ versus TTT wins and mean gain. If the semantic-specific
gate fails, HeteroBag's performance result is attributed to broad
coordinate/initialization diversity rather than Q-PLE semantics.

