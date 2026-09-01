# Orthogonal nuisance-cover model-selection protocol

Status: frozen before outcome computation on 2026-08-28.

## Question

Does a strength-2 nuisance cover improve an actual downstream decision—model
selection—at the same 16-fit-per-candidate budget, rather than only reducing
squared distance to a quotient predictor?

## Panels

Analyze three already-complete exact-factorial panels separately:

1. the 11-dataset confirmation panel;
2. the six-dataset changed nuisance-menu repeat;
3. the six-dataset changed-subsample repeat.

The two repeats share dataset identities with the confirmation panel and are
replications, not new independent datasets. Every dataset has the same five
candidate learning algorithms and validation/test splits.

## Frozen procedure

For each dataset and each of 1,024 deterministic random draws, give every
candidate algorithm the same realized set of nuisance coordinates (common
random numbers), average its 16 member predictions, and select the algorithm
with lowest validation Brier score (classification) or MSE (regression). Score
the selected 16-member ensemble on held-out test data.

Compare four equal-fit actions:

- one randomized strength-2 OA(16) cover;
- 16 IID joint configurations;
- four independent randomized strength-1 OA(4) covers;
- four complete seed blocks with randomly chosen schema coordinates.

Ties use the fixed model order in the panel config. Report:

- held-out loss of the selected realized ensemble (end-to-end outcome);
- held-out loss of the full quotient predictor for the selected algorithm
  (selection-only outcome);
- agreement with the full-quotient validation winner;
- validation quotient regret and held-out quotient regret;
- selection entropy and win counts by dataset/panel.

The primary comparison is the equal-dataset mean difference in end-to-end
held-out proper loss, strength-2 minus each control. Use a deterministic
100,000-draw dataset-cluster bootstrap within each panel. A claim requires a
negative point estimate against all three controls and at least 60% of dataset
means lower against each; confidence intervals are descriptive because the
dataset panels are fixed and partly reused.

No validation-material screen is used. This protocol tests model selection on
the full available panel. The exact full quotient is a diagnostic target, not
an available 16-fit method.

## Prospective no-duplicate addendum

Frozen before computing this additional control. Add uniform simple random
sampling of 16 distinct nuisance-product cells (SRSWOR-16), using the same
candidate coupling and 1,024 draws. It is a fifth equal-fit method and enters
the same held-out-loss gate. This prevents the result from relying on IID's
occasional duplicate configurations. All original four-method outcomes remain
reported; no seed or threshold changes.

## Prospective QMC-selection addendum

Frozen after the residual-only QMC audit and before QMC selection outcomes.
Add independently scrambled Sobol-16 and Latin-hypercube-16 actions, discretized
to the same mixed nuisance product and coupled across candidates. Preserve all
previous random streams. Apply the same panel-wise requirement—negative mean
held-out loss difference and at least 60% dataset wins—to each QMC control,
but report this as a separate addendum gate so it cannot alter the already
frozen core/SRSWOR result.
