# Real-classification failure diagnostic

Status: frozen on 2026-09-01 after the transfer outcomes and before computing auxiliary
classification metrics. This is an immutable-prediction mechanism diagnostic.

## Question

When the synthetic-tuned competence mixture fails on real binary data, does it primarily
lose ranking information, probability calibration, or both relative to the fixed mixture?

## Data and methods

Use all nine completed real binary identities from `real_panel_competence_55553b7ffd`
and `openml_breadth_competence_48170161d0`. Reconstruct only the frozen `fixed` and
`competence` predictions from stored expert predictions and CV losses. No model is fit,
no weight is tuned, and no dataset is excluded.

Per episode compute:

- log loss (parent-result parity metric);
- Brier score;
- ROC AUC;
- calibration-in-the-large error, `abs(mean(p) - mean(y))`;
- ten-bin equal-width expected calibration error (ECE);
- sharpness, `mean(abs(p - 0.5))`.

For loss/error metrics, gain is fixed minus competence; for AUC and sharpness, gain is
competence minus fixed. Aggregate with equal dataset weight and a 20,000-draw
hierarchical dataset/episode bootstrap, seeds 185001 onward. Stored-prediction parity
with parent log losses must be within `1e-5`.

## Interpretation

- Positive AUC with negative log-loss/Brier and worse calibration errors is evidence for
  a calibration-dominant failure, not proof of a causal decomposition.
- Negative AUC identifies a ranking component to the failure.
- ECE is a finite-bin descriptive statistic and cannot by itself establish calibration.
- Results cannot authorize real-data tuning or restore a classification performance claim.

## Tail-loss addendum

Frozen after the six metrics above were computed and before inspecting pointwise loss
tails. Aggregate AUC was near zero and ECE did not explain the log-loss direction, so a
single diagnostic extension asks whether rare confident errors are responsible.

For each method and episode, compute the mean of the largest 10% pointwise negative
log-likelihoods, the mean of the remaining 90%, and the fraction with NLL above 2.0.
Use the same fixed-minus-competence direction and hierarchical bootstrap. Report both
the complete nine-dataset panel and the six-dataset unseen breadth panel, whose parent
log-loss interval was significantly negative. The addendum is explanatory and post-result.
