# Multiclass cross-score scope protocol

Status: protocol frozen before inspecting cross-score outcomes; analysis
complete.

## Question

Do the independent-cover cross-score and its efficiency advantage extend to
genuinely vector-valued multiclass Brier risk? The earlier two-source
multiclass model-selection gate had no headroom because every method selected
the exact winner.

## Design

Reuse the prospectively fit Vehicle (4 classes) and Segment (7 classes)
tensors, with three candidate models per source and a `4 x 1 x 4 x 4` nuisance
product. At 32 fits compare two independent strength-2 cover means through
their cross-score against a complete IID-U32 score over 1,024 actions. Measure
candidate score bias/RMSE against exact full-quotient multiclass Brier loss.
Selection agreement/regret remain descriptive because of the known ceiling.

## Frozen gate

Pass if the cover cross-score has lower RMSE in at least 4/6 candidate cells
and on both source means, and every finite cover standardized bias is at most
3 in absolute value.

This is a small scope addendum, not source-generalization evidence.

## Outcome

The frozen gate **passes**.

- Cover cross-score RMSE is lower in all 6/6 candidate cells.
- Segment mean RMSE falls from `6.73e-4` to `5.47e-4` (18.7%); Vehicle falls
  from `4.40e-3` to `2.75e-3` (37.4%).
- The maximum absolute cover standardized bias is only 0.424.
- As anticipated, all methods retain 100% selection agreement and zero regret;
  this panel supports multiclass score efficiency, not selection improvement.

This closes the output-space scope across scalar MSE, binary Brier, and
4-/7-class vector Brier, while retaining the two-source limitation.
