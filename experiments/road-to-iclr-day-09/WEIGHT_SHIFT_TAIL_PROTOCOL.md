# Weight-shift versus tail-risk diagnostic

Status: frozen on 2026-09-01 after the opposite-sign tail result and before computing
weight-divergence associations. This is exploratory and uses immutable predictions.

## Question

Within a dataset, is the tail effect stronger on episodes where competence weights move
farther from the fixed mixture, and does that association have opposite signs for real
classification and regression?

## Frozen analysis

- Classification: the six unseen breadth identities with a significant negative parent
  result. Regression: all 16 completed identities.
- Reconstruct competence and fixed predictions and weights exactly. Define weight shift
  as `KL(w_competence || w_fixed)`; all fixed weights are strictly positive.
- Define tail gain as fixed minus competence worst-decile pointwise NLL for classification
  and squared error for regression. Positive means competence improves the tail.
- Within each dataset, report Spearman correlation between KL shift and tail gain.
- Within each dataset, split episodes into equal-count KL quintiles and compute
  high-shift minus low-shift tail gain. Average dataset statistics equally and use a
  20,000-draw dataset bootstrap, seeds 205001 and 205101.
- Also report total-variation weight shift and maximum competence weight descriptively.

Negative classification association/high-minus-low contrast paired with positive
regression values would support a task-dependent cost/benefit of stronger adaptation.
The analysis is associational, post-result, and cannot identify a safe threshold.

## Numerical correction

V1 produced no valid regression correlation because exact-zero softmax weights evaluated
`0 * log(0 / fixed)` as NaN. V2 applies the defining KL convention that zero-mass terms
contribute zero. No finite result, analysis choice, seed, or gate changed; V1 artifacts
are invalid and overwritten by V2.
