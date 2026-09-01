# HeteroBag ambiguity decomposition

Status: frozen after the Phase-2 predictions were generated/partially generated
and before this decomposition was computed.

This is a descriptive mechanism analysis on the fixed Phase-2 triplet, not a
new predictive confirmation.

For classification, map member logits to positive-class probabilities; for
regression, retain standardized predictions. For each three-member ensemble,
use squared loss and the exact ambiguity identity

`ensemble loss = mean member loss - mean member disagreement from ensemble`.

Compare `T(A)+T(B)+alternate(C)` with `T(A)+T(B)+T(C)`. Decompose the
heterogeneous ensemble's squared-loss gain into:

- member-quality gain: control mean-member loss minus heterogeneous
  mean-member loss;
- diversity gain: heterogeneous ambiguity minus control ambiguity.

Their sum must reconstruct the ensemble gain numerically. Repeat for the
coordinate-only transformed-T placebo. Report task/cell means, signs, and
correlation with the primary log-loss/RMSE gain. This tests whether semantic
representation heterogeneity helps because the alternate member is stronger,
because its errors are complementary, or both.

