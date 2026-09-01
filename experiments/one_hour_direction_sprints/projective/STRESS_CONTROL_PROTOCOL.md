# Post-gate direct-comparator stress controls

This addendum was frozen after the primary gate passed but before any control
outcome was generated.  It cannot change the primary pass/fail result.

Three stronger attacks on the result are run for all original seeds:

1. `direct_long`: the original 138,850-parameter direct DeepSets comparator,
   trained four times longer (20,000 rather than 5,000 updates) on the same
   sparse query support.
2. `direct_broad`: the original direct comparator trained for 5,000 updates
   with dense and scaled-dense queries included in training.  This grants it
   privileged knowledge of the evaluation query families.
3. `direct_moment`: a larger direct architecture whose query encoder explicitly
   supplies signed first-order and squared-coefficient second-order summaries,
   trained for 5,000 updates on the original sparse support.  It is not forced
   to obey projective identities.

Evaluation reuses the original deterministic held-out episodes for point,
dense, and scaled-dense queries across all four task families and five
covariate domains.

The primary mechanism is called stress-robust only if, against the per-cell
best of all three new controls, the projective model retains at least 0.05 nat
mean advantage and 70% wins on *unscaled dense* queries, with no more than 0.02
nat point-query degradation.  Best-of-controls is deliberately favorable to
the comparator and is not a deployable selector.
