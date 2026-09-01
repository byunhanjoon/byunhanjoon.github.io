# Real task tail-risk contrast

Status: frozen on 2026-09-01 after the classification tail diagnostic and before
computing any regression tail metric. This is a retrospective immutable-prediction
diagnostic, not a new performance confirmation.

## Question

Classification transfer fails through rare high-log-loss errors. Does the confirmed
regression improvement act on the analogous squared-error tail, and in the opposite
direction?

## Evidence and metrics

Use all 16 completed real regression identities from the small, breadth, and independent
confirmation panels. Reconstruct frozen fixed and competence predictions with no fitting
or tuning. For every episode and method compute total MSE, mean of the largest 10% of
pointwise squared errors, mean of the remaining 90%, and fraction of squared errors above
4.0 (absolute standardized error above two).

Gain is fixed minus competence. Give datasets equal weight and use a 20,000-draw
hierarchical dataset/episode bootstrap with seeds 195001 onward. Recomputed parent MSE
must agree within `1e-5`.

The classification comparator is the already frozen six unseen-identity result; it is
not re-estimated or treated as prospectively selected.

## Interpretation

Positive regression tail intervals paired with the negative classification tail result
support an asymmetric tail-risk account. They do not prove a causal mechanism, establish
distributional robustness, or authorize a tail-tuned method. The threshold and decile
are descriptive; total MSE remains the performance endpoint.
