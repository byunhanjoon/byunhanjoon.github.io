# PROJECTIVE PILOT IMPLEMENTATION NOTE

The first projective run placed a constant `1e-4` numerical variance floor
after projecting the joint covariance onto a query. A post-projection constant
does not scale as `c^2` when a query is scaled by `c`, so the implementation
itself violated the frozen projective identity and produced maximum residual
`2.24e-4`.

That run is retained under `projective_invalid_postprojection_floor/`. The
correction removes the post-projection constant. Positivity remains guaranteed
by the strictly positive diagonal of the joint covariance, so every nonzero
query used by the protocol has positive variance. No data, seeds, training
steps, model widths, queries, metrics, or thresholds changed.
