# Exact compute-efficiency audit

Status: frozen before outcomes.

This analysis reuses the exact full-product fANOVA tensors from the frozen
confirmation panel. It does not fit or select another model.

Screen material cells using validation only, with the existing threshold
`joint risk / mean member loss >= 0.005`, and report their held-out test
results. For each cell define:

- iid-equivalent fits: `joint_risk / strength2_residual`, since an iid
  size-`B` quotient estimate has expected residual `joint_risk / B`;
- strength-1-equivalent fits:
  `16 * four_strength1_residual / strength2_residual`, because the stored
  comparator is already the average of four randomized four-run strength-1
  blocks (16 total fits);
- seed-block-equivalent fits analogously from the stored four-block,
  16-total-fit seed comparator.

The realized strength-2 action costs 16 fits. Report medians, equal-cell means,
and source-group means. Residuals below
`max(1e-18, 1e-12 * joint_risk)` are treated as numerical zero and reported as
infinite equivalence separately; finite summaries exclude them. This is an
expected squared-prediction-risk comparison, not wall-clock acceleration.
