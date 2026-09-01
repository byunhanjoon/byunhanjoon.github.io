# High-dimensional strength-3 field-and-row cover

Status: frozen before outcomes on 2026-08-28.

The OA-32 field-and-row result beat IID but failed against the independently
marginal-balanced control. This prospective escalation tests whether explicit
three-factor balance repairs that failure.

## Design

- same frozen Bank Marketing, German Credit, and Compustat datasets;
- same ordinal forest, native CatBoost, and one-hot Adam MLP;
- same feature-order, seed, target-numbering, per-categorical-field numbering,
  and training-row-order nuisances;
- 128 fits per ensemble and four independent design repetitions;
- methods: one strength-3 OA-128, four independent strength-2 OA-32 blocks,
  four independently marginal-balanced-32 blocks, and IID-128.

The linear OA-128 enumerates `GF(2)^7`. Two independent linear forms encode
each four-level factor and one encodes each binary factor. Coefficients are
frozen in code, support at least 28 binary factors, and the complete mixed
array is exhaustively verified to balance every one-, two-, and three-factor
margin before training. Randomization independently permutes each factor's
level names.

Primary outcome is between-repetition squared prediction risk on held-out test
predictions. Strength-3 must beat all three controls in at least 6/9 cells and
on at least 2/3 dataset-pooled means. Brier score is secondary. The fixed-panel
bootstrap is descriptive and is run only after the frozen cell summary.

## Post-gate uncertainty addendum

After evaluating the frozen gate, resample the four independent repetitions
50,000 times within every method and cell. Report percentile intervals and
bootstrap probabilities for pooled risk ratios and Brier-score differences.
This is conditional fixed-panel sensitivity analysis, not a replacement for
the predeclared cell-and-dataset gate and not population-level inference.
