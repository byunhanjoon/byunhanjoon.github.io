# Real task tail-risk contrast

The regression success is the opposite of the classification failure at the
pointwise-loss tail.

| Regression metric, fixed minus competence | Gain | 95% hierarchical CI | Positive datasets |
|---|---:|---:|---:|
| total standardized MSE | +0.217965 | [+0.039075, +0.545568] | 14/16 |
| bottom-90% mean squared error | +0.028780 | [+0.011267, +0.050244] | 13/16 |
| worst-decile mean squared error | +1.887379 | [+0.225772, +5.036851] | 14/16 |
| squared-error > 4 rate | +0.004831 | [+0.000910, +0.010452] | 12/16 |

Competence helps across the error distribution, but its largest absolute effect is on
the worst decile, and it prevents about 0.48 percentage points of errors beyond two
standard deviations. This contrasts with the six unseen classification identities,
where worst-decile NLL worsened by 0.025122 [0.008502, 0.045252] and NLL>2 events became
more frequent.

The resulting mechanism hypothesis is a task-dependent tail-risk sign flip: adaptive
soft weights suppress regression catastrophes but amplify rare confident classification
errors. This is more precise than an average-risk statement, yet remains diagnostic:
the deciles are method-specific order statistics, and the analysis does not prove why
the sign differs.

Artifacts: `TAIL_RISK_CONTRAST_PROTOCOL.md`,
`results/processed/tail_risk_contrast_detail_v1.csv`, and
`results/processed/tail_risk_contrast_audit_v1.json`.
