# Context-rescaled confirmation robustness

The fresh 500-episode / 12,000-fit audit passes for light classification shrinkage but
not jointly for full regression competence.

| Task and frozen method | Gain over fixed | 95% hierarchical CI | Positive datasets | Gate |
|---|---:|---:|---:|---|
| classification, 10% competence | +0.000732 | [+0.000142, +0.001680] | 5/5 | pass |
| regression, full competence | +0.092896 | [-0.001456, +0.238104] | 4/5 | fail |

Episode-level feature scaling was fit only on the 96 context rows; regression target
scaling used only context labels. This algebraically cancels the original outer-fold
affine transforms. Outer-train label-free feature selection and median imputation remain.

Classification is stronger than in its first confirmation: every dataset is positive.
Regression preserves almost the same point effect and sign pattern—Auction Verification
is again negative—but the five-dataset interval narrowly includes zero. The joint gate
therefore fails. The real regression result remains valid under its predeclared
outer-training-fold normalization, but strict context-rescaled robustness is suggestive,
not confirmed. No unfavorable identity is removed.

Artifacts: `CONTEXT_RESCALED_CONFIRMATION_PROTOCOL.md`, config hash `c83c219597...`, raw
run `context_rescaled_confirmation_c83c219597`, and
`results/processed/context_rescaled_confirmation_audit_v1.json`.
Raw reconstruction matches parent losses within `3.24e-7`; the raw bundle SHA-256
begins `7c51679617ff`.
