# Regression competence CV-budget result

Status: **two-fold retains performance versus fixed, but the frozen noninferiority gate
versus three-fold fails**.

| Router | Standalone fits/episode | Mean MSE | Gain vs fixed (95% hierarchical CI) |
|---|---:|---:|---:|
| Fixed mixture | 6 | 1.066803 | — |
| 2-fold competence | 18 | 0.975910 | +0.090893 [0.004574, 0.230806] |
| 3-fold competence | 24 | 0.968428 | +0.098376 [0.003187, 0.238067] |
| 5-fold competence | 36 | 0.965589 | +0.101215 [0.003629, 0.239257] |

Two-fold cuts standalone fit count by 25% relative to three-fold and still beats fixed.
However, `loss(2-fold)-loss(3-fold)` is +0.007482 with CI
[-0.002952, 0.025717]. Its upper bound exceeds the frozen 0.01 harm margin, so the
low-cost gate fails. Five folds provides only a small point improvement for 50% more fits
than three folds. Three-fold remains the defensible performance/compute default.

This result does not invalidate the two-fold performance point estimate; it rejects the
stronger claim that the cheaper rule is demonstrably noninferior on this five-dataset
panel.
