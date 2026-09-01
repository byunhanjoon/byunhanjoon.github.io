# Context-rescaled confirmation robustness

Status: frozen on 2026-09-01 before outcomes. This responds to a reviewer-style audit of
the real runners, which found that feature and regression-target scaling used the full
official training fold before context sampling. That choice was predeclared and query
labels were never used, but it is not strict few-shot preprocessing.

## Frozen correction

Reuse the five regression-confirmation identities and five independent binary-shrinkage
identities, but draw 50 fresh episodes per dataset with seed 235001, n=96, q=64, and
three CV folds. After sampling each context:

- refit feature centering/scaling on the 96 context rows and apply it to the query;
- for regression, refit target centering/scaling on the 96 context labels and apply it
  to query targets;
- retain outer-train-only label-free feature selection and median imputation;
- retain the original synthetic fixed weights, competence temperature, and experts.

The extra episode rescaling algebraically cancels the outer-fold affine scaling. No
dataset identity, method parameter, or candidate is changed.

## Gates

Primary paired gains are full competence versus fixed for regression and the frozen 10%
competence prediction versus fixed for classification. Use equal-dataset hierarchical
bootstraps with 20,000 draws, seed 235501 onward. Each task passes if its 95% interval is
above zero and at least 3/5 dataset point estimates are positive.

This is preprocessing robustness on known confirmation identities, not another
independent-identity confirmation. Failure narrows the real claim to outer-fold-scaled
benchmarks; passing supports the stronger context-rescaled scope. Outer-train median
imputation and schema selection remain explicit limitations.
