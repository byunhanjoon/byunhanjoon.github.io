# Finite-population without-replacement baseline

Status: frozen before outcomes.

Post-gate scope addendum (2026-08-28): apply the unchanged finite-population
comparison to the task-balanced and four-class panels. These panels do not
alter the original confirmation/external gate.

IID sampling with replacement is not the strongest unstructured baseline on
a finite nuisance product. A reviewer can reasonably require uniform simple
random sampling without replacement (SRSWOR), which avoids duplicate fits.

For a product with `N` cells and a uniform sample of `B <= N` distinct cells,
the exact quotient-estimator residual is

`R_SRSWOR = (R_joint / B) * (N-B)/(N-1)`.

This follows from the standard finite-population correction and is evaluated
without Monte Carlo error. Reuse the validation-screened held-out test cells
from the confirmation and untouched OpenML panels. Compare budget-16
strength-2 against SRSWOR-16. The frozen gate requires lower pooled residual
on both panels, at least 20/25 confirmation wins, and at least 9/12 OpenML
wins. Also report the budget-4 strength-1 comparison descriptively.
