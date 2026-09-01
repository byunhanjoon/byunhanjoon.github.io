# Empirical cross-score variance-identity audit

Status: protocol frozen before inspecting outcomes; analysis complete.

## Question

Does Proposition 19 quantitatively predict cross-score variance on the real
aligned prediction tensors, or is it only an algebraic toy identity?

## Design

Use all 171 candidate cells in the five model-selection panels. Audit both a
16-fit strength-2 cover estimator and a 16-fit IID-mean estimator. For each
candidate/method generate four independent 1,024-draw streams from the stored
complete tensor:

- streams A/B estimate `2<r,Cr> + tr(C^2)` through the unbiased drawwise terms
  `<r,e_A>^2 + <r,e_B>^2 + <e_A,e_B>^2`;
- disjoint streams C/D estimate the actual variance of
  `<Y-Q_hat_C,Y-Q_hat_D>`.

Report predicted/observed variance ratio and a standardized discrepancy that
combines the Monte Carlo standard errors of the component mean and sample
variance. Exclude only cells where both quantities are below `1e-18` from
ratio and standardized summaries; retain their raw values.

## Frozen gate

The identity calibration passes if at least 90% of nondegenerate cells have
absolute standardized discrepancy at most 2.58 for each method and every
panel-level geometric mean predicted/observed ratio lies in `[0.8,1.25]`.

This audit checks Proposition 19, not whether cover variance is always below
IID variance; that efficiency question is tested separately.

## Outcome

The frozen gate **passes**.

- All 141 nondegenerate IID cells and 72/73 nondegenerate cover cells have
  absolute standardized discrepancy at most 2.58.
- Median absolute standardized discrepancies are 0.647 (IID) and 0.803
  (cover).
- All ten panel/method geometric mean predicted/observed ratios pass. They lie
  in `[0.988, 1.042]`, far inside the frozen `[0.8,1.25]` range.
- The lower number of nondegenerate cover cells (73 versus 141 IID) reflects
  the many tensors whose higher-order residual is numerically annihilated by
  strength-2; all raw values remain in the cell table.

Thus the covariance-operator equation quantitatively predicts real
cross-score variance across binary, multiclass, and regression tensors, not
only the synthetic unit example.
