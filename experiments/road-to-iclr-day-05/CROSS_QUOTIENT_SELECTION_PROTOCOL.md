# Cross-quotient selection protocol

Status: **frozen before outcome computation** (2026-08-28).

## Question

The prediction average of a randomized cover is unbiased for the full nuisance
quotient, but its validation Brier/MSE is not: its expectation is the quotient
loss plus the cover residual.  Can two independent covers turn the same idea
into an unbiased *selection score*, and does the lower cover residual improve
selection at equal fit budget?

## Estimators and budget

For every candidate and randomized action, construct two independent 16-fit
strength-2 covers, `A` and `B`.  With residual vectors
`r_A = y - Q_hat_A` and `r_B = y - Q_hat_B`, score the candidate by

`L_cross = <r_A, r_B>`.

This uses 32 fits.  It is compared at the same 32-fit budget with:

1. the same cross-score from two independent IID-16 actions;
2. the order-2 IID U-statistic using all 32 members, a stronger unbiased IID
   baseline;
3. ordinary proper loss of the mean of the two covers;
4. ordinary proper loss of the mean of the two IID halves.

All candidates within a dataset receive the same randomized action indices.
Only validation labels determine selection.  Test labels are used afterward
for the selected full-quotient loss and the realized 32-member ensemble loss.

## Panels

Run the already materialized complete tensors from five selection panels:

- confirmation;
- changed nuisance menu and model seeds;
- changed data subsample;
- external binary OpenML;
- task-balanced external OpenML (four classification, four regression).

No model is refit and no test result is used to define a panel or candidate.

## Frozen primary gate

Relative to the IID U-statistic baseline, `strength2_cross32` must:

1. have higher mean validation-quotient-winner agreement in at least four of
   five panels;
2. have lower mean validation quotient regret in at least four of five panels;
3. have lower mean selected full-quotient test loss in at least three of five
   panels.

The first two clauses test the estimator's intended target.  The third is kept
separate because validation-to-test target shift can reverse a better
validation selector.  Results against IID cross-score and ordinary 32-member
loss are descriptive secondary comparisons.

## Interpretation boundary

The cross-product identity is classical second-moment algebra, not a claim of
inventing unbiased risk estimation.  The possible contribution is its use for
selection over a declared equivalence×seed quotient, coupled to randomized
low-strength covers and tested end to end.  A cross-score can be negative and
need not dominate ordinary ensemble loss at finite variance.
