# Unbiased cross-score budget frontier

Status: **frozen before 64-fit outcomes**.

The two-cover cross-score is the degree-2 U-statistic over two independent
cover estimates. At 64 fits, draw four independent strength-2 covers of 16
fits and average the six unordered cross-products of their validation
residuals. Compare it with the complete degree-2 U-statistic over 64 IID
nuisance members. Both scores are exactly unbiased for full quotient
Brier/MSE.

Run 512 actions on the same five panels. Selection is validation-only; report
full-quotient and realized 64-member test loss afterward. Candidate-level
score RMSE is computed around the exact quotient validation loss.

The frontier gate requires:

1. cover-block-U64 lower mean score RMSE than IID-U64 on at least four panels;
2. cover-block-U64 agreement no lower than cover-cross32 on at least four;
3. cover-block-U64 validation regret no higher than cover-cross32 on at least
   four.

The 32-fit values are imported from the already frozen primary experiment.
Test-loss transfer is descriptive because of the known validation/test shift.

This is a classical U-statistic construction over independent randomized
blocks. Its role here is to turn a nuisance-cover schedule into an unbiased
quotient-risk and selection schedule; the U-statistic itself is not new.
