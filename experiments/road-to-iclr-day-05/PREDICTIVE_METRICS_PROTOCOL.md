# Predictive-metric diagnostic for randomized strength-2 covers

Status: post-confirmation diagnostic, frozen before these nonlinear metrics
were computed.

On validation-screened confirmation cells, draw 128 randomized realizations
per action using RNG seed `2026082802`. Compare strength 2, iid-16, four
strength-1 covers, and four seed blocks on test Brier/MSE, classification
log-loss, ROC AUC, and accuracy, or regression R-squared. Test labels are used
only for this evaluation. Brier/MSE remains primary because its quotient-risk
identity is exact; nonlinear metrics are supporting performance context.

## High-precision uncertainty addendum

Because 128 realizations can visibly reverse tiny cell-balanced differences,
repeat the diagnostic independently with 2,048 realizations per action and
cell, streamed in batches of 32. Report Monte Carlo standard errors and normal
95% Monte Carlo intervals. Verify the simulation against exact expected
Brier/MSE from the incidence-covariance calculation; these intervals describe
simulation error conditional on fixed test sets, not dataset uncertainty.
