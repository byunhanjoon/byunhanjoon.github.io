# Frozen diagnostic: quotient margins and the accuracy boundary

For the 23 non-exhaustive binary candidates, compute the complete quotient's
absolute class-probability margin and the fractions of validation examples
within `0.001, 0.005, 0.01, 0.02, 0.05` of a tie. Relate those quantities to
the paired-minus-control accuracy-metric RMSE difference at pair32 and pack64
using Spearman correlation and a deterministic 100,000-permutation test.

Also report near-tie mass separately for strict accuracy wins and non-wins.
This is an explanatory diagnostic with no performance gate. The theoretical
claim is only margin-conditioned: small prediction error can still flip an
argmax where the quotient margin is small.
