# Post-outcome validation-to-test selection diagnostic

Status: explanatory audit specified after the external selection gate failed.
It cannot rescue or redefine that gate.

Post-outcome decomposition addendum: for every randomized method, decompose
held-out quotient regret exactly into the test regret of the exact validation
winner (the target-shift floor) plus the method's nuisance-selection term.
The latter can be negative when noisy validation selection accidentally picks
a model preferred by the test split; this diagnostic cannot change any gate.

For each dataset, compute validation and held-out test Brier/MSE for every
candidate's exact full nuisance quotient. Report whether the validation
quotient winner is also the test quotient winner, its held-out oracle regret,
and candidate-rank correlation. This separates nuisance-estimation error—the
quantity a cover controls—from validation-to-test target shift, which it
cannot control.
