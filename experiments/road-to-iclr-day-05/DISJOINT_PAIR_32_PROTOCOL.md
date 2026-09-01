# Packed disjoint-pair 32-fit protocol

Status: frozen after the 64-fit antithetic result and before inspecting these
32-fit outcomes.

## Question

Does packing two disjoint strength-2 covers improve the actual 32-fit quotient
prediction and its ordinary validation-loss selector, relative to two
independent strength-2 covers at the same budget?

For each of 1,024 paired actions per dataset, share a uniform first cover. The
packed method chooses its second cover uniformly among disjoint graph
neighbors; the control chooses a fresh independent uniform cover. Average the
two covers and evaluate ordinary Brier/MSE. Report candidate score RMSE/bias,
prediction residual, winner agreement, validation regret, and held-out loss.

On a 32-cell nuisance product, the unique disjoint partner completes an exact
partition, so the packed average must equal the full quotient. On larger
products this is an antithetic estimator, not exact enumeration, and its
ordinary loss retains nonnegative residual bias.

## Frozen gate

The gate passes if packed pairs have lower panel-mean candidate score RMSE and
prediction residual in all five panels, and no lower agreement/no higher
validation regret in at least four of five. Also verify exact numerical closure
for every candidate whose nuisance product has 32 cells or fewer.

This post-gate result may strengthen the prediction/computation frontier. It
does not replace the unbiased cross-score when the product is not exactly
partitioned.
