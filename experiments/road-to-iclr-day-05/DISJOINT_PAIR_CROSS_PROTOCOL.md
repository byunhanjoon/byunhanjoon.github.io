# Disjoint-pair antithetic cover cross-score protocol

Status: frozen before inspecting tensor outcomes.

## Construction

For the `4 x 4 x 2 x 4` nuisance product, deduplicate the randomized
strength-2 family into its 1,728 distinct 16-cell covers. Join two covers when
they are cell-disjoint. The graph is regular: every cover has 485 neighbors.
Therefore sampling a uniform first cover and then a uniform disjoint neighbor
gives uniform marginals for both covers.

Apply the same construction to factor-collapsed products. Their deduplicated
graphs are also regular (degrees 1 or 21 in the observed nontrivial shapes).
When the 16-row cover already exhausts a 16-cell product, use that exact cover
for both positions; both methods are then exact and should tie.

Generate two independent disjoint pairs `(A,B)` and `(C,D)`. Let
`Q_AB=(Q_A+Q_B)/2` and `Q_CD=(Q_C+Q_D)/2`, and score

`<Y-Q_AB, Y-Q_CD>`.

The within-pair dependence is allowed; the two pair averages are independent
and marginally unbiased, so the score remains exactly unbiased for quotient
Brier/MSE. Total compute is 64 fits per candidate.

## Comparison and outputs

Over 1,024 actions on all five established selection panels, compare with the
complete block-U score over four independent strength-2 covers at the same 64
fits. Report:

- candidate-level quotient-score bias and RMSE;
- winner agreement and validation quotient regret;
- residual loss of the four-cover prediction average;
- exact fANOVA covariance multipliers of an independent cover mean and a
  disjoint-pair mean.

## Frozen gate

The antithetic gate passes if it has lower panel-mean score RMSE, no lower
winner agreement, no higher validation regret, and lower prediction residual
than independent block-U64 on at least four of five panels for each clause.

Held-out transfer remains diagnostic. A disjoint cross-score between `A` and
`B` alone would generally be biased; exact unbiasedness here comes from
crossing two *independent pair averages*.
