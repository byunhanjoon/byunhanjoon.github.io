# Frozen real-regression CV-budget experiment

Frozen: 2026-09-01, before fresh seed-165001 outcomes.

## Question

Can context competence retain its confirmed regression gain with two rather than three
CV folds, reducing routing compute?

## Design

Use all five independent-confirmation identities, 40 fresh repeats, 96 context rows, 96
shared query rows, and seed family 165001. Fit the six query experts once per episode.
Estimate context competence separately with 2, 3, and 5 folds, using the same frozen
temperature 0.1 and no shrinkage.

Total expert fits per episode are 18 for a standalone two-fold router (6 full + 12 CV),
24 for three-fold, and 36 for five-fold. Thus two-fold reduces fit count by 25% versus
the confirmed three-fold rule.

## Frozen analysis and gate

Report dataset-balanced loss for fixed and every fold count with 10,000 hierarchical
paired bootstrap draws. The low-cost gate passes only if:

1. two-fold competence beats fixed with a strictly positive interval; and
2. the upper 95% bound of `loss(two-fold) - loss(three-fold)` is at most 0.01 MSE.

Five-fold is diagnostic. No fold-specific temperature tuning is allowed. This is a
post-confirmation efficiency experiment on seen identities, not external confirmation
or algorithmic novelty.
