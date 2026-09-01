# Independent screen-then-cross allocation protocol

Status: frozen before inspecting outcomes.

## Question

Can the cover cross-score be converted into a compute-saving candidate
allocation rule without optional-reuse bias?

## Design

For every dataset in the five established selection panels, use two disjoint
randomization stages:

1. score every candidate with two independent strength-2 covers (32 fits);
2. retain the two lowest pilot scores;
3. score only those two candidates with two fresh independent strength-2
   covers (32 additional fits each), and select the lower deployment score.

The deployment score is independent of the pilot screening event and remains
unbiased conditional on the retained set.  With `M` candidates, total compute
is `32M+64`, versus `64M` for four blocks on every candidate.  The saving is
`1/2-1/M`: 30% for five, 25% for four, and 16.7% for three candidates.

Use 1,024 paired actions per dataset.  Report pilot exact-winner inclusion,
final exact-winner agreement, quotient-validation regret, and quotient-test
loss.  Compare panel means with the already frozen equal-allocation 32- and
64-fit cover selectors.  This is a post-core prospective allocation test; it
does not alter earlier gates.

## Frozen gate

The allocation gate passes if:

- it saves at least 15% of fits versus equal 64-fit allocation in every panel;
- pilot top-two inclusion is at least 98% in at least four of five panels;
- final agreement is no lower and validation regret no higher than the
  equal-32 cover selector in at least four of five panels;
- final validation regret is no more than 25% above equal-64 in at least four
  of five panels.

Held-out transfer remains a separate diagnostic because pilot/deployment
independence does not address validation-to-test rank instability.
