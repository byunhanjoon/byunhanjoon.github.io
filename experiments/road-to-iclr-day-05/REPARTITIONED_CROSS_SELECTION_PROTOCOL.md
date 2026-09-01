# Repartitioned cross-score selection protocol

Status: protocol frozen before inspecting outcomes; analysis complete.

## Question

Does the strong validation-side efficiency of the independent-cover
cross-score improve held-out model choice after averaging over evaluation
partitions, or does finite-sample validation winner's curse remain dominant?

## Design

Use the eight external OpenML sources and the eight task-balanced OpenML
sources. Pool only their untouched validation and test predictions; training is
not repeated or accessed. Generate 1,024 paired deterministic draws per source.
Each draw contains:

1. a new evaluation partition with the original validation size and, for
   classification, the original validation class counts;
2. two independent 16-fit strength-2 covers, scored by their residual inner
   product;
3. 32 IID fits, scored by the complete pairwise U-statistic.

Use common partitions and nuisance coordinates between methods. Candidate
scores use the redrawn validation rows. Selection is evaluated against both
the exact 128-fit quotient winner on that validation partition and the exact
quotient winner on its held-out complement. Report exact validation regret,
exact test regret, and agreement with both winners. The target is quotient
Brier/MSE, not the realized finite ensemble.

## Frozen interpretation

- **Transfer pass:** cover cross-score has lower mean validation regret and
  lower mean test regret than IID-U in both panels, with favorable mean test
  regret on at least 4/8 sources in each panel.
- **Validation-only pass:** validation regret is lower in both panels but the
  transfer clause fails.
- **Method failure:** validation regret is not lower in both panels.

This is a conditional repeated-partition analysis of already trained models.
It estimates sensitivity to evaluation sampling; it is not a repeated-training
or new-source generalization experiment.

## Outcome

The frozen outcome is **validation-only pass**.

- External: cover validation agreement is 99.34% versus 98.55% for IID-U and
  regret is `1.09e-5` versus `3.47e-5`. Mean test regret is slightly favorable
  (`-1.13e-4` cover-minus-IID; 4/8 favorable, one tie), but the equal-source
  interval `[-3.73e-4, 4.32e-5]` crosses zero.
- Task-balanced: cover validation agreement is 98.02% versus 96.06% and regret
  is `2.41e-5` versus `7.28e-5`. Mean test regret is essentially tied/slightly
  adverse (`+6.51e-6`; 1/8 favorable, four ties), with source interval
  `[-5.17e-5, 5.80e-5]`. Restricting to its four classification sources gives
  `+1.30e-5` and only 1/4 favorable.
- Validation-regret source intervals exclude zero favorably in both panels;
  test-regret source intervals exclude zero in neither.

Thus score efficiency reliably solves nuisance estimation conditional on a
sampled validation partition, but does not reliably solve evaluation-sample
model-selection noise. Proposition 22 gives the exact complement identity
behind this boundary.
