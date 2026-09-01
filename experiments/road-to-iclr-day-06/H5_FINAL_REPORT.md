# H5 final report — Cross-Perturbation Fragility Transfer

Status: **FINAL**

## Verdict

**FALSIFIED AS WRITTEN.**

H5 reuses all 324 H4 bundles and performs no outcome-dependent retraining.  In
the FT cells, the epoch-2 semantic shadow versus epoch-20 independent-seed
prediction variance has Spearman at least .60 in 0/3 datasets: Bank .364,
Credit .441, and FreMTPL .322.  Equal-dataset pooled Spearman is .375,
improvement over the constant epoch-zero control is .375, and equal-dataset
mean top-quartile AUROC is .704.  Only the improvement gate and the complete-
matrix/matched-initialization gate pass.

## Interpretation

Neither score uses test labels: the early statistic is prediction disagreement
between exact conjugate schema paths, and the target is canonical prediction
variance across independent training seeds.  This makes a pass a transfer
result between perturbation sources, not a disguised test-loss selection rule.
It still would require new seeds, optimizers, datasets, and equal-cost
baselines before a practical screening claim.

Ranks are computed within dataset and ties use the frozen average-rank rule;
constant scores have association zero.  The twelve configurations and three
seeds are repeated measurements.  Results on MLP and ResNet are scope controls,
not gates.

## Decision

Discard cross-perturbation fragility transfer as a Day-6 paper claim.  Do not
tune a new score on this completed tensor panel; any revisit requires new
datasets, seeds, optimizer settings, and equal-cost baselines.
