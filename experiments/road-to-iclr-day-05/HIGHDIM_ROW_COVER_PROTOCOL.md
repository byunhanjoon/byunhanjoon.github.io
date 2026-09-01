# High-dimensional field-and-row strength-2 cover

Status: frozen before outcomes.

## Question

Does pairwise-balanced nuisance coverage remain useful when the declared
pipeline includes training-row order, and is its advantage explained by
marginal balance alone?

## Frozen design

- datasets: Bank Marketing, German Credit, Compustat Korea direction;
- models: ordinal random forest, native CatBoost, one-hot Adam MLP;
- factors: four feature orders, four model seeds, binary target numbering,
  one binary factor per categorical field, and binary training-row order;
- row-order level 0 is the loaded training order; level 1 is one fixed
  nonidentity permutation, applied jointly to training features and labels;
- train/validation/test caps: 10,000/1,000/1,000;
- equal budget: 32 fits per ensemble;
- methods: randomized pairwise-balanced OA-32, independently
  marginal-balanced-32, and iid-32;
- 8 independent design repetitions; ensemble predictions are retained.

The OA is the same full `GF(2)^5` construction as the field-wise experiment,
with a distinct nonzero linear form assigned to row order. The marginal
control has exactly eight copies of each four-level value and sixteen copies
of each binary value, but independent column shuffles destroy pair balance.
All three designs are unbiased for the full finite nuisance product after
their random level-name permutations.

Primary outcome: unbiased between-repetition squared prediction risk on the
test split. The OA must beat both controls to support the pairwise-interaction
mechanism. Validation is analyzed identically as a transfer check. Brier score
is secondary; no predictive-performance improvement is required by the
variance-reduction theorem.

