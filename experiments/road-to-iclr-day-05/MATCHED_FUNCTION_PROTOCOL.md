# Exact matched-function schema control

Status: frozen before running or inspecting this control's outcomes on
2026-08-28. This is a late completion of the matched-initial-function control
required by the Day-5 program. It does not retroactively alter any earlier
gate.

## Question

For a categorical-embedding MLP, how much prediction variation across exact
schema representatives remains when the transformed model is initialized as
the exact same semantic function as the canonical model?

## Frozen panel

- datasets: Australian Credit, Bank Marketing, German Credit, LendingClub;
- task: binary classification;
- fixed train/validation/test arrays from the Day-5 Tier-1 source;
- caps: 3,000 train, 1,000 validation, 1,000 test rows;
- model seeds: 101, 202, 303, 404;
- model: float64 categorical-embedding MLP, 8-dimensional embeddings, one
  64-unit ReLU hidden layer, full-batch Adam for 40 epochs;
- nuisance representatives: identity plus three deterministic combined
  transformations of numeric-column order, every categorical ID map, and
  target-label ID;
- numerical columns are standardized once in canonical semantic order before
  their order is permuted; categorical mappings are fitted on training data
  and reused unchanged on validation/test.

The numeric-column permutation is a genuine but restricted column-permutation
control; categorical-field order is not permuted. The declared four-state menu
is not the full natural symmetry group.

## Arms

1. **Ordinary transformed initialization:** reset the same positional
   parameter arrays from the seed, so new numeric positions, category IDs, and
   target IDs inherit the positional parameters that happen to occupy them.
2. **Matched-function transformed initialization:** permute first-layer input
   columns, embedding rows, and output rows/biases so aligned canonical and
   transformed predictions represent the same initial function.

Both arms use identical examples, full-batch update counts, optimizer,
hyperparameters, and seed. Predictions are aligned back to canonical target
IDs before comparison.

## Integrity gates

- maximum aligned matched-initial prediction gap below `1e-10`;
- identity predictions agree exactly between arms;
- category maps and class maps are permutations;
- validation/test use the training category dictionary and nuisance map;
- all aligned probabilities are finite, in `[0,1]`, and sum to one;
- every configured dataset×seed×representative×arm fit is present.

## Endpoints and interpretation

For each dataset×seed, compute prediction-space variance over the four aligned
representatives for both arms, then report the pooled fraction removed,
dataset means, and the maximum post-training aligned matched gap.

Gate F is supported only if ordinary schema variance is material and some
non-numerical fraction remains after function matching. If matched variance is
at numerical zero, report that this controlled optimizer/training recipe is
equivariant after the parameter transformation and that this experiment does
not support a residual optimization-path effect. Do not generalize that null
to unmatched architectures or stochastic minibatch pipelines.
