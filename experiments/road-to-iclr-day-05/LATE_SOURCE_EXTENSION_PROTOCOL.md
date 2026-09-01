# Frozen protocol: late untouched source extension

Frozen before dataset download, tensor fitting, or outcome inspection on
2026-08-28 (Asia/Seoul).

## Purpose and fixed sources

The strongest remaining empirical weakness is the small number of independent
non-exhaustive sources.  Add four OpenML binary-classification datasets not
used by any prior Day-5 cover panel, fixed by data ID:

- kr-vs-kp (3);
- credit-approval (29);
- sick (38);
- tic-tac-toe (50).

They were selected as a compact, classical mixed/categorical-input panel that
can plausibly finish inside the remaining sprint.  No alternative source will
replace a failure after outcomes are seen.

## Frozen fit design

For `onehot_linear`, `ordinal_forest`, and `onehot_adam_mlp`, fit the complete
`4 feature-order x 4 category-ID x 2 class-ID x 4 seed = 128` nuisance product
using the same split, preprocessing, budgets, and implementation as the prior
task-balanced OpenML panel.  Store every validation/test prediction and
manifest.  Dataset download, schema, fit, or convergence failures are part of
the result.

## Predeclared analyses

1. Verify complete tensors, probability validity, and exact fANOVA identity.
2. At 16 fits compare randomized strength-2 covers with equal-fit IID joint
   draws, four strength-1 blocks, and four seed blocks using 2,048 actions.
3. At 32 and 64 fits compare disjoint pair/four-pack prediction RMSE with their
   frozen controls; at 128 fits compare exhaustive quotient evaluation with
   randomized packed cross-score only as a closure check.
4. For validation-only model selection, compare packed/unbiased scores on
   exact-winner agreement and quotient regret.  Report test transfer
   separately; it is not allowed to redefine a validation gate.

The primary late-source gate requires strength-2 residual risk below IID and
strength-1 in at least 10/12 candidate cells and all four equal-source means.
Seed-block comparison, packing, and selection are secondary scope checks.

## Post-failure implementation note

The inherited loader rejected IDs 3 and 50 before fitting because they have no
numerical fields. The failure is retained. A categorical-only adapter was then
added: it represents the numeric block as zero columns, preserves every
categorical column, and uses the same stratified split seed. This is a schema
compatibility repair, not a replacement dataset or outcome-conditioned model
change.
