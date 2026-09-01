# Mutually disjoint four-cover packing protocol

Status: frozen after the disjoint-pair result and before inspecting four-pack
outcomes.

## Construction

Sample a uniform cover `A`; sample `B` uniformly among its disjoint neighbors;
sample `C` uniformly among common disjoint neighbors of `A,B`; and sample `D`
uniformly among common disjoint neighbors of `A,B,C`, resampling an earlier
choice only if extension is impossible. The algorithm is equivariant under
the transitive factor-level permutation group, so the expected incidence of
the unordered four-pack is uniform over product cells and its prediction mean
is quotient-unbiased.

For 64-cell products, four mutually disjoint 16-cell covers partition the
whole nuisance product and must close exactly. For 32/16-cell products, use
the already exact disjoint pair/full cover. For the 128-cell product, compare
the genuine four-pack with two independently sampled disjoint pairs and four
independent covers.

## Experiment and frozen gate

Over 1,024 actions on all five panels, evaluate direct prediction residual,
ordinary-loss RMSE/bias, exact-winner agreement, validation regret, and held-out
loss. The four-pack gate passes if, versus two disjoint pairs at the same 64-fit
budget, prediction residual and score RMSE are no higher in all five panels and
strictly lower in at least three, and agreement/regret improve or tie in at
least four. Verify exact numerical closure on every product of at most 64
cells.

This is a post-gate compute-frontier extension. Ordinary loss is not exactly
unbiased on the 128-cell product; an independent outer cross of two four-packs
would require 128 fits.
