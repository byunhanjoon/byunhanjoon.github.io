# Prospective multiclass target-relabeling cover

Status: frozen before multiclass outcomes.

Post-gate neural addendum (frozen before its outcomes on 2026-08-28): add the
unchanged one-hot Adam MLP to both sources. The initial linear/forest gate is
not altered; report MLP materiality and held-out cover comparisons separately
and then update the descriptive multiclass scope totals.

The prior exact panels exercise binary or singleton target-ID factors. Test
the four-level construction on OpenML vehicle (4 semantic classes) and segment
(7 semantic classes), with four distinct target-ID permutation maps treated as
the declared nuisance factor. Use four feature orders, no category-ID factor
when the data are numeric-only, four seeds, and fixed linear/forest models.
Every prediction is realigned to original semantic class coordinates before
scoring.

Store all `4 x 1 x 4 x 4 = 64` fits per cell. The construction gate requires
exhaustive strength-2 and strength-3 margin verification for a four-level class
factor and ambiguity/fANOVA reconstruction below `1e-10`. The empirical gate
requires strength-2 lower pooled held-out residual than IID-16, four
strength-1 blocks, and four seed blocks, and at least 3/4 cells beating all.
This small panel closes a task-type scope gap; it is not independent evidence
of broad dataset generalization.
