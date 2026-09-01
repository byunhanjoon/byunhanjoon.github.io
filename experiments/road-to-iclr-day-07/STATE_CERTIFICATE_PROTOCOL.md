# Post-hoc protocol — state-level family-wise certificate audit

Status: **EXPLORATORY; DESIGNED AFTER THE SEALED MLP OUTCOME AND BEFORE THE
NEW BACKBONE OUTCOMES WERE ANALYZED JOINTLY**

## Motivation

The frozen MLP experiment used five state folds and treated their aggregate
gains as the uncertainty sample. That gave only five observations per
operator, blurred heterogeneous state effects, and did not account for
choosing the best of nine operators. A pre-backbone prototype showed that
twenty folds improved MLP ranking correlation, motivating this audit.

## Fixed refinement

- deterministically split observed states into twenty folds;
- rebuild every operator after removing a fold;
- retain a gain for each held-out state, then average within fold so shared
  training data are not counted as independent evidence;
- choose the operator with the highest mean fold gain;
- compare positive mean, pointwise `1.96 SE` lower bound, and a Bonferroni
  family-wise 95% lower bound over the nine searched operators;
- replay each decision on the untouched outer states of all four datasets,
  two splits, and four backbones.

The family-wise threshold is `Phi^-1(1 - 0.05/(2K))`, with `K=9`.

## Interpretation constraints

This is a mechanism audit, not confirmation. It can determine whether coarse
folding and uncorrected operator search explain observed harm. It cannot
support a distribution-free certificate: target residual alignment is absent
without labels, folds share learned residual bases, sources are reused, and
semantic-state exchangeability remains an assumption. Any successful rule
must be frozen and tested on genuinely new source families next.
