# Frozen protocol — disjoint-state backbone-robust certificate

Status: **FROZEN BEFORE OUTCOMES FROM THE DISJOINT-STATE BASES**

Freeze date: 2026-08-30 (Asia/Seoul).

## Question

Does the apparent safety of the post-hoc backbone-worst rule survive a clean
three-way semantic-state split in which neither the base nor the operator has
seen certificate or test states?

## Fixed design

For each of the four development sources and two existing split indices:

1. use the manifest `train` states (60%) to fit the ordinary-covariate base and
   estimate operator residual state effects;
2. use the disjoint `validation` states (20%) only to estimate operator gain
   and form a decision;
3. use the untouched `test` states (20%) once to measure deployed gain.

Within construction states, three-fold row OOF predictions estimate residual
state effects. One full construction-state fit predicts validation and test
rows. The semantic field and every derivative remain excluded from the base.
The MLP, ResNet, FT-Transformer, TabM, optimizer, row cap, epoch budget, and
nine operators are exactly those in `BACKBONE_PROTOCOL.md`.

## Fixed decision rules

- `mean`: per backbone, choose the operator with largest validation gain and
  deploy it iff its mean is positive;
- `per_backbone_lcb`: same, but require a 95% Gaussian LCB with Bonferroni
  correction over nine operators;
- `backbone_worst`: choose one operator per task-split by maximizing the
  minimum LCB over the four backbones, using one joint correction over
  `4 × 9`; deploy it for all backbones iff that minimum is positive.

Validation gain observations and standard errors use semantic states, not
rows, as the unit. Test gain is state-balanced standardized MSE improvement.

## Frozen development gates

1. `backbone_worst` has no practically harmful test cell below `-0.002`;
2. it has positive mean test gain;
3. it selects at least two source-split cells from at least two sources;
4. it causes fewer harmful cells than `mean` and no more than
   `per_backbone_lcb`;
5. all OOF and output integrity checks pass.

Passing all gates would justify a fresh-source confirmatory protocol, not a
paper claim. Failure of gate 3 means the robust rule is statistically
non-actionable even if it abstains safely.

## Known inferential limitation

The Gaussian LCB is a development approximation for unbounded squared loss.
A submission must either use a bounded proper loss, establish a justified
tail condition, or adopt a finite-sample robust mean bound. The existing test
states have appeared in related experiments, so this run is structurally
nested but not prospectively novel at the source level.
