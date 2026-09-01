# ICLR 2027 paper freeze card

## Primary decision

**Learning on the Schema Quotient: Attributing Arbitrary-Representation Risk
Across Tabular Pipelines** (**OrbitANOVA**)

Readiness: **3.5/5**. Develop this as the one primary paper. The paper is not
submission-ready until the frozen broad audit and held-out action tests pass.

### Thesis

A tabular pipeline should be evaluated over a declared quotient of schema
spellings that preserve the task, not on one lucky representation. Aligned
prediction dispersion gives a label-free proper-loss tax; product fANOVA says
which schema and randomness factors create it; the same audit should predict a
targeted repair.

### Four headline contributions

1. **Declared schema quotient.** Separate exact group symmetries, equivalent
   within-field function bases, and semantics-backed unit charts. Align output
   labels before comparing complete fitted pipelines.
2. **Risk-valued attribution.** For Brier/squared loss, schema prediction
   variance is exactly the proper-loss gain from quotient averaging. Attribute
   the total to schema factors/interactions and cross it with seed/split/search
   randomness. Treat log-loss ambiguity separately with its geometric centroid.
3. **The tuning path is part of the pipeline.** Audit fixed-recipe and
   selection-rule orbits separately. Decompose configuration switching exactly
   as `Delta SR = Var(d) + 2 Cov(p_frozen,d)` and fANOVA the one-hot decisions.
   For sampled development menus, include menu seed and schema×menu effects;
   only a complete uniform group has no menu-sampling randomness.
4. **Audit predicts targeted action.** Test pooled validation selection on
   held-out nuisance levels, OrbitCover against iid schema/seed ensembles,
   covariant training, and OrbitCascade. Report proper loss, schema risk, every
   randomness component, and compute; no action is assumed to dominate.

### Strongest current evidence

- Material conventional cells on Adult/Churn/Otto remove 0.21%--1.13% of mean
  member Brier under quotient averaging; hard labels change on 2.3%--11.6% of
  rows. Neural same-seed chart tax is 0.63%--8.20% of mean proper loss, with
  chart×seed contributing 54%--65% of joint variance.
- In a 3-dataset × 3-family HPO pilot, selection changes within the orbit in
  3/9 baseline cells and raises schema risk in all three. Seven unseen
  validation splits confirm higher risk in 7/7 Adult-CatBoost, 6/7
  Churn-forest, and 7/7 Churn-CatBoost cases (magnitude/binomial-sign
  `p=.0156/.0156`, `.0469/.125`, `.0156/.0156`). The
  amplification is primarily schema×split coupling, not persistent mean bias.
- Prospective screens complete the original 3×3 panel. Adult forest switches
  on 2/7 new splits and Otto forest on 1/7; Adult/Churn/Otto HistGB and Otto
  CatBoost remain stable on 7/7. Across the panel, forests ever switch in 3/3
  cells, CatBoost in 2/3, and HistGB in 0/3 (5/9 total). This is strong
  family-structured pilot evidence, not a population prevalence estimate.
- The promoted Otto-forest switch is driven 100% by feature order. Reselection
  improves Brier by `.00510` but raises schema risk 2.35× (`.00321→.00755`)
  and hard flips `13.8%→16.9%`, extending the trade-off beyond mixed binary data.
- Development-sub-orbit pooled selection reduces held-out same-split schema
  risk by 28%/35%/32% in the three confirmation cases (magnitude/binomial-sign
  `p=.0313/.125`, `.0313/.0313`, `.0625/.219`)
  without a resolved Brier contrast. It can increase split-main variance, so
  this is targeted quotient repair and risk relocation—not total stability.
- That result uses one frozen nuisance partition. Across all 36 balanced
  partitions per split, the pooled decision matches the full-menu choice only
  88%/85%/74% of the time and is held-out-validation optimal 76%/75%/57%.
  This is a decision diagnostic on shared validation rows, not an independent
  performance test. The paper must draw independent development/evaluation
  menus, cross menu seed in the ANOVA, and never select a favorable partition.
- In decision fANOVA, menu×split interactions already explain 28%/44%/51% of
  balanced-menu choice variance in the three cases (pure menu main only
  3%/5%/10%). This is selected-case mechanism evidence, not prevalence.
- At output level, averaging the 36 menu-specific held-out comparisons within
  each split reduces schema risk on 7/7 splits in all three cases (both exact
  tests `p=.015625`), by 29%/38%/35% overall. Individual-menu success is only
  228/252, 180/252, and 198/252, and Brier moves against pooling on most splits.
  Claim a menu-distribution quotient-risk frontier, never uniform dominance.
- Under a 2× adverse density-ratio tilt over menus, splitwise favorable counts
  fall to 7/7, 4/7, and 5/7 (at 4×: 6/7, 2/7, 4/7). Always show sensitivity to
  the declared `mu`; uniform weighting is part of the estimand.

### Claims that are forbidden

- “We discovered that tabular models are representation sensitive.”
- “OrbitANOVA/fANOVA/Bregman information/Rao--Blackwellization is a new generic
  mathematical identity.”
- “Arbitrary schemas broadly reverse architecture rankings.” All four selected
  three-seed reversals disappeared on 13 new seeds.
- “Same integer seed defines intrinsic schema distance.” It is an operational
  coupling; report persistent and same-seed quantities separately.
- “Pooled HPO is invariant” for a sampled non-closed menu, or “more stable
  overall” when variance merely moves from schema×split to split main.
- “252 menus are independent replications.” Balanced menus overlap; average
  within split and use prospective splits/tasks as the replication units.
- “OrbitCover, OrbitCascade, or stable HPO is a standalone new method.”

## Minimum frozen study

- 12--15 datasets; exact metadata provenance; classification and regression.
- Exact-group controls plus declared category/unit/field-chart extensions.
- Classical, boosted-tree, neural, and tabular-foundation pipelines with exact
  versions, metadata interfaces, internal ensembles, and deterministic flags.
- Sequential total-risk screen; exact fANOVA on small products and frozen
  pick-freeze estimators only for promoted cases.
- At least 8--16 paired seeds for shortlisted conclusions. Dataset—not row—is
  the cross-task replication unit.
- Selection substudy: 3 datasets × 3 sensitive families × 16 representatives
  × 12 semantic split seeds × six candidates and three final paths. Use
  predeclared development/evaluation menu draws; average menu-level estimands
  within split, and cache full refits by `(representative, configuration)`.
  Approximate conservative full-study ceiling:
  **24,992 fits/passes** before caching and early stopping. Twelve split seeds
  let a 10/12 direction pass a two-sided binomial sign test; eight do not.
- Prospective action choices frozen on development dataset/model cases and
  evaluated on unseen cases at matched fits, inference passes, wall time,
  memory, and stored-model count.

### Four main figures

1. Quotient declaration, output alignment, and exact proper-risk identity.
2. Dataset × pipeline × nuisance risk atlas with invariant and seed controls.
3. Selection-rule path: decision fANOVA, switch/covariance identity, and
   schema×split×menu confirmation.
4. Held-out action frontiers: menu-distribution pooled selection and OrbitCover
   primary; covariant/OrbitCascade closures secondary.

### Submit gate

Submit only if the broad atlas establishes heterogeneous material effects in
multiple conventional and modern families, selection-path prevalence survives
the full predeclared panel, and at least one audit-guided action transfers to
unseen dataset/model or nuisance cases at matched resources.

## Conditional second direction

**Features Are Function Spaces, Not Coordinates: Chart-Covariant Field
Training** (FieldRiesz)

Current status: strong causal intervention inside OrbitANOVA; conditional
standalone idea only.

- Existing Adult/Diamond/Black-Friday pilots close chart risk to numerical
  precision; field-VectorAdam removes 99.97% of Black Friday conditional chart
  risk while retaining raw-member MSE.
- A 27-cell fixed-strength Bayes suite shows controlled selectivity: matched
  path/ring metrics beat permuted topology in all 18 ordinal/cyclic cells;
  isotropic beats every fixed false topology in all nine nominal cells.
- A post-freeze calibration stress test overturns the tempting stronger claim.
  When each family tunes stiffness over `{0,1,4,16}`, false path, ring, and
  permuted-path topologies significantly beat isotropic on 8/9, 8/9, and 9/9
  nominal cells. Validation exploits accidental adjacency; it cannot infer
  semantic topology. Field type must be declared metadata, with tuning only
  inside the declared family.
- Classical splines/Galerkin metrics, natural/reparameterization-invariant
  optimization, VectorAdam, Lie-symmetry frameworks, and topology matching are
  prior art. The only plausible novelty is their tabular input-field-specific
  composition.

Promote it to a second paper only if at least three neural architectures and
ten datasets show >95% chart closure, competitive single-model proper loss,
selective benefit from declared path/ring/nominal metadata, robustness to
equivalent charts, and advantage beyond whitening+SGD/field-VectorAdam
baselines. Do not require or claim that unconstrained validation reliably
rejects wrong topology—the calibration experiment falsifies that premise.
