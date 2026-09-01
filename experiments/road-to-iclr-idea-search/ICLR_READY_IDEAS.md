# ICLR-ready idea decision

Date: 2026-08-26  
Target: ICLR 2027

## Decision

Develop one primary paper:

> **Learning on the Schema Quotient: Attributing Arbitrary-Representation
> Risk Across Tabular Pipelines**

Framework: **OrbitANOVA**. Supporting interventions: **OrbitCover** for
factor-level compute allocation and **OrbitCascade** for row-adaptive
approximation of a finite quotient predictor.

Do not split OrbitCover or OrbitCascade into separate papers. Do not lead with
“tabular models are sensitive to representation”: Liu et al. establish that
premise for LLMs and tabular foundation models, and PREF independently studies
broad preprocessing volatility. The ICLR contribution must be the complete
measurement-to-action chain over *semantically equivalent* schema choices,
plus the surprising real-data result that conventional tabular pipelines are
not uniformly protected.

Keep one conditional method direction:

> **Chart-Covariant Field Training**: render each semantic field in a whitened
> function-space chart, transport initialization, and use field-block
> rotation-equivariant optimization.

This is already a strong causal intervention inside the primary paper. It
should become an independent idea only if a frozen benchmark shows competitive
single-model performance and a selective benefit from *declared* field
topology. Validation cannot be allowed to search over semantic topologies:
the post-freeze calibration below shows that it exploits accidental adjacency
even when the field is nominal.
Generic optimizer invariance and VectorAdam are prior art, so exact trajectory
closure alone is insufficient.

Classical spline/Galerkin penalties already turn semantic derivatives into
basis-coordinate quadratic forms; [Otto et al.
(2025)](https://www.jmlr.org/papers/v26/24-1315.html) give a broad Lie-symmetry
framework covering basis regression, neural networks, and fields; and
[PH-Reg](https://proceedings.mlr.press/v235/zhang24z.html) already promotes
topology matching for regression representations. Any standalone novelty must
therefore be the tabular field-metadata construction and its joint covariant
initialization/regularization/optimization—not topology, function-space
penalties, or symmetry enforcement separately.

## The primary paper in one sentence

A benchmark should score a tabular pipeline over the quotient of schema
spellings declared equivalent by the task, then report which arbitrary factor
and interaction creates the removable prediction risk and whether the model
comparison survives that quotient.

## Why this can clear ICLR

The paper is not one new theorem. Its value is a new empirical object and a
coherent chain of consequences:

1. aligned predictions from equivalent schema representatives define a
   label-free proper-score tax;
2. a product declaration attributes the squared tax exactly to schema factors
   and interactions;
3. crossing schema with training randomness distinguishes persistent bias from
   a same-seed operational witness;
4. an end-to-end selection-rule orbit attributes when validation tuning
   amplifies schema×search coupling and where configuration switches enter;
5. the attribution predicts targeted repairs—pooled validation selection,
   factor marginalization, covariant training, and rowwise compute allocation;
6. quotient model comparisons remain a downstream test of whether a reported
   conclusion is schema-identifiable, not the headline empirical promise.

Each mathematical ingredient—Bregman ambiguity, functional ANOVA, group
averaging, submodular coverage, and optimal transport—is established. The
novel claim is their schema-specific composition, benchmark evidence, and
audit-to-action validation. This is ICLR-shaped only if the broad study is
large, frozen, and surprising.

## Abstract draft

Tabular pipelines are usually evaluated on one spelling of each dataset, even
when feature order, target numbering, opaque category identifiers, measurement
units, or within-field coordinates are arbitrary. We treat these choices as a
declared product of schema nuisances and audit the complete fitted pipeline
after aligning its outputs. For squared and Brier loss, prediction dispersion
over equivalent representatives is exactly the average proper-score risk
removed by quotient averaging and therefore requires no evaluation labels.
Functional ANOVA attributes this schema-representation risk to individual
choices and their interactions, while an explicit seed factor separates
persistent representation effects from coupling-dependent optimization
effects. Across [frozen datasets] and [pipelines], we find that classical
invariant controls close at numerical precision, but forests, boosters,
tabular neural networks, and foundation models exhibit distinct failure
profiles; equivalent coordinates can also change whether an architecture
comparison is statistically supported, even when its seed-marginal direction
does not reverse. Validation selection can amplify schema×search coupling;
pooling validation decisions over a declared schema menu exposes a measurable
stability--accuracy frontier. Finally, audit-guided factor marginalization and
row-adaptive schema cascades approximate quotient predictions more efficiently
than uniform alternatives. These results turn representation sensitivity from
a pairwise stress test into an attributable, label-free benchmark of whether
a tabular conclusion is identified by the data rather than its spelling.

## Exact estimands

Let `z ~ mu` index an evaluation schema representative and `s` algorithmic
randomness. For a fixed recipe, `p_{z,s}` is the aligned prediction vector on
unlabeled query rows. If the end-to-end rule selects on a sampled development
menu `M_m`, write `p_{z,s,m}`: the menu seed `m` is part of the pipeline, not a
fixed benchmark convenience.

`A` must be named as either a fixed recipe or an end-to-end validation
selection rule. The former is conditional on one configuration; the latter
reruns a frozen search within every `z` and includes search randomness in `s`.
Do not generalize a fixed-recipe result to an estimator family.

An exploratory 3-dataset x 3-family pilot shows why both estimands are
necessary. A four-configuration validation rule changes choice somewhere in
the orbit in 3/9 cells. All three have higher selected-pipeline schema risk
(ratios `1.26`, `1.54`, and `1.77`), but only Churn forest has a clearly
nonzero Brier improvement. Its choice changes in 37.5% of representatives,
orbit-mean Brier improves by `0.00129`, and schema risk rises from `0.00137` to
`0.00241` (paired-row ratio interval `1.69--1.85`). Thus HPO is neither a
generic repair nor a nuisance to average away: it can trade predictive loss
against quotient stability. Three selected unstable cases have minimum exact
two-sided sign `p=.25`, so the common risk increase is descriptive, not a
general claim about search spaces.

For unstable selection, write `d_z = p_{z,h(z)}-p_{z,h_0}`. Then the change is
exactly `Var(d)+2Cov(p_fixed,d)`, giving a label-free attribution of the tuning
path. In all three unstable pilots, switching dispersion is partly corrective
(negative cross-covariance cancels 40%--71%) but still increases net risk.
An fANOVA of one-hot configuration decisions localizes the discontinuity to
class-ID main effects (Adult CatBoost), feature×category (Churn forest), or
feature×class (Churn CatBoost). This is promising depth evidence for auditing
the complete pipeline and motivated the prospective split repetition below.

That conditional repetition now exists across seven unseen semantic
train/validation splits. All three baseline-selected cells remain
selection-unstable in 7/7 splits; schema risk rises in 7/7 Adult CatBoost
(magnitude/sign `p=.0156/.0156`), 6/7 Churn forest (`.0469/.125`), and 7/7
Churn CatBoost (`.0156/.0156`). No mean Brier contrast is significant
across splits. The effect is mainly coupling, not a persistent mean shift:
schema×split fractions of joint selected-path variance increase from
`39%→62%`, `16%→43%`, and `32%→54%` relative to identity-select-then-freeze.
Decision fANOVA is likewise dominated by schema×split interactions. This
supports the paper's randomness layer while still being conditional on three
cells selected from the baseline 3×3 panel.

A complementary prospective screen first follows three baseline-stable binary
cells over the same seven unseen splits. Adult forest becomes unstable on 2/7
splits—once changing configuration on 87.5% of representatives—and raises
schema risk both times. Adult and Churn HistGB remain exactly stable on 7/7.
A later decision-only screen completes the three untouched Otto cells: forest
switches on 1/7 splits, while HistGB and CatBoost remain stable on 7/7. Across
the complete original 3×3 panel, all forests ever switch (3/3), two CatBoost
cells do (2/3), and no HistGB cell does (0/3). Five of nine cells ever switch.
This supports family-structured heterogeneity rather than ubiquity, but nine
cells are still not a population prevalence estimate.

The one promoted Otto switch is a clean multiclass numerical endpoint case.
Feature order alone moves 4/16 representatives from forest config 0 to 3;
schema risk rises from `0.00321` to `0.00755` (2.35×), hard-label flips rise
from 13.8% to 16.9%, and orbit-mean Brier improves by `0.00510`. Conditional
row-bootstrap intervals exclude zero for both changes. This is one prospective
split, so it demonstrates the trade-off outside mixed binary data without
estimating its frequency.

An elementary validation-margin certificate provides a mechanistic screen.
The maximum schema shift in competitor gaps exceeds the identity winning
margin in all 24 unstable-case orbits (`delta/gamma=1.55--132`), while the
certificate holds for two stable HistGB controls (`.56` and `.39`). It fails
conservatively for one stable forest control (`5.00`), so passing certifies
stability on the measured orbit but failure does not imply a switch.

As an audit-predicted repair, choose one configuration by averaging validation
loss over the declared menu, then freeze it across representatives. On the
seven unseen splits this reduces selected-path same-split schema risk by 27%,
35%, and 36%, with magnitude/sign `p=.0156/.0156`, `.0469/.125`, and
`.0156/.0156`. Average Brier moves
slightly against the repair but is unresolved (`p=.0625`, `.344`, `.0625`),
so this is a stability--accuracy frontier rather than a free improvement. The
pilot uses sampled menus, not closed groups; it establishes menu-relative
pooling, not globally invariant HPO. Full-group averaging would be invariant
to the starting representative, while sampled menus need held-out orbit tests.

The held-out test chooses on one `2×2×2` development sub-orbit and evaluates
different feature/category levels. Risk falls on 6/7 Adult-CatBoost splits
(magnitude/sign `p=.0313/.125`), 6/7 Churn-forest splits (`.0313/.0313`), and
5/7 Churn-CatBoost splits (`.0625/.219`), reducing held-out same-split schema
risk by 28%, 35%, and 32%.
Brier contrasts remain unresolved. This is not total reproducibility: pooling
increases split-main variance for Adult CatBoost and Churn forest while
reducing schema×split interaction, effectively relocating some randomness.
The paper must report the complete partition and claim targeted quotient
repair, not overall stability dominance.

A post-frozen decision-level stress test enumerates all 36 disjoint `2-of-4`
feature by `2-of-4` category development partitions on each of the seven new
splits. The development choice matches the full-menu choice in 88%, 85%, and
74% of the 252 partition/split cases, but is held-out-validation optimal in
only 76%, 75%, and 57%. Only 4/7, 2/7, and 3/7 splits choose one configuration
under every partition. These representatives share validation rows, so this is
not independent-sample confirmation. It shows that the original one-partition
risk result is promising but partition-sensitive. In the frozen study,
sampled development and evaluation menus must be independent draws from `mu`,
the menu seed must enter the product ANOVA, and no favorable partition may be
selected after seeing outcomes. Balanced folds can diagnose transfer; their
mean candidate loss equals the full-menu mean when every representative has
equal inclusion count.

Decision fANOVA over those balanced-menu choices shows why menu seed should be
explicit. Pure menu main effects are modest (3%, 5%, 10% of decision variance),
but menu×validation-split interactions contribute 28%, 44%, and 51%; split
main effects contribute the remaining 69%, 51%, and 39%. This is conditional
on the same three selected cases and overlapping balanced subsets, not a
prevalence estimate. It nevertheless identifies a previously hidden
randomness axis in the complete tuning path.

Propagating every menu choice through its uniform full-data refit closes the
output-level loop. After averaging over the 36 balanced menus within each
split, held-out-representative schema risk falls on 7/7 splits in all three
cases (magnitude/binomial-sign `p=.015625/.015625` each), by 29.2%, 37.5%, and
34.7% in the grand menu average. The reduction occurs in 228/252, 180/252, and
198/252 individual menu/split cases, so it is not a per-menu guarantee. Mean
Brier moves against pooling in 6/7, 5/7, and 5/7 splits; none passes both exact
tests. Menu-involving terms account for 20%, 27%, and 40% of joint
schema×split×menu prediction variance. This supports an audit-predicted
quotient-risk action over a declared menu distribution while exposing its
accuracy and menu-randomness costs.

The action also depends on the declared menu measure. Under adversarial
reweighting with density ratio at most two relative to uniform (the worst 18
of 36 menus receive all mass), the repair remains favorable on 7/7, 4/7, and
5/7 splits; at ratio four the counts are 6/7, 2/7, and 4/7. Mean worst-case
contrasts remain negative in all three cases, but the splitwise guarantee
weakens. Report this bounded-reweighting curve as measure sensitivity, not as
a new distributionally robust optimization result.

- Persistent schema risk: `Var_z(E_s p_{z,s})`.
- Same-seed conditional schema risk: `E_s Var_z(p_{z,s})`.
- Joint squared dispersion: `Var_{z,s}(p_{z,s})`, decomposed by fANOVA into
  schema main effects, seed main effect, and interactions.
- Sampled-menu dispersion: `Var_{z,s,m}(p_{z,s,m})`, with menu main,
  schema×menu, and schema×split×menu terms. It vanishes only when the menu is
  fixed by design, such as a complete uniformly weighted finite group.
- Quotient score: `Theta_a = E_{z,s} R(p_{a,z,s})`.
- Claim contrast: `Delta_ab = E_z E_s[R(p_{a,z,s})-R(p_{b,z,s})]`, accompanied
  by the supported representative range and a frozen ROPE.
- Finite-set robustness companion: schema radius squared, the maximum schema
  variance over all representative weights, equivalently the minimum
  enclosing prediction-ball radius squared.

For Brier/squared loss,

`E_z R(p_z) - R(E_z p_z) = E_z ||p_z-E_z p_z||^2`.

For log loss, use the normalized geometric mean and reverse-KL ambiguity as a
separate scalar audit. Do not apply the Euclidean fANOVA decomposition to it.

## Admissibility contract

The benchmark needs visible strata:

| Tier | Examples | Permitted claim |
| --- | --- | --- |
| exact group symmetry | feature/row permutations; aligned target-ID permutations; bijections of opaque nominal IDs | strongest zero-risk invariance test and true group symmetrization |
| same field-function space | invertible cumulative, local, spectral, and whitened bases within one field | coordinate-dependent inductive-bias audit; quotient requires a raw-field renderer |
| semantics-backed chart | physical unit/origin transformations | valid only with explicit metadata and reported separately |

Exclude feature deletion, lossy rounding, semantic category strings replaced
by opaque IDs, arbitrary nonlinear warps, and global rotations that mix field
semantics from the primary nuisance product. They can be stress tests, not
equivalent-schema evidence.

## Evidence already in hand

### Measurement and mechanisms

- TabPFN v2.5: exact Brier identities and three-factor ANOVA on Breast Cancer,
  Wine, and Adult; its default ensemble reduces but does not eliminate risk.
- Seven conventional Adult pipelines: logistic regression and LightGBM close
  at numerical precision, while CatBoost, random forest, HistGB, XGBoost, and
  an MLP have distinct feature/category/class/interaction profiles.
- Across the saved Adult/Churn/Otto conventional cells, 12/17 are material;
  quotient averaging removes a median 0.53% of mean Brier (range 0.21–1.13%)
  and hard predictions change on 2.3–11.6% of rows. Across five neural chart
  grids, same-seed schema tax is 0.63–8.20% of mean proper loss.
- HistGB mechanism: binary target relabeling interacts with the fixed-side
  handling of rare categorical levels; a minimal reproducer closes when the
  rare level crosses the support threshold.
- Five-chart neural grids: Adult MLP (32 seeds), Adult ResNet (16), Diamond
  MLP/ResNet (16), and Black Friday MLP (16). Black Friday uses 100,000 train
  rows and has conditional chart risk `0.004567`, 72.3% of joint prediction
  variance.
- Persistent chart risk is only 10–21% of same-seed conditional risk across
  the four many-seed grids, and chart×seed accounts for 54–65% of joint
  variance across five neural grids, making the coupling qualification
  empirically essential.

### Scientific-claim consequence

- Adult: MLP beats ResNet under all five charts; the quotient Brier contrast is
  `-0.001479` with interval `[-0.001878,-0.001080]`. Sensitivity does not imply
  claim failure.
- Diamond: only cumulative coding supports a detectable ResNet win; the
  quotient interval crosses zero and representative point differences change
  direction.
- Frozen Day-3 reuse: the point-estimate MLP/ResNet winner changes between two
  equivalent bases on 7/25 datasets at three seeds. A conditionally prospective
  13-seed extension of all four cases crossing both sides of the 0.1% ROPE
  finds zero persistent opposite-chart mean rankings. HELOC and Polish-2 have
  stable quotient winners; Churn and Compustat remain uncertain. Same-seed
  chart winner changes still occur in 30.8% of the new paired cells. There is
  no confirmed broad architecture reversal; the surviving signal is
  chart×seed interaction and representation-dependent detection.

### Audit-to-action consequence

- OrbitCover's corrected hybrid frontier chooses which factors to
  Rao--Blackwellize before sampling their complement. On a label-free Churn
  500/500 query-row split, all 18 choices across six pipelines and budgets
  2/4/8 match the evaluation-row oracle; this is an internal pilot, not the
  required held-out dataset/model gate.
- Covariant closure: chart risk reaches numerical precision under transported
  whitening plus SGD on Adult/Diamond; field-VectorAdam removes 99.97% of
  Black Friday's conditional chart risk while retaining raw-member MSE.
- OrbitCascade: compared with row-independent escalation at matched realized
  cost, residual to the five-chart centroid falls by 54–88% on Adult, 19–26%
  on Black Friday, 40–43% on Diamond MLP, and 51–61% on Diamond ResNet. Every
  held-out two-way interval is positive.
- Schema radius is only 1.04–1.22 times uniform chart risk across split halves,
  so the declared uniform distribution is not hiding one catastrophic chart
  in the current cases.

## Novelty scorecard

Scores are out of five and assume the proposed frozen study, not only the
current pilots.

| Component | Novelty | Value | Evidence | Risk | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| discovery of task-irrelevant tabular sensitivity | 1.0 | 4.0 | 4.5 | 5.0 | explicitly concede to PNAS Nexus |
| broad preprocessing robustness card | 1.0 | 3.5 | 3.0 | 5.0 | explicitly concede to PREF |
| aligned label-free schema proper-risk object | 3.0 | 4.0 | 4.5 | 2.0 | keep; underlying identity is established |
| product-factor and schema×seed attribution | 3.5 | 4.5 | 4.0 | 2.5 | central technical/empirical claim |
| selection-rule quotient path and schema×search attribution | 3.5 | 4.5 | 4.0 | 3.0 | central depth result; full 3×3 pilot is family-structured, broad prevalence still unknown |
| schema-identifiable model comparisons | 3.0 | 4.5 | 2.5 | 3.5 | central consequence, but the selected broad reversal failed 13-seed confirmation |
| OrbitCover factor allocation | 1.5 | 4.0 | 3.0 | 3.5 | operational validity only; budgeted Rao--Blackwellization and ANOVA-guided preintegration are prior art |
| OrbitCascade row allocation | 2.0 | 3.5 | 4.0 | 3.0 | supporting experiment; adaptive TTA is prior art |
| chart-covariant field training | 2.5 | 4.0 | 4.0 | 4.0 | causal closure inside primary; conditional standalone |
| multi-marginal coupling-free estimator | 3.5 | 4.0 | 1.5 | 5.0 | do not headline; empirical OT bias is unresolved |

Overall current readiness: **3.5/5, ICLR-plausible but not submission-ready**.
It becomes a 4/5 paper if a frozen 12–15 dataset audit establishes large,
heterogeneous conventional-pipeline effects, estimates selection-path
prevalence on all predeclared cells, and if an audit-guided action predicts a
matched-compute result on held-out dataset/model cases.

### Exact novelty boundary

| Work | Branch universe | Primary object | Simultaneous interactions | Randomness treatment | OrbitANOVA delta |
| --- | --- | --- | --- | --- | --- |
| PNAS Nexus 2026 | one task-irrelevant variation at a time, including names/format/rounding | relative label-dependent error change | no product attribution | LLM reruns; conventional synthetic comparison | aligned proper-risk value, product attribution, schema×seed |
| PREF 2026 | broad preprocessing choices, some information-changing | absolute validation/test metric deltas | mainly single-knob sensitivity/volatility | repeated pipelines, not a schema-coupling estimand | equivalent-schema-only quotient and aligned predictions |
| ML multiverse | plausible analysis branches | distribution of scientific conclusions | branch analysis | workflow dependent | branches are required to denote the same semantic task |
| metamorphic testing | hand-specified semantic relations | pass/fail or pairwise output inconsistency | generally pairwise | not the target | proper-risk total, interactions, and action validation |
| [HPO overtuning](https://proceedings.mlr.press/v293/schneider25a.html) / [stable tuning](https://www.jmlr.org/papers/v14/sun13b.html) | validation noise or resampled datasets | test regret or selected-variable stability | search-space factors, not schema quotient | resampling instability | equivalent-schema selection path, aligned predictor switch decomposition |
| [SmoothDARTS](https://proceedings.mlr.press/v119/chen20f.html) | perturbations of architecture weights/search landscape | architecture/search stability | not schema-product attribution | search trajectory | discrete complete-pipeline equivariance audit, not a new generic stabilization method |

The Brier/Bregman identity, fANOVA, and Rao--Blackwellization are prior art.
The contribution lives or dies on the schema-specific estimand, admissibility
contract, cross-pipeline empirical map, and evidence that the audit predicts a
repair.

## Frozen experiment plan

### E0: invariance and estimator integrity

- Exact linear/logistic, canonical one-hot, and constant-predictor controls.
- Output-alignment tests for every class/category permutation.
- Brier/squared identities, log-loss ambiguity, fANOVA reconstruction, schema
  radius duality, and pick-freeze calibration.
- Declare representative distributions, renderer, random-seed coupling, and
  ROPE before broad outcomes.

### E1: five-dataset gate

Use five mixed classification/regression datasets with nominal, ordinal, and
continuous fields. Include one small, one medium, and Black Friday-scale case.

Models: logistic/ridge, random forest, HistGB, LightGBM, XGBoost, CatBoost,
MLP, ResNet, TabM, TabPFN v2.5, and one current non-TabPFN TFM if available.

Screen total risk with 16–32 iid schema representatives. For stochastic
learners use at least four seeds; promote material cases to eight or sixteen.
Stop numerical-zero controls early. Retain every aligned row prediction.

Pass only if:

1. at least three non-LLM model families show persistent Tier-1 or Tier-2 risk
   beyond a frozen practical threshold on at least two datasets;
2. at least two pipelines have interaction profiles that change the best
   factor-level action;
3. effects survive production metadata and correct output alignment;
4. at least one model comparison becomes non-identifiable and one sensitive
   comparison remains identifiable.

### E2: frozen broad audit

- 12–15 datasets from at least three sources and all three task types.
- Exact-group core reported separately from field-chart extensions.
- Dataset-first aggregation and dataset bootstrap; never treat rows or schema
  members as independent scientific replicates.
- Persistent, same-seed, seed-only, joint, hard-flip, schema-radius, and
  representative-claim endpoints.
- Compare model-family factor profiles, not just one grand mean.
- Recheck shortlisted ranking conclusions with 8–16 paired seeds.

### E3: prospective action test

Freeze factor choices from development-only profiles. On held-out dataset/model
pairs compare:

- OrbitCover hybrid: exactly marginalize a selected factor subset, then average
  independent draws of its complement within the remaining budget;
- iid schema sampling;
- ordinary seed/checkpoint ensembling;
- each library's default preprocessing/permutation ensemble;
- the best single fixed representative;
- a training-time or native invariance repair where available.

Match total fits, stored models, inference passes, wall time, and memory as
separate resources. For non-group chart menus, ensure deployment starts from
the semantic raw-field renderer.

Primary action endpoint: residual schema risk. Secondary endpoints: quotient
proper score, task metric, calibration, and latency. Do not promise improvement
over a lucky reference member.

## Paper figures

1. **Schema quotient and exact risk identity.** One semantic table branches
   into aligned representatives, predictions, centroid, and factor ANOVA.
2. **Pipeline × nuisance risk atlas.** Heatmap of persistent main/interaction
   components with invariant controls and seed reference columns.
3. **The tuning path is part of the pipeline.** Configuration-decision fANOVA,
   exact switch/covariance decomposition, and schema×split×menu amplification
   over prospective validation splits and development-menu draws.
4. **Audit predicts targeted action.** Predeclared menu-distribution pooled-
   selection frontier plus held-out OrbitCover and covariant/
   OrbitCascade closures.

Representative and quotient architecture/model contrasts move to a secondary
figure or appendix unless the frozen study confirms a robust persistent case.

## Reviewer attack matrix

| Attack | Required response/evidence |
| --- | --- |
| “The transformations are not really equivalent.” | admissibility tiers, field metadata, exact output alignment, same-function-space proofs, and separate reporting |
| “This is just PNAS Nexus for more models.” | conventional real-data contradiction; exact label-free risk; interactions; randomness coupling; quotient actions |
| “This is PREF with ANOVA.” | restrict to equivalence-preserving actions; aligned row predictions; exact proper-risk value; simultaneous interactions; zero invariance target |
| “Bregman diversity and fANOVA are old.” | concede both; claim the schema-specific estimand, benchmark, and consequences |
| “The result depends on arbitrary schema weights.” | declare `mu`, show weight sensitivity, and report schema radius as a companion |
| “Same integer seeds are arbitrary.” | report persistent and conditional endpoints; state the coupling; do not call same-seed risk intrinsic |
| “A constant predictor wins the robustness metric.” | always pair schema risk with centroid/member fit; explicitly state it is not total quality |
| “Averaging is generic ensembling.” | compare at equal compute with seed/checkpoint ensembles and optimize removal of a declared nuisance component |
| “The benchmark is computationally infeasible.” | sequential total-risk screen, early stopping of zero controls, pick-freeze only for material cases, decision-regret stopping |
| “The ranking changes are seed noise.” | paired 8–16 seed confirmation, frozen ROPE, dataset as replication unit, representative and quotient intervals |
| “HPO instability and stable model selection are old.” | concede overtuning/stability selection; distinguish an equivalent-schema quotient, output-aligned switch risk, decision fANOVA, schema×search partition, and held-out-nuisance repair |
| “The adaptive method already exists.” | present OrbitCascade only as a schema-quotient consequence; do not claim adaptive TTA novelty |
| “The optimizer repair sacrifices useful axis bias.” | report performance and stability jointly; retain raw centroid comparator and random-rotation negative control |

## Kill criteria

Abandon the ICLR framing if any condition holds after E1/E2:

1. direct prior work is found with the same equivalent-schema product,
   aligned proper-risk gap, and interaction attribution;
2. conventional pipeline effects vanish with correct field metadata;
3. persistent risk is negligible after separating same-seed coupling;
4. broad effects reduce to TabPFN feature order or one library bug;
5. representative model conclusions almost always agree under the frozen
   ROPE;
6. audit-guided actions do not beat iid schema sampling or ordinary seed
   ensembling at matched resource cost;
7. the broad result requires mixing exact symmetries with transformations that
   change information or semantics.

## Conditional second idea gate

Chart-Covariant Field Training becomes a standalone method paper only if all
of the following hold prospectively:

1. a frozen synthetic suite shows matching declared topology helps smooth/
   ordinal/cyclic fields, while nominal fields are constrained to the
   isotropic family rather than selected among candidate graphs;
2. the field metric and optimizer close chart risk by at least 95% across at
   least three neural architectures and ten datasets;
3. single-model proper loss is noninferior to tuned raw-coordinate training
   under a frozen ROPE;
4. it beats sample whitening plus ordinary SGD and field-VectorAdam ablations
   in either performance, compute, or the selective-semantic trade-off;
5. the contribution is demonstrably beyond generic reparameterization-
   invariant optimization.

The fixed-strength experiment passes a controlled semantic-selectivity check.
In a 27-cell Bayes-prior suite, the matched path/ring metric beats a permuted
path in all 18 ordinal/cyclic cells; it beats isotropic in 6/9 cells for each
topology, while isotropic beats every fixed false topology in all nine nominal
cells. A post-frozen stress test then lets each topology family tune stiffness
over `{0,1,4,16}`. Correct path/ring still beats isotropic in all 18 structured
cells, but a tuned false topology significantly beats isotropic in 8/9 nominal
cells for path, 8/9 for ring, and 9/9 for a permuted path. The false families
choose nonzero stiffness about 37% of the time. Thus validation can exploit a
realized but semantically arbitrary adjacency; it does not identify topology.
This is a useful failure, not a passed standalone gate: field type must be
metadata, and only stiffness *within* that family may be calibrated. The
direction still fails the real-data, multi-architecture, and competitive-
performance bars. Treat it as a strong intervention in the primary paper
unless those gates clear.

## Closest work that must appear in the introduction

- [Liu, Yang, and Adomavicius, “Robustness is important,” PNAS Nexus 2026](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)
- [PREF, “Preprocessing Robustness in Heterogeneous Tabular Learning,” 2026](https://openreview.net/pdf?id=1JhhSxdBS1)
- [EquiTabPFN, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html)
- [Biloš et al., mechanistic permutation invariance in TFMs, 2026](https://arxiv.org/abs/2605.21288)
- [Bell et al., ML multiverse analysis, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html)
- [Gruber and Buettner, Bregman information, AISTATS 2023](https://proceedings.mlr.press/v206/gruber23a.html)
- [Wood et al., ensemble diversity, JMLR 2023](https://www.jmlr.org/papers/v24/23-0041.html)
- [Lengerich et al., functional ANOVA, AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html)
- [Xie et al., metamorphic classifier testing, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144/)
- [Liu et al., budgeted Rao--Blackwellization versus minibatching, ICML 2019](https://proceedings.mlr.press/v97/liu19c.html)
- [Liu and Owen, ANOVA/Sobol-guided preintegration, 2023](https://epubs.siam.org/doi/10.1137/22M1479129)

## Bottom line

OrbitANOVA is the only current direction that combines credible novelty,
strong existing evidence, and a feasible path to ICLR by the deadline. The
paper should be sold as a new standard for whether tabular predictions and
comparisons are identified by the semantic dataset—not as the discovery that
representations matter, a new ANOVA theorem, or a universal accuracy method.
