# OrbitANOVA broad-audit protocol draft

Status: draft for human review; do not call preregistered until dataset files,
versions, transformations, thresholds, and hashes are frozen.

## Primary question

For complete tabular learning pipelines, how much aligned prediction risk is
caused by schema choices declared arbitrary by the task, which choices and
interactions cause it, and does the resulting scientific/model comparison
survive averaging over those choices?

## Candidate panel

Freeze 12–15 datasets before broad model outcomes. A defensible candidate set
from the existing harness is:

- binary mixed-schema: Adult, Australian Credit Approval, Bank Marketing,
  Churn, Credit Card Default, German Credit, HELOC;
- multiclass: Otto, Covtype, Helena or Jannis;
- regression: Black Friday, Diamond, California Housing, House Sales,
  FREM-TPL claim count.

Require at least three source families, at least two datasets per task type,
and a size range from below 5,000 to at least 100,000 training rows. Preserve
official/chronological splits. Dataset eligibility depends on field metadata,
not observed representation sensitivity.

## Pipeline panel

Core conventional controls and families:

1. logistic/ridge with canonical one-hot and standardized numeric fields;
2. ordinal-code random forest;
3. native HistGradientBoosting;
4. LightGBM;
5. XGBoost;
6. CatBoost;
7. MLP;
8. ResNet;
9. TabM or FT-Transformer;
10. TabPFN v2.5;
11. one current non-TabPFN tabular foundation model if licensed and available.

Record exact library versions, deterministic flags, metadata interfaces,
preprocessing, early stopping, and internal ensemble settings. The full
pipeline—not only the estimator class—is the benchmark unit.

Predeclare two distinct estimands and never call the first a model-family
property:

1. **fixed-recipe orbit:** one frozen configuration (including validation
   stopping) is rerun across schema representatives; this is the scalable core;
2. **selection-rule orbit:** a frozen validation-only hyperparameter search is
   rerun inside every representative, with search RNG included in `s`; run this
   smaller end-to-end substudy on at least three datasets and three sensitive
   families.

No configuration may be selected from test outcomes or from the full schema
grid. Report tuning fits separately from training fits. If a default or tuned
configuration was chosen on one reference spelling, label that selection path
explicitly; it is part of the pipeline and may itself be representation
dependent.

For the selection-rule substudy, additionally report the selected-
configuration distribution and entropy, the fraction of representatives that
depart from the identity-chart choice, and paired contrasts of schema risk and
proper loss against select-on-identity-then-freeze. Treat these as two
endpoints: a validation rule can improve proper loss while worsening quotient
stability. In the exploratory 3-dataset x 3-family panel, selection changes in
3/9 cells and schema risk rises in all three, but only Churn forest has a
clearly nonzero loss improvement (`-0.00129` orbit-mean Brier and `1.77x`
schema risk). The three-cell direction is underpowered (`p=.25` minimum exact
two-sided sign test), so freeze the larger substudy rather than promoting the
pilot pattern; neither endpoint may stand in for the other.

When selection changes, set `d_z=p_{z,h(z)}-p_{z,h_0}` and report the exact
label-free switch decomposition
`Delta SR = Var_z(d_z)+2Cov_z(p_{z,h_0},d_z)`. Also fANOVA-decompose the one-hot
configuration decisions. This distinguishes direct switching dispersion from
corrective or reinforcing covariance with the frozen path, and identifies
which nuisance factors make the tuning rule discontinuous.

Record a simple selection-margin certificate. For candidate set `H`, identity
winner `h_0`, and validation loss `L_z(h)`, define

`gamma = min_{h != h_0} [L_e(h)-L_e(h_0)]`

and

`delta = max_{z,h != h_0} |[L_z(h)-L_z(h_0)]-[L_e(h)-L_e(h_0)]|`.

If `delta < gamma`, then `h(z)=h_0` throughout the measured orbit. Report the
ratio and the exact minimum orbit gap even when the sufficient certificate
fails. This is elementary argmin stability, not a new theorem, but it predicts
whether small schema perturbations can trigger a discontinuous tuning path.

For a finite group `G`, uniform full-orbit pooling satisfies
`|G|^-1 sum_g L_{g g_0}(h)=|G|^-1 sum_g L_g(h)` for every starting element
`g_0`; with a fixed tie rule its chosen configuration is therefore invariant
to the starting spelling. This reindexing argument does not apply to sampled
menus, nonuniform weights, or non-group chart families.

Cross each selection-rule cell with at least twelve frozen semantic
train/validation split or search seeds. When the development orbit is sampled,
also predeclare independent menu seeds `m`; the fitted output is then
`p_{z,s,m}`. Partition the joint probability ANOVA into persistent schema,
search/split main, menu main, schema×search, schema×menu, and higher-order
components, and apply the same partition to one-hot configuration decisions.
For a complete uniformly weighted finite group the menu is deterministic and
`m` disappears. Test paired
selected-versus-identity-frozen contrasts over prospective split seeds with a
magnitude sign-flip test and the weaker binomial sign test. State their null
assumptions and show both when they disagree. Query-row bootstrap intervals are
conditional on fitted models and cannot replace this search-randomness layer.

Add a predeclared repair comparison: pool each candidate's validation loss over
a development schema sub-orbit, select one configuration, and freeze it across
evaluation representatives. Evaluate on disjoint nuisance levels whenever the
orbit is sampled. Compare identity-only selection, per-representative
selection, pooled selection, and (exploratorily) validation loss plus a frozen
schema-risk penalty. Report the proper-loss versus schema-risk Pareto frontier
and all tuning costs. Also report split/search main variance: pooled selection
can reduce schema×search coupling while increasing coherent search-main
variation, so quotient repair is not total-stability dominance. Only a uniform
complete group orbit is invariant to the
starting spelling by construction; do not give that claim to a sampled finite
menu.

For inference over several development menus, compute each held-out proper-
loss/schema-risk contrast first, average those contrasts within the same split
seed, and apply paired tests to split-level (and ultimately dataset-level)
averages. Overlapping menus are not independent replication. Cache
deterministic full refits by `(evaluation representative, configuration)`;
the validation split changes the chosen index, not a full-data refit with the
same model seed.

For a sampled menu, predeclare multiple balanced nuisance folds before fitting.
For each fold, select on its development product and score the decision on the
complement. Average candidate losses across the predeclared folds, then use a
fixed tie break for the final test refit. For the complete set of equal-sized
subsets, this average is exactly the full-menu mean because every
representative has equal inclusion probability; the folds diagnose transfer
rather than manufacture a different selector. Report choice agreement and
held-out validation regret across folds. Better still, draw independent
development and evaluation menus from the declared `mu` and replicate menu
seeds. Do not select the fold:
an exploratory 36-partition enumeration yields only 88%/85%/74% agreement with
the full-menu decision and 76%/75%/57% complement-optimal choices in the three
current cases. These fold diagnostics reuse validation rows and therefore do
not replace split-level, menu-seed, or test-level inference.

Because `mu` is declared rather than discovered, add a measure-sensitivity
curve for promoted actions. On a finite uniform menu, maximize the mean
contrast over weights satisfying `0 <= w_i <= kappa/|M|` and `sum_i w_i=1`
for predeclared `kappa` values (for divisible integer `kappa`, this is the mean
of the worst `|M|/kappa` contrasts). This is a bounded density-ratio stress
test, not a replacement estimand or a novel robust-optimization method.

## Schema factors

### Tier 1: exact group core

- feature positions: identity plus seven frozen permutations;
- opaque nominal IDs: identity plus seven independent within-field
  permutations where such fields exist;
- target IDs: all binary swaps; for multiclass, identity plus seven frozen
  permutations;
- unordered context row positions: identity plus seven frozen permutations,
  evaluated only for in-context or explicitly order-sensitive pipelines.

Every action is applied jointly to train/context and validation/test/query
representations. Feature metadata follows feature permutations. Probabilities
are aligned to reference target coordinates before any metric.

### Tier 2: field-function charts

For datasets with ordinal or binned-continuous fields, render a separately
reported five-chart set: local, cumulative, standardized cumulative,
path-spectral, and sample-whitened. Verify rank, reconstruction, and fieldwise
span equality below a frozen numerical tolerance. Access to the raw semantic
field renderer is required.

### Tier 3: semantics-backed units

Use only fields with unambiguous measurement metadata. Freeze bounded scale
and origin distributions. Do not merge these results with Tier 1.

## Sampling design

### Screen

- deterministic pipelines: 32 iid product representatives per eligible
  dataset/pipeline;
- randomized pipelines: 16 representatives × 4 seeds using a declared
  same-seed coupling;
- exact/near-zero controls stop after a frozen sequential upper bound;
- retain aligned row predictions for every member.

Estimate total schema risk using unbiased Hilbert sample variance. Bootstrap
evaluation rows for the conditional query-population uncertainty and orbit
members for Monte Carlo uncertainty; preserve their two axes rather than
pooling them.

### Attribution

Promote only material screen cases. Use either a balanced factorial for small
products or pick-freeze with `N=16`, then increase `N` until the selected
OrbitCover action's regret/interval meets the frozen tolerance. Scientific
factor profiles retain uncertainty even if the action has stabilized.

### Randomness

For promoted stochastic cases use 8–16 paired seeds. Report:

- `Var_z(E_s p)` with a cross-seed U-statistic or split estimator;
- `E_s Var_z(p|s)` under the explicit common-random-number coupling;
- seed main and schema×seed components;
- a same-reference seed orbit.

Do not use naive high-dimensional empirical Wasserstein distance as a primary
endpoint at these sample sizes.

## Endpoints

### Primary label-free endpoints

- persistent and same-seed conditional schema risk;
- fANOVA main and interaction fractions;
- root schema risk `sqrt(SR)` in probability-vector or standardized-target
  units;
- fraction of mean member proper loss removed by exact quotient averaging,
  `SR / mean_member_loss`;
- hard-prediction flip fraction and row-risk quantiles;
- schema radius/uniform-risk ratio for finite representative sets.

The relative-loss fraction can be unstable for near-perfect predictors, while
raw risk is not comparable across regression target units. Therefore report
both, standardize regression targets using training statistics, and aggregate
dataset-level summaries rather than raw rows or cells.

### Claim endpoints

For frozen model pairs and task metrics:

- quotient contrast across schema and seeds;
- representative minimum/maximum or central quantiles;
- paired uncertainty;
- classification under a predeclared normalized ROPE.

A comparison is schema-identifiable only if the quotient contrast and all
supported representative contrasts lie beyond the same ROPE boundary.

Do not call a representative contrast nonzero from a three-seed percentile
bootstrap: its minimum attainable exact two-sided paired sign-flip `p` is
`0.25`. Shortlisted claim comparisons require at least 8--16 paired seeds;
report a paired randomization/sign-flip test alongside the interval, and keep
the seed-marginal contrast distinct from the fraction of same-seed winner
changes. The Compustat conditional confirmation is the motivating negative
control: the selected three-seed opposite ranking disappeared on 13 new seeds.

### Action endpoints

Primary: residual label-free schema risk at matched resource cost.

OrbitCover actions use a hybrid conditional-Monte-Carlo frontier. For a
selected factor subset `J` with exact enumeration cost `c_J` and budget `B`,
average `m=floor(B/c_J)` independent draws of the complementary factors after
exactly marginalizing `J`. Its expected residual is `SR(Q_J p)/m`; `J=empty`
is the iid-schema baseline. Freeze the chosen `J` on development cases, and
report realized cost plus any unused remainder. This is Rao--Blackwellization
guided by the audit, not a claimed new Monte Carlo identity; only closed action
orbits should be called group symmetrization.

Secondary: quotient proper score, official task metric, calibration, wall
time, peak memory, stored-model count, and inference passes. Training fits and
deployment passes are distinct resources and must not be conflated.

## Aggregate inference

- Dataset is the scientific replication unit.
- Average seeds and schema samples within dataset/pipeline before any
  cross-dataset mean.
- Use dataset bootstrap intervals and report medians, wins, full distributions,
  and leave-one-dataset-out sensitivity.
- Stratify by task, sample size, metadata type, and model family only when the
  stratum was frozen or is labeled exploratory.
- Control multiplicity for confirmatory pairwise claims; do not apply a
  multiple-testing correction to descriptive heatmaps and then call them
  confirmatory.

## Compute budget estimate

For 15 datasets:

| Stage | Approximate work |
| --- | ---: |
| six deterministic pipelines × 32 representatives | 2,880 fits |
| three stochastic neural pipelines × 16 reps × 4 seeds | 2,880 fits |
| two TFM pipelines × 32 representatives | 960 contextual passes |
| 20 promoted cases × five-factor pick-freeze at N=16 | 2,240 fits/passes |
| 10 shortlisted comparisons × two models × 12 additional paired seeds × two reps | 480 fits |
| 9 selection-rule cells × 16 schema reps × 12 split seeds × (6 candidate + 3 final paths) | 15,552 fits |

The ceiling is about 24,992 fits/passes before early stopping or reuse when
identity-frozen and pooled configurations coincide. The 16-representative
selection menu must contain frozen balanced development and held-out products;
partition nuisance levels before the pooled-selection repair.
Existing Day-3 runs
show that the two idle H100 NVL GPUs and capped datasets make the neural part
feasible. Tree CPU parallelism, prediction storage, and TFM licensing/cache
are separate bottlenecks to audit before freeze.

## Gates

### Measurement gate

Pass if three non-LLM model families show persistent material risk on at least
two datasets, with at least two distinct dominant factor profiles and at least
one reproducible interaction-driven action change.

### Claim gate

Pass if at least one frozen model comparison is non-identifiable and one
sensitive comparison remains identifiable after 8–16 seed confirmation. The
point is diagnosis, not maximizing reversal count.

### Action gate

Pass if development-selected OrbitCover beats iid schema sampling and ordinary
seed/checkpoint ensembling in residual schema risk at matched resources on
held-out dataset/model cases, without a quotient proper-loss degradation beyond
the frozen ROPE.

### Paper gate

Proceed only if the exact-group core alone supports a meaningful result.
Tier-2 charts may deepen the mechanism and interventions but may not rescue a
failed Tier-1 benchmark by being pooled into it.

## Integrity checklist before freeze

- hash all dataset arrays/splits and transformation lists;
- serialize raw semantic field metadata and output alignments;
- test inverse/permutation closure on synthetic predictions;
- pin libraries, model checkpoints, and deterministic settings;
- freeze thresholds, ROPEs, model pairs, aggregation, and stopping rules;
- verify disk budget for all aligned row predictions;
- create a failure table and never silently drop cells;
- reserve an untouched action-test subset before inspecting broad actions.
