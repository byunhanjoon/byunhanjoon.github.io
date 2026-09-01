# AGENT.md — FINAL ICLR CLOSURE EXPERIMENTS

## Mission

Finish the OrbitCover project.

Do **not** stop after implementing experiments.

Do **not** stop after smoke tests.

Do **not** stop after generating raw tensors.

Do **not** stop because one model fails.

Do **not** ask for confirmation unless execution is literally impossible because required credentials/data/hardware are unavailable.

Troubleshoot failures, resume interrupted runs, rerun corrupted cells, complete all mandatory experiments, analyze them, regenerate figures/tables, and finally write:

```text
results.md
```

The final `results.md` is the primary deliverable.

The goal is to resolve the remaining reviewer-critical questions raised by the current evidence.

---

# 0. Start by reading existing evidence

Before changing code, read:

```text
results.md
FINAL_COMPLETION_PROTOCOL.md
COMPLETION_PROTOCOL_DEVIATIONS.md
EXPERIMENT_LEDGER.md
INTEGRITY_AUDIT.md
```

and all existing summary JSON/CSV files relevant to:

```text
completion neural panel
matched-function controls
fANOVA decomposition
strength-2 / strength-3
SRSWOR
packing
ranking/model selection
TabPFN
CatBoost / XGBoost / LightGBM / HistGradientBoosting
```

Do not rerun completed experiments unless required for one of the new comparisons.

Preserve all prior evidence grades.

Do not retroactively relabel exploratory/adaptive results as prospective.

---

# 1. Current scientific state

Treat the following as established background to be tested/extended, not assumptions that must remain true:

1. All 144 modern-neural dataset×split×model cells showed material finite nuisance variance.

2. Strength-2 beat IID-16 in 144/144 cells.

3. Strength-2 beat LHS, Sobol, and strength-1 broadly.

4. Strength-2 versus SRSWOR was positive in aggregate but heterogeneous:
   - 101/144 cell wins;
   - positive clustered mean reduction;
   - only 8/12 source means positive.

5. Matched initialization removed approximately 98% of pooled ordinary schema variance:
   - MLP essentially closes;
   - ResNet essentially closes;
   - FT-Transformer retains a meaningful residual;
   - TabM retains a small residual.

6. Validation ranking improves strongly, but held-out model-selection gains are small because validation/test partition shift dominates.

The remaining paper-level question is therefore no longer:

> Do arbitrary equivalent representations change neural optimization?

The stronger final question is:

> Can semantic symmetries be used as structured randomization dimensions to estimate the symmetrized prediction of a randomized learning pipeline more efficiently than ordinary independent retraining?

---

# 2. Final experiments

There are THREE mandatory closure experiments.

All three must finish.

```text
A. Independent canonical-seed ensemble comparison
B. Realistic-scale / convergence experiment
C. Interaction-spectrum prediction of OrbitCover vs SRSWOR
```

A fourth experiment is highly preferred and should be completed unless computationally impossible:

```text
D. Coupling-mechanism decomposition
```

Do not stop before A–C are complete and `results.md` is written.

---

# 3. Freeze protocol before examining outcomes

Create:

```text
experiments/final_closure/FINAL_CLOSURE_PROTOCOL.md
experiments/final_closure/final_closure_config.json
```

The protocol must contain:

- dataset choices;
- model choices;
- split seeds;
- training budgets;
- convergence criteria;
- RNG definitions;
- nuisance factors;
- canonical-seed baseline definitions;
- OrbitCover definitions;
- SRSWOR definitions;
- metrics;
- primary analyses;
- success/failure criteria;
- statistical aggregation;
- planned figures;
- planned tables.

Hash both files.

Write the hash to:

```text
experiments/final_closure/PROTOCOL_HASH.txt
```

Do not modify the frozen protocol after looking at final outcomes.

Implementation corrections go in:

```text
experiments/final_closure/PROTOCOL_DEVIATIONS.md
```

---

# 4. EXPERIMENT A — Independent canonical-seed ensemble

## Why this experiment is mandatory

The matched-function result implies that much of measured schema variance may come from how a fixed random state is assigned to semantic coordinates.

Therefore a reviewer can argue:

> Why not simply train the canonical representation with additional independent random seeds?

This must be answered directly.

The baseline must use **genuinely independent full-pipeline randomness**, not a small reused finite seed menu.

---

# 5. Models for Experiment A

Mandatory:

```text
MLP
ResNet
FT-Transformer
TabM
```

Also include if implementation is already stable:

```text
CatBoost
XGBoost
```

Do not delay completion to add optional architectures such as SAINT or TabR.

TabPFN does not need to be forced into the seed experiment if its stochastic structure is inference-ensemble based rather than independent retraining.

Its previous external-cover experiment remains separate evidence.

---

# 6. Dataset panel for Experiment A

Use all 12 datasets from the completed modern-neural panel if computationally feasible.

Because the existing models train cheaply on the available H100s, all 12 are preferred.

Use all three established split seeds.

Thus the target panel is:

```text
12 datasets
× 3 splits
× 4 neural models
= 144 primary cells
```

Do not choose only favorable datasets.

---

# 7. Full-pipeline independent RNG definition

A "fresh independent seed" must randomize all ordinary stochastic components that would naturally change between independent training runs.

At minimum where applicable:

```text
parameter initialization
dropout RNG
minibatch permutation
dataloader worker/order RNG
stochastic preprocessing
model-specific stochastic training operations
```

Generate a master integer seed for every independent replicate and deterministically derive sub-seeds.

Example:

```text
master_seed
  -> init_seed
  -> dataloader_seed
  -> dropout_seed
  -> augmentation_seed
```

Do not reuse the previous two-seed finite product as the independent-seed baseline.

Use at least:

```text
64 independent master seeds
```

per dataset×split×model if compute allows.

Minimum acceptable reference pool:

```text
32
```

Preferred:

```text
128
```

if runtime remains manageable.

---

# 8. Methods to compare in Experiment A

For equal total model-fit budgets compare:

## A1. Canonical independent seeds

Always use the canonical schema.

Train with independently sampled full-pipeline master seeds.

Call this:

```text
CANONICAL-INDEPENDENT
```

---

## A2. IID symmetry × independent RNG

For every fit independently sample:

```text
schema action g
fresh master seed s
```

Call this:

```text
IID-JOINT
```

---

## A3. SRSWOR schema × independent RNG

Choose schema transformations without replacement where applicable.

Use a fresh independent master seed for every fit.

Call:

```text
SRS-JOINT
```

---

## A4. Strength-1 OrbitCover × independent RNG

Balanced schema design.

Fresh independent master seed per fit.

Call:

```text
OC1-INDEPENDENT
```

---

## A5. Strength-2 OrbitCover × independent RNG

Balanced strength-2 schema design.

Fresh independent master seed per fit.

Call:

```text
OC2-INDEPENDENT
```

---

## A6. Existing coupled OrbitCover

Retain the original declared finite schema×RNG product construction where the seed/order factors themselves participate in the balanced design.

Call:

```text
OC2-COUPLED
```

This distinction is critical.

It tests whether benefit arises from:

```text
balancing schema alone
```

or

```text
jointly balancing/coupling schema and training randomness.
```

---

## A7. Packed/disjoint OrbitCover

At larger budgets include the existing disjoint block construction:

```text
OC2-PACKED
```

when technically applicable.

---

# 9. Experiment A budgets

Mandatory:

```text
B = 4
8
16
32
64
```

Preferred:

```text
B = 1
2
4
8
16
32
64
128
```

The headline comparison is:

```text
B = 16
```

because most existing results use 16 fits.

---

# 10. Reference predictor for Experiment A

The reference should approximate the expectation of the randomized learning pipeline under the declared distribution.

Construct high-quality references separately for:

```text
canonical independent-seed expectation
joint schema × independent-seed expectation
finite coupled quotient where exactly available
```

Do NOT silently assume these expectations are identical.

This is itself a key scientific test.

Measure:

```text
|| Q_canonical_seed - Q_joint_schema_seed ||²
```

for every cell.

Report whether the difference is statistically/materially distinguishable from Monte Carlo error.

This tests the hypothesis:

> Schema averaging changes only the variance of estimating the same expectation.

versus:

> Schema averaging changes the target expectation itself.

---

# 11. Main Experiment A metrics

For every method and budget compute:

```text
prediction-space residual to appropriate reference
Brier/MSE of ensemble prediction
variance across repeated estimator constructions
IID-equivalent fit budget
```

Also compute:

```text
canonical-seed reference vs joint-symmetrized reference distance
```

and, where useful:

```text
accuracy
log loss
AUROC
MAE
R²
```

but keep quadratic prediction error primary.

---

# 12. Replications of sampling estimators

For every cell/method/budget, generate enough independent estimator replicates to estimate expected quotient residual reliably.

Target:

```text
>= 256 estimator replicates
```

when predictions are already cached.

If this requires retraining unnecessarily, instead cache a sufficiently large trained pool and resample from it appropriately.

Do not treat overlapping finite-pool samples as independent observations in statistical tests.

---

# 13. Experiment A primary questions

Answer all of these:

### Q1

At B=16, does `OC2-INDEPENDENT` beat `CANONICAL-INDEPENDENT`?

### Q2

At B=16, does `OC2-COUPLED` beat `CANONICAL-INDEPENDENT`?

### Q3

Does joint balancing of seed/order factors provide additional benefit beyond schema balancing with fresh independent seeds?

### Q4

Does `OC2-COUPLED` outperform `OC2-INDEPENDENT`?

### Q5

Does the canonical independent-seed expectation equal the schema-symmetrized expectation within uncertainty?

### Q6

How many canonical independent training runs are needed to match 16 OrbitCover fits?

### Q7

Are gains architecture dependent?

Pay particular attention to:

```text
MLP
ResNet
FT-Transformer
TabM
```

because matched-initialization results differ greatly.

---

# 14. Experiment A possible outcomes

## Outcome A — strongest possible

OrbitCover beats truly independent canonical-seed ensembles at equal fit budget.

Interpretation:

> Semantic symmetries provide useful structured randomization beyond ordinary independent retraining.

This is a strong ICLR result.

---

## Outcome B — still interesting

`OC2-COUPLED` beats canonical independent seeds but `OC2-INDEPENDENT` does not.

Interpretation:

> The main benefit comes from structured coupling between semantic symmetry and training randomness.

This is potentially the most mathematically interesting result.

---

## Outcome C — weaker

Canonical independent seeds match OrbitCover.

Interpretation:

> The main practical advantage of the previous experiments arose from balancing a finite stochastic nuisance menu rather than outperforming ordinary independent retraining.

Report this clearly.

It weakens the central method claim.

---

# 15. EXPERIMENT B — realistic scale and convergence

## Why this is mandatory

The existing broad panel uses:

```text
2,048 training examples
20 epochs
```

A reviewer can argue that nuisance variation is simply transient undertraining.

The final project must test whether the phenomenon and OrbitCover efficiency persist as training becomes more realistic.

---

# 16. Dataset selection for Experiment B

Use at least:

```text
6 datasets
```

with:

```text
3 classification
3 regression
```

Select them BEFORE final outcomes.

Choose them to span:

```text
small/large nuisance risk
mostly numerical/mixed categorical
low/high dimensionality
low/high main+pair interaction fraction
OrbitCover strong/weak versus SRSWOR
```

Do not select only large OrbitCover wins.

Include at least one existing source-level SRSWOR loss.

Include FT-Transformer because it had the most high-order mass.

Include MLP or TabM because they had strong low-order structure.

---

# 17. Models for Experiment B

Mandatory:

```text
MLP
ResNet
FT-Transformer
TabM
```

All four must run.

---

# 18. Training-size scaling

For each selected dataset, use nested training subsets.

Target:

```text
N = 2,048
8,192
32,768
full available training set
```

If the dataset has fewer rows:

use all feasible nested levels and clearly document them.

Do not duplicate examples to reach a larger N.

---

# 19. Optimization-budget scaling

At each feasible training size, evaluate increasingly complete training.

Use:

```text
20 epochs
50 epochs
100 epochs
200 epochs
```

plus:

```text
early stopping / convergence-trained
```

where appropriate.

The converged condition should use a generous maximum epoch budget such as:

```text
500
```

with validation-based early stopping.

Freeze patience and stopping criteria prospectively.

For models whose standard training recipe differs, use a fair architecture-appropriate convergence criterion and record it.

---

# 20. Convergence diagnostics

Record:

```text
training loss
validation loss
gradient norm where easy
epoch of best checkpoint
epoch stopped
parameter-update norm if practical
```

The key scientific question is not merely whether validation loss improves.

Measure nuisance variance versus convergence.

For each training budget compute:

```text
total nuisance variance
schema-only variance
stochastic-only variance
schema×stochastic interaction
main+pair fraction
OC2/SRS residual ratio
OC2/canonical-independent ratio
```

---

# 21. Experiment B design

It is unnecessary to run the entire original full nuisance tensor at every scale if prohibitive.

Use a fixed scientifically valid subset of nuisance actions large enough to estimate the relevant quantities.

Preferred:

```text
4 feature-order actions
4 category-ID actions where valid
2 target-ID actions for classification
>= 8 independent training master seeds
```

Use strength-2 and SRS/independent baselines at:

```text
B=16
```

and optionally:

```text
B=32
```

The emphasis is scaling behavior, not reconstructing every earlier figure.

---

# 22. Experiment B primary analyses

For each architecture fit trends of:

```text
log nuisance risk
vs
log training size
```

and:

```text
log nuisance risk
vs
training budget / convergence
```

Do not overinterpret fitted scaling exponents with only a few points.

Plot raw trajectories.

Determine whether:

### Hypothesis B1

Absolute nuisance variance shrinks with more training.

### Hypothesis B2

OrbitCover's relative estimator efficiency survives even when absolute nuisance variance shrinks.

### Hypothesis B3

High-order interaction fraction changes during optimization.

### Hypothesis B4

FT-Transformer's matched-path residual survives convergence more than MLP/ResNet.

---

# 23. Matched-function convergence subexperiment

This is mandatory on a smaller subset.

Use at least:

```text
2 datasets
× 4 models
```

with one classification and one regression dataset.

Repeat exact matched-initial-function transformations at:

```text
20 epochs
100 epochs
convergence
```

Measure:

```text
ordinary schema variance
matched-function variance
fraction removed
```

This determines whether the residual FT-Transformer/TabM effect grows, shrinks, or remains stable with training.

---

# 24. EXPERIMENT C — explain OrbitCover vs SRSWOR failures

## Why this matters

The strongest remaining baseline challenge is SRSWOR.

Strength-2 has a positive aggregate effect but loses on some dataset means.

The paper becomes stronger if those losses follow the interaction-order theory.

---

# 25. Use ALL available exact/broad cells

Do not restrict this analysis to favorable cases.

Combine all valid existing neural tensors for which both of these exist:

```text
fANOVA spectrum
strength-2 vs SRSWOR residual
```

Include new Experiment A/B cells where compatible.

---

# 26. Predictive variables

For every cell calculate:

```text
main-effect fraction
pairwise fraction
main+pair fraction
triple fraction
higher-order fraction
effective interaction order
total nuisance variance
finite population size
sampling fraction B/N
number of nuisance factors
architecture
dataset
task family
```

Define:

```text
OrbitCover gain over SRS
=
1 - residual_OC2 / residual_SRS
```

Positive means OrbitCover wins.

---

# 27. Theory-first analysis

Before fitting arbitrary predictors, test the direct expected relationships.

Primary:

```text
OrbitCover gain
vs
main+pair fANOVA fraction
```

Secondary:

```text
OrbitCover gain
vs
higher-order fraction
```

Expected signs:

```text
higher main+pair mass -> larger OC2 advantage
higher triple/higher mass -> smaller OC2 advantage
```

Use:

```text
Spearman correlation
dataset-clustered bootstrap intervals
architecture-stratified plots
```

---

# 28. Cell-level failure analysis

Create a table containing every cell where:

```text
SRSWOR beats strength-2
```

For each report:

```text
dataset
split
architecture
population size
main fraction
pair fraction
triple fraction
higher fraction
OC2 residual
SRS residual
relative loss
```

Then answer:

> Are failures concentrated in high-order interaction cells?

Do not hide exceptions.

---

# 29. Source-level failure analysis

The earlier result was:

```text
8/12 source means positive vs SRSWOR.
```

Identify all four negative source means.

For each source compare its ANOVA spectrum with the eight positive sources.

If the negative sources have systematically higher high-order mass, quantify it.

If not, say so.

---

# 30. Simple predictive model

Optionally fit a transparent model such as:

```text
gain ~ main_pair_fraction
     + sampling_fraction
     + architecture
```

or a small hierarchical regression.

Do not use a complex black-box predictor.

Use leave-one-dataset-out evaluation if making predictive claims.

The paper does not need a perfect predictor.

The goal is to test whether interaction structure explains the observed boundary.

---

# 31. Counterfactual strength comparison

Where strength-3 is available, examine cells where strength-2 loses to SRS.

Ask:

```text
Does strength-3 recover the loss?
```

especially when triple-order mass is elevated.

This directly tests:

> match design strength to interaction order.

Report both successful and unsuccessful recoveries.

---

# 32. EXPERIMENT D — coupling mechanism decomposition

Complete this unless technically infeasible.

The purpose is to identify exactly what OrbitCover balances.

Construct controlled variants where balancing is applied to:

```text
schema only
initialization seed only
dataloader/order only
schema × initialization
schema × order
initialization × order
all factors jointly
```

Use a representative subset:

```text
>= 4 datasets
× 4 models
× 3 splits if affordable
```

At equal B=16.

Measure quotient-estimation residual.

This should answer:

> Which pairwise interactions account for most of OrbitCover's advantage?

Relate the answer directly to fANOVA component mass.

---

# 33. Data reuse and computational efficiency

Reuse existing trained predictions whenever mathematically legitimate.

Do not retrain identical `(dataset, split, model, schema, RNG)` configurations.

Create a persistent registry keyed by:

```text
dataset
split
model
model config hash
schema transformation hash
master RNG seed
training size
training budget
```

Before launching any training job:

```text
check registry
```

If identical completed artifact exists:

```text
reuse it
```

If partial/corrupt:

```text
rerun it
```

---

# 34. Execution must finish

The agent must operate autonomously until completion.

If a training job crashes:

```text
inspect error
repair
resume
```

If one GPU job OOMs:

```text
reduce batch size
preserve effective training recipe where possible
rerun affected jobs
document deviation
continue
```

If a library model fails:

```text
debug
retry
```

Do not skip a mandatory architecture merely because the first attempt fails.

If a dataset has an unavoidable incompatibility:

1. document the exact reason;
2. attempt a principled fix;
3. only exclude if technically impossible;
4. never replace it with a more favorable dataset after viewing outcomes.

---

# 35. Parallel execution

Use both H100 GPUs efficiently.

Run independent cells in parallel.

Avoid oversubscribing GPU memory.

Parallelize CPU analysis separately.

Keep deterministic logs.

Use restartable manifests so an interrupted command continues incomplete cells rather than restarting the entire project.

---

# 36. Integrity tests

Add/retain tests for:

```text
schema transforms preserve semantics
fresh master seeds are actually unique
sub-seeds derive deterministically
canonical-independent baseline never changes schema
OC2-independent uses fresh independent master RNG per fit
OC2-coupled uses exactly the declared balanced product
equal-budget comparisons use equal fit counts
training-size subsets are nested
matched initialization preserves initial functions
prediction alignment is exact
fANOVA reconstruction holds
strength-2 formal balance holds
maximum-row uniqueness holds
no test leakage
```

Run all tests before final analysis.

The final report must state:

```text
tests passed / total tests
```

---

# 37. Statistical unit

Do not treat individual fit-level outcomes as independent evidence.

Primary aggregation:

```text
dataset
```

Secondary:

```text
dataset × split
```

Use:

```text
paired source-level differences
dataset-clustered bootstrap
task-balanced means
architecture-stratified results
```

Do not make a sign test the sole arbiter of a claim.

Report:

```text
effect magnitude
uncertainty
win count
heterogeneity
```

together.

---

# 38. Required headline comparisons

The following numbers MUST appear in the final report.

At B=16:

```text
OC2-COUPLED vs CANONICAL-INDEPENDENT
OC2-INDEPENDENT vs CANONICAL-INDEPENDENT
OC2-COUPLED vs OC2-INDEPENDENT
OC2-COUPLED vs SRS-JOINT
OC2-COUPLED vs IID-JOINT
```

For each give:

```text
cell wins / total
source wins / total
mean relative reduction
median relative reduction
dataset-clustered 95% interval
architecture-stratified reductions
```

---

# 39. Required convergence numbers

For every model report nuisance variance at:

```text
small / 20 epochs
small / convergence
largest feasible N / 20 epochs
largest feasible N / convergence
```

Also report:

```text
OC2/SRS ratio
```

at those points.

Answer explicitly:

> Does nuisance risk disappear with realistic optimization?

and:

> Does OrbitCover's relative efficiency disappear with realistic optimization?

---

# 40. Required mechanism numbers

Report:

```text
Spearman(main+pair fraction, OC2 gain vs SRS)
Spearman(higher-order fraction, OC2 gain vs SRS)
```

with clustered/bootstrap uncertainty.

Compare:

```text
mean high-order fraction in OC2 wins
vs
mean high-order fraction in OC2 losses
```

Also report separately for:

```text
MLP
ResNet
FT-Transformer
TabM
```

---

# 41. Figures

Generate at least these final paper-candidate figures.

## Figure 1 — independent-seed showdown

X axis:

```text
number of fits
```

Y axis:

```text
quotient-estimation residual
```

Curves:

```text
Canonical independent
IID joint
SRS joint
OC1 independent
OC2 independent
OC2 coupled
OC2 packed
```

Use dataset-balanced aggregation with uncertainty.

---

## Figure 2 — architecture comparison at B=16

For:

```text
MLP
ResNet
FT-Transformer
TabM
```

show relative error against canonical independent seeds.

---

## Figure 3 — are the expectations the same?

Plot:

```text
distance between canonical infinite-seed reference
and schema-symmetrized reference
```

per dataset/model.

Include Monte Carlo uncertainty.

---

## Figure 4 — convergence

Nuisance variance versus:

```text
training epochs / convergence
```

faceted or separately plotted by model.

---

## Figure 5 — training scale

Nuisance variance versus:

```text
training sample size
```

---

## Figure 6 — OrbitCover survives convergence?

Plot:

```text
OC2 residual / SRS residual
```

versus training progress.

Value below 1 favors OrbitCover.

---

## Figure 7 — interaction spectrum predicts gain

X:

```text
main+pair fraction
```

Y:

```text
OrbitCover gain over SRSWOR
```

Mark architecture.

---

## Figure 8 — failure cells

Show fANOVA decomposition for:

```text
strongest OC2 wins
strongest SRS wins
```

---

## Figure 9 — matched path over convergence

Ordinary vs matched-function nuisance variance at:

```text
20 epochs
100 epochs
convergence
```

---

## Figure 10 — mechanism decomposition

Equal-budget residual when balancing:

```text
none
schema
seed
order
pairs
all factors
```

---

# 42. Tables

Generate:

```text
table_A_independent_seed_comparison.csv
table_B_convergence.csv
table_C_interaction_prediction.csv
table_D_coupling_ablation.csv
table_E_final_claims.csv
```

Also produce markdown versions.

---

# 43. Final scientific decision rules

Do not manipulate these after results are visible.

## Strong support

Use:

```text
SUPPORTED
```

if all of the following broadly hold:

1. OrbitCover beats canonical independent-seed ensembles at equal fit budget in dataset-balanced aggregate.

2. The advantage survives the realistic/converged training study.

3. SRSWOR heterogeneity is meaningfully related to interaction-order structure.

4. There is no major architecture where OrbitCover systematically fails without an explainable boundary.

---

## Partial support

Use:

```text
PARTIALLY SUPPORTED
```

if:

- OrbitCover beats IID/SRS finite designs but not genuine independent canonical-seed ensembling;
- or the advantage disappears close to convergence;
- or interaction theory fails to explain SRSWOR heterogeneity.

---

## Not supported

Use:

```text
NOT SUPPORTED
```

if canonical independent-seed ensembles match or dominate OrbitCover broadly and no distinct target expectation or coupling advantage remains.

Do not rescue this by changing the thesis after seeing results.

---

# 44. Final `results.md`

After ALL experiments finish, overwrite/create the final top-level:

```text
results.md
```

The report must stand alone.

Use the following structure exactly.

---

# RESULTS — FINAL ICLR CLOSURE

## 1. Executive verdict

Choose exactly:

```text
SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED
```

Give the answer immediately.

Then summarize the decisive evidence in 8–15 sentences.

---

## 2. What changed relative to the previous results

Summarize the previous state and what the closure experiments resolved.

Explicitly discuss the earlier matched-function finding.

---

## 3. Independent canonical-seed showdown

This is the most important section.

Report B=16 headline table:

| Method | Mean residual | Relative to canonical independent | Cell wins | Source wins |
|---|---:|---:|---:|---:|

Include:

```text
Canonical independent
IID joint
SRS joint
strength-1
OC2 independent
OC2 coupled
OC2 packed if available
```

Give clustered confidence intervals.

---

## 4. Does schema symmetrization change the expectation?

Compare:

```text
Q_canonical_independent
Q_schema×independent
Q_finite_coupled
```

State clearly whether they are statistically/materially different.

This section determines whether OrbitCover is:

```text
estimating a genuinely different symmetrized target
```

or primarily:

```text
a variance-reduction coupling for the same expectation.
```

---

## 5. What does the 98% matched-function result mean now?

Interpret using the new independent-seed evidence.

Discuss separately:

```text
MLP
ResNet
FT-Transformer
TabM
```

Do not return to a universal optimization-path claim if unsupported.

---

## 6. Training-scale and convergence

Report all mandatory scaling numbers.

State explicitly:

```text
Does nuisance variance persist at convergence?
Does OrbitCover efficiency persist at convergence?
Does dataset size change the story?
```

---

## 7. Interaction spectrum explains successes/failures

Report:

```text
main+pair fraction vs OC2/SRS gain
higher-order fraction vs OC2/SRS gain
```

List all source-level SRS wins.

Answer whether the 8/12 source result is theoretically explained.

---

## 8. Strength hierarchy

Update the interpretation of:

```text
strength-1
strength-2
strength-3
```

using the interaction analysis.

State whether:

> match strength to interaction order

is empirically supported.

---

## 9. Coupling mechanism

If Experiment D completed, report which balancing dimensions matter most.

Answer:

> Is OrbitCover principally balancing schema, RNG, or schema×RNG interactions?

---

## 10. Architecture-specific conclusions

Create one subsection each:

```text
MLP
ResNet
FT-Transformer
TabM
TabPFN
CatBoost/GBDT
```

For TabPFN and classical models, integrate prior completed evidence rather than unnecessarily rerunning everything.

---

## 11. Practical compute efficiency

Report:

```text
IID-equivalent independent-seed budget
GPU-hours
wall clock
number of complete fits
```

Provide statements such as:

```text
16 OrbitCover fits match approximately X canonical independent fits.
```

Only if supported.

---

## 12. Ranking/model-selection implications

Integrate the previous ranking evidence.

Do not headline small test-regret improvements.

State clearly that partition shift is separate from nuisance Monte Carlo.

---

## 13. Failure cases

Mandatory.

List:

- cases canonical independent seeds win;
- cases SRSWOR wins;
- cases strength-3 does not recover strength-2 failures;
- cases nuisance variance vanishes with convergence;
- architectures with negligible matched residual;
- datasets where OrbitCover provides little practical benefit.

---

## 14. Final defensible theorem/claim target

Write the strongest empirical/theoretical statement justified by the final evidence.

Do not claim novelty for orthogonal arrays, group averaging, or generic antithetic sampling.

---

## 15. Recommended final paper thesis

Choose one.

### Thesis A

```text
Semantic symmetries provide structured randomization that estimates the
expectation of randomized learning pipelines more efficiently than independent
retraining.
```

### Thesis B

```text
Semantic symmetrization defines a distinct quotient predictor, and
interaction-balanced designs estimate it efficiently.
```

### Thesis C

```text
Finite nuisance balancing is useful only for restricted seed/schema menus;
the broader independent-training claim does not survive.
```

Choose according to evidence.

---

## 16. Best paper title

Give 3 titles ranked strongest to weakest.

---

## 17. ICLR readiness

Score 1–5:

```text
novelty
theory
empirical breadth
baseline strength
mechanism
realistic-scale evidence
prospective validity
story coherence
reproducibility
```

Then choose exactly:

```text
READY TO WRITE ICLR
ONE TARGETED EXPERIMENT REMAINS
MAJOR ISSUE
PIVOT
```

Because this AGENT.md is intended as the final experiment program, choose:

```text
ONE TARGETED EXPERIMENT REMAINS
```

only if an unavoidable new scientific ambiguity genuinely appears.

Do not invent additional experiments merely because results are imperfect.

---

## 18. Five strongest reviewer objections

For each provide:

```text
objection
evidence addressing it
remaining weakness
best response
```

Use serious reviewer objections.

---

## 19. Final recommendation

Choose exactly:

```text
COMMIT TO PAPER
PIVOT PAPER THESIS
ABANDON ORBITCOVER AS MAIN METHOD
```

Explain in one paragraph.

---

# 45. Completion requirements

The task is NOT complete until all of the following exist:

```text
FINAL_CLOSURE_PROTOCOL.md
PROTOCOL_HASH.txt
PROTOCOL_DEVIATIONS.md
all mandatory raw outputs
all summary CSV/JSON files
all required figures
all required tables
integrity audit
results.md
```

Run a final audit verifying:

```text
no missing mandatory cells
no duplicate supposedly-independent RNG seeds
no corrupted predictions
no unequal fit-budget comparisons
no unrecorded protocol deviations
all statistical summaries regenerate from raw outputs
all figures regenerate
all tables regenerate
```

Write:

```text
experiments/final_closure/FINAL_AUDIT.md
```

The very last step is to regenerate `results.md` from final audited artifacts.

Do not stop before that point.

---

# 46. Final perspective

The purpose of these experiments is not to accumulate more positive results.

The purpose is to answer the last decisive question:

> Does interaction-balanced symmetrization offer a genuine efficiency advantage over simply training more independent models on the canonical representation?

If yes, quantify when and why.

If only structured schema×RNG coupling wins, make that the paper.

If no, state it clearly.

Finish the experiment program and let the evidence determine the final thesis.