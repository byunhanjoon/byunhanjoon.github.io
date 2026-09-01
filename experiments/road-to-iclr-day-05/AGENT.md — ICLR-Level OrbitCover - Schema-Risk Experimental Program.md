# AGENT.md — ICLR-Level OrbitCover / Schema-Risk Experimental Program

## Mission

Run a rigorous, comprehensive, largely prospective experimental program testing the following paper thesis:

> Modern tabular learning pipelines are not invariant to semantically equivalent representations of the same dataset. This representation-induced nuisance variation interacts with ordinary training randomness. By explicitly defining the symmetrized/quotient predictor over these nuisance variables and using interaction-balanced finite designs such as OrbitCover, we can estimate that predictor, compare models, and perform model selection substantially more efficiently than ordinary IID ensembling.

The goal is **not** to maximize leaderboard performance.

The goal is to determine whether this thesis survives strong modern architectures, exact equivariance controls, realistic schema transformations, alternative sampling designs, repeated data splits, and prospective evaluation.

Be aggressive about falsification.

Do not rescue failed hypotheses after seeing results.

---

# 0. Required outputs

Create and maintain:

```text
experiments/
  protocols/
  raw/
  processed/
  figures/
  tables/
  logs/
  checkpoints/

results.md
```

The primary final deliverable is:

```text
results.md
```

`results.md` must summarize the scientific findings, not implementation details.

Also save machine-readable experiment outputs as CSV/Parquet/JSON wherever appropriate.

Do not bury results only inside logs.

---

# 1. Core research questions

The experiments must answer the following questions.

## RQ1 — Does exact schema nuisance risk exist broadly?

For semantically equivalent representations of a dataset, how much do fitted predictions change?

Study at minimum:

- feature/column permutation;
- categorical-ID relabeling;
- target-label permutation for classification;
- category ordering where ordering is semantically arbitrary;
- model seed;
- initialization seed if separable;
- dataloader/order seed if separable.

Do not describe all of these as mathematical symmetries if they are not.

Distinguish:

- semantic/schema transformations `g`;
- stochastic training variables `s`.

Conceptually define:

```text
f_{g,s}(x)
```

after predictions have been aligned back to the canonical schema/output convention.

Define the quotient/symmetrized predictor:

```text
Q(x) = E_{g,s}[ f_{g,s}(x) ]
```

over a clearly declared finite nuisance distribution.

When practical, also discuss the distinction between:

```text
finite declared nuisance set
```

and

```text
the much larger/full natural symmetry group
```

Do not call a small finite menu "the full orbit" unless it is literally exhaustive.

---

# 2. Models

Use modern tabular architectures and strong classical baselines.

At minimum include:

## Neural

- MLP
- ResNet-style tabular network
- FT-Transformer
- TabM

Strongly preferred if available:

- TabR
- SAINT or another attention-based tabular model

## Foundation model

- TabPFN

For TabPFN explicitly evaluate:

1. normal/default inference;
2. built-in ensembling or permutation handling if exposed;
3. external OrbitCover-style nuisance averaging where meaningful.

Avoid double-counting transformations already internally averaged by TabPFN.

Document exactly which transformations TabPFN performs internally.

## Tree / boosting models

At minimum:

- CatBoost
- HistGradientBoosting or LightGBM
- XGBoost if practical

If LightGBM/XGBoost are unavailable, use sklearn HistGradientBoosting and CatBoost.

## Negative-control / nearly invariant models

Include at least one model expected to have minimal sensitivity to certain schema nuisances.

Examples:

- one-hot encoded logistic/linear regression;
- one-hot ridge;
- deterministic tree pipeline where sensible.

These are scientifically important.

If a transformation should theoretically leave a pipeline invariant, test whether empirical measured nuisance risk approaches numerical zero.

---

# 3. Dataset panel

Construct a broad untouched panel.

Target:

```text
12–20 datasets minimum
```

Preferred:

```text
16–24 datasets
```

Include both:

```text
classification
regression
```

Aim for approximately balanced task families.

Datasets should vary in:

- number of rows;
- number of features;
- categorical fraction;
- categorical cardinality;
- numerical/categorical mix;
- dataset size;
- class count;
- class imbalance;
- feature dimensionality.

Prefer TabArena/OpenML datasets or another public standardized benchmark collection.

Avoid choosing datasets because previous OrbitCover experiments performed well.

Create the panel before running the main prospective experiment.

Save a freeze file containing:

```text
dataset IDs
dataset versions
preprocessing
splits
model configurations
nuisance menus
budgets
evaluation metrics
success criteria
random seeds
```

Hash the freeze file.

---

# 4. Data splits

Do NOT rely on one train/validation/test split.

Use multiple independent splits when computationally feasible.

Recommended:

```text
5 splits per dataset
```

Minimum:

```text
3 splits per dataset
```

Important conceptual separation:

```text
schema/training nuisance variance
```

versus

```text
dataset split / finite-sample variance
```

Do not imply OrbitCover fixes split instability.

Analyze both.

If compute is prohibitive:

1. use one fixed split for exhaustive nuisance-tensor experiments;
2. use 3–5 independent splits for the main prospective benchmark.

---

# 5. Exact nuisance transformations

Implement transformations that preserve dataset semantics.

## 5.1 Column permutation

Randomly permute input feature order.

Alignment is straightforward.

Test models with:

- shared feature processing;
- feature-specific parameters;
- transformer feature tokens;
- tree models.

---

## 5.2 Category-ID permutation

For categorical variable `j`, map category IDs through a permutation:

```text
π_j : {1,...,K_j} -> {1,...,K_j}
```

The underlying category identity remains unchanged.

Predictions should represent the same mathematical learning problem after proper input transformation.

Do NOT accidentally change train/test mappings independently.

Use the same category mapping consistently.

---

## 5.3 Target-label permutation

For multiclass classification:

```text
π_y : classes -> classes
```

Train under relabeled targets.

At evaluation, permute output probabilities back to canonical class order.

This is especially important for TabPFN and classifiers whose training behavior may depend on label identity.

---

## 5.4 Combined transformations

Study interaction effects between:

```text
feature permutation
category relabeling
target relabeling
training seed
```

Use a manageable declared factorial product.

An example:

```text
4 feature permutations
4 category relabeling states
2 target relabelings
4 model seeds
```

or another design appropriate to each task.

For regression omit target-label permutation and replace it, if scientifically appropriate, with another genuine nuisance factor.

Never invent transformations merely to fill a factorial design.

---

# 6. Exact matched-function / equivariance controls

This is one of the highest-priority experiments.

A reviewer can argue that category-ID permutation changes which random embedding vector each category receives, making schema nuisance equivalent to another initialization seed.

Test this directly.

For every transformation where possible, construct a corresponding parameter transformation so that the network represents the **exact same initial function** before optimization.

---

## 6.1 Category embedding relabeling

If category IDs are permuted by:

```text
π
```

then permute embedding-table rows so that each semantic category starts with exactly the same vector as under the canonical representation.

Verify numerically before training:

```text
max |f_original(x) - aligned(f_transformed(gx))| < 1e-7
```

or an appropriately strict floating-point tolerance.

Compare:

```text
ordinary transformed initialization
vs
matched-function transformed initialization
```

Measure how much nuisance risk disappears.

---

## 6.2 Feature permutation

For models with feature-specific parameters, permute the corresponding:

- embeddings;
- normalization parameters;
- first-layer input weights;
- feature tokens;
- feature-specific biases;
- any metadata or learned feature identities.

Verify exact initial-function equivalence.

---

## 6.3 Target permutation

For classification, permute final-layer/output parameters appropriately so that canonical and relabeled models represent exactly aligned probabilities at initialization.

---

## 6.4 Interpretation

Separate nuisance sensitivity into approximately:

```text
initial function / parameter-assignment effect
```

and

```text
optimization-path / stochastic-training effect
```

Do not overclaim causal decomposition unless the intervention supports it.

This experiment should directly connect Day-3 optimization-geometry findings to OrbitCover.

---

# 7. Exact nuisance tensor experiments

For a subset of manageable dataset × model combinations, exhaustively enumerate the declared nuisance product.

Store aligned predictions:

```text
P[g1, g2, ..., seed, sample, output]
```

Compute exactly:

- quotient prediction `Q`;
- member-to-quotient squared prediction deviation;
- quotient Brier/MSE;
- nuisance-induced excess quadratic risk;
- variance contribution by factor;
- pairwise interaction contribution;
- higher-order interaction contribution.

Use prediction-space functional ANOVA.

Verify reconstruction numerically.

For quadratic prediction norm:

```text
Total nuisance variance
≈
sum of all fANOVA components
```

Require reconstruction error close to floating-point tolerance.

---

# 8. Interaction-order characterization

Determine which nuisance interaction orders dominate.

For each complete nuisance tensor, measure fractions attributable to:

```text
main effects
pairwise interactions
triple interactions
4-way/higher interactions
```

Aggregate by:

- dataset;
- architecture;
- task type;
- categorical fraction;
- number of categorical features.

Core hypothesis:

```text
real nuisance tensors are often dominated by low-order interactions,
making strength-matched balanced designs efficient.
```

But explicitly search for counterexamples.

Identify datasets/models where:

```text
high-order interactions dominate
```

These are expected failure cases for low-strength OrbitCover.

---

# 9. OrbitCover designs

Implement balanced estimators over the declared nuisance product.

At minimum:

```text
strength-1
strength-2
strength-3 when feasible
```

Use randomized orthogonal arrays or equivalent balanced covering designs.

Clearly distinguish:

```text
classical OA machinery
```

from

```text
the proposed application to complete-pipeline nuisance integration.
```

For each estimator:

```text
Q_hat = average predictions over chosen nuisance configurations
```

Measure:

```text
||Q_hat - Q||²
```

or held-out approximation when exhaustive `Q` is impossible.

---

# 10. Sampling baselines

This section is mandatory.

Compare OrbitCover against equal-fit-budget baselines.

At minimum:

1. IID sampling with replacement
2. simple random sampling without replacement
3. seed-only averaging
4. schema-only averaging
5. strength-1 balanced designs
6. Latin hypercube sampling
7. Sobol / QMC where mathematically applicable
8. strength-2 OrbitCover
9. strength-3 OrbitCover
10. canonical representation only
11. exhaustive quotient when feasible

Also compare against:

```text
built-in ensembling of TabPFN
```

when available.

Every comparison must use exactly the same number of model fits unless the purpose is explicitly a budget-equivalence curve.

---

# 11. Budget curves

Use multiple fit budgets.

Recommended:

```text
B = 1, 2, 4, 8, 16, 32, 64
```

and:

```text
128
```

where feasible.

For each method plot:

```text
quotient estimation error vs number of fits
```

Also estimate:

```text
IID-equivalent fit budget
```

Example question:

```text
How many IID fits are required to match the error of a 16-fit strength-2 OrbitCover?
```

Report distributions across datasets, not only averages.

---

# 12. Strength vs interaction order synthetic sanity checks

Construct exact synthetic prediction tensors with known ANOVA structure.

Examples:

```text
pure main effect
pure pairwise interaction
pure triple interaction
pure four-way interaction
mixed realistic spectrum
```

Verify theoretically predicted behavior.

Expected:

- strength-1 annihilates relevant main-effect components;
- strength-2 handles low-order structure but can fail badly for pure higher-order tensors;
- strength-3 improves when third-order interactions matter;
- no design is universally optimal.

Use this to establish a clear failure boundary.

---

# 13. Model-ranking experiments

The quotient should be treated as a stable target for model comparison.

For each dataset/split, define the reference ranking from the most accurate available quotient estimate.

Then compare rankings produced by equal-budget:

```text
IID
seed averaging
strength-1
strength-2
strength-3
Sobol
LHS
```

Metrics:

- Spearman correlation;
- Kendall correlation;
- pairwise order accuracy;
- inversion count;
- top-1 winner agreement;
- top-k overlap.

Analyze as a function of true model-performance margin.

Hypothesis:

```text
OrbitCover matters especially when models are close and nuisance variance is non-negligible.
```

---

# 14. Model-selection experiments

This is a major paper component.

For every candidate model/hyperparameter configuration:

1. estimate validation performance under a fixed budget;
2. select the apparent winner;
3. evaluate that selected model on held-out test data using a high-quality quotient estimate;
4. compare selection methods.

Methods:

```text
single canonical fit
seed averaging
IID nuisance averaging
strength-1
strength-2
strength-3
```

Measure:

- quotient-winner agreement;
- selected model test loss;
- regret to true quotient winner;
- inversion probability;
- frequency of selecting different architectures.

Use independent nuisance draws between candidate models in at least one experiment to show results do not depend entirely on common random numbers.

---

# 15. Independent-cover quadratic cross-score

For Brier and MSE, test the Day-5 unbiased cross-score result.

Given independent unbiased quotient estimates:

```text
Q_hat_A
Q_hat_B
```

evaluate:

```text
<Y - Q_hat_A, Y - Q_hat_B>
```

Confirm empirically that it is unbiased for:

```text
||Y - Q||²
```

within Monte Carlo uncertainty.

Compare:

```text
ordinary squared loss of Q_hat
```

which should contain the estimator-variance bias.

---

# 16. Cross-score variance mechanism

Test the predicted variance formula:

```text
Var(score)
=
2 <r, C r> + tr(C²)
```

where:

```text
r = Y - Q
C = covariance of quotient estimator prediction error
```

Estimate:

- residual-aligned term;
- covariance self-interaction;
- total predicted variance;
- empirical score variance.

Do this for:

```text
IID
strength-1
strength-2
strength-3
```

where feasible.

Report predicted/observed variance ratios.

This creates a mechanism chain:

```text
balanced design
→ covariance reduction
→ cross-score variance reduction
→ better selection
```

---

# 17. Negative dependence / packed covers

Test the late Day-5 idea separately from ordinary OrbitCover.

Construct successive cover blocks with negative dependence when mathematically supported.

Examples:

```text
disjoint cover blocks
regular disjoint-cover graphs
resolvable OA cosets
```

Compare against:

```text
independent balanced covers
IID covers
Gaussian-antithetic-style reference simulations
```

Do not claim general antithetic optimality.

The purpose is to determine whether structured finite-product negative dependence provides additional variance reduction before exhaustive closure.

Plot risk versus:

```text
16
32
64
128
```

fits.

At exact resolution closure, verify the estimator equals the finite quotient where applicable.

---

# 18. Finite nuisance approximation vs larger symmetry distribution

The declared nuisance menu is only a finite approximation of potentially huge transformation spaces.

Study this explicitly.

For selected datasets, create increasingly large nuisance menus:

```text
M = 4
8
16
32
64
```

states per relevant factor where practical.

Estimate whether results stabilize.

Separate:

```text
finite-menu approximation error
```

from

```text
sampling error within a fixed menu.
```

For column permutations, sample from the natural uniform permutation distribution.

For category relabeling, sample from valid category permutations.

Where exact exhaustive integration is impossible, use a very large Monte Carlo reference.

---

# 19. TabPFN-specific study

TabPFN deserves a dedicated section.

Determine experimentally:

1. sensitivity to feature permutations;
2. sensitivity to target-label permutations;
3. sensitivity to category encoding choices;
4. effect of internal/default TabPFN ensembling;
5. remaining external nuisance risk after its own ensemble;
6. whether external OrbitCover reduces that residual risk more efficiently than more IID TabPFN calls.

Compare:

```text
TabPFN canonical
TabPFN default ensemble
TabPFN IID external nuisance ensemble
TabPFN OrbitCover external nuisance ensemble
```

Be careful to count actual inference/training compute fairly.

If TabPFN has no training fit cost comparable to neural training, report:

```text
number of forward/inference ensemble members
```

separately from:

```text
number of fitted models
```

Do not make misleading fit-budget comparisons.

---

# 20. CatBoost / GBDT study

Study CatBoost and at least one conventional boosting model.

Important questions:

- Does CatBoost have lower category-ID sensitivity because it treats categorical variables specially?
- Does feature permutation matter?
- Does seed still interact with schema choices?
- Are interactions lower order than in neural models?
- Does OrbitCover help equally, less, or not at all?

Include:

```text
CatBoost
HistGradientBoosting or LightGBM
XGBoost if available
```

One-hot linear/logistic models serve as useful near-invariant controls.

A strong paper should show where OrbitCover is unnecessary.

---

# 21. Representation-sensitive vs invariant architectures

Measure nuisance risk per architecture.

Create an architecture table:

```text
model
schema risk
seed risk
interaction risk
fraction pairwise+
OrbitCover improvement
```

Test whether models with architectural equivariance/invariance exhibit lower nuisance risk.

This can make the paper broader than tabular benchmarking:

```text
OrbitCover is useful precisely when the learning algorithm fails to respect an available task symmetry.
```

---

# 22. Relationship between nuisance risk and OrbitCover gain

Across all cells, test whether OrbitCover gain correlates with:

- total nuisance variance;
- pairwise interaction mass;
- higher-order interaction mass;
- category fraction;
- number of features;
- model architecture;
- dataset size;
- canonical-vs-quotient prediction gap.

A particularly important test:

```text
Does strength-2 gain increase when pairwise interaction mass is large relative to higher-order mass?
```

This should connect empirical gains directly to the ANOVA theory.

---

# 23. Calibration / discrimination / non-quadratic metrics

Primary theoretical metrics:

Classification:

```text
Brier score
```

Regression:

```text
MSE / RMSE
```

Secondary metrics:

Classification:

```text
accuracy
log loss
AUROC where appropriate
ECE / calibration error
```

Regression:

```text
MAE
R²
```

Do not imply the exact quadratic theory automatically extends to log loss, accuracy, or AUROC.

Use these to test whether prediction-space stabilization translates to useful downstream metrics.

---

# 24. Efficiency

Measure:

```text
wall-clock time
GPU-hours
CPU-hours
peak memory
number of fits
```

OrbitCover should have almost no overhead apart from choosing which configurations to run.

Report whether balancing itself has negligible computational cost.

For TabPFN, separate inference cost from fitted-model cost.

---

# 25. Statistical analysis

The experimental unit is not always an individual model fit.

Avoid pseudoreplication.

Aggregate primarily at:

```text
dataset
dataset × split
```

levels.

Use:

- dataset-level bootstrap intervals;
- task-balanced averages;
- classification/regression stratification;
- paired differences;
- sign counts where appropriate.

Do not report every nuisance cell as statistically independent.

Whenever showing win counts, state the aggregation level.

---

# 26. Prospective protocol

Before the final major benchmark:

Create:

```text
experiments/protocols/final_prospective_protocol.md
```

containing:

- exact dataset list;
- exact versions;
- all models;
- all hyperparameters;
- nuisance factors;
- nuisance levels;
- sampling methods;
- budgets;
- seeds;
- metrics;
- primary hypotheses;
- success/failure criteria;
- intended analyses.

Hash it.

After hashing, do not modify the protocol based on partial results.

If an implementation bug is discovered:

1. record it;
2. fix it;
3. rerun all affected cells;
4. document the deviation.

---

# 27. Recommended primary success criteria

Freeze exact criteria before the prospective run.

Suggested criteria:

## Gate A — nuisance phenomenon

Across modern neural architectures:

```text
schema nuisance must be materially nonzero on a majority of datasets.
```

Negative controls should show substantially less risk.

## Gate B — OrbitCover estimator

At budget 16:

```text
strength-2 OrbitCover must lower quotient-estimation error versus
IID without replacement on a majority of datasets
and improve the dataset-balanced mean.
```

Prefer a high bar such as:

```text
>= 75% dataset-level wins
```

if previous evidence suggests this is realistic.

## Gate C — model rankings

Strength-2 must improve:

```text
winner agreement
and/or
pairwise ranking accuracy
```

relative to IID at the same budget.

## Gate D — selected test performance

OrbitCover-based validation selection should reduce average selected-test regret versus IID selection.

Do not require every dataset to improve.

## Gate E — mechanism

Observed gain must track predicted low-order interaction structure.

If strength-2 improves everywhere but has no relationship to interaction decomposition, investigate alternative explanations.

## Gate F — matched-function control

At least some measured schema risk should remain after exact initial-function matching.

Otherwise the phenomenon may reduce primarily to reassignment of random initialization.

If almost all risk disappears, report that honestly and reconsider the thesis.

---

# 28. Critical falsification cases

Actively search for conditions where OrbitCover fails.

Examples:

- purely high-order nuisance interactions;
- highly invariant models;
- very low nuisance-risk datasets;
- huge categorical spaces;
- models where category relabeling is intrinsically equivariant;
- cases where a single canonicalization matches quotient performance;
- cases where Sobol/LHS outperform strength-2;
- cases where better quotient validation does not improve test selection;
- cases where dataset-split variance dominates all schema effects.

These belong in the paper.

Do not hide them.

---

# 29. Avoid these claims

Do NOT claim:

```text
Orthogonal arrays are novel.
```

Do NOT claim:

```text
negative dependence for risk estimation is novel.
```

Do NOT claim:

```text
OrbitCover is universally optimal.
```

Do NOT claim:

```text
the finite declared nuisance menu is the entire mathematical symmetry group.
```

Do NOT claim:

```text
all schema transformations form a group action together with seed.
```

Do NOT claim:

```text
OrbitCover always improves predictive accuracy.
```

Do NOT claim:

```text
Brier/MSE theory applies unchanged to arbitrary proper losses.
```

Do NOT claim:

```text
TabPFN is not permutation-aware.
```

Instead measure its remaining sensitivity after its existing mechanisms.

---

# 30. Key figures to generate

At minimum produce publication-quality figures for:

## Figure 1

Distribution of prediction changes caused by semantically equivalent schema transformations.

Show several models.

---

## Figure 2

fANOVA decomposition:

```text
main
pair
triple
higher
```

across datasets/models.

---

## Figure 3

Quotient estimation error vs fit budget:

```text
IID
without replacement
LHS
Sobol
strength-1
strength-2
strength-3
```

---

## Figure 4

IID-equivalent compute budget.

Example:

```text
16 OrbitCover fits ≈ X IID fits
```

dataset distribution.

---

## Figure 5

Model ranking fidelity vs budget.

---

## Figure 6

Selected-test regret vs validation estimator.

---

## Figure 7

Matched-function control:

```text
ordinary schema transform
matched-initial-function transform
```

and remaining nuisance variance.

---

## Figure 8

OrbitCover gain vs pairwise/higher-order ANOVA mass.

---

## Figure 9

Architecture comparison:

```text
MLP
ResNet
FT-Transformer
TabM
TabPFN
CatBoost
GBDT
```

---

## Figure 10

Failure-boundary synthetic tensors showing when strength-2 fails.

---

# 31. Tables

Create at minimum:

## Table 1 — datasets

Columns:

```text
dataset
task
n
d
# numerical
# categorical
mean category cardinality
classes
```

## Table 2 — models

Columns:

```text
model
parameter count
training budget
schema handling
seed count
```

## Table 3 — nuisance risk

Per architecture:

```text
total risk
schema-only
seed-only
interaction
```

## Table 4 — estimator comparison

At budget 16:

```text
IID
SRSWOR
LHS
Sobol
strength-1
strength-2
strength-3
```

## Table 5 — ranking/model selection

```text
winner agreement
Spearman
pairwise accuracy
test regret
```

## Table 6 — matched-function control

```text
ordinary nuisance variance
matched-function nuisance variance
fraction removed
fraction remaining
```

---

# 32. Implementation integrity tests

Write automated tests for:

1. schema transformations preserve semantic feature values;
2. category relabeling is invertible;
3. target relabeling is inverted correctly at prediction time;
4. feature permutation alignment is exact;
5. matched-parameter transformations preserve the initial prediction function;
6. quotient predictions are invariant to enumeration order;
7. OA designs have the declared strength;
8. all requested factor levels are balanced;
9. fANOVA components reconstruct total variance;
10. independent-cover samples are actually independent where required;
11. disjoint covers are truly disjoint where claimed;
12. no test data is used for hyperparameter selection;
13. equal-budget comparisons use equal numbers of fits.

Fail loudly.

---

# 33. Compute strategy

Use staged execution.

## Stage 1 — smoke test

Use:

```text
2 datasets
2 models
2 nuisance factors
```

Validate implementation.

Do not interpret scientifically.

---

## Stage 2 — exact tensor subset

Use:

```text
4–6 datasets
4–6 models
```

where exhaustive nuisance products are affordable.

Develop theory/interaction figures.

---

## Stage 3 — broad screening

Use the entire dataset/model panel at modest budget.

Identify implementation failures, not scientific winners.

---

## Stage 4 — frozen prospective benchmark

Run all final methods with no adaptive changes.

This is the main evidence.

---

## Stage 5 — targeted mechanism experiments

Run:

```text
matched-function controls
cross-score validation
negative-dependent packed covers
larger nuisance-menu experiments
```

Preferably according to protocols defined before examining those specific results.

---

# 34. Hyperparameters

Avoid expensive independent HPO for every nuisance configuration.

Use one of these approaches:

```text
published/default architecture configurations
```

or:

```text
hyperparameters selected once from the canonical training representation
and then frozen across nuisance configurations
```

Do not tune hyperparameters separately for each schema transformation unless that is the experiment being studied.

For model-selection experiments, define a separate candidate grid prospectively.

---

# 35. results.md requirements

At the end, generate:

```text
results.md
```

Do not merely concatenate logs.

It must be a concise scientific report.

Use this structure.

---

# RESULTS

## 1. Executive verdict

State:

```text
SUPPORTED
PARTIALLY SUPPORTED
or
NOT SUPPORTED
```

for the overall ICLR thesis.

Then give 5–10 sentences explaining why.

---

## 2. Main findings

List the strongest quantitative findings.

Include:

- nuisance risk breadth;
- architecture breadth;
- strength-2 performance;
- strength-3 performance;
- IID-equivalent budgets;
- ranking improvements;
- model-selection improvements;
- matched-function results;
- TabPFN results;
- CatBoost/GBDT results;
- repeated-split findings.

---

## 3. Prospective benchmark

Explicitly distinguish:

```text
exploratory
adaptive
prospective
```

experiments.

Report final prospective results separately.

---

## 4. Nuisance-risk phenomenon

Answer:

- Which models are most sensitive?
- Which transformations matter?
- How large is the prediction effect?
- How much is explained by seed?
- How much comes from schema × seed interactions?

---

## 5. Matched-function results

Report:

```text
fraction of schema variance removed by exact initial-function matching
fraction remaining after matching
```

Interpret cautiously.

This section is mandatory.

---

## 6. fANOVA interaction structure

Report average/median shares from:

```text
main
pairwise
triple
higher
```

State whether the empirical tensors actually justify strength-2 balancing.

---

## 7. OrbitCover estimator results

At each budget compare all methods.

Highlight budget 16.

Report:

```text
mean reduction
median reduction
dataset-level wins
bootstrap interval
```

against at least:

```text
IID
SRS without replacement
LHS
Sobol
seed-only
strength-1
```

---

## 8. Strength hierarchy

Report:

```text
strength-1
strength-2
strength-3
```

Explain when higher strength helps and when it does not.

Include synthetic failure cases.

---

## 9. TabPFN

Report:

- default sensitivity;
- effect of internal ensembling;
- residual nuisance risk;
- effect of external OrbitCover;
- whether OrbitCover remains useful after built-in symmetry handling.

---

## 10. CatBoost / GBDT

Report whether:

- schema risk is smaller;
- category relabeling matters;
- feature ordering matters;
- OrbitCover still helps.

Emphasize models where the method is unnecessary.

---

## 11. Ranking and model selection

Report:

```text
winner agreement
pairwise inversion
Spearman/Kendall
selected-test regret
```

Compare all sampling schemes.

---

## 12. Cross-score results

Report:

- empirical bias of ordinary quadratic validation;
- unbiasedness of independent-cover cross-score;
- theoretical vs empirical variance;
- downstream selection effects.

---

## 13. Split instability

Compare:

```text
schema/seed nuisance
vs
train-validation-test split instability
```

Answer whether split variance becomes dominant after quotient estimation improves.

---

## 14. Failure cases

This section is mandatory.

List all meaningful cases where:

- OrbitCover loses;
- Sobol/LHS wins;
- strength-3 does not help;
- canonicalization is enough;
- nuisance risk is negligible;
- better validation quotient does not improve selected test performance;
- high-order interactions break low-strength designs.

---

## 15. Ablations

Summarize:

- category permutation;
- feature permutation;
- target permutation;
- seed;
- interactions;
- matched initialization;
- independent vs common nuisance draws.

---

## 16. Best defensible paper claim

Write the strongest claim supported by evidence.

It should resemble:

> Semantically equivalent schema representations induce measurable prediction variation in modern tabular learning pipelines. This variation contains substantial low-order interactions with training randomness. A strength-matched balanced estimator over these nuisance variables approximates the schema-symmetrized predictor substantially more efficiently than IID averaging and improves finite-budget model ranking and selection.

Modify this only according to actual results.

---

## 17. Claims that failed

Explicitly list hypotheses that were falsified.

Examples:

```text
strength-2 always wins
semantic representations explain all ensemble gains
matched-function initialization removes no risk
OrbitCover improves accuracy universally
TabPFN is fully invariant
```

Only include claims actually tested.

---

## 18. ICLR readiness assessment

Give scores from 1–5 for:

```text
novelty
empirical strength
dataset breadth
architecture breadth
theoretical support
mechanism evidence
baseline quality
prospective validity
story coherence
reproducibility
```

Then state:

```text
Ready to write
Needs one more experiment
Major issue remains
Abandon direction
```

---

## 19. Biggest reviewer objections

Write the five strongest reviewer criticisms.

For each:

```text
objection
existing evidence
remaining weakness
best response
```

Do not strawman reviewers.

---

## 20. Recommended paper structure

Provide an 8–10 section paper outline based on actual findings.

---

## 21. Final recommendation

Choose exactly one:

```text
COMMIT TO ICLR PAPER
CONTINUE WITH TARGETED EXPERIMENTS
PIVOT
ABANDON
```

Give the reason.

---

# 36. Scientific discipline

Throughout the work:

Do not optimize for positive results.

Do not silently remove datasets.

Do not silently change nuisance definitions.

Do not alter the primary metric after seeing results.

Do not replace a failed baseline.

Do not rerun only unfavorable cells with new seeds.

Do not present dependent cells as independent samples.

Do not call exploratory experiments prospective.

Do not reinterpret a failed hypothesis into a new success without labeling it as a new hypothesis.

Record negative findings.

---

# 37. Final priority order if compute becomes limited

If the full program is too expensive, prioritize in this exact order:

1. prospective modern-neural panel;
2. MLP + ResNet + FT-Transformer + TabM;
3. TabPFN;
4. CatBoost + GBDT;
5. matched-initial-function controls;
6. exact nuisance tensors on a representative subset;
7. IID/SRS/LHS/Sobol/strength-1/strength-2 comparisons;
8. repeated data splits;
9. ranking/model-selection evaluation;
10. strength-3;
11. cross-score experiments;
12. negative-dependent packed covers;
13. enlarged nuisance-space experiments.

Never drop matched-function controls just to add more datasets.

---

# 38. Decision rule

The project is strongest if all of these survive:

```text
1. Equivalent schema representations measurably change fitted predictions.

2. The effect persists across modern neural models and is not merely a quirk
   of one architecture.

3. Exact function-matched initialization removes part but not all of the
   effect, supporting an optimization-path component.

4. Real nuisance tensors contain enough low-order interaction structure for
   strength-2 balancing to be well motivated.

5. Strength-2 OrbitCover beats strong equal-budget sampling controls.

6. The advantage extends to ranking/model selection, not merely prediction
   approximation.

7. TabPFN/CatBoost/GBDT provide informative boundaries rather than destroying
   the phenomenon entirely.

8. Failure cases agree with the interaction-order theory.
```

If these hold on the frozen prospective panel, the direction is plausibly strong enough to develop into an ICLR submission.

If matched-function controls eliminate essentially all nuisance risk, or strength-2 fails against SRS/LHS/Sobol on the untouched modern panel, state that clearly in `results.md` and recommend reconsidering the central thesis.

The purpose of this agent is to find out whether the paper is true.