# AGENT.md — ICLR Direction Decision Sprint

## Mission

Run the **minimum decisive set of frozen, prospective experiments** needed to decide which research direction deserves the majority of effort:

1. **Day 3 / OrbitANOVA / schema-representation risk** — incumbent primary paper.
2. **Day 4 / HeteroBag-3** — representation diversity vs seed diversity.
3. **Day 4 / FieldRiesz** — semantic field geometry / chart-covariant priors.

The goal is **not** to maximize benchmark numbers or invent more variants.

The goal is to finish with:

```text
PRIMARY_DIRECTION = DAY3 | HETEROBAG | FIELDRIESZ
SECONDARY_DIRECTION = ...
STOP = [...]
```

and enough evidence that the decision would survive skeptical ICLR review.

---

# 0. Non-negotiable scientific rules

Before running new test outcomes:

- Freeze dataset IDs.
- Freeze splits.
- Freeze architectures.
- Freeze seed lists.
- Freeze preprocessing.
- Freeze hyperparameter menus.
- Freeze candidate/control definitions.
- Freeze promotion gates.
- Hash the freeze file.

Create:

```text
experiments/iclr_direction_decision/
    DECISION_FREEZE.md
    decision_config.json
```

Do **not**:

- select datasets after seeing outcomes;
- replace failed datasets;
- change seed counts after results;
- tune a method on test data;
- rescue a failed gate with a new aggregation statistic;
- quietly mix exploratory and confirmatory results;
- run endless post-hoc architecture variants.

Any post-gate experiment must be labeled:

```text
EXPLORATORY_POST_GATE
```

Reuse existing Day 1–4 artifacts whenever they already answer a question. Verify hashes and run small reproduction spot-checks rather than wasting compute rerunning thousands of completed cells.

---

# 1. Candidate A — Day 3 / OrbitANOVA

## Core hypothesis

Equivalent schema spellings can induce meaningfully different complete fitted predictors.

The contribution is not merely:

> representations matter.

The proposed paper is:

```text
declare equivalent schema variations
→ measure prediction-space schema risk
→ decompose it into nuisance factors / interactions / randomness
→ include HPO/model-selection instability
→ use the diagnosis to select a targeted repair
→ validate that repair prospectively
```

Existing Day 3 controlled basis-conditioning results are prior evidence. Do not rerun the complete 30-dataset κ benchmark unless required for integrity checks.

The missing decisive evidence is **realistic schema risk + selection-path effects + audit-guided action transfer**.

---

## A1. Frozen schema-risk atlas

Choose **12–15 datasets** across:

- binary classification;
- multiclass classification;
- regression;
- small / medium / large;
- numerical-heavy;
- categorical-heavy;
- mixed;
- at least several official temporal/non-random splits where available.

Use at least:

```text
MLP
ResNet
FT-Transformer
TabM
```

Optional contrast baselines:

```text
CatBoost/XGBoost
TabPFN on eligible datasets
```

Use ≥3 paired seeds initially; use 5 on promoted subsets.

### Equivalent-schema factors

Only use transformations that preserve the declared prediction task.

Include where admissible:

1. feature/column order;
2. categorical ID relabeling;
3. class-ID relabeling with prediction realignment;
4. equivalent numerical units/scalings;
5. exact invertible within-field numerical basis changes;
6. exact equivalent nominal contrast bases;
7. exact equivalent ordinal bases;
8. full cyclic/Fourier equivalent bases where a genuine cyclic field exists.

Keep exact group/orbit transformations separate from merely representative equivalent families.

For every transformation record:

```text
factor_name
semantic_justification
invertibility/equivalence proof
output_alignment_rule
group_or_non_group
train_only_fit
```

---

## A2. Compute schema risk

For fitted predictor `P_z(x)` under schema spelling `z`:

```text
P_bar(x) = mean_z P_z(x)

SR = mean_(x,z) ||P_z(x) - P_bar(x)||²
```

For squared/Brier loss report the exact loss-valued interpretation.

Also report:

```text
schema risk
seed risk
split risk
schema × seed
schema × split
schema × architecture
```

Primary practical comparison:

```text
schema risk / seed risk
schema risk / ordinary HPO variance
```

The paper becomes much stronger if realistic representation choice is comparable to ordinary seed or tuning variability.

---

## A3. Functional attribution

Run balanced product/fANOVA decomposition over schema factors.

Measure:

```text
feature-order component
category-ID component
unit/basis component
ordinal/cyclic component
interactions
schema × seed
schema × split
```

Do not oversell fANOVA itself as novel.

The contribution is the schema-equivalence estimand and what the decomposition changes operationally.

---

## A4. HPO/model-selection path experiment

This is essential.

For every schema spelling, run the **same small frozen HPO menu**.

Example dimensions:

```text
learning rate
weight decay
width
dropout
numeric encoding option
optimizer option
```

Record:

```text
selected configuration h(z)
validation ranking
configuration-switch frequency
test predictor after selection
```

Compare:

```text
per-schema HPO
pooled HPO
reference-schema HPO
```

Quantify:

```text
Var_z[P_(z,h(z))]
Var_z[h(z)]
schema × menu
schema × split
```

Question:

> Can an otherwise stable learner become schema-sensitive because the tuning rule changes?

This should be tested across the full frozen panel, not just selected sensitive datasets.

---

## A5. Decisive audit-to-action experiment

Compare four diagnostic systems:

```text
1. Pairwise metamorphic violation count
2. Single-factor/PREF-style sensitivity
3. Undifferentiated total prediction variance
4. Full OrbitANOVA decomposition
```

Give every diagnostic the **same action library**:

```text
A0: abstain / baseline
A1: ordinary iid seed ensemble
A2: schema-view ensemble
A3: factor-balanced schema cover
A4: pooled HPO selection
A5: exact/canonical closure when admissible
A6: optimizer-side covariant repair when admissible
A7: FieldRiesz only when schema semantics make it admissible
```

Eligibility must be determined from metadata before outcomes.

Learn audit→action rules only on development datasets.

Evaluate using **outer dataset-level cross-fitting**:

```text
hold out all model families from dataset D
learn policy on all other datasets
use only train/validation diagnostics on D
choose action
score held-out test result
```

No test covariates or labels may influence action selection unless explicitly declared transductive.

### Day-3 promotion gate

Promote Day 3 as primary if:

- realistic schema risk is materially nonzero across multiple datasets and ≥3 neural families;
- schema variability is nontrivial relative to seed/HPO variability;
- selection-path instability appears across the frozen panel;
- OrbitANOVA attribution changes the chosen action on held-out cases;
- its chosen action beats simpler audits and/or a matched-resource iid seed ensemble on the held-out proper-risk frontier.

If the full decomposition never changes decisions or simpler diagnostics perform equally well, downgrade the contribution to an instrumentation/benchmark paper.

---

# 2. Candidate B — HeteroBag-3

## Core hypothesis

Different legitimate representations produce **structured model diversity** that can be more valuable than another random seed under identical compute.

The key comparison is not:

```text
T + Q > T
```

It is:

```text
T + T + Q > T + T + T
```

under exactly equal model count, training budget, and active parameters.

---

## B1. Second untouched prospective panel

Select **8–12 new datasets**, preferably:

```text
4–6 classification
4–6 regression
```

Do not reuse the current successful prospective panel as confirmation.

Architectures:

```text
MLP
ResNet
FT-Transformer
TabM
```

Run at least **3 independent seed triplets**.

### Primary candidate

Classification:

```text
T(seed A) + T(seed B) + Q(seed C)
```

Regression:

```text
T(seed A) + T(seed B) + Midrank(seed C)
```

Control:

```text
T(seed A) + T(seed B) + T(seed C)
```

Fixed 1/3 averaging.

---

## B2. Necessary controls

Also test:

```text
Q + Q + Q
Midrank + Midrank + Midrank
```

where appropriate.

Add a **representation-difference placebo**:

```text
T + T + transformed-T
```

where transformed-T changes coordinates but is not a distinct meaningful chart.

Purpose:

> determine whether benefit comes from meaningful representation heterogeneity or merely making members different.

Keep total compute matched.

---

## B3. Diversity mechanism

For every member pair calculate:

```text
prediction correlation
error correlation
disagreement
ambiguity / ensemble gain
representation disagreement
seed disagreement
```

Test whether:

```text
cross-representation disagreement
```

predicts ensemble improvement better than:

```text
same-representation seed disagreement.
```

Fit the predictor only on development datasets and test it on untouched datasets.

This is potentially the scientific contribution:

> representation diversity is a structured ensemble axis distinct from stochastic seed diversity.

---

## B4. HeteroBag promotion gate

Promote HeteroBag to serious paper candidate if the second prospective panel shows approximately:

```text
≥ 65% candidate wins
positive mean gain overall
positive mean in classification
positive mean in regression
positive mean on ≥ 70% of datasets
no architecture-specific catastrophic failure
gain survives multiple seed triplets
meaningful benefit beyond homogeneous alternate-representation ensembles
```

Desirable effect size:

```text
~0.5%+ average relative loss improvement
```

under exact equal compute.

Most importantly, require evidence that the gain is associated with **representation-induced error diversity**, not simply another model/seed.

If the prospective replication is weak or near zero:

```text
KEEP_AS_DAY3_CONSEQUENCE
NOT_STANDALONE_PAPER
```

---

# 3. Candidate C — FieldRiesz

## Core hypothesis

For fields with genuine semantic topology, the right field geometry can provide a useful chart-covariant neural prior.

The decisive comparison is:

```text
correct semantic geometry
vs
mass-only
vs
wrong geometry
vs
exact-spectrum randomized geometry
```

not merely:

```text
FieldRiesz vs PLE.
```

California alone is insufficient.

---

## C1. Freeze independent semantic replications

Find new datasets **without inspecting test outcomes** containing pre-existing semantic field structure.

Target at least three families if available:

```text
spatial coordinates
cyclic time/angle variables
ordinal variables
```

Prefer official chronological/spatial splits.

For every field, write its topology into the freeze file before training:

```text
nominal
path/ordered
cycle
spatial pair
```

Do not infer topology from target outcomes.

---

## C2. Matched experiment

For each eligible field compare:

```text
1. ordinary PLE
2. empirical mass only
3. correct semantic Riesz operator
4. node-permuted/wrong-semantic operator
5. exact M-isospectral randomized operator
```

For cyclic features additionally compare:

```text
correct ring
wrong path
permuted ring
```

For spatial pairs compare only mathematically audited constructions whose rank/reference-mass controls are valid.

Use:

```text
MLP
ResNet
TabM
```

and FT-Transformer if implementation is already stable.

≥3 seeds.

All parameter counts and training budgets matched.

---

## C3. Tau/strength selection

Never select `tau` from test results.

Use a frozen inner-validation grid such as:

```text
tau ∈ {0.1, 0.3, 1, 3, 10}
```

or preserve the existing frozen grid if already specified.

Report sensitivity to tau.

---

## C4. Negative-control fields

On each dataset include fields where no topology should help.

The semantic method should:

```text
activate/useful on semantically appropriate fields
not systematically improve arbitrary numerical fields
```

Otherwise the result is probably conditioning/capacity rather than semantic geometry.

---

## C5. FieldRiesz promotion gate

Promote FieldRiesz only if:

- correct semantic geometry beats mass-only;
- correct geometry beats wrong geometry;
- correct geometry beats exact-spectrum randomized geometry;
- this hierarchy replicates on at least **2 independent real datasets**;
- it appears in ≥2 architectures;
- dataset-level mean performance is positive;
- validation-only selection or conservative fallback works;
- the result survives rank/reference-mass audits.

A single spectacular California-like dataset does **not** pass.

If independent replication fails:

```text
STOP_PERFORMANCE_METHOD
KEEP_AS_MECHANISM/APPENDIX
```

---

# 4. Cross-direction comparison

After all frozen experiments complete, create:

```text
FINAL_DIRECTION_DECISION.md
final_direction_summary.csv
final_direction_summary.json
```

Score each candidate on:

| Criterion | Weight |
|---|---:|
| Novelty vs closest prior art | 25% |
| Prospective replication | 20% |
| Breadth across datasets | 15% |
| Breadth across architectures | 10% |
| Mechanistic clarity | 10% |
| Effect size / practical importance | 10% |
| Reviewer-defensible controls | 5% |
| Simplicity/coherence of paper story | 5% |

Use a 1–5 score for each.

Do not let one spectacular dataset dominate the decision.

---

# 5. Decision logic

Use the following priority rule.

## Choose Day 3 if

```text
schema risk is broad
+ selection/HPO paths are affected
+ attribution predicts useful held-out action
```

This is currently the incumbent.

## Choose HeteroBag if

```text
the second untouched replication succeeds strongly
+ equal-compute representation diversity repeatedly beats seed diversity
+ diversity diagnostics explain/predict the gain
```

If this happens, consider:

```text
Day 3 = phenomenon/mechanism paper
HeteroBag = method paper
```

rather than forcing both into one paper.

## Choose FieldRiesz if

```text
semantic-vs-isospectral hierarchy independently replicates
+ performance gains transfer across datasets and architectures
```

This has the highest mathematical novelty but requires the strongest replication before promotion.

---

# 6. Execution order

Run in this order:

```text
Phase 0 — freeze everything

Phase 1 — cheap screens
    OrbitANOVA realistic-schema atlas
    HeteroBag new-panel first seed-triplet
    FieldRiesz semantic replication first seeds

Phase 2 — only candidates passing screen
    full seeds
    all architectures
    robustness controls

Phase 3 — decisive novelty experiments
    OrbitANOVA audit→action transfer
    HeteroBag diversity prediction
    FieldRiesz exact-isospectral replication

Phase 4 — final portfolio decision
```

A candidate that clearly fails its frozen Phase-1 gate should not consume large Phase-2 compute.

---

# 7. Statistical/reporting rules

Primary unit for generalization is the **dataset**, not seeds.

Use:

```text
paired comparisons
dataset-level aggregation
dataset bootstrap / hierarchical bootstrap
confidence intervals
win rates
mean and median effects
```

Report failures as failures.

Do not treat seeds as independent dataset replications.

For every result distinguish:

```text
DEVELOPMENT
PROSPECTIVE_CONFIRMATORY
POST_GATE_EXPLORATORY
```

Do not replace a failed predeclared statistic with a more favorable one.

---

# 8. Reproducibility

Save for every fit:

```text
dataset ID + version
split hash
representation definition
transform hash
model configuration
parameter count
seed
training budget
optimizer
validation metric
selected checkpoint
test metric
runtime
GPU memory
git commit
environment
```

Store raw cell-level results before summaries.

Run unit tests for:

```text
exact representation equivalence
output realignment
invertibility/rank
no train/test leakage
equal-compute controls
parameter-count matching
schema-risk identities
FieldRiesz covariance
isospectral controls
```

---

# 9. Final report format

`FINAL_DIRECTION_DECISION.md` must begin with:

```text
PRIMARY DIRECTION:
SECONDARY DIRECTION:
STOP:
```

Then answer exactly:

1. Which direction has the strongest ICLR-level novelty?
2. Which has the strongest prospective empirical evidence?
3. Which has the clearest reviewer-defensible claim?
4. Which results replicated?
5. Which hypotheses were falsified?
6. What is the strongest paper title/one-sentence claim for each surviving direction?
7. What is the single biggest remaining rejection risk?
8. If only one project can receive the next month of work, which one and why?

End with one of:

```text
DECISION = COMMIT_DAY3
DECISION = COMMIT_HETEROBAG
DECISION = COMMIT_FIELDRIESZ
DECISION = DAY3_PRIMARY_HETEROBAG_SECONDARY
DECISION = NEED_MORE_EVIDENCE
```

Do not choose `NEED_MORE_EVIDENCE` merely because results are imperfect. Choose the best direction from the completed frozen evidence.

---

# Current prior before new experiments

Treat this only as the prior, not the conclusion:

```text
Day 3 / OrbitANOVA:
    strongest existing paper candidate

HeteroBag-3:
    strongest Day-4 performance result
    deserves one serious prospective replication

FieldRiesz:
    highest-risk / potentially highest-method-novelty direction
    currently lacks independent semantic replication
```

The purpose of this sprint is to either confirm that ranking or overturn it with prospective evidence.