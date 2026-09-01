# AGENT.md — Rapid Feasibility Sweep for Next-Gen Tabular AI Research

## Mission

Determine which of the following research directions has enough empirical signal to justify serious follow-up work:

1. **Identification-aware causal relational foundation models**
2. **Semantics-aware synthetic pretraining for relational/tabular foundation models**
3. **Prior misspecification, calibration, and failure detection for amortized causal/tabular models**

You are conducting a **preliminary scientific feasibility study**, not building production-quality implementations.

The entire investigation must finish within **3.5 hours wall-clock**.

Your goal is to find surprising, robust failure modes or unusually strong proof-of-concept improvements.

Do not optimize benchmark scores extensively.

Do not train large models.

Do not spend more than 20 minutes getting any external repository to work.

Prefer controlled synthetic experiments where the ground truth is exactly known.

---

# Final deliverable

Create:

```text
research_sweep/
├── RESULTS.md
├── results.json
├── direction_1_causal_relational/
├── direction_2_semantics/
├── direction_3_prior_shift/
└── figures/
```

`RESULTS.md` is the primary output.

It must end with a ranking:

```text
1. <direction>
2. <direction>
3. <direction>
```

For each direction assign:

- empirical signal: 1–5
- novelty potential: 1–5
- tractability: 1–5
- probability that deeper experiments reveal something publishable: 0–100%
- recommended next experiment
- verdict: KILL / MAYBE / PURSUE

Be willing to output **KILL**.

Finding that an idea does not work is a successful result.

---

# Global constraints

## Time

Hard wall-clock limit:

**3 hours 30 minutes**

Suggested allocation:

```text
00:00–00:20 environment + scaffolding
00:20–01:10 Direction 1
01:10–02:00 Direction 2
02:00–02:50 Direction 3
02:50–03:20 robustness checks
03:20–03:30 synthesis + final ranking
```

If parallel execution is easy, run independent seeds concurrently.

Do not let one direction consume time belonging to the others.

---

# Compute assumptions

Assume at most:

- 1 commodity GPU, OR
- CPU-only environment

Experiments must remain viable without a GPU.

Preferred stack:

```text
python
numpy
pandas
scikit-learn
scipy
pytorch
matplotlib
```

Optional if immediately installable:

```text
xgboost
catboost
sentence-transformers
networkx
```

Avoid heavyweight dependencies unless already installed.

---

# Scientific philosophy

We are looking primarily for:

1. **clear failure modes of current paradigms**
2. **controlled examples where the proposed direction fixes them**
3. **effects that survive random seeds**
4. **phenomena that cannot be dismissed as ordinary hyperparameter tuning**

An interesting negative result is more valuable than a tiny accuracy improvement.

Prefer experiments with interpretable ground truth.

---

# Common experiment rules

For every meaningful result:

- run at least 3 random seeds if runtime permits
- report mean and standard deviation
- preserve raw metrics in `results.json`
- save important plots
- compare against at least one deliberately stupid baseline
- check for leakage
- include a shuffled-label or shuffled-semantics sanity check where applicable

Do not claim significance from a single lucky run.

---

# Direction 1 — Identification-Aware Causal Relational Models

## Scientific question

Can a learned tabular/relational model distinguish:

> "I can estimate this causal quantity"

from

> "This quantity is not identifiable from the available observational information and assumptions"?

The key hypothesis is that ordinary predictive models will confidently answer causal questions even when two observationally indistinguishable worlds imply different intervention effects.

If this phenomenon is strong, it motivates models that explicitly reason about identifiability.

---

## Test 1A — Observational equivalence trap

Construct pairs of structural causal models that produce nearly identical observational distributions but imply different intervention effects.

Start with simple examples.

### World A

```text
U → X
U → Y
X → Y
```

with hidden confounding.

### World B

Construct another SCM producing approximately the same observational relationship between `X` and `Y`, but a materially different:

```text
ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
```

Generate many datasets from both worlds.

Do not expose the latent confounder.

Train a predictor to infer ATE from observational samples.

Candidate input representation:

For each dataset calculate:

```text
mean(X)
mean(Y)
var(X)
var(Y)
cov(X,Y)
correlation(X,Y)
simple regression coefficient
selected quantiles
```

Use:

- linear regression
- random forest
- small MLP

Target:

```text
true ATE
```

### Desired finding

Two datasets can have nearly identical observational statistics while requiring contradictory causal answers.

Measure:

```text
prediction error
prediction confidence / ensemble variance
distance between observational summaries
difference in true ATE
```

Create a scatterplot:

```text
observational-distance vs causal-effect-distance
```

Highlight pairs with:

```text
low observational distance
high causal difference
```

This is the central sanity check.

---

## Test 1B — Explicit identifiability target

Generate approximately:

```text
5,000–20,000
```

small synthetic datasets.

Mix cases such as:

1. randomized treatment
2. observed confounding
3. hidden confounding
4. valid instrumental variable
5. invalid instrumental variable
6. mediator observed
7. collider observed

Attach metadata describing which assumptions are available to the learner.

Example feature:

```text
assumption_vector = {
    randomized: 0/1,
    confounders_observed: 0/1,
    valid_iv_available: 0/1,
    ...
}
```

Ground-truth target:

```text
IDENTIFIABLE
NOT_IDENTIFIABLE
```

Train a small classifier using:

```text
observational statistics only
```

and compare with:

```text
observational statistics + explicit assumption metadata
```

The hypothesis is:

```text
observational-only identification accuracy ≈ impossible in ambiguous cases

but

observational + assumptions → high identification accuracy
```

This demonstrates why causal assumptions must be explicit model inputs.

---

# Add relational structure

If enough time remains, construct a minimal relational setting.

Example:

```text
CUSTOMERS
customer_id
region
latent_income

ORDERS
order_id
customer_id
discount
purchase_value
```

Generate causal effects where:

```text
customer properties → discount
customer properties → purchase
discount → purchase
```

Create a second schema:

```text
PATIENTS
patient_id
severity

VISITS
visit_id
patient_id
treatment
outcome
```

with an analogous causal mechanism.

Train on the first schema and evaluate equivalent inference on the second.

Represent relations using simple aggregated parent-table features.

Do NOT build a sophisticated graph transformer.

The question is merely:

> Is cross-schema causal generalization measurably harder than ordinary relational prediction?

---

# Direction 1 success criteria

Strong signal if ANY of these occur:

### A

Models confidently predict different causal quantities despite observational non-identifiability.

### B

Providing explicit causal assumptions dramatically improves identification decisions.

### C

A model trained on predictive tasks performs well observationally but catastrophically on interventions.

### D

Cross-schema relational transfer produces a qualitatively different causal failure than ordinary IID prediction.

---

# Direction 1 kill criterion

Lower priority if:

- ambiguity is trivial and already completely captured by elementary causal tests
- learned models add no interesting behavior beyond obvious theoretical impossibility
- there is no plausible foundation-model-specific research question

---

# Direction 2 — Semantics-Aware Relational Pretraining

## Scientific question

Do table names, column names, relationship roles, and textual schema descriptions provide a substantial signal for **generalization to unseen database schemas**?

We want to know whether relational models are leaving major performance on the table by treating:

```text
buyer_id
seller_id
prescriber_id
patient_id
sender_id
receiver_id
```

as merely interchangeable foreign keys.

---

# Core synthetic benchmark

Generate many small relational problems sharing the same underlying abstract task but using different schemas.

Example abstract relation:

```text
ENTITY_A --role_1--> TRANSACTION <--role_2-- ENTITY_B
```

Create semantic variants.

### Commerce

```text
buyer_id
seller_id
price
fraud
```

### Messaging

```text
sender_id
receiver_id
message_count
spam
```

### Medicine

```text
doctor_id
patient_id
dosage
adverse_event
```

The target-generating mechanism should depend on **relation role**.

Example:

```text
risk =
  + 2 * source_entity_risk
  - 1 * destination_entity_risk
  + noise
```

Thus swapping source and destination changes the prediction.

---

# Models

Implement the cheapest possible comparison.

## Model A — structure only

Encode columns/relations with arbitrary IDs.

No access to names.

## Model B — semantics aware

Provide embeddings derived from schema text.

Preferred hierarchy:

1. frozen sentence-transformer embeddings, if immediately available
2. pretrained text embedding already present locally
3. TF-IDF / hashed text features
4. manually generated semantic role vectors as final fallback

Do not spend substantial time downloading models.

Concatenate schema embedding with numeric features.

Use identical downstream architecture for Models A and B.

---

# Generalization split

Critical requirement:

Training and testing must use **different schemas**.

Example:

Train:

```text
buyer → seller
sender → receiver
```

Test:

```text
doctor → patient
lender → borrower
teacher → student
```

Do not allow exact column names to overlap unnecessarily.

---

# Experiment 2A — Semantic transfer

Measure held-out-schema performance.

Compare:

```text
structure only
vs
semantic schema embeddings
```

Metric:

classification:

```text
AUROC
accuracy
log loss
```

regression:

```text
RMSE
R²
```

---

# Experiment 2B — Shuffle semantics

Randomly permute column names while preserving values and structure.

Evaluate the semantics-aware model.

Expected:

```text
real semantics > shuffled semantics
```

If performance does not degrade, semantic embeddings probably are not contributing meaningfully.

---

# Experiment 2C — Role reversal

Construct examples where:

```text
buyer_id ↔ seller_id
sender_id ↔ receiver_id
```

are swapped.

Measure whether the model notices that the causal/predictive meaning changed.

This is particularly important.

A semantics-aware model should be substantially more robust than a structure-only model.

---

# Experiment 2D — Few-shot unseen schema

Take an unseen schema.

Evaluate performance with:

```text
0
10
50
100
```

labeled rows.

Ask whether semantic information gives a much better zero-shot or few-shot starting point.

---

# Direction 2 success criteria

Strong signal if:

```text
semantic model improves unseen-schema performance by >5 percentage points
```

or

```text
semantic model cuts error by >15%
```

without helping much on IID same-schema validation.

That pattern is particularly interesting because it indicates semantics specifically improve transfer rather than merely capacity.

Also strong:

```text
structure-only model fails badly under role reversal
semantic model remains stable
```

---

# Direction 2 kill criteria

Lower priority if:

- gains vanish under proper held-out schemas
- gains disappear after controlling model capacity
- improvement comes entirely from lexical overlap
- semantics add little compared with 10–50 labeled target examples

---

# Direction 3 — Prior Misspecification and Calibration

## Scientific question

When an amortized tabular/causal learner is trained over a synthetic task distribution, what happens when the real data-generating process falls outside that prior?

Particularly:

> Can it become confidently wrong?

And:

> Can prior mismatch be detected before ground-truth labels or counterfactuals are available?

---

# Build a tiny amortized learner

Do NOT reproduce TabPFN.

Build a toy analogue.

Generate thousands of small datasets.

Each dataset contains:

```text
n = 32–256 rows
```

with:

```text
X
T
Y
```

Train a dataset-level estimator to predict ATE.

Architecture:

```text
row MLP
→ mean pooling
→ dataset embedding
→ ATE prediction
```

This is a simple DeepSets-style estimator.

Optional uncertainty:

train an ensemble of 3–5 models.

---

# Training prior P

Sample training datasets from restricted mechanisms.

Example:

```text
X ~ Normal(0,1)

T ~ Bernoulli(sigmoid(aX))

Y = τT + βX + ε

τ ~ Uniform(-2,2)
β ~ Uniform(-2,2)
Gaussian noise
linear treatment effects
```

---

# Evaluate progressively shifted priors

## Shift 0 — IID

Same prior as training.

## Shift 1 — covariate shift

```text
X ~ StudentT
```

or mixture distribution.

## Shift 2 — treatment mechanism shift

Use nonlinear propensity:

```text
P(T=1|X) = sigmoid(aX + bX²)
```

## Shift 3 — outcome nonlinearity

```text
Y = τT + β sin(X) + ε
```

## Shift 4 — heterogeneous treatment effect

```text
τ(X) = τ0 + τ1 X
```

## Shift 5 — hidden confounding

Introduce:

```text
U → T
U → Y
```

without exposing `U`.

## Shift 6 — combinations

Combine multiple shifts.

---

# Measurements

For each shift calculate:

```text
ATE MAE
ATE RMSE
ensemble predictive variance
coverage if intervals are available
```

Plot:

```text
shift severity vs error
shift severity vs uncertainty
```

The key failure signature is:

```text
error ↑↑
uncertainty ≈ unchanged
```

That means the amortized estimator becomes confidently wrong.

---

# Misspecification detector

Construct cheap dataset-level summary features:

```text
means
variances
skewness
kurtosis
correlations
quantiles
propensity statistics
simple regression coefficients
```

Fit an anomaly detector using training-prior datasets:

```text
IsolationForest
```

or:

```text
Mahalanobis distance
```

Then ask:

> Does prior-distance predict estimator error?

Calculate:

```text
Spearman correlation(
    OOD_score,
    absolute_ATE_error
)
```

Also compute AUROC for predicting:

```text
absolute error > threshold
```

---

# Stronger variant

Compare:

```text
ensemble uncertainty
OOD score
ensemble uncertainty + OOD score
```

for failure prediction.

Potentially interesting result:

```text
model uncertainty alone fails,
but prior-distance detects failures.
```

That suggests tabular FMs may need explicit prior-mismatch diagnostics.

---

# Direction 3 success criteria

Very strong signal:

```text
OOD error > 2× IID error
```

while:

```text
uncertainty increases <25%
```

and an external mismatch score substantially predicts failure.

Also strong:

```text
Spearman(OOD score, error) > 0.5
```

across heterogeneous shifts.

---

# Direction 3 kill criteria

Lower priority if:

- ordinary ensemble uncertainty already perfectly tracks failure
- prior mismatch is trivial to detect
- failures occur only under absurd distributions
- conclusions depend heavily on one synthetic prior

---

# Cross-direction robustness phase

After all three experiments have run, spend remaining time attacking the strongest findings.

For the top two directions:

1. run additional seeds
2. vary dataset size
3. vary noise
4. reduce model capacity
5. test one alternative data-generating process
6. run shuffled/randomized controls

Try actively to destroy the effect.

A research direction should rise in ranking if the central phenomenon survives attempts to eliminate it.

---

# Scoring rubric

Score every direction from 1–5.

## 1. Empirical signal

```text
1 = nothing interesting
2 = weak/inconsistent
3 = measurable phenomenon
4 = strong and robust
5 = striking qualitative failure or improvement
```

## 2. Novelty potential

```text
1 = standard benchmark optimization
2 = minor extension
3 = potentially interesting
4 = clearly underexplored
5 = attacks a fundamental limitation
```

## 3. Tractability

```text
1 = enormous infrastructure required
2 = difficult
3 = manageable
4 = straightforward research program
5 = high leverage with modest compute
```

## 4. Scientific depth

Ask whether the question concerns:

```text
generalization
identification
uncertainty
semantics
causal reasoning
epistemic limits
```

rather than merely predictive score.

---

# Decision heuristic

Prefer a direction exhibiting a **qualitative discontinuity**.

Examples:

```text
prediction remains accurate while causal inference collapses

model remains highly confident when the answer is mathematically unidentified

relational model succeeds until foreign-key roles are semantically reversed

amortized model uncertainty remains low while prior shift causes catastrophic error
```

These are substantially more promising than:

```text
our model improves AUROC from .874 to .881
```

---

# RESULTS.md format

Use exactly this approximate structure.

```markdown
# Rapid Tabular-AI Research Sweep

## Executive conclusion

Best direction:
<NAME>

Why:
<3–6 sentences>

Most surprising observation:
<observation>

Biggest concern:
<concern>

---

# 1. Identification-Aware Causal Relational Models

## Hypothesis

...

## Experiments run

...

## Results

| experiment | baseline | proposed | effect |
|---|---:|---:|---:|

## Key figure

...

## Interpretation

...

## Failure modes / caveats

...

## Verdict

PURSUE / MAYBE / KILL

---

# 2. Semantics-Aware Relational Pretraining

...

---

# 3. Prior Misspecification and Calibration

...

---

# Comparison

| direction | empirical | novelty | tractability | depth | publishability |
|---|---:|---:|---:|---:|---:|
| causal relational | | | | | |
| semantics | | | | | |
| prior shift | | | | | |

# Ranking

1. ...
2. ...
3. ...

# Recommended next 7-day experiment

Describe ONE experiment for the winning direction.

Include:

- precise hypothesis
- required implementation
- datasets/generators
- baselines
- metric
- falsification criterion
- expected compute
```

---

# Research hygiene

Never quietly discard failed experiments.

Record:

```text
seed
parameters
runtime
dataset sizes
model sizes
errors/exceptions
```

If an external repository fails to install, document that and use a minimal local approximation.

Do not reinterpret noisy outcomes as success.

If evidence is inconclusive, say:

```text
INCONCLUSIVE
```

rather than forcing a recommendation.

---

# What NOT to do

Do not:

- reproduce giant benchmark suites
- train a transformer from scratch
- spend hours tuning hyperparameters
- use dozens of datasets
- chase SOTA
- use an LLM API as the primary experimental model
- confuse prediction with causal identification
- infer hidden causal assumptions from observational data and label them as ground truth
- let semantic train/test leakage invalidate Direction 2
- evaluate prior shift using only one arbitrarily chosen OOD distribution

---

# Preferred outcome

The ideal conclusion is not necessarily:

> "Direction X has the highest accuracy."

It is something more like:

> "Current amortized models remain extremely confident under a class of prior shifts that increases causal error 4×, and a simple prior-distance detector anticipates the failure."

or:

> "Predictive models achieve excellent held-out likelihood while giving incompatible causal answers on observationally indistinguishable relational SCMs."

or:

> "Structural relational models fail nearly catastrophically when foreign-key roles are reversed, while adding schema semantics restores most zero-shot transfer."

Those are phenomena worth turning into papers.

---

# Autonomous behavior

Proceed without asking the user questions.

Make reasonable implementation choices.

If an experiment fails technically, simplify it rather than spending the entire budget debugging.

Continuously prioritize **scientific information gained per minute**.

At the hard wall-clock limit, stop experiments and write `RESULTS.md` from whatever evidence has been collected.

The final answer should identify one direction to pursue next and explain exactly what observation drove that recommendation.