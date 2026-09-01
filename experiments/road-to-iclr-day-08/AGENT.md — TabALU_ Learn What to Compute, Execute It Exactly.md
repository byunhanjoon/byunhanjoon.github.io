# AGENT.md — TabALU: Learn What to Compute, Execute It Exactly

## Mission

Develop and rigorously evaluate a new tabular-learning architecture based on the hypothesis:

> **Tabular neural networks should learn what computation to perform, which variables to trust, and under which conditions the computation applies — rather than learning to approximate arithmetic that the computer can already execute exactly.**

Working project name:

**TabALU — Tabular Adaptive Logic Unit**

The intended decomposition is:

\[
\text{observed table}
\rightarrow
\text{operand/noise inference}
\rightarrow
\text{regime inference}
\rightarrow
\text{program induction}
\rightarrow
\text{exact typed execution}
\rightarrow
\text{optional neural residual}
\]

The goal is **not** to force a positive result. The goal is to determine which parts of this hypothesis are actually useful, eliminate weak variants, and converge on the strongest defensible ICLR-level contribution.

Do not optimize for making the method look good. Optimize for discovering the truth quickly.

---

# 1. Core Research Questions

Test these hypotheses independently before combining them.

## H1 — Exact execution improves extrapolation

If the relationship is arithmetic/compositional, learning a discrete computational graph and executing it exactly should extrapolate better than an MLP/Transformer approximating the function.

Example:

\[
y = \frac{x_1x_2}{x_3}
\]

Train on:

\[
x_j \in [-2,2]
\]

Evaluate on:

\[
|x_j| \in [3,10].
\]

Primary question:

> Does exact execution preserve prediction quality when feature magnitudes move far outside the training support?

---

## H2 — Operand inference improves robustness to measurement noise

Observed features may be corrupted:

\[
x=z+\epsilon_x.
\]

The model should infer a cleaner operand representation before executing the program.

Test whether this improves:

- measurement-noise robustness;
- missingness robustness;
- corrupted-feature robustness;
- redundant-feature robustness.

Critically test whether the operand estimator simply becomes an unrestricted neural network.

Regularize it heavily.

---

## H3 — Regime-dependent programs improve conditional and temporal shift

Many tabular relationships are not governed by one global function.

Model:

\[
r=R_\theta(x,t,c)
\]

and:

\[
\hat y=P_r(z).
\]

Test:

- abrupt regimes;
- gradual regimes;
- categorical regimes;
- latent regimes;
- temporal change points;
- covariate-dependent regimes.

Compare against ordinary mixture-of-experts architectures.

The contribution is only interesting if **the experts being routed to are sparse executable programs rather than generic neural networks**.

---

## H4 — Separate program structure from changing coefficients

Test whether many problems can be modeled as:

\[
\hat y=a_rP(x)+b_r
\]

or more generally:

\[
P(x;\beta_r)
\]

where program structure remains invariant but coefficients vary with regime.

Compare:

1. one global program;
2. global structure + regime-specific constants;
3. separate program per regime;
4. full neural mixture of experts.

Determine whether structure/parameter separation improves:

- sample efficiency;
- extrapolation;
- temporal stability;
- interpretability.

---

## H5 — Typed execution helps heterogeneous tabular data

Do not convert every feature into an unconstrained embedding.

Implement typed operations.

### Numerical

Allow operators such as:

- add
- subtract
- multiply
- protected divide
- abs
- square
- sqrt
- log
- exp, if numerically stable
- min
- max
- clip

### Ordinal

Allow:

- comparison
- threshold
- rank
- monotonic transforms
- bounded difference

### Categorical

Allow:

- equality
- inequality
- membership
- learned category grouping
- categorical conditions controlling routing

### Datetime

Allow:

- elapsed duration
- hour
- weekday
- month
- day-of-year
- periodic sine/cosine
- time difference
- before/after threshold
- time-since-event

Test whether typed inductive biases outperform simply embedding everything.

---

## H6 — A penalized neural escape hatch preserves generality

Pure symbolic programs will fail on some real datasets.

Test:

\[
\hat y =
P(x)+\lambda(x)N_\psi(x).
\]

Strongly penalize neural usage.

Desired behavior:

\[
\lambda(x)\approx0
\]

whenever the symbolic/executable component is adequate.

Compare:

- pure program;
- pure neural;
- unpenalized program + residual;
- penalized program + residual;
- adaptive residual gate.

Determine whether this produces a useful continuum:

\[
\text{exact program}
\rightarrow
\text{program + small residual}
\rightarrow
\text{neural model}.
\]

---

# 2. Architecture

Implement TabALU as modular components.

Do not start with one giant model.

Directory structure:

```text
tabalu/
    data/
    synthetic/
    models/
        operand.py
        router.py
        executor.py
        program_search.py
        residual.py
        tabalu.py
    baselines/
    training/
    evaluation/
    analysis/
    configs/
    scripts/
    tests/
```

---

# 3. Exact Program Executor

Represent the program as a directed acyclic graph.

Each node chooses:

1. an operator;
2. operand A;
3. operand B when required;
4. optional scalar constants.

Example:

```text
h0 = x1
h1 = x3
h2 = multiply(h0, x2)
h3 = safe_divide(h2, h1)
output = h3
```

## Primitive library

Start small.

Initial library:

```text
identity
add
subtract
multiply
safe_divide
abs
square
safe_sqrt
safe_log
min
max
```

Do not begin with dozens of operators.

Complex operator libraries dramatically increase search difficulty.

Add operators only after ablations demonstrate a need.

---

# 4. Differentiable Program Discovery

During training, use soft/discrete selectors.

Evaluate:

- Gumbel-Softmax;
- straight-through Gumbel;
- sparsemax/entmax if practical;
- hard-concrete / L0 gates.

Each program node learns distributions over:

- operator;
- first input;
- second input.

For node \(v\):

\[
h_v =
\sum_o p_{v,o}\,
o(h_{a_v},h_{b_v}).
\]

Anneal temperature during training.

Eventually discretize:

\[
o_v=\arg\max_o p_{v,o}.
\]

After discretization:

1. compile the program;
2. remove unused nodes;
3. merge redundant operations;
4. fine-tune scalar constants only;
5. evaluate the compiled program separately.

**Always report both soft-model and compiled-model performance.**

If the soft model works but the compiled model collapses, the method has not actually learned an executable program.

---

# 5. Program Complexity Regularization

Use an objective of approximately:

\[
\mathcal L =
\mathcal L_{\text{task}}
+\lambda_{\text{nodes}}C_{\text{nodes}}
+\lambda_{\text{features}}C_{\text{features}}
+\lambda_{\text{entropy}}H(P)
+\lambda_{\text{noise}}C_{\text{operand}}
+\lambda_{\text{router}}C_{\text{router}}
+\lambda_{\text{residual}}C_{\text{NN}}.
\]

Penalize:

- program depth;
- number of active nodes;
- number of input features;
- number of regimes;
- router entropy where appropriate;
- neural residual magnitude;
- operand modification magnitude.

We want the smallest program that explains the data.

Track prediction accuracy against program complexity as a Pareto curve.

---

# 6. Operand / Noise Estimator

Observed variable:

\[
x_j.
\]

Latent clean operand:

\[
z_j.
\]

Begin with a conservative form:

\[
z_j=
g_j(x)x_j+
(1-g_j(x))\tilde{x}_j.
\]

Where:

- \(g_j\) = confidence in observed measurement;
- \(\tilde{x}_j\) = reconstructed estimate.

Also test:

\[
z_j=x_j+\Delta_j(x).
\]

But strongly penalize:

\[
\|\Delta_j\|.
\]

The operand network must not be allowed to secretly perform the prediction itself.

### Required ablation

Compare:

```text
raw x
bounded correction
confidence-gated reconstruction
unrestricted latent encoder
```

If only the unrestricted encoder works, the conceptual hypothesis is weakened.

---

# 7. Regime Router

Implement:

\[
p(r=k|x,t,c).
\]

Start with \(K \in \{1,2,4,8\}\).

Test:

### Soft routing

\[
\hat y=\sum_k p_kP_k(x)
\]

### Hard routing

\[
r=\arg\max_kp_k
\]

\[
\hat y=P_r(x).
\]

### Temporally regularized routing

Add a penalty encouraging neighboring timestamps to remain in the same regime unless evidence indicates change.

### Sparse regime usage

Prevent every sample from becoming its own regime.

Track:

- regime entropy;
- regime utilization;
- regime stability;
- alignment with true regimes on synthetic data.

---

# 8. Global Structure vs Regime Parameters

Implement three variants.

## Variant A

Independent program per regime:

\[
P_k(x).
\]

## Variant B

Shared graph, regime-dependent constants:

\[
P(x;\beta_k).
\]

## Variant C

Global graph with context-dependent constants:

\[
\beta=H_\theta(x,t,c).
\]

Variant C can become too flexible.

Regularize heavily.

The important scientific question is whether **stable structure + adaptive constants** provides a better inductive bias than changing the entire model.

---

# 9. Neural Residual

Implement:

\[
\hat y=P(x)+\lambda(x)N(x).
\]

Variants:

```text
no residual
fixed scalar residual gate
learned global scalar
per-row residual gate
unpenalized residual
strongly penalized residual
```

Track:

\[
E[\lambda(x)].
\]

The model should reveal how much of each dataset can be explained by executable structure.

This statistic may itself become an interesting empirical result.

---

# 10. Synthetic Benchmark Suite

Build a serious synthetic benchmark rather than a handful of toy equations.

Generate thousands of tasks.

Every task must store its ground-truth program.

## Program generation

Randomly generate programs with:

```text
depth: 1–6
number of relevant features: 1–8
number of irrelevant features: 0–20
number of regimes: 1–4
```

Reject numerically unstable programs.

Generate train/validation/test datasets independently.

---

# 11. Synthetic Data Conditions

Each program should be evaluated under multiple shifts.

## A. IID

Same feature distribution.

## B. Magnitude extrapolation

Example:

```text
train: [-2, 2]
test-1: [-4, 4]
test-2: [-8, 8]
test-3: [-16, 16]
```

Plot error against extrapolation distance.

---

## C. Measurement noise

Add:

- Gaussian noise;
- Laplace noise;
- heteroscedastic noise;
- outliers;
- feature-specific noise.

---

## D. Missing data

Randomly mask:

```text
0%
5%
10%
20%
40%
```

of entries.

---

## E. Irrelevant features

Add independent nuisance features.

---

## F. Collinearity

Add:

\[
x'_j=x_j+\epsilon
\]

and linear combinations of true operands.

---

## G. Engineered redundant features

Add features such as:

\[
x_1+x_2,
\quad
x_1x_2,
\quad
x_1/x_2.
\]

Test whether the model recovers stable computations despite multiple equivalent representations.

---

## H. Regime shift

Generate:

\[
y =
\begin{cases}
P_1(x), & r=1\\
P_2(x), & r=2
\end{cases}
\]

with regimes determined by:

- category;
- feature threshold;
- time;
- hidden latent process.

---

## I. Temporal drift

Generate coefficient drift:

\[
y=a(t)P(x)+b(t).
\]

Include:

- gradual drift;
- abrupt change;
- recurring seasonal regimes.

---

## J. Categorical interaction

Example:

\[
y=
\begin{cases}
x_1x_2,&c=A\\
x_1+x_3,&c=B\\
x_4/x_2,&c=C.
\end{cases}
\]

---

## K. Ordinal interaction

Create ordered categories:

```text
low < medium < high < critical
```

and make program parameters depend monotonically on rank.

---

## L. Datetime structure

Generate targets involving:

- elapsed time;
- weekday;
- cyclic hour;
- seasonal effects;
- change points.

---

# 12. Synthetic Metrics

For every model report:

### Prediction

- MAE
- RMSE
- \(R^2\)
- normalized RMSE

### Extrapolation

Performance versus extrapolation multiplier.

### Program recovery

When ground truth exists:

- exact operator recovery;
- graph edit distance;
- feature-selection precision/recall/F1;
- program depth error;
- coefficient error.

### Regime recovery

- regime accuracy after permutation matching;
- adjusted Rand index;
- normalized mutual information.

### Robustness

Performance versus:

- measurement noise;
- corruption probability;
- missingness;
- nuisance-feature count.

### Efficiency

- train time;
- inference time;
- parameter count;
- compiled-program operation count.

---

# 13. Required Baselines

Do not compare only against neural networks.

Use at least:

## Classical

- Linear/Logistic Regression
- Random Forest
- XGBoost
- CatBoost

## Tabular neural

Where feasible:

- MLP
- ResNet-style tabular network
- FT-Transformer
- TabM
- TabR
- TabPFN

Use current maintained implementations.

## Arithmetic / equation learning

Include representative methods such as:

- NALU/NMU-style arithmetic architecture;
- Equation Learner / EQL-style model;
- PySR or another competitive symbolic-regression system.

## Mixture baseline

Implement a conventional neural mixture-of-experts:

\[
\hat y=\sum_kp_kN_k(x).
\]

This comparison is critical for the regime-routing claim.

---

# 14. Experiment 1 — Arithmetic Extrapolation

This is the first go/no-go experiment.

Use approximately 100–500 generated programs.

Compare:

```text
MLP
FT-Transformer
TabM
TabPFN
EQL/NALU-like model
PySR
TabALU
```

Evaluate increasingly extreme OOD ranges.

Success criterion:

TabALU should show a clearly flatter error curve as extrapolation distance increases.

If TabALU does not substantially outperform ordinary neural models here, stop and debug the program-induction mechanism before doing larger experiments.

---

# 15. Experiment 2 — Program Recovery

Measure whether TabALU actually identifies the underlying graph.

Test:

- clean data;
- moderate target noise;
- measurement noise;
- irrelevant features;
- correlated features.

Produce a table:

```text
Method | Prediction | Feature F1 | Operator Accuracy | Exact Program %
```

This establishes whether prediction improvements correspond to meaningful structural recovery.

---

# 16. Experiment 3 — Noise / Operand Inference

Compare:

```text
TabALU raw
TabALU + operand correction
TabALU + confidence gating
TabALU + unrestricted encoder
```

Evaluate increasing measurement noise.

Key question:

> Does explicit operand inference provide robustness beyond simply increasing model capacity?

Plot:

```text
noise strength -> normalized prediction error
```

---

# 17. Experiment 4 — Regime Discovery

Generate known regime-switching problems.

Compare:

```text
single TabALU program
TabALU program MoE
neural MoE
single MLP
TabM
FT-Transformer
tree ensemble
```

Evaluate:

- prediction;
- regime recovery;
- extrapolation within each regime;
- unseen regime proportions.

Make sure the test set changes regime frequencies.

---

# 18. Experiment 5 — Structure vs Parameters

Generate:

\[
y=a_rP(x)+b_r.
\]

Compare:

```text
global program
independent program per regime
shared program + regime coefficients
context-conditioned coefficients
neural MoE
```

Determine whether shared structure provides better sample efficiency when each regime has few samples.

---

# 19. Experiment 6 — Heterogeneous Types

Construct synthetic tasks combining:

- continuous variables;
- categorical variables;
- ordinal variables;
- timestamps.

Compare:

```text
everything embedded
manual preprocessing + neural model
typed TabALU
typed TabALU without ordinal operators
typed TabALU without time operators
typed TabALU without categorical conditions
```

This should determine whether typed execution is genuinely useful or merely aesthetically appealing.

---

# 20. Experiment 7 — Neural Escape Hatch

Run increasingly non-symbolic target functions.

Examples:

\[
y=P(x)+\alpha N^*(x)
\]

for:

\[
\alpha \in
\{0,.1,.25,.5,1\}.
\]

Measure:

- predictive error;
- learned residual gate;
- program complexity.

Ideal result:

As \(\alpha\) increases, the model gradually uses more neural residual capacity rather than catastrophically failing.

Plot:

```text
true non-symbolic fraction
        vs
learned neural residual usage
```

This could become an important figure.

---

# 21. TabularMath-Style Evaluation

Use any available benchmark specifically testing arithmetic or computational extrapolation in tabular models.

The central metric should not only be \(R^2\).

Include:

- exact or near-exact computational accuracy;
- extrapolation outside training range;
- relative error;
- operator/program recovery where possible.

This experiment directly tests the paper's central motivation.

---

# 22. Real-World Temporal Evaluation

Use temporal tabular datasets such as those provided in TabReD or comparable temporal benchmarks.

Primary question:

> Does regime-aware executable structure degrade more gracefully under temporal distribution shift?

Compare:

```text
CatBoost
XGBoost
MLP
FT-Transformer
TabM
TabR
TabPFN where feasible
TabALU
TabALU without regime router
TabALU without temporal regularization
```

Metrics:

- IID validation;
- temporal test;
- performance degradation:

\[
\Delta =
\text{IID score}-
\text{temporal score}.
\]

Also analyze learned regime assignments over time.

---

# 23. General Tabular Benchmark

Only after the core hypotheses work, evaluate general usefulness.

Use a representative subset first.

Then scale toward TabArena-compatible evaluation where practical.

Include both:

- regression;
- classification.

Do not expect TabALU to dominate every dataset.

Important questions:

1. Is it competitive?
2. On which dataset properties does it win?
3. Can we predict when it should be used?
4. Does the neural residual prevent catastrophic failure?

Collect dataset meta-features:

- sample size;
- feature count;
- numeric fraction;
- categorical fraction;
- missingness;
- temporal structure;
- apparent smoothness;
- symbolic compressibility.

Analyze performance improvement as a function of dataset characteristics.

This may yield a stronger contribution than average rank alone.

---

# 24. Symbolic Regression Benchmark

Evaluate a restricted version on SRBench or representative symbolic-regression datasets.

Do not claim to beat specialized symbolic-regression systems unless results justify it.

The goal is to establish:

> TabALU remains capable of equation recovery while also handling heterogeneous/noisy/regime-dependent tabular data.

Compare prediction and expression complexity.

---

# 25. Ablation Matrix

The final paper must contain a disciplined ablation.

Full model:

```text
operand estimator
+ typed executor
+ sparse program discovery
+ regime router
+ structure/parameter separation
+ penalized residual
```

Remove individually:

```text
- operand estimator
- exact execution
- program sparsity
- regime routing
- typed operations
- temporal regularization
- shared structure
- residual penalty
- neural residual entirely
```

Also compare program depth:

```text
2
4
6
8
```

and regimes:

```text
1
2
4
8
```

---

# 26. Exact Execution Ablation

This is especially important.

Compare the same learned architecture with:

### A

Exact arithmetic primitives.

### B

Small MLP approximating each primitive.

### C

Entire program replaced by MLP.

If exact execution is the claimed reason for extrapolation, demonstrate it directly.

---

# 27. Program Discretization Ablation

Compare:

```text
soft program
hard/discrete program
compiled program
compiled program + coefficient fine-tuning
```

The hard compiled program should retain most of the useful performance.

Otherwise do not claim successful program induction.

---

# 28. Statistical Protocol

Use at least 5 seeds for important experiments.

Prefer 10 seeds for central synthetic claims if affordable.

Report:

- mean;
- standard deviation;
- bootstrap 95% confidence intervals.

Across datasets use paired comparisons.

Where appropriate use:

- Wilcoxon signed-rank test;
- corrected multiple comparisons;
- average ranks.

Do not claim superiority based only on a few favorable datasets.

---

# 29. Hyperparameter Fairness

For every competitive baseline:

- use recommended defaults;
- perform reasonable tuning;
- give comparable compute budgets;
- document search space.

Do not massively tune TabALU while leaving baselines untuned.

Maintain:

```text
configs/baseline_search_spaces.yaml
configs/tabalu_search_space.yaml
```

---

# 30. Failure Analysis

Actively search for cases where TabALU fails.

Generate:

- highly discontinuous functions;
- chaotic/noisy targets;
- high-dimensional smooth interactions;
- image-like random projections;
- extremely categorical tables;
- functions requiring primitives absent from the library.

For each failure identify whether the bottleneck is:

```text
program search
operator library
operand estimation
routing
optimization
insufficient neural fallback
```

A strong paper should explain the method's applicability boundary.

---

# 31. Interpretability Evaluation

Do not rely on qualitative examples alone.

Measure:

### Stability

Train with multiple seeds.

Compare recovered programs.

### Fidelity

Compare compiled program prediction against full-model prediction.

### Sparsity

Number of features and operations used.

### Regime coherence

Do routed samples share meaningful characteristics?

For temporal datasets plot:

```text
time
vs
regime probability
```

alongside target/performance changes.

---

# 32. Efficiency

Measure:

- training GPU-hours;
- inference throughput;
- memory;
- compiled executor latency;
- number of arithmetic operations.

An especially interesting result would be:

> Expensive neural program discovery during training, followed by extremely cheap compiled inference.

If achieved, quantify it carefully.

---

# 33. Curriculum Training

If direct joint training is unstable, test:

## Stage 1

Train program induction on clean synthetic equations.

## Stage 2

Add target noise.

## Stage 3

Add measurement noise.

## Stage 4

Add nuisance features.

## Stage 5

Add regimes.

## Stage 6

Add heterogeneous feature types.

Do not silently use curriculum.

Report whether it is necessary.

---

# 34. Optimization Diagnostics

Log:

```text
operator entropy
feature-selection entropy
number of active nodes
router entropy
residual-gate magnitude
operand corrections
gradient norms
program changes per epoch
```

If program induction collapses early to the wrong operator, experiment with:

- higher starting temperature;
- entropy regularization;
- delayed sparsification;
- warm-started coefficients;
- progressive node activation.

---

# 35. Suggested Development Order

Do not build everything simultaneously.

## Phase A — Core proof of concept

Implement:

```text
numeric inputs
single regime
no denoiser
no residual
small operator library
```

Demonstrate arithmetic extrapolation.

### GO condition

TabALU strongly outperforms ordinary neural baselines OOD while remaining reasonably competitive IID.

---

## Phase B — Structural recovery

Add:

- feature selection;
- discrete compilation;
- graph recovery metrics.

### GO condition

Recovered programs correlate strongly with ground truth and compiled performance remains high.

---

## Phase C — Noise

Add operand estimator.

### GO condition

Clear robustness improvement under measurement noise without an unrestricted encoder.

---

## Phase D — Regimes

Add program router.

### GO condition

Beats both single-program TabALU and conventional neural MoE under regime shift.

---

## Phase E — Time

Add temporal routing and structure/parameter separation.

### GO condition

Improves temporal-OOD degradation on synthetic and real temporal datasets.

---

## Phase F — Heterogeneous types

Add categorical, ordinal and datetime operators.

Keep only components that empirically help.

---

## Phase G — Neural fallback

Add residual.

Tune penalty to preserve program usage while improving general datasets.

---

## Phase H — Broad benchmarks

Run full real-data study.

---

# 36. Kill Criteria

Do not continue blindly.

## Kill or radically redesign exact execution if:

It does not provide a substantial extrapolation advantage on known arithmetic programs.

## Kill operand inference if:

It behaves like an arbitrary latent MLP and provides no robustness beyond additional model capacity.

## Kill regime routing if:

A conventional neural MoE performs equally well under regime and temporal shifts.

## Kill typed categorical/time operators if:

Normal embeddings perform equally well.

## Kill neural residual novelty if:

It simply turns TabALU into an MLP with an unused symbolic branch.

The final architecture should include only components supported by ablation evidence.

---

# 37. Strongest Possible Paper Narrative

Do not pre-commit to this narrative.

Earn it empirically.

Ideal final argument:

### Observation

Modern tabular neural networks approximate computations end-to-end even when parts of the data-generating mechanism consist of exact arithmetic or conditional logic.

### Problem

This creates poor:

- computational extrapolation;
- structural recovery;
- adaptation under regime shift.

### Principle

Separate:

\[
\text{inference of computation}
\]

from:

\[
\text{execution of computation}.
\]

### Method

TabALU performs:

1. uncertainty-aware operand inference;
2. sparse regime inference;
3. typed program induction;
4. exact deterministic execution;
5. optional penalized neural residual.

### Result

Compared with neural tabular models, symbolic regression, and neural mixtures of experts, TabALU combines:

- strong IID prediction;
- arithmetic extrapolation;
- robustness to noisy measurements;
- regime adaptation;
- interpretable compiled computation.

That would constitute the core ICLR story.

---

# 38. Minimum Evidence Needed for a Serious Submission

Do not start writing strong claims until we have:

### Claim 1 — Computational extrapolation

Hundreds of synthetic programs or a benchmark demonstrating strong OOD arithmetic gains.

### Claim 2 — Exact execution matters

Direct exact-vs-neural-execution ablation.

### Claim 3 — Regime routing matters

Controlled piecewise/temporal experiments against neural MoE.

### Claim 4 — Noise handling matters

Measurement-error experiment with clean ground truth.

### Claim 5 — Real-world relevance

Meaningful results on temporal and general tabular benchmarks.

### Claim 6 — Program induction is real

Compiled graphs preserve performance.

### Claim 7 — Components are justified

Full ablation table.

---

# 39. Figures to Produce

Create publication-quality versions automatically.

## Figure 1

Architecture diagram:

```text
Observed Table
     ↓
Operand Inference
     ↓
Regime Router
     ↓
Program Selector
     ↓
Typed Exact Executor
     ↓
Prediction
      ↘ optional Neural Residual
```

## Figure 2

Error vs extrapolation distance.

This is likely the main paper figure.

## Figure 3

Performance vs measurement noise.

## Figure 4

Regime probabilities over time.

## Figure 5

Prediction accuracy vs program complexity.

## Figure 6

Non-symbolic target fraction vs learned neural residual usage.

## Figure 7

Real benchmark average-rank / performance summary.

---

# 40. Main Tables

## Table 1

Synthetic arithmetic extrapolation.

## Table 2

Program and feature recovery.

## Table 3

Noise + regime robustness.

## Table 4

Temporal datasets.

## Table 5

General tabular benchmarks.

## Table 6

Ablations.

## Table 7

Efficiency and model complexity.

---

# 41. Claim–Evidence Matrix

Maintain:

```text
CLAIMS.md
```

Example:

| Claim | Required evidence | Status |
|---|---|---|
| Exact execution extrapolates | Synthetic + arithmetic benchmark | TODO |
| Operand inference handles feature noise | Noise sweep | TODO |
| Regime routing handles shifts | Synthetic + temporal | TODO |
| Typed operators help heterogeneous data | Typed ablation | TODO |
| Program remains executable | Compilation experiment | TODO |
| Residual preserves generality | General benchmarks | TODO |

Never put a claim into the paper if the corresponding evidence is weak.

---

# 42. Experiment Tracking

Every run must record:

```text
git commit
dataset
seed
split
model
hyperparameters
training time
validation metric
test metric
OOD metric
program complexity
regime count
residual usage
compiled metric
```

Save to machine-readable CSV/Parquet even if using W&B.

Never rely exclusively on W&B.

---

# 43. Reproducibility

Set:

- Python seeds;
- NumPy seeds;
- PyTorch seeds;
- deterministic splits.

Store dataset-generation seeds.

Synthetic tasks must be exactly regenerable.

Create:

```text
scripts/reproduce_main_tables.sh
scripts/reproduce_main_figures.sh
```

or equivalent Python entry points.

---

# 44. Paper Development Files

Continuously maintain:

```text
paper/
    ABSTRACT_DRAFT.md
    INTRODUCTION_NOTES.md
    RELATED_WORK.md
    METHOD.md
    EXPERIMENTS.md
    RESULTS.md
    LIMITATIONS.md
    CLAIMS.md
    FIGURES.md
    PAPER_OUTLINE.md
```

Do not wait until experimentation finishes to document results.

After every major experiment append:

```text
Hypothesis
Setup
Result
Interpretation
Possible confound
Next experiment
```

to:

```text
EXPERIMENT_LOG.md
```

---

# 45. Negative Results

Create:

```text
NEGATIVE_RESULTS.md
```

Record failed ideas.

Examples:

- operators that destabilize training;
- useless denoising variants;
- excessive regime counts;
- poor discretization methods;
- datasets where symbolic bias hurts;
- failure of categorical operators.

These are useful for deciding what the final paper should actually claim.

---

# 46. Do Not Cherry Pick

Never:

- discard failed seeds;
- select datasets after seeing results;
- report only favorable extrapolation ranges;
- compare against weak baseline settings;
- hide compiled-program degradation;
- hide datasets where the residual dominates.

If the method works only in a particular domain, narrow the paper's claim.

A focused true contribution is better than a broad false one.

---

# 47. Research Pivot Rules

Possible outcomes:

## Outcome A — Excellent arithmetic extrapolation + strong temporal adaptation

Proceed with full TabALU paper.

## Outcome B — Excellent arithmetic extrapolation, weak general tabular performance

Reframe around:

> Computational extrapolation in tabular learning.

The neural residual becomes secondary.

## Outcome C — Strong regime adaptation, mediocre equation recovery

Reframe around:

> Executable program mixtures for non-stationary tabular prediction.

## Outcome D — Strong noise robustness

Potential framing:

> Learning latent operands rather than arbitrary representations.

## Outcome E — No clear advantage over symbolic regression or neural MoE

Stop the project or substantially redesign it.

Do not force an ICLR paper out of a null result.

---

# 48. Reviewer Attacks We Must Preempt

Before finalizing, explicitly answer these.

### “Isn't this just symbolic regression?”

Need comparison against strong symbolic regression and evidence of capability on:

- heterogeneous tables;
- noisy operands;
- regimes;
- time;
- neural fallback.

### “Isn't this just NALU/EQL?”

Need show:

- tabular setting;
- explicit program compilation;
- regime-conditioned structure;
- operand uncertainty;
- heterogeneous types;
- broader empirical study.

### “Isn't the regime component just MoE?”

Compare directly against neural MoE.

### “Does the denoiser secretly perform the prediction?”

Use bounded corrections and strong regularization.

Visualize corrections.

### “Does the hard program actually work?”

Report compiled-program performance.

### “Why not CatBoost?”

Include CatBoost.

### “Why not TabPFN/TabM?”

Include them where computationally feasible.

### “Does this work outside synthetic equations?”

Real-data experiments are mandatory.

---

# 49. Final ICLR Paper Target

Potential title:

**Learning What to Compute: Exact Program Execution for Adaptive Tabular Prediction**

Alternatives:

**TabALU: Learning Programs Instead of Arithmetic for Tabular Prediction**

**Neural Control, Exact Execution: Adaptive Program Induction for Tabular Learning**

Possible abstract structure:

1. Tabular models approximate arithmetic and conditional relationships end-to-end.
2. This can produce strong interpolation but fragile computational extrapolation.
3. Introduce TabALU, which separates inference from execution.
4. Neural modules infer operands, regimes and sparse typed programs.
5. Programs are compiled and executed using exact deterministic operators.
6. Experiments evaluate arithmetic extrapolation, measurement noise, regime shift, temporal shift and real tabular prediction.
7. State only the empirical advantages actually demonstrated.

---

# 50. Immediate Work

Begin now.

Do not start with TabArena.

Perform in this order:

```text
1. Build synthetic program generator.
2. Build typed exact executor.
3. Implement differentiable operator/input selection.
4. Demonstrate program compilation.
5. Run arithmetic extrapolation study.
6. Compare against MLP, tree model, symbolic regression and arithmetic-learning baseline.
7. Produce extrapolation curves.
8. Add operand corruption.
9. Add operand estimator.
10. Produce noise curves.
11. Add regime generator.
12. Implement router.
13. Compare against neural MoE.
14. Add temporal regimes.
15. Test shared program + adaptive coefficients.
16. Add categorical/ordinal/datetime operators.
17. Add penalized neural residual.
18. Run real temporal datasets.
19. Run representative general tabular datasets.
20. Run full ablations.
21. Scale experiments only after positive pilot results.
22. Generate all tables and figures automatically.
23. Build the final claim-evidence matrix.
24. Draft the paper around the strongest surviving hypothesis.
```

---

# 51. Autonomous Agent Behavior

You are the primary research engineer.

Do not repeatedly ask the user what to do next.

Make reasonable technical decisions autonomously.

When something fails:

1. diagnose it;
2. run the smallest experiment capable of distinguishing explanations;
3. modify the implementation;
4. document what changed;
5. rerun;
6. preserve the negative result.

Favor experiments over speculation.

Favor cheap falsification before expensive benchmarks.

Favor simple variants before complex variants.

Do not spend large amounts of compute optimizing an architecture whose central hypothesis has not passed the synthetic tests.

---

# 52. Final Success Standard

The project is worth pushing as an ICLR paper only if we can establish at least one major result that existing general tabular neural models do not naturally provide.

The strongest target is:

\[
\boxed{
\text{neural-level interpolation}
+
\text{symbolic-level extrapolation}
+
\text{regime adaptation}
}
\]

Ideally with:

\[
\boxed{
\text{measurement-noise robustness}
+
\text{compiled interpretable programs}
}
\]

The scientific contribution is **not that CPUs and GPUs can do arithmetic**.

The contribution must be evidence for a stronger statement:

> **For an important class of tabular problems, separating neural inference of computation from deterministic execution is a better inductive bias than end-to-end function approximation.**

Every experiment should either strengthen or falsify that statement.