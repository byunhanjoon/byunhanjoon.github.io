# AGENT.md — Kill Experiment 2: Semantic Symmetries and Representation Orbits in Tabular Models

## Mission

Test whether modern tabular models give materially different predictions for **information-equivalent / semantics-preserving representations of the same table**, especially transformations associated with feature semantics rather than ordinary column permutation.

Central question:

> Does a model predict the same function when the data are expressed through a different but equivalent representation?

We need evidence of:

1. significant prediction/schema sensitivity;
2. sensitivity beyond already-studied column permutation;
3. a simple feature-type-aware repair or orbit method that reduces the sensitivity and preferably improves predictive performance.

Final required output: **`results.md`** plus raw CSV/JSON and plots.

---

# 1. Models

Required:

1. **TabICLv2**
2. **TabPFN-2.6**
3. **CatBoost**

Strongly preferred if installation is practical:

4. **TabM** through TabArena/current implementation

Optional:

5. RealMLP / FT-Transformer

Do not wait on optional models if installation blocks experiments.

Use fixed seeds:

```text
0, 1, 2
```

For TFMs use frozen checkpoints.

For trainable models retrain separately under each representation using the same train/validation split and seed.

---

# 2. Real datasets

Use these because they expose different semantic feature types.

### adult — classification

Focus:

- nominal categoricals
- numerical scale/affine transforms

### bank-marketing — classification

Focus:

- nominal variables
- month/time-like categorical representation if present
- numerical recoding

### diamonds — regression

Known ordinal categories:

```text
cut:
Fair < Good < Very Good < Premium < Ideal

color:
J < I < H < G < F < E < D

clarity:
I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF
```

Focus:

- ordinal recoding
- nominal-vs-ordinal treatment
- numerical units

### bike-sharing — regression

Use a version containing hour/month/weekday if possible.

Known cyclic variables:

```text
hour: period 24
weekday: period 7
month: period 12
```

### california_housing — regression

Focus:

- numerical unit/affine transformations
- rank/monotone transformations

### wine-quality-red or equivalent — regression

Focus:

- numerical transformations

Resolve datasets via sklearn/OpenML and record exact OpenML IDs/versions.

Add 2 more TabArena datasets only if time permits.

---

# 3. Synthetic sanity dataset

Before real experiments, create one synthetic dataset with:

- one nominal feature
- one ordinal feature
- one cyclic feature
- one ratio numerical feature
- one interval numerical feature
- noise variables

Generate target from semantic structure.

Example:

\[
y
=
f_{nominal}(x_n)
+0.7\,rank(x_o)
+\sin(2\pi x_c/P)
+0.5\log(1+x_r)
+0.3x_i
+\epsilon.
\]

Use this only to verify transformation code and confirm that every transformation preserves the true target-generating function.

Do not treat synthetic success as paper evidence.

---

# 4. Evaluation philosophy

For every model/dataset:

1. train/fit on representation \(T(D_{train})\);
2. predict \(T(D_{test})\);
3. compare against predictions from the canonical/original representation.

Transform train/validation/test consistently.

Never alter target labels.

For every transformation generate at least:

```text
8 random orbit members
```

Use fewer only if compute requires it.

---

# 5. Metrics

## Classification prediction disagreement

For original probabilities \(p\) and transformed probabilities \(p_T\):

- mean absolute probability difference
- Jensen-Shannon divergence
- predicted-label flip rate

Also task performance:

- log loss
- ROC-AUC
- accuracy

Define orbit performance span:

\[
\Delta_{orbit}
=
\max_T L(T)-\min_T L(T).
\]

## Regression disagreement

\[
D_{pred}
=
RMSE(\hat y,\hat y_T)/std(y_{test}).
\]

Also:

- Pearson prediction correlation
- Spearman prediction correlation
- RMSE
- MAE
- orbit RMSE span

Report both **prediction instability** and **performance instability**. Prediction disagreement is important even when average accuracy is similar.

---

# 6. Transformation families

## T0. Column permutation — CONTROL ONLY

Randomly permute feature columns.

Run 8 permutations.

This verifies previous findings but is NOT the intended novelty.

---

## T1. Nominal category relabeling

For every nominal feature, randomly permute category identities.

Examples:

```text
red -> 7
blue -> 2
green -> 9
```

or random opaque strings.

Apply identical mapping to train/validation/test.

The mapping must remain bijective.

Test:

- one feature at a time
- all nominal features simultaneously

Do not accidentally convert nominal categories into meaningful numeric magnitude without preserving categorical metadata when the model supports categorical metadata.

Record both:

- native-categorical pipeline
- numeric-code pipeline where relevant

---

## T2. Numerical unit transformations

For numerical feature \(x\):

### scaling

\[
x'=ax,\quad a>0
\]

Sample:

\[
\log_{10}a\sim U(-1,1).
\]

### affine recoding

\[
x'=ax+b
\]

with:

\[
b=c\cdot std(x),\qquad c\sim U(-3,3).
\]

Test:

- one numerical column at a time
- all numerical columns simultaneously

These should preserve predictive information exactly.

---

## T3. Monotone numerical recoding

Apply strictly increasing transformations such as:

- signed log-like mapping
- rank-to-quantile mapping
- random monotone piecewise-linear spline

Preserve ordering exactly.

Interpret this as an **order-scale** transformation, not a universal semantic invariance for every numerical variable.

Measure which models implicitly depend on numerical spacing.

---

## T4. Ordinal spacing recoding

Primary dataset: diamonds.

For ordered categories \(1,\dots,m\), replace canonical equally spaced codes by random strictly increasing values:

```text
0, 0.03, 0.41, 0.55, 3.7
```

Order is unchanged; spacing is arbitrary.

Generate 8 random monotonic codings.

Compare:

1. categorical treatment
2. naive integer treatment
3. ordinal-rank treatment

This is a core experiment.

---

## T5. Cyclic origin recoding

For known cyclic variable of period \(P\):

\[
x'=(x+s)\bmod P.
\]

Use multiple random shifts \(s\).

IMPORTANT: record the transformation metadata. This is testing representation dependence given known cyclic semantics; do not claim a raw model can infer the changed physical origin without metadata.

For sin/cos encoding:

\[
[\sin\theta,\cos\theta]
\]

also apply equivalent 2D rotation:

\[
z'=R_\alpha z.
\]

The rotation is invertible and preserves the circle geometry exactly.

---

## T6. Equivalent basis transformations

This directly tests the representation-orbit idea.

For a numerical feature construct a fixed basis:

- 8 RBF basis functions OR
- 8 piecewise-linear/bin basis functions.

Call it:

\[
\phi(x)\in\mathbb R^8.
\]

Generate a random invertible, reasonably conditioned matrix:

\[
A\in\mathbb R^{8\times8}
\]

with condition number preferably `<10`.

Compare:

\[
\phi(x)
\]

against

\[
A\phi(x).
\]

These contain exactly the same information.

Run 8 random matrices.

Do NOT use pathological nearly-singular matrices in the headline analysis.

Optional variants:

- orthogonal A
- diagonal rescaling A
- general well-conditioned invertible A

This is another core experiment.

---

# 7. Baseline preprocessing / repair methods

We need more than demonstrating failure.

## R0. Original representation

No repair.

## R1. Standardization

Training-set z-score all numerical features.

Tests whether affine sensitivity disappears.

## R2. Quantile/rank frontend

Training-set empirical CDF / quantile encoding.

Tests approximate invariance to monotone transformations.

## R3. Nominal canonicalization

Construct category codes from training data using TARGET-FREE statistics:

1. category frequency
2. tie-break using a deterministic signature of other-feature distributions

Do not use target labels.

Goal: make arbitrary category names irrelevant.

## R4. Ordinal canonicalization

Given known order:

\[
rank(c)/(m-1).
\]

This should remove arbitrary ordinal spacing.

## R5. Cyclic semantic frontend

Given period/origin metadata, map to canonical:

\[
(\sin(2\pi x/P),\cos(2\pi x/P)).
\]

For deliberately shifted representations, use the known shift metadata to canonicalize first.

## R6. Orbit prediction ensemble

For each original sample average predictions across `4–8` semantically equivalent representations.

Classification:

\[
p_{ens}=\frac1M\sum_T p_T.
\]

Regression:

\[
\hat y_{ens}=\frac1M\sum_T\hat y_T.
\]

Compare equal-compute seed/model ensemble if possible.

Key question:

> Is representation/orbit diversity useful beyond random model diversity?

---

# 8. Trainable-model repair ablations

Run only on the 2–3 datasets with strongest schema sensitivity.

Use TabM if convenient; otherwise a solid MLP.

## A. Orbit augmentation

During training randomly sample one legal semantic transformation each epoch/batch.

## B. Consistency regularization

For two equivalent representations \(x,T(x)\):

Classification:

\[
L=L_{task}+\lambda KL(p(x)\Vert p(T(x))).
\]

Regression:

\[
L=L_{task}+\lambda(\hat y(x)-\hat y(T(x)))^2.
\]

Try:

```text
lambda = 0.1, 1.0
```

## C. Raw + canonical dual view

Model both:

- raw representation
- type-aware canonical representation

Combine by averaging predictions or a small learned gate.

This tests whether invariance can be added without destroying information that raw coordinates sometimes provide.

---

# 9. Important ablations

For each strong effect isolate:

### Which feature type?

- nominal
- ordinal
- cyclic
- ordinary numerical

### Which transformation strength?

Numerical affine:

```text
scale ranges:
[0.5, 2]
[0.1, 10]
```

Basis transforms:

```text
orthogonal
condition number <= 3
condition number <= 10
```

### One feature vs all features

This distinguishes local instability from accumulated schema changes.

### Prediction instability vs performance instability

A model can retain accuracy while producing very different individual predictions. Report both.

### Architecture dependency

Compare:

- TabICLv2
- TabPFN
- CatBoost
- TabM

A cross-family effect is much more interesting.

---

# 10. If headline transformations show no signal

Do not immediately stop.

Try:

1. more orbit samples: `8 -> 20`
2. all-features-at-once transforms
3. basis transforms instead of simple affine scaling
4. ordinal nonlinear spacing
5. combine multiple semantic recodings simultaneously
6. compare individual predictions rather than aggregate accuracy
7. examine calibration/logloss instead of accuracy
8. subgroup effects: rare categories, distribution tails
9. out-of-distribution split if dataset naturally supports time/order
10. representation ensemble even when mean accuracy is stable

However:

- do not manufacture signal using extreme numerical overflow;
- do not use singular basis transforms;
- do not call information-destroying transforms “equivalent.”

---

# 11. Strongest research tests

These questions matter most.

### Q1

Does an arbitrary nominal relabeling change predictions for current SOTA TFMs?

### Q2

Does arbitrary **ordinal spacing**, while preserving order, change predictions?

### Q3

Do well-conditioned invertible basis changes alter predictions/performance despite identical information?

### Q4

Is sensitivity feature-type dependent?

### Q5

Can simple canonicalization reduce sensitivity without losing accuracy?

### Q6

Does orbit ensembling improve mean or worst-case performance?

### Q7

Does training with semantic-orbit consistency outperform generic augmentation?

---

# 12. Kill criteria

## Strong GO

Recommend pursuing if there is a repeated effect satisfying roughly:

1. at least 3 real datasets;
2. at least 2 strong model families;
3. transformation is clearly information-equivalent / semantically justified;
4. prediction disagreement is substantial, e.g.
   - classification label flips >~3% or meaningful probability divergence;
   - regression prediction disagreement >~0.05 target standard deviations;
5. AND canonicalization/orbit ensembling reduces instability substantially or improves worst-case/average predictive performance.

Especially compelling:

- ordinal spacing;
- nominal relabeling;
- well-conditioned basis transforms;
- feature-type-aware repair beating generic standardization.

## Interesting but incomplete

If equivalent representations cause large disagreement but none of the repair methods help, report:

```text
FOUNDATIONAL SIGNAL, METHOD UNSOLVED
```

This may still justify deeper method work.

## NO-GO

Recommend dropping/pivoting if:

- only ordinary column permutation matters;
- only pathological/ill-conditioned transforms create effects;
- modern TFMs are already effectively invariant;
- effects occur only in weak MLP baselines;
- repairs provide no performance/robustness benefit;
- results disappear under multiple seeds.

---

# 13. Required figures

Generate:

1. orbit prediction-disagreement heatmap:
   `model × transformation × dataset`
2. task-performance range across orbit members
3. original vs transformed prediction scatter
4. repair effectiveness
5. orbit ensemble vs ordinary ensemble
6. feature-type sensitivity plot
7. basis condition number vs disagreement
8. worst-case orbit performance vs average performance

---

# 14. Required results.md

Write exactly:

```markdown
# Semantic Symmetries / Representation Orbits — Kill Experiment

## Executive Verdict
GO / WEAK-GO / FOUNDATIONAL-SIGNAL-METHOD-UNSOLVED / NO-GO

## One-Paragraph Summary

## Experimental Setup
- hardware
- runtime
- packages
- model versions
- datasets / OpenML IDs
- seeds
- transformations

## Main Schema-Sensitivity Table
dataset | model | transformation | pred disagreement | task metric original | orbit mean | orbit worst | orbit span

## Results by Transformation
### Column permutation control
### Nominal relabeling
### Numerical affine/unit transforms
### Monotone transforms
### Ordinal spacing
### Cyclic recoding
### Equivalent basis changes

## Model Comparison
TabICLv2 vs TabPFN vs CatBoost vs TabM

## Repair Results
standardization | quantile | nominal canonicalization | ordinal canonicalization | cyclic frontend | orbit ensemble

## Training-Time Ablations
orbit augmentation / consistency / dual-view if run

## Strongest Positive Finding

## Strongest Negative Finding

## Information-Equivalence Sanity Checks
Explain why each important transformation really preserves information.

## Failures / Unexpected Results
Report all.

## Does This Look Like an ICLR/ICML/NeurIPS-Level Direction?
Give evidence-based YES / MAYBE / NO and why.

## Best Next Method
If signal exists, propose the simplest model suggested by the results.

## Files Produced
List scripts, raw CSV/JSON and figures.
```

All raw measurements must be saved so another analyst can reproduce tables.

Do not cherry-pick orbit members, seeds, models or datasets. The purpose is to decide whether this direction deserves months of research.