# AGENT.md — Kill Experiment 1: Interaction-Aware Context Selection for Tabular Foundation Models

## Mission

Determine whether TFM context utility is genuinely **non-additive** and whether that interaction structure can be predicted and exploited to choose better context sets at the same context budget.

Core hypothesis:

\[
U(S) \neq c+\sum_{i\in S}a_i
\]

and a useful approximation is

\[
U(S)\approx c+\sum_i a_i+\sum_{i<j}b_{ij}.
\]

This is a **kill experiment**, not a polished paper benchmark. Run the most informative experiments possible in a few GPU-hours. Do not stop because one implementation fails; try the specified ablations.

Final required output: **`results.md`** plus raw CSV/JSON and plots.

---

## 1. Primary model and setup

Primary TFM:

- **TabICLv2**, official `tabicl` package.
- Use classification and regression.
- Frozen pretrained checkpoint. No fine-tuning.

Secondary validation if compute permits:

- **TabPFN-2.6** official `tabpfn` package.

Do not make the experiment depend on proprietary/API-only models.

Use fixed random seeds: `0, 1, 2`.

Create:

```text
experiments/
results/
plots/
results.md
```

Record package versions, GPU, runtime, dataset sizes, seeds.

---

## 2. Datasets

Start with 6 manageable real datasets:

Classification:
- adult
- bank-marketing
- credit-g
- electricity

Regression:
- california_housing
- diamonds

Use OpenML/sklearn. If an exact OpenML name fails, resolve the closest canonical dataset programmatically and record its ID.

If these run quickly, add 2–4 more TabArena datasets with:

- 2,000–30,000 rows
- <=100 features
- no text/images
- mixture of classification/regression

For each dataset create once:

- context candidate pool: 256 rows
- selector/meta-validation queries: 128–256 rows
- final test queries: 256+ rows if available

Never use final test labels for context selection or surrogate fitting.

Stratify classification splits. For regression, stratify approximately using target quantile bins.

Use the same candidate pool for every selector.

---

## 3. Context budgets

Test:

```text
K = 16, 32, 64
```

If a classification task has many classes, require every generated context to contain at least one example from every observed class when feasible.

For each `(dataset, K)` sample initially:

```text
512 random context sets
```

Evaluate every set on the SAME selector-validation queries.

Primary utility:

Classification:
\[
U(S)=-\text{logloss}
\]

Regression:
\[
U(S)=-RMSE/\operatorname{std}(y_{train})
\]

Also record:

- accuracy
- ROC-AUC where appropriate
- RMSE/MAE
- runtime

Cache every context membership vector and its utility. Reuse model predictions aggressively.

---

# 4. Experiment A — Is context utility non-additive?

Represent each sampled context by binary membership vector:

\[
x_S\in\{0,1\}^{256}.
\]

Split context sets themselves 70/30 into surrogate-train and surrogate-test.

### A0 — constant baseline

Predict mean utility only.

### A1 — additive context model

Ridge regression:

\[
\hat U(S)=c+\sum_i a_i x_i.
\]

Tune ridge regularization only on surrogate-train CV.

### A2 — factorization-machine interaction

Fit:

\[
\hat U(S)
=
c+\sum_i a_i x_i+
\sum_{i<j}x_ix_jv_i^\top v_j.
\]

Test interaction ranks:

```text
r = 2, 4, 8, 16
```

Use PyTorch, AdamW, early stopping, strong L2 regularization.

This gives:

\[
b_{ij}=v_i^\top v_j.
\]

### A3 — pairwise residual model

First fit A1.

Fit interaction model only to residual:

\[
U(S)-\hat U_{A1}(S).
\]

This tests whether pairwise structure explains information specifically missing from additive utility.

### A4 — DeepSets control

Fit a small set predictor:

\[
\hat U(S)=\rho\left(\sum_{i\in S}\phi(z_i)\right)
\]

where \(z_i\) is a row representation defined below.

This detects higher-order set structure without explicit \(b_{ij}\).

Report held-out:

- \(R^2\)
- Spearman correlation
- MAE

Most important number:

```text
ΔR² = R²(pairwise) - R²(additive)
```

---

# 5. Row representation z_i

Construct a selector-only representation; do NOT modify TFM input because of this.

For each candidate row:

1. standardized numerical features
2. one-hot/ordinal encoded categoricals
3. PCA to 16 dimensions
4. target:
   - classification: one-hot class
   - regression: standardized y
5. optional distances to class/target-bin centroids

Call the resulting vector \(z_i\).

---

# 6. Experiment B — Different b_ij formulations

Test all of these. Do not stop at the first failure.

## B1. ID-factor interaction

\[
b_{ij}=v_i^\top v_j
\]

from A2.

This is the highest-capacity dataset-specific diagnostic.

## B2. Feature-bilinear

Learn:

\[
v_i=g(z_i), \qquad b_{ij}=v_i^\top v_j
\]

with a 2-layer MLP `g`.

Ranks:

```text
4, 8, 16
```

This asks whether interaction is predictable from row properties rather than point identity.

## B3. Signed bilinear

\[
b_{ij}=z_i^\top UV^\top z_j
\]

low rank.

Allow positive AND negative interactions.

Compare against a constrained purely-negative/diversity version.

## B4. Geometry/diversity

\[
b_{ij}=-\lambda\,\mathrm{sim}(z_i,z_j)
\]

Try:

- cosine
- RBF
- Euclidean-neighbor penalty

Tune \(\lambda\) on selector-validation only.

## B5. Label/target complementarity

Classification:

- reward class coverage
- penalize excessive same-class redundancy

Regression:

- reward target-bin coverage
- reward diverse residual/target regions

Combine with geometry:

\[
b_{ij}
=
-\lambda_1 sim(z_i,z_j)
+\lambda_2 complement(y_i,y_j).
\]

## B6. DPP/logdet diversity

Use kernel \(K_{ij}\) from \(z_i\).

Select contexts approximately maximizing:

\[
\log\det(K_S+\epsilon I).
\]

This is an important non-additive diversity baseline.

---

# 7. Direct interaction diagnostic

On the 2 fastest datasets, `K=32`, sample ~100 pairs.

For random base context \(B\) of size `K-2`, calculate:

\[
I_{ij}(B)
=
U(B\cup\{i,j\})
-U(B\cup\{i\})
-U(B\cup\{j\})
+U(B).
\]

Repeat across 3 base contexts where compute permits.

Report:

- distribution of \(I_{ij}\)
- fraction positive/negative
- median \(|I_{ij}|\)
- stability across base contexts
- relationship to similarity, label agreement, distance, etc.

Plot interaction heatmaps for the strongest dataset.

This provides direct evidence of non-additivity independent of surrogate quality.

---

# 8. Context-selection comparison

Every method gets identical context budget K.

Required:

1. random stratified — 20 seeds
2. additive top-K using learned \(a_i\)
3. k-center/farthest-point
4. k-medoids
5. nearest-neighbor context to query/query-cluster
6. MMD subset / **CRUMB-like** selection
7. latent-medoid / **LUCoS-like** selection
8. DPP/logdet
9. pairwise FM greedy
10. pairwise FM greedy + 1-swap local search
11. feature-bilinear greedy
12. strongest complementarity variant

Also attempt official CRUMB/LUCoS/VIP-COP code if straightforward. If integration becomes a major blocker, continue with faithful lightweight implementations and clearly label them "`-like`"; never claim they are official reproductions.

Useful references:

- additive black-box baseline: estimate each point's average marginal utility across sampled contexts and select top-K.
- this is an especially important comparison because it directly tests **individual importance ranking vs joint set optimization**.

Greedy pairwise objective:

\[
\Delta(i\mid S)
=
a_i+\sum_{j\in S}b_{ij}.
\]

After greedy selection, perform repeated 1-out/1-in swaps until no surrogate improvement.

---

# 9. Oracle/headroom controls

These are diagnostics, NOT legitimate final baselines:

### Oracle best-of-random

Take the best selector-validation context among the 512 random sets.

Then evaluate that frozen context on final test.

### Direct TFM local search

Start from additive top-K.

Using selector-validation labels only, directly evaluate TFM utility for candidate swaps and greedily improve the context.

Run on only 1–2 datasets if expensive.

Interpretation:

- direct search succeeds but learned interactions fail → interaction signal exists, surrogate is inadequate
- direct search also fails → probably little exploitable context-set signal

---

# 10. If pairwise FM fails

Do not stop.

Try in this order:

1. reduce candidate pool `256 -> 128`
2. increase sampled contexts to `1024` on 2 datasets
3. stronger L2 regularization
4. FM ranks `2/4/8/16`
5. fit interaction residual after additive rather than jointly
6. signed vs diversity-only \(b_{ij}\)
7. query-cluster-conditioned interactions
8. direct TFM swap search
9. DeepSets set utility predictor
10. pairwise + simple 3-way correction:

\[
\hat U(S)=A(S)+B(S)+\rho\left(\sum_i\phi(z_i)\right)
\]

Do not run huge hyperparameter sweeps. The goal is identifying whether a robust signal exists.

---

# 11. Statistical comparison

For each dataset/K report test performance relative to random and strongest non-interaction selector.

Calculate:

- win/tie/loss
- mean normalized improvement
- median normalized improvement
- average rank
- bootstrap 95% CIs across test rows where sensible

Most important plots:

1. surrogate held-out \(R^2\): additive vs interaction
2. performance vs context budget
3. interaction magnitude histogram
4. selector win/loss heatmap
5. predicted utility vs actual utility
6. performance of direct-search oracle vs learned selector

---

# 12. Kill criteria

## Strong GO

Recommend pursuing the paper if BOTH are approximately true:

1. Pairwise/set model improves held-out context-utility prediction substantially:
   - roughly `ΔR² >= 0.10` on >= half the datasets, OR another clearly consistent effect.

2. An interaction-aware selector beats the strongest equal-budget non-interaction selector on >=60% of dataset/budget combinations, with meaningful aggregate improvement.

Especially strong if it transfers from TabICLv2 to TabPFN.

## Scientific signal, method not solved

Report this explicitly if:

- direct interactions are large;
- direct TFM local search finds much better contexts;
- but current \(b_{ij}\) surrogates fail to predict/select them.

This is still valuable and suggests better interaction modeling.

## NO-GO

Recommend dropping/pivoting if:

- pairwise models add <~0.03 held-out \(R^2\) almost everywhere;
- direct interaction magnitudes are tiny;
- direct TFM swap search cannot outperform additive/random selectors;
- apparent gains disappear on final test.

---

# 13. Required results.md

Write **`results.md`** with exactly these sections:

```markdown
# Interaction-Aware Context Selection — Kill Experiment

## Executive Verdict
GO / WEAK-GO / METHOD-FAILS-BUT-SIGNAL / NO-GO

## One-Paragraph Summary

## Experimental Setup
- hardware
- runtime
- packages
- exact datasets / OpenML IDs
- splits
- TFM versions
- context budgets
- number of context evaluations

## Main Result Table
dataset | task | K | random | additive | CRUMB-like | LUCoS-like | DPP | best pairwise | direct-search oracle

## Utility Prediction
dataset | K | additive R2 | FM R2 | feature-FM R2 | DeepSets R2 | ΔR2

## Direct Interaction Diagnostic
Include distribution/statistics of I_ij.

## b_ij Ablations
rank, parameterization, regularization, signed/diversity/complementarity.

## Selector Results

## Cross-Model Check
TabICLv2 vs TabPFN if run.

## Failures and Negative Results
Be explicit.

## Strongest Evidence FOR the Hypothesis

## Strongest Evidence AGAINST the Hypothesis

## Recommended Next Research Direction
Explain what should be built next if GO, or why to abandon/pivot if NO-GO.

## Files Produced
List CSVs, JSON, plots and scripts.
```

Save raw results in tidy CSV form so another analyst can recompute every table.

Do not hide failed ablations. Do not selectively report only positive datasets.