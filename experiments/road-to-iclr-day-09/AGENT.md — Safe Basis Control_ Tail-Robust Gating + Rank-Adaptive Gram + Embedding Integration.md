# AGENT.md — Safe Basis Control: Tail-Robust Gating + Rank-Adaptive Gram + Embedding Integration

## Mission

Turn the current promising method signal into a paper-ready method.

Previous tournament established:

- **GramAnchor**: 100% median orthogonal-basis disagreement reduction, +0.90% median prospective task cost.
- **Raw+GramAnchor@0.75**: 75% disagreement reduction with approximately -0.10% median task change and best prospective predictive ranking.
- **BlockAdam+DataInit**: mechanistically valid but architecture-dependent.
- Main weakness: **catastrophic tail failures** on some datasets despite good median statistics.
- Orthogonal/natural-basis invariance is the primary target; general non-orthogonal GL(k) invariance is secondary.

Do NOT spend this round rediscovering the phenomenon.

Main questions:

1. Can validation safely choose how much invariant representation to use?
2. Can GramAnchor be made minimal/rank-adaptive and less disruptive?
3. Are catastrophic failures caused by information loss, optimization, or changed inductive bias?
4. Does basis sensitivity and the proposed fix persist **inside standard numerical embeddings**?

Required output:

```text
results.md
results/raw/
results/processed/
figures/
configs/
```

Target runtime: a few GPU-hours.

---

# 1. Primary candidate methods

Prioritize:

1. **SafeGram** — validation-controlled raw/invariant mixture
2. **RankAdaptiveGram** — minimal invariant Gram coordinates
3. **SafeRankGram** — combination of 1 + 2
4. Numerical-embedding integration of the winning method

Keep fixed-alpha Raw+GramAnchor@0.75 and pure GramAnchor as baselines.

BlockAdam remains a mechanistic baseline only.

---

# 2. Dataset protocol

## Development panel

Use previous difficult / informative datasets:

```text
steel-plates-fault
wilt
eeg-eye-state
satimage
space-ga
california_housing
house_16H
phoneme
```

Include Steel Plates specifically because it showed catastrophic Gram failures.

## NEW untouched prospective panel

Choose 8–12 datasets NEVER used in previous rounds.

Preferred criteria:

```text
1,000–50,000 rows
<=100 raw columns preferred
classification + regression
mixed task difficulty
no text/images
```

Before method selection write:

```text
configs/NEW_TAIL_PROSPECTIVE_PANEL.json
```

Do not load prospective outcomes until finalists are frozen.

---

# 3. Representation setup

For each numerical feature construct the same 8-D RBF/hat block used previously:

\[
z_j=\phi_j(x_j)\in\mathbb R^8.
\]

Primary transformations:

- 8 random orthogonal basis rotations
- local-hat ↔ spectral-hat
- one-hot ↔ Helmert where categoricals exist
- Fourier-origin transformation where cyclic metadata exists

All exact-equivalence checks must pass `<1e-6`.

Do NOT prioritize condition<=3 general transforms.

---

# PART A — SAFETY-AWARE RAW / GRAM GATING

# 4. SafeGram

Let:

\[
p_\alpha
=
(1-\alpha)p_{raw}
+
\alpha p_{gram}
\]

for classification, or

\[
\hat y_\alpha
=
(1-\alpha)\hat y_{raw}
+
\alpha\hat y_{gram}
\]

for regression.

Test:

```text
alpha = [0, 0.25, 0.5, 0.75, 1.0]
```

Do NOT select alpha by test performance.

---

# 5. Safety-normalized excess risk

Relative percentage loss is unstable when raw loss is nearly zero.

Define trivial baseline loss:

Classification:

\[
L_{trivial}
=
\text{log-loss of training class-prior predictor}.
\]

Regression:

\[
L_{trivial}
=
RMSE(\bar y_{train},y).
\]

Define normalized excess cost:

\[
C_\alpha
=
\frac{L_\alpha-L_{raw}}
{\max(L_{trivial}-L_{raw},\epsilon)}.
\]

Use:

```text
epsilon = 1e-8
```

This is the primary safety quantity.

---

# 6. Safe alpha selection

Goal:

\[
\max \alpha
\]

subject to validation evidence that predictive cost is acceptably small.

Test safety thresholds:

```text
tau = 0
0.005
0.01
0.02
```

Select largest alpha satisfying:

\[
UCB_{95}(C_\alpha)\le\tau.
\]

Estimate uncertainty by bootstrap over validation rows:

```text
500 bootstrap resamples
```

If no nonzero alpha passes:

```text
alpha = 0
```

This fallback is essential.

Call:

```text
SafeGram-t0
SafeGram-t005
SafeGram-t01
SafeGram-t02
```

Primary expected candidate:

```text
SafeGram-t01
```

---

# 7. Alternative gate ablations

If UCB gating is overly conservative, test:

## G1. point-estimate gate

Largest alpha with:

\[
C_\alpha\le0.01.
\]

## G2. one-standard-error gate

Use one-SE instead of 95% UCB.

## G3. validation-loss minimizer

Choose alpha purely minimizing validation loss.

This is a control, not preferred method.

## G4. constrained objective

Choose:

\[
\arg\min_\alpha
L_{val,\alpha}
+
\lambda(1-\alpha)
\]

where larger alpha is rewarded.

Test:

```text
lambda = [0.001, 0.01, 0.05] * L_trivial
```

Only keep if SafeGram UCB rule fails.

---

# PART B — RANK-ADAPTIVE GRAM

# 8. RankAdaptiveGram

Current GramAnchor uses fixed `m=16`.

Instead estimate empirical block rank:

\[
r_j
=
\#\{\sigma_i:
\sigma_i/\sigma_1>\epsilon_r\}.
\]

Test:

```text
epsilon_r = 1e-6
1e-4
1e-3
```

Use invariant Gram-matrix eigenspectrum / SVD.

Anchor counts:

```text
m_j = r_j
m_j = r_j + 1
m_j = min(2*r_j, 16)
fixed m=16 baseline
```

Use deterministic Gram-pivot anchor selection only.

No target labels.

---

# 9. RankAdaptiveGram requirements

For every feature block report:

```text
raw block dimension
empirical rank
anchor count
condition number of selected anchor Gram matrix
coordinate dimension
```

Check whether invariant coordinates retain the empirical block information.

For selected anchors \(B\), compute reconstruction of the original training block from Gram coordinates by a diagnostic-only least-squares inverse.

Report:

\[
E_{recon}
=
\frac{\|Z-\hat Z\|_F}{\|Z\|_F}.
\]

This inverse is diagnostic only; it is NOT used by the model.

Important question:

> Are catastrophic task failures occurring even when invariant coordinates are empirically information-preserving?

---

# 10. Normalization ablations

For Gram coordinates test:

### N0
raw inner products

\[
zB^\top
\]

### N1
anchor-norm normalized

\[
\frac{z b_i^\top}{\|b_i\|+\epsilon}
\]

### N2
cosine-like normalized

\[
\frac{z b_i^\top}
{(\|z\|+\epsilon)(\|b_i\|+\epsilon)}
\]

### N3
global block RMS normalization

Normalize invariant coordinate block to match average training RMS of original basis block.

Do not run all combinations blindly.

Run on Steel Plates + 3 representative datasets first.

---

# PART C — WHY DO TAIL FAILURES OCCUR?

# 11. Failure diagnosis panel

Take the five worst pure-Gram task failures from the previous tournament, including:

```text
steel-plates-fault
```

and four automatically selected worst cells.

For each compare:

```text
raw
GramAnchor
RankAdaptiveGram
SafeGram
```

Record:

- training error
- validation error
- test error
- disagreement
- empirical rank
- reconstruction error
- feature dimension
- anchor condition
- optimization convergence
- model confidence/calibration

---

# 12. Interpret failure type

Automatically classify each failure:

## Type A — information/interface loss

If Gram training error is much worse than raw training error AND reconstruction error is non-negligible.

## Type B — optimization difficulty

If representation is reconstructible but training error is substantially worse.

## Type C — altered generalization / inductive bias

If training performance is similar but validation/test performance degrades.

## Type D — metric denominator artifact

If raw error is extremely close to zero and relative percentage looks catastrophic despite small absolute excess risk.

Report both:

```text
absolute task difference
relative task difference
normalized excess risk C
```

This is critical.

---

# 13. Optimization rescue for Type B failures

If Gram representation preserves information but training error is worse, test only:

```text
learning rate = 0.5x, 1x, 2x
weight decay = baseline, 0
standardize Gram coordinates yes/no
```

Equal tuning budget for raw and Gram.

Do not perform larger HPO.

---

# PART D — NUMERICAL EMBEDDING INTEGRATION

# 14. Goal

Demonstrate that basis ambiguity exists **inside normal tabular numerical embedding pipelines**, not only as external preprocessing.

Use at least:

```text
controlled MLP
TabM-D
FT-Transformer or ResNet-style tabular network if implementation available
```

Do not block the experiment if FT-Transformer is difficult; MLP + TabM are required.

---

# 15. Embedding types

Test:

## E1. Piecewise Linear Encoding / PLE

Use 8-dimensional numerical embedding.

## E2. RBF encoding

Same dimensionality.

Optional:

## E3. Fourier numerical features

where reasonable.

For every embedded numerical feature:

\[
e_j(x_j)\in\mathbb R^8.
\]

Compare:

\[
e_j(x_j)
\]

against:

\[
e_j(x_j)Q_j
\]

with orthogonal \(Q_j\).

The downstream model receives exactly the same information.

---

# 16. Embedding conditions

Compare:

```text
Raw embedding
Rotated embedding
Gram-after-embedding
RankAdaptiveGram-after-embedding
SafeRankGram prediction hybrid
```

The invariant interface must operate **between embedding and backbone**.

This is a central experiment.

---

# 17. Embedding questions

Answer:

1. Do PLE/RBF embeddings show the same basis sensitivity?
2. Does sensitivity grow with embedding dimension?
3. Does Gram control remove it?
4. Does task performance remain competitive?
5. Does the best basis sometimes outperform the default embedding basis?
6. Does SafeGram retain that useful inductive bias?

---

# 18. Embedding dimension ablation

On 3 development datasets test:

```text
k = 4, 8, 16, 32
```

Measure:

\[
\text{basis disagreement vs embedding dimension}.
\]

Hypothesis:

larger internal feature spaces may increase arbitrary basis dependence.

Do not run this on the whole benchmark.

---

# PART E — OPTIONAL LEARNED GATING

# 19. Dataset-level target-free gate predictor

Only run if SafeGram works but is too conservative.

Try predicting safe alpha from training-only descriptors:

```text
n rows
n raw features
feature-block empirical ranks
Gram spectrum entropy
condition numbers
class imbalance
categorical fraction
raw-vs-Gram validation prediction disagreement
```

Allowed target during development:

```text
best safe alpha determined from development validation
```

Model:

```text
tiny ridge regression / shallow tree
```

Do NOT use complex meta-learning.

Prospective comparison:

```text
fixed alpha 0.75
SafeGram validation gate
descriptor gate
```

If descriptor gate does not clearly outperform validation gating, discard it.

---

# PART F — NEW PROSPECTIVE TEST

# 20. Development finalists

Choose at most 4:

```text
GramAnchor
best RankAdaptiveGram
best SafeGram
best SafeRankGram
```

The final list may contain fewer.

Freeze:

```text
configs/TAIL_FINALISTS.json
```

Include:

```text
alpha rule
tau
rank threshold
anchor formula
normalization
all HPO settings
```

Compute SHA256.

Only then load the new prospective panel.

---

# 21. Primary prospective criteria

The method should satisfy BOTH median and tail criteria.

## Median

Desired:

```text
disagreement reduction >= 70%
median normalized excess risk <= 0.01
```

## Tail safety

Desired:

```text
95th percentile normalized excess risk <= 0.05
```

and:

```text
no dataset/model cell with C > 0.20
```

A method does NOT count as paper-ready solely because median task cost is good.

This is the key change from the previous tournament.

---

# 22. Raw fallback rate

For SafeGram report:

\[
P(\alpha=0)
\]

and distribution of selected alpha.

A good safety method may legitimately fall back to raw on difficult datasets.

Report:

```text
alpha=0 fraction
alpha=.25 fraction
alpha=.5 fraction
alpha=.75 fraction
alpha=1 fraction
```

Interpretability is a feature.

---

# 23. Baselines

Required:

```text
Raw
Pure GramAnchor m=16
Raw+GramAnchor@0.5
Raw+GramAnchor@0.75
PCA canonicalization
best RankAdaptiveGram
SafeGram
SafeRankGram
```

For trainable embedding experiments include:

```text
standard embedding + AdamW
rotated embedding + AdamW
```

BlockAdam may appear as mechanism baseline but is not required in full prospective ranking.

---

# 24. Statistical reporting

Primary unit:

```text
dataset × model
```

Report:

- median disagreement reduction
- median normalized excess risk
- 90th/95th percentile excess risk
- maximum excess risk
- fraction of cells improved in task performance
- W/T/L
- bootstrap CI across datasets

Also report results excluding cells where:

```text
L_trivial - L_raw < 1e-6
```

as a denominator-sensitivity check.

Never remove those cells from the primary table.

---

# 25. Method ranking

Produce five rankings.

## Ranking A — Safety First

Eligible only if:

```text
median C <= 0.01
95th percentile C <= 0.05
max C <= 0.20
```

Rank by disagreement reduction.

## Ranking B — Invariance

Rank by disagreement reduction.

## Ranking C — Predictive Performance

Rank by task metric.

## Ranking D — Tail Robustness

Rank by:

```text
95th percentile C
max C
fallback effectiveness
```

## Ranking E — Paper Candidate

Score:

\[
S
=
R_D
-3\max(\operatorname{median}C,0)
-2\max(P95(C)-0.05,0)
-2\max(C_{max}-0.20,0).
\]

Report raw components.

Do not hide methods excluded from Safety-First ranking.

---

# 26. Kill / GO criteria

## PAPER-READY METHOD SIGNAL

Use if a method on the NEW prospective panel achieves approximately:

```text
>=70% median disagreement reduction
median C <=1%
95th percentile C <=5%
no catastrophic tail >20%
```

AND works for:

```text
>=3 model families
```

AND succeeds for at least one natural basis pair.

---

## STRONG METHOD, TAIL UNSOLVED

Use if median results are strong but rare catastrophic cells remain.

---

## SAFE BUT TOO CONSERVATIVE

Use if SafeGram avoids damage by selecting alpha=0 on most datasets and median disagreement reduction falls below ~40%.

---

## REPRESENTATION METHOD FAILS

Use if even adaptive gating cannot obtain meaningful invariance without predictive cost.

Then recommend a phenomenon/mechanism paper instead.

---

# 27. Critical plots

Generate:

### Figure 1
Disagreement reduction vs normalized excess risk.

### Figure 2
Tail CDF of normalized excess risk.

Compare:

```text
GramAnchor
fixed 0.75
SafeGram
SafeRankGram
```

### Figure 3
Selected alpha histogram.

### Figure 4
Steel Plates failure diagnosis.

Train/validation/test error for raw vs invariant methods.

### Figure 5
RankAdaptiveGram anchor count vs performance.

### Figure 6
Embedding-basis sensitivity.

```text
PLE/RBF original vs rotated
```

### Figure 7
Embedding dimension vs disagreement.

### Figure 8
Development vs prospective safety.

---

# 28. Required results.md

Produce exactly:

```markdown
# Safe Basis Control — Tail-Robust Method Round

## Executive Verdict
PAPER-READY-METHOD-SIGNAL /
STRONG-METHOD-TAIL-UNSOLVED /
SAFE-BUT-TOO-CONSERVATIVE /
REPRESENTATION-METHOD-FAILS

## One-Paragraph Summary

## Frozen Protocol
- git commit
- hardware
- versions
- seeds
- development datasets
- NEW prospective datasets
- TAIL_FINALISTS SHA256

## 1. Previous Result Being Addressed
State:
- GramAnchor median success
- Raw+GramAnchor@0.75 success
- catastrophic tail failures
- why median relative task change is insufficient

## 2. SafeGram Development Results

tau | median alpha | reduction | median C | p95 C | max C | raw fallback rate

## 3. Gate Ablations

UCB / point / one-SE / validation-min / constrained objective

## 4. RankAdaptiveGram

rank threshold | anchor rule | coordinate count | disagreement | task | C

## 5. Normalization Ablations

## 6. Catastrophic Failure Diagnosis

dataset | model | method | train error | val error | test error | recon error | rank | failure type

Explicitly classify Type A/B/C/D.

## 7. Steel Plates Deep Dive

Give absolute loss, relative loss, normalized excess risk.

Explain whether failure is information loss, optimization, or inductive bias.

## 8. Numerical Embedding Basis Test

dataset | model | embedding | k | original task | rotated task | disagreement

## 9. Gram Inside Numerical Embeddings

method | disagreement reduction | task cost

## 10. Embedding Dimension Ablation

k | disagreement | task effect

## 11. Development Finalist Ranking

## 12. Frozen Finalists

List exact configs and SHA.

## 13. NEW Prospective Results

dataset | model | method | alpha | disagreement reduction | raw task | method task | C

## 14. Prospective Aggregate Results

method | median reduction | median C | p90 C | p95 C | max C | task W/T/L | alpha=0 rate

## 15. Safety-First Ranking

## 16. Invariance Ranking

## 17. Predictive Ranking

## 18. Tail-Robustness Ranking

## 19. Paper-Candidate Ranking

## 20. Natural-Basis Validation

local/spectral + one-hot/Helmert + Fourier if available

## 21. Strongest Positive Result

## 22. Strongest Negative Result

## 23. Does Adaptive Gating Actually Solve Tail Risk?

YES / PARTLY / NO

Support with:
- p95
- max
- fallback behavior
- catastrophic-cell outcomes

## 24. Is RankAdaptiveGram Better Than Fixed m=16?

YES / PARTLY / NO

## 25. Does the Phenomenon Exist Inside Standard Numerical Embeddings?

YES / NO

This section is important for paper framing.

## 26. Recommended Paper Method Candidates

Rank at most top 3:

rank | method | median invariance | median C | p95 C | model breadth | embedding success | complexity

Do NOT select the final one automatically.

## 27. Reviewer Attack Audit

### "Median performance hides catastrophic failures."
### "The gate is just validation overfitting."
### "The invariant representation throws away information."
### "Why not simply use the original scalar feature?"
### "Random rotations are artificial."
### "The method doubles inference cost."
### "The method just falls back to raw everywhere."
### "This is only relevant to handcrafted preprocessing."

## 28. Recommended Next Experiment for Top-3

Give one decisive experiment each.

## 29. Files Produced
```

---

# 29. Compute priority

Run in this order:

```text
1. SafeGram gating
2. Steel Plates / tail diagnosis
3. RankAdaptiveGram
4. SafeRankGram
5. numerical embedding integration
6. NEW prospective panel
7. optional learned descriptor gate
```

Do not drop prospective evaluation to run more ablations.

---

# 30. Scientific framing

Do NOT optimize for exact invariance alone.

The intended method principle is:

> Basis-dependent inductive bias can sometimes help prediction, but arbitrary basis sensitivity creates unstable models. Use an invariant feature view to control that sensitivity while retaining the raw view only when validation evidence indicates that it is useful.

The key target is:

\[
\boxed{
\text{maximum basis control subject to predictive safety}
}
\]

The strongest result would therefore be a method that automatically becomes highly invariant on safe datasets and falls back toward the raw representation on datasets where invariance damages predictive performance.