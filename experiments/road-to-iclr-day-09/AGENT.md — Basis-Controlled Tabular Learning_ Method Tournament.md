# AGENT.md — Basis-Controlled Tabular Learning: Method Tournament

## Mission

Find a practical method that reduces arbitrary within-feature basis dependence **without sacrificing predictive performance**.

Previous experiments have already established:

- orthogonal condition-number-1 basis changes produce broad prediction disagreement;
- natural equivalent bases reproduce the phenomenon;
- equal HPO does not remove it;
- function-matched SGD preserves basis equivalence;
- function-matched AdamW recreates disagreement;
- removing AdamW weight decay does not fix it;
- PCA canonicalization gives near-perfect orthogonal invariance but can significantly hurt task performance;
- consistency training failed broadly;
- pseudoinverse-based AnchorCanonical fails under rank-deficient feature blocks.

Do NOT rerun those as discovery experiments.

This round is a **method tournament**.

The final output must rank candidate methods so another researcher can decide which one deserves full paper development.

Required output:

```text
results.md
results/raw/
results/processed/
figures/
configs/
```

Target runtime: a few GPU-hours, not days.

---

# 1. Scientific target

For a feature block

\[
Z_j=\phi_j(x_j)\in\mathbb R^{n\times k},
\]

consider an equivalent basis

\[
Z'_j=Z_jQ,\qquad Q^\top Q=I.
\]

A strong method should achieve:

\[
f(Z)\approx f(ZQ)
\]

while retaining the predictive benefits of adaptive optimization / raw feature geometry.

Primary success target:

```text
median disagreement reduction >= 70%
median relative task degradation <= 1%
```

Strongest target:

```text
>= 90% disagreement reduction
and
<= 0.5% task degradation
```

Do NOT assume perfect invariance is optimal.

---

# 2. Two method tracks

Evaluate separately:

## Track A — Trainable-model optimization methods

Goal:

preserve Adam-like predictive strength while eliminating arbitrary coordinatewise optimizer dependence.

Primary models:

- controlled MLP
- TabM-D

## Track B — Representation/interface methods

Goal:

construct non-oracle feature interfaces that can also be supplied to frozen models.

Models:

- controlled MLP
- TabM-D
- TabICLv2
- TabPFN-2.6
- CatBoost

Do not force one method to solve both tracks.

The final report should identify:

1. best trainable optimizer method;
2. best universal representation method;
3. best hybrid method overall.

---

# 3. Dataset protocol

## Development method-search panel

Use exactly 6 datasets from the completed development pool:

```text
california_housing
house_16H
bike-sharing
phoneme
credit-g
wine-quality-red
```

This provides:

- regression
- classification
- weak/moderate/strong basis sensitivity

Do method design only here.

## NEW locked prospective panel

Do NOT reuse the previous seven prospective datasets.

Before looking at results, select 6–8 new OpenML/TabArena-style datasets not previously used.

Preferred candidate pool:

```text
abalone
kin8nm
spambase
vehicle
satimage
mfeat-factors
superconductivity
cpu_act
```

If unavailable, replace with datasets satisfying:

```text
1,000–30,000 rows preferred
<=100 raw columns
classification or regression
no text/images
not used in any previous experiment
```

Freeze the final dataset list before method development finishes.

Write:

```text
configs/NEW_PROSPECTIVE_PANEL.json
```

Do NOT access prospective results until the winning configurations are frozen.

---

# 4. Feature representation

Use the same primary basis construction as the previous confirmation study.

For every continuous feature:

```text
8-dimensional RBF or hat basis block
```

Primary orbit:

```text
8 random orthogonal Q matrices
```

Also use one natural basis test:

```text
local hat basis <-> spectral/DCT hat basis
```

For categorical datasets where practical:

```text
one-hot <-> full Helmert basis
```

Every equivalent pair must reconstruct to relative error `<1e-6`.

---

# 5. Metrics

Primary disagreement:

Regression:

\[
D=
RMSE(\hat y,\hat y_Q)/std(y).
\]

Classification:

use probability-vector RMSE or equivalent previously frozen metric.

Also record:

- log loss / RMSE
- accuracy/AUC where applicable
- worst-orbit task error
- orbit mean task error
- training time
- GPU memory

Define:

\[
R_D = 1-\frac{D_{method}}{D_{raw}}
\]

and

\[
C =
\frac{Error_{method}-Error_{raw}}
{Error_{raw}}.
\]

Report both.

Do NOT hide a method with great invariance but poor task performance.

---

# PART I — OPTIMIZER METHODS

# 6. O0 — AdamW baseline

Controlled MLP / TabM using standard optimizer.

This is the baseline.

---

# 7. O1 — BlockScalarAdam

Hypothesis:

Adam's coordinatewise second moment breaks rotational equivariance.

Within the first-layer weights associated with each feature block, replace coordinatewise second moments with ONE scalar second moment.

For block gradient \(G_B\):

\[
v_B
=
\beta_2v_B+
(1-\beta_2)
\frac{\|G_B\|_F^2}{|G_B|}.
\]

First moment remains matrix-valued:

\[
M_B
=
\beta_1M_B+
(1-\beta_1)G_B.
\]

Update:

\[
W_B
\leftarrow
W_B-
\eta
\frac{\hat M_B}
{\sqrt{\hat v_B}+\epsilon}.
\]

Because Frobenius norm is orthogonally invariant, this should preserve basis equivariance.

IMPORTANT:

Use BlockScalarAdam ONLY on the basis-sensitive first-layer block weights.

Use ordinary AdamW for all subsequent model parameters.

This should preserve most of Adam's usefulness.

---

# 8. O2 — PerOutputBlockAdam

BlockScalarAdam may remove too much adaptivity.

For each output neuron \(h\), aggregate second moment across the k coordinates inside one feature block:

\[
v_{B,h}
=
\beta_2v_{B,h}
+
(1-\beta_2)
\frac{1}{k}
\sum_{r=1}^k G_{r,h}^2.
\]

Under an orthogonal rotation of the input coordinates, each gradient-column norm is preserved.

This yields one adaptive denominator per:

```text
feature block × output channel
```

rather than per coordinate.

This is probably the highest-priority optimizer candidate.

Call it:

```text
BlockAdam
```

in final tables.

---

# 9. O3 — MatrixAdam / FullBlockAdam

For small basis blocks (`k=8`), use a full second-moment matrix.

For block gradient:

\[
G_B\in\mathbb R^{k\times h}
\]

estimate:

\[
V_B
=
\beta_2V_B
+
(1-\beta_2)
\frac{G_BG_B^\top}{h}.
\]

Update using:

\[
\Delta W_B
=
-\eta
(V_B+\epsilon I)^{-1/2}
M_B.
\]

Under orthogonal coordinate changes:

\[
V'_B=Q^\top V_BQ
\]

and the inverse square root transforms correspondingly.

This should be rotation-equivariant while preserving anisotropic adaptivity.

Implementation:

- eigendecomposition for each `8×8` block
- eigenvalue floor
- bias correction
- float32 training; float64 matrix inverse-square-root allowed

Try only on first-layer block weights.

Ordinary AdamW elsewhere.

Call:

```text
MatrixAdam
```

---

# 10. O4 — Data-equivariant initialization

Standard random initialization is isotropic in distribution but not pathwise matched.

Construct first-layer weights from training data:

For block:

\[
Z_B\in\mathbb R^{n\times k}
\]

sample frozen:

\[
R_B\in\mathbb R^{n\times h}
\]

and initialize:

\[
W_B
=
\frac{Z_B^\top R_B}
{\sqrt n}.
\]

Under:

\[
Z'_B=Z_BQ
\]

we obtain automatically:

\[
W'_B=Q^\top W_B.
\]

No knowledge of Q is used.

Normalize variance to approximately match default initialization.

Test:

```text
default init + BlockAdam
data-equivariant init + BlockAdam

default init + MatrixAdam
data-equivariant init + MatrixAdam
```

UNIT TEST:

initial predictions under equivalent bases should agree to `<1e-5` for the data-equivariant initialization.

---

# 11. O5 — SoftBlockAdam

If pure block adaptivity preserves invariance but hurts task performance, construct a controlled interpolation.

Let:

\[
v^{coord}
\]

be ordinary Adam second moment and

\[
v^{block}
\]

the broadcast block-invariant moment.

Use:

\[
v^\star
=
\alpha v^{coord}
+
(1-\alpha)v^{block}.
\]

Test:

```text
alpha = 0
0.1
0.25
0.5
```

Interpretation:

- `alpha=0`: fully block-invariant
- larger alpha: restore coordinate-specific adaptivity

This deliberately traces a:

```text
performance <-> invariance Pareto frontier
```

Do NOT choose alpha using prospective datasets.

---

# 12. Optimizer ablation cascade

If O1/O2/O3 underperform raw AdamW by >2%, try:

### A
First-layer-only block optimizer vs first two layers where applicable.

### B
Learning-rate multipliers:

```text
0.5×
1×
2×
```

relative to frozen AdamW LR.

### C
Epsilon:

```text
1e-8
1e-6
1e-4
```

### D
Block second-moment normalization:

```text
mean squared gradient
vs
sum squared gradient / sqrt(k)
```

### E
First moment:

```text
standard momentum
vs
no first-moment momentum
```

### F
SoftBlockAdam alpha frontier.

Do NOT perform larger sweeps.

---

# PART II — REPRESENTATION METHODS

# 13. R0 — Raw basis

Baseline.

---

# 14. R1 — PCA canonicalization

Include only as previous best invariance baseline.

Do not retune extensively.

Expected role:

```text
high-invariance / potentially-high-task-cost reference
```

---

# 15. R2 — GramAnchor

Do NOT use pseudoinverses.

For feature block \(z\) select training anchor vectors:

\[
b_1,\ldots,b_m.
\]

Represent:

\[
\psi(z)
=
[
\langle z,b_1\rangle,
\dots,
\langle z,b_m\rangle
].
\]

Under orthogonal Q:

\[
\langle zQ,b_iQ\rangle
=
\langle z,b_i\rangle.
\]

Thus it is exactly orthogonal-basis invariant.

No Q is known.

No inverse is required.

---

## Anchor selection

Test:

### random-index anchors

Fixed target-free training row indices.

### Gram-pivot anchors

Use pivoted Cholesky/QR based ONLY on:

\[
K=ZZ^\top.
\]

Since K is basis-invariant, anchor selection is basis-invariant.

Prefer Gram-pivot if implementation is reliable.

Test:

```text
m = 8
16
32
```

or:

```text
m = min(32, max(8, 2*empirical_rank))
```

---

# 16. R3 — GramDistance

Instead of inner products use:

\[
\psi_i(z)
=
\|z-b_i\|_2^2.
\]

Orthogonal invariant.

Try:

```text
raw squared distance
RBF(distance / median_distance)
```

This may work better for nonlinear/local basis structure.

---

# 17. R4 — NyströmGram

Construct invariant kernel coordinates.

Training Gram matrix:

\[
K=ZZ^\top.
\]

Using selected anchors \(A\):

\[
K_{AA}=U\Lambda U^\top.
\]

Represent each sample using Nyström coordinates:

\[
\psi(z)
=
K_{zA}
U_r
\Lambda_r^{-1/2}.
\]

Choose rank using:

```text
99% spectral energy
maximum rank 8
```

This uses only inner products, so it is orthogonal-basis invariant.

Regularize:

```text
lambda_floor = 1e-6 * largest_eigenvalue
```

This is a rank-robust alternative to failed pseudoinverse AnchorCanonical.

---

# 18. R5 — Degeneracy-aware PCA

Previous PCA gave invariance but hurt performance.

Try a less destructive canonicalization.

For each feature block:

1. eigendecompose training covariance;
2. find eigenvalue groups with relative gap:

\[
\frac{\lambda_i-\lambda_{i+1}}
{\lambda_i}
>\tau.
\]

Test:

```text
tau = 0.01
0.05
0.10
```

Canonicalize ONLY well-separated eigendirections.

For nearly degenerate subspaces:

```text
leave them in invariant Gram coordinates
```

rather than choosing an arbitrary eigenvector orientation.

Call:

```text
HybridSpectral
```

---

# PART III — HYBRID METHODS

# 19. H1 — Raw + invariant prediction mixture

Train/evaluate:

```text
raw model
best invariant-interface model
```

Combine predictions:

Regression:

\[
\hat y
=
(1-\alpha)\hat y_{raw}
+
\alpha\hat y_{inv}.
\]

Classification:

average probabilities.

Test:

```text
alpha = 0.25
0.5
0.75
1.0
```

Select alpha on development validation only.

This is a simple way to trade useful basis prior against invariance.

---

# 20. H2 — Raw + invariant feature branch

Only for controlled MLP / TabM if easy.

Build:

\[
h_{raw}=f_r(z)
\]

and

\[
h_{inv}=f_i(\psi(z)).
\]

Combine:

\[
h
=
h_{inv}
+
\alpha h_{raw}.
\]

Use learned scalar alpha initialized to:

```text
0.1
```

Regularize:

\[
\lambda_\alpha\alpha^2
\]

with:

```text
lambda_alpha = 1e-3
1e-2
```

No explicit orbit-consistency loss.

Previous generic consistency training failed; do NOT repeat it.

---

# 21. H3 — BlockAdam + invariant branch

If BlockAdam is strong and Gram/Nyström representation is strong, test one combined method:

```text
raw branch optimized by BlockAdam
+
NyströmGram/GramAnchor branch
```

Do this only on the 3 most informative development datasets.

Do not build a complicated architecture.

---

# 22. Method pruning

Run methods in stages.

## Stage 1 — 3-dataset smoke test

Datasets:

```text
california_housing
phoneme
wine-quality-red
```

Run:

```text
BlockScalarAdam
BlockAdam
MatrixAdam
GramAnchor
GramDistance
NyströmGram
PCA
```

A method survives if either:

### invariance route

```text
median disagreement reduction >= 50%
task degradation <= 3%
```

OR

### performance route

```text
task performance improves
while disagreement does not worsen >20%
```

Kill others unless an explicitly specified ablation can rescue them.

---

# 23. Stage 2 — full development panel

Run survivors on all 6 development datasets, 3 seeds.

Then run:

- data-equivariant init
- SoftBlockAdam
- HybridSpectral
- raw/invariant mixtures

only where relevant.

---

# 24. Stage 3 — freeze finalists

Choose maximum:

```text
3 finalist methods
```

based ONLY on development data.

Freeze:

```text
configs/FINALIST_CONFIGS.json
```

Record SHA256.

Then access NEW prospective panel.

No tuning after this point.

---

# 25. Important mechanism checks

For BlockAdam and MatrixAdam, use a matched-function pair on 2 datasets.

Verify:

```text
epoch 0 disagreement < 1e-5
```

Save disagreement after:

```text
epoch 0
1
2
5
10
20
final
```

A genuinely equivariant optimizer should remain near numerical tolerance under matched minibatch order.

If disagreement grows:

- implementation is not truly equivariant;
- inspect optimizer state transformations.

Do not report it as an invariant optimizer until fixed.

---

# 26. Natural-basis validation

Finalist interface methods must also be tested on:

```text
local hat <-> spectral hat
```

and where applicable:

```text
one-hot <-> Helmert
```

A method that only handles random Q but not natural equivalent coordinates receives a penalty in final ranking.

---

# 27. General-invertible exploratory test

Orthogonal invariance is primary.

For finalists only, test condition≤3 transforms.

Important:

GramAnchor inner products are NOT generally invariant to arbitrary non-orthogonal A.

Do not pretend otherwise.

Report condition≤3 separately.

Possible exploratory general-linear interfaces:

### covariance-normalized Gram

For block covariance:

\[
\Sigma=Z^\top Z/n
\]

use:

\[
k(z,b)
=
z
(\Sigma+\lambda I)^+
b^\top.
\]

This resembles a Mahalanobis inner product.

Because rank deficiency is known to be an issue, test:

```text
lambda = 1e-6
1e-4
1e-2
```

and report rank sensitivity.

Call:

```text
MahalanobisGram
```

This is exploratory only.

If unstable, stop.

---

# 28. Fair optimization control

For each surviving optimizer method use equal development HPO:

```text
learning rate:
0.5× baseline
1× baseline
2× baseline
```

No other tuning unless required by specified ablations.

Raw AdamW gets the same three-trial LR budget.

This prevents giving the new optimizer a tuning advantage.

---

# 29. Ranking system

Do NOT declare one winner based on one metric.

Produce four rankings.

## Ranking A — Performance-preserving invariance

Eligible only if:

```text
median relative task degradation <= 1%
```

Rank by:

1. prospective median disagreement reduction;
2. prospective worst-orbit task error;
3. development consistency.

This is the PRIMARY ranking.

---

## Ranking B — Pareto frontier

Plot:

```text
x = median task degradation
y = median disagreement reduction
```

Identify non-dominated methods.

---

## Ranking C — Predictive performance

Rank methods purely by task error.

This reveals whether basis dependence is buying useful performance.

---

## Ranking D — Paper-method score

Use:

\[
Score
=
R_D
-
5\max(C,0)
-
0.25F,
\]

where:

- \(R_D\) = median disagreement reduction;
- \(C\) = median positive relative task-cost fraction;
- \(F\) = fraction of dataset×model units where method crashes/fails/increases disagreement by >20%.

Also report the raw components.

The score is descriptive, not proof.

---

# 30. Recommended interpretation categories

For each method assign:

```text
KEEP
PROMISING
NICHE
FAIL
```

Definitions:

## KEEP

Prospective:

```text
>=70% median disagreement reduction
<=1% median task cost
works across >=2 model families or is clearly superior in its intended track
```

## PROMISING

```text
>=50% reduction
<=3% task cost
```

or compelling optimizer mechanism with near-raw performance.

## NICHE

Works strongly on one architecture/task family but not broadly.

## FAIL

Poor invariance, >5% task cost, instability, or frequent implementation/rank failures.

Do not make final project recommendation beyond ranking these methods. Another researcher will choose after reviewing `results.md`.

---

# 31. Specific fallback logic

## If BlockAdam is invariant but loses performance

Try:

1. first-layer-only application;
2. SoftBlockAdam alpha 0.1/0.25;
3. data-equivariant initialization;
4. LR ×2;
5. MatrixAdam.

## If MatrixAdam is unstable

Try:

```text
eigenvalue floor 1e-4
EMA beta2 = 0.99 instead of 0.999
```

If still unstable, mark FAIL.

## If GramAnchor loses performance

Try:

1. pivoted anchors;
2. m=16/32;
3. normalize inner products:
   \[
   \frac{\langle z,b\rangle}
   {\|b\|+\epsilon}
   \]
4. NyströmGram;
5. raw+invariant prediction mixture.

## If GramDistance loses performance

Try RBF distances and NyströmGram.

## If PCA remains best invariance but high-cost

Try:

1. HybridSpectral;
2. raw/PCA prediction mixture;
3. canonical branch + weak raw residual.

Do NOT claim PCA itself solved the problem.

## If every invariant interface hurts frozen TFMs

Record that clearly.

Then separate paper method into:

```text
trainable basis-controlled optimization
```

while keeping frozen TFMs as evidence of the broader phenomenon.

That is an acceptable outcome.

---

# 32. Required figures

Generate:

### Figure 1
Method Pareto frontier:

```text
task cost vs disagreement reduction
```

### Figure 2
Optimizer comparison:

```text
AdamW
BlockScalarAdam
BlockAdam
MatrixAdam
SGD control
```

### Figure 3
Matched-function disagreement over epochs.

### Figure 4
Representation methods:

```text
raw
PCA
GramAnchor
GramDistance
NyströmGram
HybridSpectral
```

### Figure 5
Natural-basis results.

### Figure 6
Development vs NEW prospective performance.

### Figure 7
Per-model-family method ranks.

### Figure 8
Worst-orbit task performance.

---

# 33. Required results.md

Produce exactly:

```markdown
# Basis-Controlled Tabular Learning — Method Tournament

## Executive Summary

Do not choose a final paper method.

State:
- strongest optimizer candidate
- strongest representation candidate
- strongest hybrid candidate
- whether any method satisfies KEEP criteria

## Frozen Protocol
- git commit
- hardware
- packages
- seeds
- development datasets
- NEW prospective datasets
- FINALIST_CONFIGS SHA256

## Previous Findings Treated as Fixed
Briefly state:
- orthogonal/natural basis effect established
- AdamW mechanism evidence
- SGD predictive weakness
- PCA task-cost issue
- consistency failure
- AnchorCanonical rank failure

## 1. Stage-1 Method Screening

method | disagreement reduction | task change | runtime | verdict

## 2. Optimizer Methods

### AdamW
### BlockScalarAdam
### BlockAdam
### MatrixAdam
### Data-equivariant initialization
### SoftBlockAdam

method | dataset | disagreement | reduction | task metric | task change

## 3. Optimizer Equivariance Audit

method | epoch0 | epoch1 | epoch5 | final disagreement

Explicitly state whether each claimed equivariant optimizer actually preserved matched-function equivalence.

## 4. Representation Methods

### PCA
### GramAnchor
### GramDistance
### NyströmGram
### HybridSpectral
### MahalanobisGram if run

method | model | dataset | disagreement | reduction | task change

## 5. Anchor / Rank Ablations

m | selection method | empirical rank | disagreement | task change

## 6. Natural Equivalent Basis Results

### Local vs spectral hat
### One-hot vs Helmert

## 7. Hybrid Methods

raw/invariant mixtures and branch methods.

## 8. Equal-HPO Control

## 9. Development Ranking

### Performance-Preserving Invariance Ranking
### Pareto Ranking
### Predictive Performance Ranking
### Paper-Method Score

## 10. Frozen Finalists

List at most 3 methods and exact frozen configs.

## 11. NEW Prospective Results

Clearly mark these datasets as untouched until configs were frozen.

dataset | model | method | raw disagreement | method disagreement | reduction | raw task | method task | relative task change

## 12. Prospective Rankings

### Ranking A — Performance-Preserving Invariance

rank | method | median reduction | median task cost | worst-orbit gain | win/tie/loss

### Ranking B — Pareto Frontier

### Ranking C — Predictive Performance

### Ranking D — Paper-Method Score

## 13. Method-by-Model Matrix

method | controlled MLP | TabM | TabICL | TabPFN | CatBoost

Use:
KEEP / PROMISING / NICHE / FAIL

## 14. Strongest Result

## 15. Strongest Negative Result

## 16. Failed Methods and Why

Do not omit negative results.

## 17. Mechanistic Interpretation

Answer:

- Can blockwise adaptivity retain Adam's task performance?
- Does full matrix adaptivity help beyond scalar BlockAdam?
- Does data-equivariant initialization matter after optimizer correction?
- Can invariant Gram interfaces preserve predictive information?
- Is raw-coordinate information genuinely useful?
- Is a hybrid preferable to full invariance?

## 18. Reviewer Attack Audit

### "You merely replaced Adam with SGD."
### "The new optimizer loses Adam's performance."
### "The representation throws information away."
### "PCA already solves this."
### "The method only handles random rotations."
### "It only works for MLPs."
### "The method was tuned on the test datasets."
### "Basis dependence may actually be beneficial."

## 19. Ranked Candidates for Human Decision

Produce a final table:

rank | method | type | prospective reduction | task cost | model breadth | natural-basis success | condition<=3 behavior | complexity | recommendation

Do NOT automatically choose the paper method.

Rank all serious candidates.

## 20. Suggested Next Experiment for Each Top-3 Method

Give one specific experiment that would most increase or decrease confidence in each.

## 21. Files Produced
```

---

# 34. Decision philosophy

The goal is NOT:

> maximize invariance.

The goal is:

> remove harmful dependence on arbitrary coordinates while retaining useful inductive bias and predictive strength.

A method with:

```text
80% reduction
0% task cost
```

is preferable to one with:

```text
100% reduction
10% task cost.
```

Likewise, if a mildly basis-dependent BlockAdam consistently matches or improves AdamW performance while reducing disagreement by 70–90%, that may be a much stronger paper method than exact canonicalization.

Optimize for a compelling scientific and practical tradeoff, not mathematical purity.