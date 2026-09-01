# AGENT.md — Basis Dependence of Tabular Learning: Falsification + Mechanism + Remedy

## Mission

Stress-test and develop the strongest finding from the previous semantic-orbit experiment:

> Modern tabular learners can produce substantially different predictions when the coordinates used to represent the **same feature representation** are changed by an invertible, well-conditioned — even orthogonal — basis transformation.

The goal is NOT to accumulate more positive examples.

The goal is to determine whether this survives the strongest reviewer objections and whether a useful **non-oracle remedy** exists.

Run for a few GPU-hours. Prioritize decisive experiments over exhaustive sweeps.

Required final output:

```text
results.md
results/raw/
results/processed/
figures/
configs/
```

Do not cherry-pick datasets, transformations, seeds, or methods.

---

# 0. Central mathematical setup

For raw feature \(x_j\), construct a multi-dimensional feature block

\[
z_j=\phi_j(x_j)\in\mathbb R^k.
\]

Create an equivalent representation

\[
z'_j=z_jQ,
\qquad
Q^\top Q=I.
\]

Only rotate coordinates **within one feature's block**.

Never rotate unrelated raw features together.

Primary transformations:

1. orthogonal \(Q\), condition number exactly 1;
2. well-conditioned invertible \(A\), condition ≤3;
3. natural equivalent basis pairs described below.

The headline claim should live or die primarily on orthogonal transforms.

---

# 1. Models

## Frozen models

Required:

- TabICLv2
- TabPFN-2.6

Use official pinned checkpoints.

These are important because representation sensitivity cannot be blamed on retraining randomness.

## Trainable tabular models

Required:

- TabM
- CatBoost
- controlled MLP

Controlled MLP:

```text
3 hidden layers
width = 256
activation = ReLU or GELU, frozen globally
batch norm = OFF
dropout = 0
```

Do not silently tune architecture per dataset.

Seeds:

```text
0, 1, 2
```

---

# 2. Dataset panel and prospective design

Use 18 real datasets if runtime permits; minimum 14.

Use a broad TabArena/OpenML-style panel containing:

- binary classification
- multiclass classification
- regression
- numerical-heavy tables
- mixed categorical/numerical tables
- datasets with cyclic/ordinal variables where known

Reuse:

- adult
- bank-marketing
- diamonds
- bike-sharing
- california_housing
- wine-quality-red

Add 8–12 manageable TabArena/OpenML datasets.

Prefer:

```text
1,000–30,000 rows
<=100 raw columns
no text/image modalities
```

Cap training rows if necessary so the whole protocol completes.

---

## DEVELOPMENT / PROSPECTIVE split

Before running experiments, randomly assign datasets using fixed seed `20260901`:

```text
development panel: ~60%
prospective holdout: ~40%
```

Do NOT inspect holdout results until:

1. method choices are finished;
2. hyperparameters are frozen;
3. `configs/FROZEN_METHOD_CONFIG.json` is written.

Compute SHA256 of this file and record it.

Then run the untouched holdout exactly once.

No changes after seeing holdout results.

If something crashes, only repair implementation errors; record every repair.

---

# 3. Feature-block construction

For each continuous raw feature create:

## RBF block

\[
\phi(x)\in\mathbb R^8
\]

using 8 centers fitted on training quantiles and a fixed width based on adjacent center spacing.

Also create:

## Piecewise-linear / hat basis

8 knots at training quantiles.

The transformation always occurs AFTER constructing the block.

All train/validation/test partitions use identical training-fitted basis parameters.

Never fit transformation statistics using test data.

---

# 4. Experiment A — Does the orthogonal effect replicate broadly?

For every development dataset/model:

Generate:

```text
8 random orthogonal matrices per feature block
```

Use QR decomposition of seeded Gaussian matrices and enforce determinant/sign convention consistently.

Test:

### A1 — one-block rotation

Rotate exactly one numerical feature block.

Choose:

- highest-variance feature;
- or first valid numerical feature under frozen deterministic rule.

### A2 — all-block rotation

Independently rotate every numerical feature block.

### A3 — condition≤3 general invertible transform

Generate singular values in:

```text
[1/√3, √3]
```

with random orthogonal left/right factors.

Audit measured condition number.

---

## Metrics

Classification:

- probability RMSE / mean absolute probability difference
- Jensen-Shannon divergence
- predicted-label flip %
- log loss
- accuracy
- ROC-AUC where appropriate

Regression:

\[
D =
RMSE(\hat y,\hat y_Q)/std(y)
\]

plus:

- prediction correlation
- RMSE
- MAE

For every orbit report:

```text
mean disagreement
max disagreement
performance mean
performance worst
performance span
```

---

# 5. Experiment B — Is this just optimization geometry?

Run controlled MLP on 4 development datasets:

- 2 strongest regression datasets
- 1 classification dataset
- 1 moderate-effect dataset

Use original basis and 4 orthogonal orbit members.

The key test uses **function-matched initialization**.

Suppose:

\[
z'=zQ.
\]

For the first-layer weights operating on that block:

\[
W' = WQ
\]

or the algebraically equivalent orientation required by the implementation.

UNIT TEST BEFORE TRAINING:

For 1,000 rows verify:

\[
\max |f_{\theta}(z)-f_{\theta'}(zQ)| < 10^{-5}
\]

before the first optimization step.

If this fails, fix orientation before proceeding.

---

## B1. Same-seed ordinary initialization + AdamW

Existing default-like behavior.

## B2. Function-matched initialization + AdamW

Tests whether matching the initial function removes the difference.

## B3. Function-matched initialization + SGD

Use:

```text
momentum = 0.9
```

Same minibatch order between the two basis runs.

## B4. Function-matched initialization + plain SGD

```text
momentum = 0
weight_decay = 0
```

Same minibatches.

This is an important near-equivariant control.

## B5. Function-matched + AdamW without weight decay

Tests optimizer coordinate adaptivity separately from weight decay.

---

## Interpretation

If:

```text
same seed differs
function-matched SGD ≈ invariant
function-matched AdamW still differs
```

then we have evidence for TWO sources of basis dependence:

1. initialization/function prior;
2. coordinatewise adaptive optimization.

Plot disagreement vs training epoch.

Save predictions at:

```text
epoch 0
1
2
5
10
20
final
```

---

# 6. Equal-HPO objection

A reviewer may claim one representation simply needs different hyperparameters.

On 3 development datasets run a small equal-budget search separately for original and transformed representations.

MLP/TabM:

```text
learning_rate = [3e-4, 1e-3, 3e-3]
weight_decay = [0, 1e-5, 1e-4]
```

Maximum 9 trials per representation.

Use identical HPO budget.

Select by validation metric only.

Report whether basis disagreement/performance differences remain after representation-specific HPO.

Do NOT expand the search after seeing results.

---

# 7. Experiment C — Natural equivalent bases

Random rotations are scientifically clean but may be criticized as artificial.

Test naturally interpretable coordinate systems spanning exactly the same feature space.

Every pair must pass explicit reconstruction/equivalence tests.

---

## C1. Nominal: one-hot ↔ Helmert basis

For a categorical variable with \(m\) categories:

Original:

\[
z_{onehot}\in\mathbb R^m.
\]

Construct a full \(m\times m\) orthogonal contrast matrix containing:

- constant direction;
- Helmert contrast directions.

Then:

\[
z_H=z_{onehot}H.
\]

Exact inverse:

\[
z_{onehot}=z_HH^\top.
\]

Use at least 3 datasets containing useful categoricals.

Feed both representations as numeric blocks so the information is exactly controlled.

---

## C2. Cyclic: Fourier origin changes

For known cyclic variables such as:

- hour
- weekday
- month

use Fourier block:

\[
[\sin\theta,\cos\theta,\sin2\theta,\cos2\theta,\ldots].
\]

Change cycle origin.

Each frequency pair undergoes an exact 2-D orthogonal rotation.

Compare predictions across origins.

---

## C3. Numerical spline: local basis ↔ spectral basis

Construct an 8-dimensional piecewise-linear/hat basis.

Create a deterministic orthogonal spectral coordinate system using a DCT-like orthogonal matrix:

\[
z_{spectral}=z_{local}Q_{DCT}.
\]

These span exactly the same feature representation.

This gives an interpretable comparison:

```text
localized coordinates
vs
distributed/spectral coordinates
```

---

## C4. Polynomial sanity family

Where numerically stable, construct degree ≤4 polynomial space using:

- monomial basis;
- Legendre basis.

Verify the exact linear mapping numerically.

Only retain pairs with:

```text
condition number <= 10
reconstruction relative error < 1e-6
```

This is secondary.

---

# 8. Information-equivalence audit

Before any model result can enter a table, verify:

\[
\|Z-Z'Q^{-1}\|/\|Z\| < 10^{-6}
\]

or equivalent reconstruction appropriate to the basis pair.

Save:

```text
results/processed/equivalence_audit.csv
```

Columns:

```text
dataset
feature
basis_a
basis_b
dimension
condition_number
reconstruction_error
passes
```

Any failed equivalence audit is excluded and explicitly reported.

---

# 9. Experiment D — Non-oracle remedies

Do NOT use knowledge of the hidden \(Q\) for the primary remedy.

Oracle inverse remains a ceiling only.

Test the following.

---

## D0. Raw representation

No repair.

---

## D1. Standardization

Per-coordinate training-set z-score.

Important baseline.

---

## D2. Whitening

Training covariance whitening inside each feature block.

This removes scale/correlation but leaves an orthogonal ambiguity.

Test whether that is enough.

---

## D3. PCA/SVD canonical coordinates

For each feature block training matrix:

\[
Z=U\Sigma V^\top.
\]

Represent samples as:

\[
z_{canon}=zV.
\]

For transformed input:

\[
Z'=ZQ
\]

the learned right singular vectors should transform correspondingly, giving approximately identical canonical coordinates.

Implement deterministic:

- descending singular-value ordering;
- sign fixing;
- degeneracy detection.

Sign convention:

For each component, find training row with largest absolute component and force that score positive.

If adjacent singular values differ by <1%, flag the block as potentially degenerate.

Report invariant error separately for degenerate vs non-degenerate blocks.

This method uses NO \(Q\).

---

# 10. D4 — Anchor-coordinate invariant representation

This is a particularly important candidate remedy.

For feature block:

\[
z\in\mathbb R^k
\]

choose a fixed target-free anchor matrix:

\[
B\in\mathbb R^{m\times k}
\]

from training rows.

Use:

```text
m = max(2k, 16)
```

anchors selected by fixed RNG on row indices, NOT by feature geometry or labels.

Define:

\[
c(z)=zB^+,
\]

where \(B^+\) is the Moore-Penrose pseudoinverse.

For an invertible basis transform:

\[
z'=zA,\qquad B'=BA,
\]

we should have:

\[
z'A(B')^+ \text{(implementation form corrected)}
\]

and mathematically verify the intended identity:

\[
(zA)(BA)^+=zB^+.
\]

UNIT TEST this numerically.

Target:

```text
relative coordinate difference < 1e-5
```

for:

- orthogonal Q
- condition≤3 A

If numerical instability occurs:

- increase anchors;
- use float64 for pseudoinverse;
- audit rank and condition number.

Do NOT use labels.

Call this method:

```text
AnchorCanonical
```

for reporting.

This is potentially the strongest non-oracle method because it targets invariance to general invertible basis changes, not just rotations.

---

# 11. D5 — Orbit consistency training

Controlled MLP and TabM where practical.

For original representation \(z\) and random equivalent \(zQ\):

Classification:

\[
L
=
L_{task}
+
\lambda JS(p(z),p(zQ)).
\]

Regression:

\[
L
=
L_{task}
+
\lambda
\frac{(\hat y(z)-\hat y(zQ))^2}{Var(y)}.
\]

Use:

```text
lambda = 1.0
```

because the previous experiment strongly favored this value.

Do not retune on prospective holdout.

At each batch sample a fresh orthogonal Q per feature block from a fixed seeded pool.

---

# 12. D6 — Dual-view model

Test only if D3/D4 are promising.

Inputs:

```text
raw feature blocks
+
AnchorCanonical or PCA-canonical feature blocks
```

Train two branches:

\[
h_{raw}
\]

and

\[
h_{canon}.
\]

Combine:

\[
h=h_{canon}+\alpha h_{raw}
\]

where \(\alpha\) is a learned scalar initialized to 0.1.

Add consistency penalty to final predictions.

Question:

> Can the model preserve useful raw-coordinate inductive bias while preventing catastrophic basis dependence?

Do not build a large new architecture.

---

# 13. Oracle ceiling

For diagnostics only:

Given known transformation \(Q\), explicitly invert it before model input.

This should give essentially zero disagreement.

Label everywhere:

```text
ORACLE INVERSE — NOT A METHOD
```

Do not use it in average-rank claims.

---

# 14. Repair evaluation

For every repair report:

\[
Reduction
=
1-
\frac{D_{repair}}{D_{raw}}.
\]

Also report task cost:

Classification:

\[
\Delta logloss
\]

Regression:

\[
\frac{RMSE_{repair}-RMSE_{raw}}{RMSE_{raw}}.
\]

A useful repair should ideally achieve:

```text
>=70% median disagreement reduction
<=1% median relative performance degradation
```

and should work on prospective datasets.

---

# 15. Experiment E — Is basis sensitivity beneficial?

Do not assume invariance is always desirable.

For each trainable model compare across orbit members:

- validation performance
- test performance
- prediction disagreement

Ask:

> Do particular bases consistently perform better?

Compute whether an oracle-best basis materially outperforms the canonical/original basis.

Also ask whether a validation-selected basis transfers to test.

If some bases genuinely provide useful inductive bias, the paper should advocate:

```text
controlled / learnable basis handling
```

rather than absolute invariance.

This is important intellectually.

---

# 16. Prospective holdout

After development experiments:

1. choose one primary method:
   - AnchorCanonical;
   - PCA-canonical;
   - consistency;
   - or simplest winning combination;

2. freeze all settings;

3. save:

```text
configs/FROZEN_METHOD_CONFIG.json
```

4. record SHA256 in `results.md`;

5. run all prospective datasets.

Prospective comparison:

```text
raw
standardization
whitening
primary proposed method
oracle inverse
```

Models:

```text
TabICLv2
TabPFN-2.6
TabM or controlled MLP
CatBoost where feasible
```

Do not add a new method after seeing prospective outcomes.

---

# 17. Statistical reporting

Primary unit:

```text
dataset × model
```

Do not pretend orbit members are independent datasets.

Report:

- dataset-level medians
- average ranks
- win/tie/loss
- bootstrap CI across datasets
- seed variability

Primary hypothesis:

\[
H:
D_{orthogonal}>0
\]

and proposed repair reduces \(D\).

Also report separate results for:

```text
random orthogonal bases
natural equivalent bases
```

Do not pool them only.

---

# 18. Crucial plots

Generate:

## Figure 1
Orthogonal basis disagreement:

```text
dataset × model heatmap
```

## Figure 2
Original prediction vs rotated-basis prediction.

## Figure 3
Disagreement vs training epoch:

```text
same-seed AdamW
function-matched AdamW
function-matched SGD
```

## Figure 4
Natural basis-pair results:

```text
onehot↔Helmert
local↔spectral spline
Fourier origin
```

## Figure 5
Repair comparison:

```text
raw
standardize
whiten
PCA canonical
AnchorCanonical
consistency
```

## Figure 6
Task performance vs invariance tradeoff.

## Figure 7
Development vs prospective results.

## Figure 8
Random-basis vs natural-basis effect size.

---

# 19. Kill / GO criteria

## STRONG GO

Recommend full ICLR/ICML/NeurIPS paper development if most of the following occur:

### Phenomenon

Orthogonal basis sensitivity repeats on:

```text
>= 50% of real datasets
>= 3 model families
```

with meaningful prediction disagreement.

AND natural equivalent basis pairs also reproduce nontrivial sensitivity.

### Mechanism

Function matching and optimizer experiments explain at least part of the effect, e.g.:

```text
ordinary init + AdamW: large disagreement
function-matched SGD: much smaller disagreement
```

or another clear mechanistic pattern.

### Remedy

A NON-ORACLE method reduces median disagreement by roughly:

```text
>=70%
```

with:

```text
<=1% relative median task degradation
```

and succeeds prospectively.

A repair that also improves worst-orbit performance is especially strong.

---

## PHENOMENON STRONG / METHOD UNSOLVED

Use this verdict if:

- orthogonal + natural basis effects replicate broadly;
- mechanism is convincing;
- but non-oracle repairs materially hurt predictive performance.

Still potentially publishable, but more method development is needed.

---

## WEAK GO

Use if:

- random rotations are strong;
- natural basis effects are weaker;
- repairs partially work.

Would need a narrower framing such as numerical embedding basis dependence.

---

## NO-GO

Recommend abandoning the main direction if any of these dominate:

- broad replication disappears;
- only CatBoost/tree models are sensitive;
- only badly conditioned transforms matter;
- natural equivalent bases show almost no effect;
- equal HPO removes the phenomenon;
- prospective datasets fail badly;
- meaningful invariance requires oracle knowledge of Q;
- performance improvements from favorable bases clearly outweigh any rationale for treating basis as nuisance and no principled handling emerges.

---

# 20. Implementation integrity checks

Write unit tests for:

```text
orthogonality: ||QᵀQ-I|| < 1e-6

condition-number constraint

basis reconstruction error < 1e-6

function-matched initial predictions < 1e-5 difference

AnchorCanonical invariance < 1e-5 where full-rank

no test-data fitting

identical row hashes across orbit members

identical target hashes

same minibatch ordering for optimizer mechanism tests
```

A failed integrity check invalidates the corresponding result.

---

# 21. Compute discipline

Priority order:

```text
1. Broad orthogonal replication
2. Natural equivalent bases
3. Function-matched optimizer mechanism
4. AnchorCanonical/PCA remedy
5. Prospective holdout
6. Equal HPO
7. Dual-view refinement
```

If runtime is running long, drop:

```text
polynomial basis
extra HPO
dual-view
```

before dropping prospective validation.

Never spend the budget on huge tuning sweeps.

---

# 22. Required results.md

Produce exactly this structure:

```markdown
# Basis Dependence of Tabular Learning — Confirmation Round

## Executive Verdict
STRONG-GO / PHENOMENON-STRONG-METHOD-UNSOLVED / WEAK-GO / NO-GO

## One-Paragraph Conclusion

## Frozen Protocol
- git commit
- hardware
- package versions
- seeds
- model checkpoints
- development datasets
- prospective datasets
- frozen method config SHA256

## 1. Orthogonal Basis Replication

dataset | model | one-block disagreement | all-block disagreement | task original | orbit mean | orbit worst

## 2. Condition<=3 Results

## 3. Natural Equivalent Bases

dataset | model | basis pair | reconstruction error | disagreement | performance delta

### One-hot vs Helmert
### Local spline vs spectral spline
### Fourier-origin changes
### Other valid natural basis pairs

## 4. Mechanism: Initialization and Optimizer

dataset | optimizer | initialization | epoch0 disagreement | final disagreement | task metric

Explicitly answer:

- Does function matching eliminate the initial difference?
- Does SGD preserve equivalence better than AdamW?
- Does AdamW reintroduce coordinate dependence?
- What role does weight decay play?

## 5. Equal-HPO Control

original vs transformed after equal independent HPO budget.

## 6. Non-Oracle Repairs

method | disagreement | reduction | task metric | relative task change

Include:

- standardization
- whitening
- PCA/SVD canonical
- AnchorCanonical
- consistency
- dual view if run
- oracle inverse separately

## 7. AnchorCanonical Audit

Report numerical invariance under:

- orthogonal transformations
- condition<=3 transformations

and rank/conditioning failures.

## 8. Is Basis Sensitivity Sometimes Helpful?

Report oracle-best basis and validation-selected basis results.

## 9. Prospective Holdout

THIS SECTION MUST CLEARLY DISTINGUISH DATASETS NEVER USED DURING DEVELOPMENT.

dataset | model | raw disagreement | proposed disagreement | reduction | task raw | task proposed

## 10. Development vs Prospective Summary

## 11. Strongest Evidence FOR the Hypothesis

## 12. Strongest Evidence AGAINST the Hypothesis

## 13. Reviewer Attack Audit

Answer directly:

### "Random rotations are artificial."
Do natural bases reproduce the phenomenon?

### "This is just poor numerical conditioning."
What happens for condition-number-one orthogonal transforms?

### "You used the wrong hyperparameters."
What happens after equal HPO?

### "This is only optimization noise."
What happens in frozen TFMs and under function matching?

### "Of course inverse canonicalization works."
Does a non-oracle method work?

### "Maybe basis choice is useful rather than nuisance."
What does Experiment E show?

### "The method was tuned to these datasets."
What happened prospectively?

## 14. ICLR/ICML/NeurIPS Assessment

Give:
- novelty assessment
- empirical strength
- method strength
- biggest remaining weakness
- estimated paper direction

## 15. Recommended Next Step

If STRONG-GO:
state the minimal paper contribution and next experiments.

If METHOD-UNSOLVED:
state exactly what the missing method must accomplish.

If NO-GO:
state why this should be abandoned.

## 16. Files Produced
```

---

# 23. Final scientific framing

Do NOT conclude:

> Neural networks should be invariant to all invertible transformations.

That claim is too broad.

The hypothesis under test is narrower:

> When multiple coordinates jointly represent a single underlying feature, the arbitrary basis used inside that feature representation can introduce a hidden inductive prior. Modern tabular learners may therefore depend on arbitrary representational coordinates even when the feature subspace and information content are unchanged.

The strongest version of the project is:

> characterize the hidden basis prior, explain where it comes from, and construct a basis-independent or basis-controlled interface that preserves predictive performance.

Optimize the experiments for deciding whether THAT claim deserves a full top-conference paper.