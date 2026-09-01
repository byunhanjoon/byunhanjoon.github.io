# AGENT.md — Guarded Basis Control: High-Control, Low-Tail-Risk Methods

## Mission

Develop the final candidate method for a paper on hidden basis priors in tabular numerical embeddings.

Previous frozen evidence already establishes:

- information-equivalent orthogonal feature-basis changes alter predictions across modern tabular learners;
- natural bases such as local/spectral splines and one-hot/Helmert reproduce the effect;
- basis sensitivity exists inside standard PLE/RBF numerical embeddings;
- sensitivity generally grows with embedding dimension;
- pure Gram coordinates remove orthogonal basis dependence exactly;
- pure Gram can alter inductive bias and cause severe generalization failures despite essentially exact information reconstruction;
- `Raw + GramAnchor @ 0.75` achieved:
  - 75% median control;
  - median normalized excess risk ≈ 0.001;
  - p95 risk ≈ 0.050;
  - max risk ≈ 0.128;
- `SafeRankGram` achieved:
  - 50% median control;
  - median risk 0;
  - p95 ≈ 0.005;
  - max ≈ 0.010;
- `SafeGram` is safer than fixed alpha but too conservative;
- catastrophic Gram failures are primarily **Type C: changed generalization / inductive bias**, not information loss.

Do NOT rerun broad discovery experiments.

This round asks:

1. Can we retain ~60–75% basis control while achieving SafeRank-like tail safety?
2. Can we selectively remove basis dependence only from feature blocks where doing so is safe?
3. Can the method work with a **single trained model**, avoiding two full inference paths?
4. Does the result replicate broadly inside PLE/RBF numerical embeddings?

Required output:

```text
results.md
results/raw/
results/processed/
figures/
configs/
```

Target runtime: several GPU-hours, not days.

---

# 1. Success criteria

Primary target on a NEW untouched prospective panel:

```text
median disagreement reduction >= 60%
median normalized excess risk C <= 0.005
p95 C <= 0.02
max C <= 0.10
```

Strong target:

```text
median disagreement reduction >= 70%
median C <= 0.005
p95 C <= 0.01
max C <= 0.05
```

A method with 65% control and tiny tails is preferable to one with 100% control and dangerous failures.

---

# 2. Core metric

Use normalized excess risk:

\[
C =
\frac{L_{method}-L_{raw}}
{\max(L_{trivial}-L_{raw},10^{-8})}.
\]

Classification:

\[
L = \text{log loss}.
\]

`L_trivial` = training-class-prior predictor.

Regression:

\[
L = RMSE.
\]

`L_trivial` = training-mean predictor.

Always also report:

- raw task loss;
- method loss;
- absolute loss difference;
- relative loss difference;
- C.

Never judge safety from relative percentage error alone.

---

# 3. Dataset protocol

## Development datasets

Use:

```text
steel-plates-fault
cardiotocography
sylvine
debutanizer
wall-robot-navigation
california_housing
phoneme
space-ga
```

These include:

- known catastrophic Gram cells;
- positive Gram cells;
- classification and regression;
- small and substantial raw basis sensitivity.

## NEW untouched prospective panel

Select 10–14 OpenML/TabArena datasets never used in ANY earlier round.

Maintain a blacklist containing all previously used datasets.

Preferred constraints:

```text
1,000–50,000 rows
<=100 raw columns preferred
mix classification/regression
no text/image modalities
```

Before any prospective outcome is loaded, write:

```text
configs/GUARDED_PROSPECTIVE_PANEL.json
```

and record SHA256.

---

# 4. Models

Required for general interface experiments:

```text
controlled MLP
TabM-D
CatBoost
TabICLv2
TabPFN-2.6
```

Required for internal numerical-embedding experiments:

```text
controlled MLP
TabM-D
ResNet-style tabular model
```

Strongly preferred if straightforward:

```text
FT-Transformer
```

Do not delay the full experiment if FT-Transformer integration becomes a blocker.

Seeds:

```text
0, 1, 2
```

If runtime is excessive, use 2 seeds during development and 3 for prospective finalists.

---

# 5. Representation

For numerical features use standard 8-D blocks:

```text
PLE
RBF
```

External-interface experiments may continue using the frozen RBF/hat representation from earlier work.

Orthogonal orbit:

```text
8 random orthogonal matrices
```

Natural controls:

```text
local hat <-> spectral hat
one-hot <-> Helmert where available
Fourier-origin shift where metadata exists
```

All equivalence audits must pass `<1e-6`.

---

# PART A — GUARDEDGRAM

# 6. Motivation

Current methods expose a tradeoff:

```text
fixed alpha=.75:
high control, moderate tail risk

SafeGram:
very safe, too conservative
```

Develop a gate that uses `alpha=.75` by default and backs off only when validation indicates danger.

Call this:

```text
GuardedGram
```

---

# 7. Candidate alpha set

Use:

```text
[0.75, 0.50, 0.25, 0]
```

Do NOT allow alpha=1 in the primary GuardedGram method.

Pure Gram remains a baseline.

---

# 8. GuardedGram-G1 — Harm-detection gate

For candidate alpha \(a\), compute paired validation-row excess losses:

\[
d_i(a)
=
\ell_i(a)-\ell_i(raw).
\]

Convert aggregate loss to normalized excess risk \(C_a\).

Start at:

\[
a=0.75.
\]

Test:

\[
H_0:C_a\le\tau
\]

versus

\[
H_1:C_a>\tau.
\]

Use one-sided paired bootstrap:

```text
1000 resamples
```

Back off only if there is significant evidence of harmful excess risk:

```text
p < 0.05
```

Then test 0.50, 0.25, then 0.

Thresholds:

```text
tau = 0
0.005
0.01
```

Call:

```text
GuardedGram-G1-t0
GuardedGram-G1-t005
GuardedGram-G1-t01
```

This is intentionally more permissive than SafeGram.

---

# 9. GuardedGram-G2 — Tunable confidence guard

Estimate:

\[
\hat C_a+\gamma SE(C_a).
\]

Use alpha if:

\[
\hat C_a+\gamma SE(C_a)\le\tau.
\]

Test:

```text
gamma = 0
0.5
1.0
1.64
```

and:

```text
tau = 0.005
0.01
0.02
```

Start from alpha .75 and back off.

This creates an explicit:

```text
control <-> statistical conservatism
```

frontier.

Do NOT run every combination prospectively.

Select one configuration on development only.

---

# 10. GuardedGram-G3 — Two-stage danger detector

Stage 1:

compare only:

```text
raw
alpha=.75
```

If .75 is clearly safe, use it.

If clearly unsafe, test lower alphas.

If ambiguous, default to `.5`, not zero.

Definitions:

### clearly safe

\[
UCB_{80}(C_{.75})\le0.01.
\]

### clearly unsafe

\[
LCB_{80}(C_{.75})>0.01.
\]

### ambiguous

```text
alpha=.5
```

For unsafe cases, recursively test `.5`, then `.25`.

The 80% bounds are intentionally weaker than earlier SafeGram bounds.

Evaluate whether this improves control without sacrificing tail safety.

---

# 11. Existing gating baselines

Required:

```text
Raw
Raw+Gram@.5
Raw+Gram@.75
SafeGram-t01
SafeRankGram-t01
```

Do not retune them.

---

# PART B — FEATURE-BLOCK SELECTIVE INVARIANCE

# 12. Motivation

The current global alpha assumes every feature should retain the same amount of raw-coordinate prior.

That is unlikely to be optimal.

Develop a method that leaves dangerous feature blocks raw while converting safe blocks to Gram coordinates.

Call:

```text
BlockGuard
```

Unlike prediction mixtures, BlockGuard produces ONE input representation and requires only ONE model at inference.

---

# 13. One-block validation intervention

For every feature block \(j\):

1. keep every feature raw;
2. replace only block \(j\) with Gram coordinates;
3. train/evaluate on validation;
4. record:

\[
C_j
\]

and basis-disagreement reduction contributed by block j.

Also record target-free descriptors:

```text
empirical rank
block dimension
spectrum entropy
largest/smallest nonzero singular value
condition proxy
mean Gram diagonal
embedding type
raw one-block orbit disagreement
```

Do this on development data only.

---

# 14. BlockGuard-Greedy

Goal:

maximize number / importance of invariant blocks subject to validation risk.

Start from raw representation.

For each candidate block estimate:

```text
benefit_j = basis disagreement removed
cost_j = validation normalized excess risk
```

Sort roughly by:

\[
score_j
=
\frac{benefit_j}
{\max(cost_j+\epsilon,\epsilon)}.
\]

Because interactions may exist, do not assume costs are additive.

Add blocks in batches and re-evaluate cumulative validation loss.

Stop before cumulative:

\[
C>\tau.
\]

Test:

```text
tau = 0
0.005
0.01
0.02
```

Maximum number of retraining stages per dataset:

```text
8
```

If there are many feature blocks, group adjacent ranks in the sorted list.

---

# 15. BlockGuard-Grouped

Cheaper alternative.

Divide blocks into four groups according to one-block danger score:

```text
very safe
safe
uncertain
dangerous
```

Construct candidate representations:

```text
raw only
very-safe Gram
very-safe + safe Gram
all except dangerous Gram
all Gram
```

Select the most invariant representation satisfying validation safety.

This should require at most five models per dataset.

Compare to greedy BlockGuard.

---

# 16. BlockGuard metrics

Report:

```text
fraction of feature blocks made invariant
fraction of numerical embedding dimensions made invariant
overall prediction disagreement reduction
C
p95/max C
inference cost
```

A key figure should show:

\[
\text{fraction Gram-controlled features}
\rightarrow
\text{basis sensitivity / task risk}.
\]

---

# PART C — SINGLE-MODEL DUAL VIEW

# 17. Motivation

Prediction-level Raw+Gram mixtures require two full model fits and two inference passes.

Test whether the tradeoff can be obtained inside one model.

Call:

```text
DualViewGram
```

Use only:

```text
controlled MLP
TabM-D
ResNet-style model
```

---

# 18. Dual-view feature encoder

For each feature block:

Raw branch:

\[
h_j^{raw}
=
W_j^{raw}z_j.
\]

Invariant branch:

\[
h_j^{gram}
=
W_j^{gram}g_j.
\]

Both map to the same hidden width.

Mix:

\[
h_j
=
(1-\alpha_j)h_j^{raw}
+
\alpha_j h_j^{gram}.
\]

Then pass through the ordinary backbone.

---

# 19. DualView variants

## D1 — global fixed gate

```text
alpha=.5
alpha=.75
```

## D2 — global guarded gate

Use best GuardedGram development alpha-selection rule.

Train separate candidate models only for required alphas.

## D3 — BlockGuard binary gates

Use:

\[
\alpha_j\in\{0,1\}
\]

from BlockGuard selections.

## D4 — regularized learnable gates

Initialize:

```text
alpha_j = 0.5
```

Parameterize:

\[
\alpha_j=\sigma(a_j).
\]

Train with penalty:

\[
L =
L_{task}
+
\lambda\sum_j \alpha_j(1-\alpha_j)
\]

to encourage clear raw/invariant choices.

Test:

```text
lambda = 1e-4
1e-3
1e-2
```

This is exploratory.

Do not send D4 to prospective evaluation unless it clearly dominates simpler gates on development.

---

# 20. Efficiency accounting

Report:

```text
training time
inference time
parameter count
peak GPU memory
```

Normalize to raw baseline.

Required comparison:

```text
two-model Raw+Gram prediction mixture
vs
single-model DualViewGram
vs
BlockGuard single representation
```

---

# PART D — NUMERICAL EMBEDDING CONFIRMATION

# 21. Primary question

Do standard numerical embeddings contain a practically meaningful hidden basis hyperparameter?

Use:

```text
PLE
RBF
```

and:

```text
k = 4, 8, 16, 32
```

on a development subset of 4 datasets.

On the full embedding benchmark use:

```text
k = 8 and 16
```

---

# 22. Backbones

Required:

```text
controlled MLP
TabM-D
ResNet-style tabular network
```

Preferred:

```text
FT-Transformer
```

Use the same train/validation/test splits across basis variants.

---

# 23. Conditions

For every embedding:

```text
default basis
8 random orthogonal bases
Gram-controlled basis
best GuardedGram method
best BlockGuard method where applicable
best DualView method where applicable
```

---

# 24. Basis selection headroom

For each dataset/model/embedding:

report:

```text
default task error
mean random-basis error
best random-basis error
worst random-basis error
validation-selected random-basis error
```

Critical question:

> Is default PLE/RBF basis systematically optimal?

Expected answer may be no.

Do not optimize basis matrices directly yet.

---

# 25. Dimension scaling analysis

Measure:

\[
D(k)
\]

for:

```text
k=4,8,16,32
```

Fit simple log-linear descriptive relationship:

\[
D(k)
=
a+b\log_2k.
\]

Report per model family and embedding type.

No strong theoretical claim required.

---

# PART E — SAFE BASIS SEARCH AS OPTIONAL HIGH-UPSIDE EXPERIMENT

# 26. Motivation

Previous results show that some equivalent bases significantly outperform the default basis.

If cheap, test whether validation can exploit this.

This is OPTIONAL and runs only after core experiments.

---

# 27. Basis portfolio

For each numerical embedding sample:

```text
8 orthogonal bases + default
```

Choose best using validation metric only.

Evaluate selected basis on test.

Compare:

```text
default
validation-selected basis
oracle best test basis
Gram invariant
GuardedGram
```

If validation selection consistently improves the default, this suggests a second interpretation:

> basis choice itself is an underused hyperparameter.

If it fails to transfer, preserve as negative result.

---

# PART F — DEVELOPMENT PRUNING

# 28. Stage 1

Use:

```text
steel-plates-fault
cardiotocography
california_housing
phoneme
```

Test:

```text
GuardedGram G1
GuardedGram G2
GuardedGram G3
BlockGuard grouped
BlockGuard greedy
DualView fixed .75
```

A method survives if:

```text
median control >= 50%
p95 C <= 0.03
max C <= 0.10
```

OR

it has significantly better efficiency with:

```text
>=40% control
p95 C <=0.02
```

---

# 29. Stage 2

Run survivors over full development panel.

Choose max:

```text
4 finalists
```

Possible finalists should ideally include:

```text
best GuardedGram
best BlockGuard
best DualViewGram
fixed Raw+Gram@.75 baseline
```

SafeRankGram remains a safety reference.

---

# PART G — FINAL PROSPECTIVE FREEZE

# 30. Freeze

Before prospective data:

write:

```text
configs/GUARDED_FINALISTS.json
```

Include every:

```text
threshold
confidence level
candidate alpha
feature grouping rule
embedding setting
rank/anchor setting
optimizer
learning rate
architecture setting
```

Compute and store SHA256.

No modifications after prospective data are accessed.

---

# 31. NEW prospective evaluation

Run all finalists on the 10–14 untouched datasets.

Required model breadth:

General interface:

```text
MLP
TabM
CatBoost
TabICLv2
TabPFN
```

Embedding-specific method:

```text
MLP
TabM
ResNet
FT-Transformer if available
```

---

# 32. Prospective primary metrics

For every method:

```text
median disagreement reduction
p25/p75 disagreement reduction
median C
p90 C
p95 C
max C
task W/T/L
fraction C<0
fraction C>0.01
fraction C>0.05
raw fallback rate
mean predictive rank
```

For BlockGuard:

```text
median invariant-feature fraction
```

For DualView:

```text
inference overhead
```

---

# 33. Tail analysis

Explicitly list the 10 worst method cells ranked by C.

For each report:

```text
dataset
model
task
raw error
method error
C
selected alpha
fraction Gram blocks
validation C
test C
```

We need to know whether the gate fails because:

```text
validation missed a real failure
validation noise
distribution shift
model instability
```

---

# 34. Ranking rules

Produce SIX rankings.

## Ranking A — Paper Safety

Eligible only if:

```text
median C <= .005
p95 C <= .02
max C <= .10
```

Rank by disagreement reduction.

## Ranking B — Strict Safety

Eligible only if:

```text
p95 C <= .01
max C <= .05
```

Rank by disagreement reduction.

## Ranking C — Basis Control

Rank purely by disagreement reduction.

## Ranking D — Predictive Performance

Rank by task performance / average rank.

## Ranking E — Efficiency

Rank by:

```text
inference overhead
training overhead
parameter overhead
```

among methods achieving >=50% control.

## Ranking F — Overall Paper Candidate

Use:

\[
S=
R
-
3\max(\text{median}C,0)
-
3\max(P95(C)-0.01,0)
-
2\max(C_{max}-0.05,0)
-
0.05\log_2(\text{inference multiplier}).
\]

Report every raw component.

Do NOT automatically choose the paper method.

---

# 35. Interpretation thresholds

## FINAL-METHOD-SIGNAL

At least one method achieves on prospective data:

```text
>=60% median control
median C <= .005
p95 C <= .02
max C <= .10
>=3 model families
```

Strongest case:

```text
>=70% control
p95 <= .01
max <= .05
```

## SAFE-BUT-CONSERVATIVE

Best adaptive method:

```text
<50% median control
```

despite strong tail safety.

## FIXED-MIXTURE-WINS

Fixed `.75` remains substantially better than adaptive methods and still passes tail criteria.

## BLOCK-SELECTION-WINS

BlockGuard obtains:

```text
>=60% control
p95 <=.02
single-model inference
```

This would be especially attractive.

## METHOD-STILL-UNSOLVED

No method reaches 60% control without meaningful tail failure.

---

# 36. Required figures

Generate:

## Figure 1

Pareto plot:

```text
x = p95 C
y = median basis control
```

Highlight `.75`, SafeRank, GuardedGram, BlockGuard.

## Figure 2

Tail CDF of C.

## Figure 3

GuardedGram selected-alpha histogram.

## Figure 4

BlockGuard:

```text
fraction invariant feature blocks
vs
task risk / disagreement
```

## Figure 5

Embedding dimension k vs basis disagreement.

## Figure 6

Default basis vs best/validation-selected equivalent basis.

## Figure 7

Dual-model vs single-model method efficiency.

## Figure 8

Development vs prospective tradeoff.

---

# 37. Required results.md

Produce EXACTLY:

```markdown
# Guarded Basis Control — Final Method Search

## Executive Verdict
FINAL-METHOD-SIGNAL /
SAFE-BUT-CONSERVATIVE /
FIXED-MIXTURE-WINS /
BLOCK-SELECTION-WINS /
METHOD-STILL-UNSOLVED

## One-Paragraph Summary

## Frozen Protocol
- commit
- hardware
- packages
- seeds
- development datasets
- untouched prospective datasets
- finalist config SHA256

## 1. Prior Evidence Treated as Fixed

State:
- pure Gram exact invariance
- fixed .75 tradeoff
- SafeRank safety
- Type-C failures
- numerical-embedding sensitivity

## 2. GuardedGram Development

method | median alpha | control | median C | p95 C | max C | fallback

### G1 Harm Detection
### G2 Confidence Guard
### G3 Two-Stage Guard

## 3. GuardedGram Ablations

confidence/threshold | control | p95 | max | predictive rank

## 4. BlockGuard

method | invariant-block fraction | control | median C | p95 | max | inference multiplier

## 5. Which Features Stay Raw?

Summarize feature descriptors for:
- Gram-selected blocks
- raw-retained blocks

Look for patterns in:
rank, spectrum entropy, dimension, etc.

## 6. DualViewGram

method | control | C | predictive metric | params | inference time

## 7. Efficiency Comparison

raw
two-fit fixed .75
GuardedGram
BlockGuard
DualViewGram

## 8. Numerical Embedding Confirmation

dataset | backbone | embedding | k | default-vs-rotated disagreement | task span

## 9. Embedding Dimension Scaling

embedding | model | k | disagreement

## 10. Gram/Guard Methods Inside Embeddings

method | control | median C | p95 C | max C

## 11. Basis Portfolio / Basis Search
If run.

default vs validation-selected vs oracle best.

## 12. Stage-1 Pruning

## 13. Full Development Ranking

## 14. Frozen Finalists

List <=4 exact configs and SHA.

## 15. NEW Untouched Prospective Results

dataset | model | method | selected alpha | invariant-block fraction | control | raw task | method task | C

## 16. Prospective Aggregate Table

method | median control | median C | p90 | p95 | max | W/T/L | fallback | predictive rank | inference multiplier

## 17. Worst 10 Tail Cells

dataset | model | method | validation C | test C | alpha | invariant fraction | explanation

## 18. Paper-Safety Ranking

## 19. Strict-Safety Ranking

## 20. Basis-Control Ranking

## 21. Predictive Ranking

## 22. Efficiency Ranking

## 23. Overall Paper-Candidate Ranking

## 24. Does GuardedGram Beat SafeGram?

YES / PARTLY / NO

## 25. Does Feature-Level Selection Beat Global Gating?

YES / PARTLY / NO

## 26. Can We Avoid Two Full Models?

YES / PARTLY / NO

## 27. Does Basis Sensitivity Grow With Embedding Dimension?

YES / PARTLY / NO

## 28. Is the Default Numerical-Embedding Basis Usually Optimal?

YES / NO / MIXED

## 29. Strongest Positive Finding

## 30. Strongest Negative Finding

## 31. Reviewer Attack Audit

### "The adaptive rule is just validation overfitting."
### "The method still has catastrophic tails."
### "The method requires twice the compute."
### "The phenomenon is caused by artificial preprocessing."
### "Why not just use scalar features?"
### "Why should basis dependence be removed if some bases are better?"
### "The method is too conservative."
### "The method is architecture-specific."

## 32. Ranked Final Candidates for Human Decision

rank | method | control | median C | p95 | max | breadth | single-model? | embedding evidence | complexity | recommendation

Do NOT automatically pick the final paper method.

## 33. Best Next Step for Each Top-3 Candidate

One decisive next test each.

## 34. Files Produced
```

---

# 38. Compute priority

Run in this order:

```text
1. GuardedGram G1/G2/G3
2. BlockGuard grouped
3. BlockGuard greedy
4. Full development comparison
5. Embedding confirmation
6. DualView if promising
7. Freeze finalists
8. NEW prospective panel
9. Optional basis portfolio
```

Never sacrifice the untouched prospective test for more hyperparameter searching.

---

# 39. Scientific interpretation

Do NOT frame the objective as:

> make the model invariant.

Frame it as:

> control dependence on arbitrary within-feature coordinates while preserving useful basis-dependent inductive bias.

The desired method should behave approximately as:

```text
basis prior useful:
retain raw coordinates

basis prior harmful/arbitrary:
use invariant Gram view
```

The central methodological question is therefore:

\[
\boxed{
\text{How much basis dependence should each table—or each feature—be allowed to retain?}
}
\]

A successful method should answer that using validation evidence without seeing test outcomes.