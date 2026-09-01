# AGENT.md — Forensic Metric Audit Before Paper Submission

## Mission

Audit every reported task loss and normalized excess-risk metric `C` in the latest Guarded Basis Control experiment.

DO NOT TRAIN ANY MODELS.

DO NOT CHANGE ANY METHOD.

DO NOT RUN NEW SCIENTIFIC EXPERIMENTS.

The goal is to determine whether reported `C`, tail metrics, rankings, and the `FINAL-METHOD-SIGNAL` verdict are mathematically correct and reproducible from raw predictions.

Primary suspicious example to investigate:

```text
OnlineNewsPopularity / CatBoost /
GuardedGram-G2-g0p0-t01

reported raw task error    = 5254.3235
reported method task error = 5250.5999
reported C                 = +0.2469
```

If lower task error means better performance, a positive excess-risk value appears inconsistent unless aggregation across seeds explains it.

Required output:

```text
metric_audit_results.md
results/audit/
```

---

# 1. Locate source artifacts

Find the exact raw files used to produce:

```text
results.md
prospective_general_cells.csv
prospective_embedding_cells.csv
prospective_six_rankings.csv
prospective_worst_10_tail_cells.csv
```

Also locate:

- per-row targets
- raw predictions
- method predictions
- seed IDs
- train/validation/test indices
- trivial-predictor statistics
- code implementing `C`
- aggregation/ranking code

Record file paths and git hashes.

---

# 2. Recompute losses from raw predictions

For EVERY:

```text
dataset × model × method × seed
```

recompute task loss directly from test predictions.

## Classification

Use exactly:

\[
L=\text{multiclass/binary log loss}.
\]

Verify:

- class ordering
- probability normalization
- clipping epsilon
- sample weights if any

## Regression

Use:

\[
L=RMSE.
\]

Do not trust cached scalar losses until they match raw predictions.

Save:

```text
results/audit/per_seed_losses.csv
```

Columns:

```text
dataset
model
method
seed
problem_type
n_test
raw_loss_recomputed
raw_loss_stored
method_loss_recomputed
method_loss_stored
absolute_loss_mismatch
```

Tolerance:

```text
absolute <= 1e-8 OR
relative <= 1e-6
```

Flag anything larger.

---

# 3. Recompute trivial predictor

Classification:

Use TRAINING class frequencies only.

\[
p_c=N_c/N.
\]

Evaluate this fixed predictor on the corresponding test set:

\[
L_{trivial}.
\]

Regression:

Use TRAINING target mean only.

Evaluate:

\[
L_{trivial}
=
RMSE(y_{test},\bar y_{train}).
\]

Save per seed.

Verify no validation/test labels were used to construct the trivial predictor.

---

# 4. Recompute canonical C per seed

The protocol defined:

\[
C
=
\frac{L_{method}-L_{raw}}
{\max(L_{trivial}-L_{raw},10^{-8})}.
\]

Compute this EXACTLY per seed.

Save:

```text
dataset
model
method
seed
L_raw
L_method
L_trivial
numerator
raw_headroom = L_trivial-L_raw
denominator_used
denominator_clipped
C_recomputed
C_stored
C_difference
```

---

# 5. Mandatory sign invariant

When:

```text
L_trivial > L_raw
```

the denominator is positive.

Therefore:

```text
L_method < L_raw  => C < 0
L_method = L_raw  => C = 0
L_method > L_raw  => C > 0
```

Assert this automatically.

Any violation at the SAME SEED LEVEL is a BUG.

Output:

```text
results/audit/sign_violations.csv
```

---

# 6. Diagnose aggregation

The displayed tables may show averages while C may use another aggregation.

For every dataset/model/method calculate all of:

## A. mean of per-seed C

\[
C_A
=
\frac1S\sum_s C_s
\]

## B. median of per-seed C

\[
C_B=\operatorname{median}_s C_s
\]

## C. ratio of mean losses

\[
C_C
=
\frac{
mean_s L_{method,s}-mean_s L_{raw,s}
}{
\max(
mean_s L_{trivial,s}-mean_s L_{raw,s},
10^{-8})
}.
\]

## D. ratio of median losses

analogous definition.

Determine EXACTLY which aggregation generated the existing report.

For every suspicious apparent sign mismatch, state whether:

```text
BUG
or
AGGREGATION-EXPLAINED
```

---

# 7. Audit denominator pathology

This is extremely important.

Count all cells where:

\[
L_{trivial}-L_{raw}\le0.
\]

Also count:

```text
headroom < 1e-8
headroom < 1e-6
headroom < 1e-4 * L_trivial
```

These cells make the current C metric unstable because the denominator is clipped to epsilon.

Report:

```text
number
percentage
datasets affected
models affected
largest |C| values caused by clipping
```

Specifically inspect giant values such as the earlier ±1e5–1e6-scale C values.

Determine whether any:

```text
median C
p90 C
p95 C
max C
method ranking
paper verdict
```

depends materially on denominator-clipped cells.

---

# 8. Compute a stable secondary metric

Do NOT replace C silently.

Keep original C, but compute a stable sensitivity metric:

\[
C_{\text{stable}}
=
\frac{L_{method}-L_{raw}}
{\max(L_{trivial},10^{-8})}.
\]

Interpretation:

> method-induced change as a fraction of trivial-predictor loss.

Also report absolute degradation:

\[
\Delta L=L_{method}-L_{raw}.
\]

For regression additionally report:

\[
\Delta_{\sigma}
=
\frac{RMSE_{method}-RMSE_{raw}}
{std(y_{test})}.
\]

For classification report:

\[
\Delta_{\logloss}
=
L_{method}-L_{raw}.
\]

Do not invent a new acceptance threshold yet.

We only want to know whether conclusions are robust.

---

# 9. Recompute prospective summaries

Using the EXACT intended aggregation, recompute for every method:

```text
median disagreement reduction
median C
p90 C
p95 C
max C
fraction C > .01
fraction C > .05
W/T/L
fallback rate
mean predictive rank
paper score
```

Then recompute the same summary with:

```text
C_stable
```

and with denominator-clipped C cells removed as a SENSITIVITY ANALYSIS only.

Never remove cells from the primary corrected analysis.

---

# 10. Priority methods

Pay special attention to:

```text
GuardedGram-G2-after-RBF-k16
GuardedGram-G2-g0p0-t01
SafeGram-t01
SafeRankGram-t01
Raw+Gram@0.75
PureGram
BlockGuard-Greedy-t01
```

The most important question is whether:

```text
GuardedGram-G2-after-RBF-k16
```

still has approximately:

```text
75% median control
p95 C ~ 0.0026
max C ~ 0.0064
```

after the audit.

---

# 11. Forensic table for suspicious rows

Create:

```text
results/audit/forensic_examples.csv
```

Include at minimum:

```text
OnlineNewsPopularity / CatBoost
OnlineNewsPopularity / controlled_mlp
OnlineNewsPopularity / TabICLv2
Brazilian_houses
SoilKsatDB
2dplanes
```

For EVERY seed show:

```text
raw loss
method loss
trivial loss
headroom
denominator
C
stored C
```

Then show the aggregated values appearing in `results.md`.

---

# 12. Recompute paper verdict

Return exactly one:

## AUDIT-PASS

All per-seed mathematics is correct, apparent inconsistencies are fully explained by aggregation, and rankings/verdict remain effectively unchanged.

## AUDIT-PASS-WITH-METRIC-CAVEAT

Calculations are correct, but C becomes pathological when raw fails to beat the trivial predictor. Main ranking survives under stable sensitivity metrics.

## AGGREGATION-BUG-BUT-CONCLUSIONS-SURVIVE

A real implementation/aggregation bug exists, but corrected rankings and main embedding-method result remain qualitatively the same.

## MATERIAL-METRIC-BUG

Corrected calculations materially change tail-risk conclusions, method rankings, or `FINAL-METHOD-SIGNAL`.

---

# 13. Required metric_audit_results.md

Produce exactly:

```markdown
# Guarded Basis Control — Metric Audit

## Executive Verdict
AUDIT-PASS /
AUDIT-PASS-WITH-METRIC-CAVEAT /
AGGREGATION-BUG-BUT-CONCLUSIONS-SURVIVE /
MATERIAL-METRIC-BUG

## One-Paragraph Summary

## 1. Files and Code Audited

## 2. Exact Metric Definitions

## 3. Raw Prediction -> Loss Verification

rows checked | mismatches | maximum mismatch

## 4. Trivial Predictor Verification

## 5. Per-Seed C Verification

cells | exact matches | mismatches | sign violations

## 6. Aggregation Semantics

Explain exactly how:
- seed losses
- C
- displayed task errors
- dataset/model units
- global summaries

were aggregated.

## 7. OnlineNewsPopularity / CatBoost Forensic Example

seed | raw | method | trivial | numerator | headroom | C

Explain the previously apparent contradiction.

## 8. Other Suspicious Cells

## 9. Denominator Pathology

headroom condition | cells | percentage

List clipped-denominator cells.

## 10. Corrected General Prospective Summary

method | control | median C | p90 | p95 | max

## 11. Corrected Embedding Prospective Summary

method | control | median C | p90 | p95 | max

## 12. Stable-Metric Sensitivity Analysis

method | median C_stable | p95 C_stable | max C_stable

## 13. Ranking Before vs After Audit

old rank | new rank | method | changed?

## 14. Does GuardedGram-G2-after-RBF-k16 Still Hold?

YES / PARTLY / NO

Give exact corrected metrics.

## 15. Does FINAL-METHOD-SIGNAL Still Hold?

YES / NO

## 16. Recommended Metric for the Paper

State:
- what should be primary
- what should be secondary
- how denominator-degenerate cases should be reported

Do not silently discard difficult cells.

## 17. Bugs Fixed

List code changes, if any.

## 18. Files Produced
```

---

# 14. Integrity requirement

Do not modify original experiment files.

Write corrected outputs under:

```text
results/audit/
```

If a bug is found, create a separate patch and record the diff.

The audit must be reproducible from raw saved predictions without model retraining.