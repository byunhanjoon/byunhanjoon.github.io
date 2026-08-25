# AGENT.md — Day 3: Basis Geometry, Schema Equivalence, and Structured Tabular Features

## Role

You are an autonomous research/coding agent continuing an existing tabular deep-learning project aimed at an ICLR submission.

Your job for **Day 3** is not to invent as many new encoders as possible. Your job is to run a **mechanism-first falsification campaign** around one central thesis:

> **Information-equivalent tabular representations can induce very different optimization problems because their basis geometry, conditioning, and interaction with regularization/optimizers differ.**

Day 1/2 found that:
- broad “numeric ↔ categorical continuum” encoders were not broadly successful;
- state-local identity corrections sometimes helped substantially;
- those improvements did not generalize broadly enough to justify a standalone encoder paper;
- PLE-like and identity-like representations can sometimes span the same empirical function space while producing very different optimization behavior.

Day 3 must determine whether that observation can become a general paper.

A successful Day 3 ends with a rigorous answer to:

1. **Do information-equivalent basis changes systematically alter neural tabular training?**
2. **Is condition number / spectrum a causal driver, or merely correlated with performance?**
3. **Do whitening/canonicalization or representation-invariant regularization remove that sensitivity?**
4. **Does the same mechanism extend cleanly to nominal categorical and ordinal features?**
5. **Can known feature topology—ordered paths for ordinals and circles for cyclic time—be separated from arbitrary basis choice?**
6. **Which categorical/structured-feature ideas belong in the core paper, and which should be dropped?**

Do not optimize for a positive result. Optimize for a decisive result.

---

# 0. Non-negotiable scientific rules

## 0.1 Preserve prior experimental protocol

Before changing code:

1. Inspect the repository structure.
2. Locate all Day 1 / Day 2 reports, especially the most recent report.
3. Locate the exact dataset splits, seeds, preprocessing code, model configs, and evaluator used previously.
4. Reproduce at least one known Day-2 anchor result before launching new experiments.
5. Reuse existing train/validation/test splits exactly wherever possible.

Do **not** silently replace previously weak datasets with favorable ones.

If Day 2 defined a prospective untouched dataset set, preserve it as a named evaluation tier.

## 0.2 No leakage

All transformations must be fit on training data only.

For any target-dependent categorical encoding:
- use cross-fitting / out-of-fold construction;
- never compute category target statistics using the row being encoded;
- never fit target encoders on validation/test targets;
- for nested procedures such as “numeric residual -> target encode residual”, use a genuinely leakage-safe nested or cross-fitted procedure.

If there is uncertainty, choose the more conservative implementation.

## 0.3 Separate three scientifically different intervention classes

Every experiment must be labeled as one of:

### A. Exact/equivalent reparameterization
The representation contains the same information and spans the same relevant space.
Examples:
- invertible change of basis;
- whitening followed by an invertible map on the retained full-rank subspace;
- categorical contrast recoding;
- ordinal local-state ↔ cumulative-threshold ↔ full-rank orthogonal contrast recoding;
- full real-Fourier recoding of a finite cyclic variable when every frequency component is retained;
- block residualization `[N, C] -> [N, C - NB]` while retaining `N`.

These are the strongest experiments for the paper.

### B. Preconditioning / optimization intervention
The predictor class is intended to stay comparable, but the optimization rule changes.
Examples:
- representation-invariant first-layer regularization;
- frequency-aware row learning rates;
- gradient preconditioning.

These test mechanism and provide remedies.

### C. Feature augmentation / changed inductive bias
The representation changes the ease or effective hypothesis class.
Examples:
- target encoding;
- residual target encoding;
- explicit numeric × categorical cross-atoms.

These can be useful, but **must not be presented as evidence of exact basis invariance**.

Never mix these categories in tables without explicit labels.


## 0.4 Respect declared feature semantics

Do not infer semantic structure from the target.

For **ordinal** columns:
- use a declared ordering from dataset metadata when available;
- otherwise use only an unambiguous domain ordering justified without looking at labels;
- never sort categories by target mean and then call them ordinal.

For **datetime/timestamp** columns:
- use declared datetime metadata or a lossless parse of the raw feature;
- preserve timezone/calendar semantics where present;
- do not manufacture target-relative dates or event times using label information.

Crucially distinguish:

- **ordered representation**: the encoder knows `c1 < c2 < ... < cK`;
- **monotonic prediction constraint**: the model is forced to make predictions monotone in that order.

These are not the same. Do not impose monotonic prediction constraints unless the task semantics genuinely guarantee monotonicity.

For datetimes, distinguish:
- absolute/relative trend coordinates;
- cyclic coordinates such as hour-of-day or day-of-week;
- non-equivalent low-frequency smoothness priors.

---

## 0.5 Statistical hygiene

For every headline claim:
- run multiple seeds;
- report paired differences using the same splits/seeds;
- report mean, standard deviation, and confidence interval or bootstrap interval;
- keep hyperparameter budget matched across compared variants;
- do not tune the proposed variant substantially more than the baseline.

Prefer:
- 5 seeds for fast/medium experiments;
- at least 3 seeds for expensive model × dataset validation;
- more seeds on the small anchor datasets if variance is high.

## 0.6 Do not overclaim “orthogonality”

“Residual target encoding” is **not automatically Gram-Schmidt orthogonalization**.

Only call a representation orthogonalized if you directly verify the relevant cross-Gram term is near zero, e.g.

`||N^T C_perp|| / (||N|| ||C_perp||)`

or the appropriate covariance-normalized equivalent.

---

# 1. Core Day-3 thesis

The paper candidate is:

> **Tabular models are not invariant to information-equivalent schema/basis choices. Conditioning and optimization geometry explain a substantial part of the variation, and canonicalization or invariant optimization can reduce it.**

The core mathematical setup:

Let `z = phi(x)` be a representation and let

`z' = A z`

for invertible `A`.

For a first linear layer

`h = W z`,

the same function is represented under the new basis by

`W' = W A^{-1}`.

Therefore prediction capacity is unchanged.

However standard parameter L2/weight decay is generally not invariant:

`||W'||_F^2 = ||W A^{-1}||_F^2 != ||W||_F^2`.

Likewise SGD trajectories can differ drastically when the feature covariance

`Sigma = E[z z^T]`

has a different spectrum.

A function-space first-layer penalty

`R_func(W, Sigma) = tr(W Sigma W^T) = E ||Wz||^2`

is invariant under the simultaneous transformation

`z' = A z`,
`W' = W A^{-1}`,
`Sigma' = A Sigma A^T`.

Verify this identity numerically in unit tests before using it experimentally.


## 1.1 Structured-feature geometry map

Use this conceptual map to keep the project coherent:

```text
continuous / ordered numerical states -> line/path geometry
ordinal levels                        -> path geometry
nominal categories                    -> discrete/simplex state geometry
cyclic datetime components            -> circle geometry
```

The key distinction is:

```text
feature topology / semantics
        !=
arbitrary coordinate basis used to represent that topology
```

Examples:
- local vs cumulative coordinates can represent the same ordered states;
- different categorical contrasts can represent the same nominal states;
- full Fourier vs one-hot coordinates can represent the same finite cyclic states;
- truncated Fourier features, monotonic constraints, target encodings, and explicit crosses change inductive bias and therefore belong in a different intervention class.

If the data support it, this distinction should become part of the eventual paper framing.

---

# 2. Priority order

Do the work in this order.

## P0 — MUST DO
1. Numeric exact-basis condition-number sweep.
2. Whitening / canonicalization of PLE-vs-identity.
3. Representation-invariant first-layer regularization.
4. Nominal categorical exact-basis / contrast condition-number sweep.
5. **Ordinal local-state ↔ cumulative-threshold ↔ orthogonal/whitened basis sweep.**
6. Exact block residualization of categoricals against numerics.
7. Produce a unified geometry-vs-performance analysis across numerical, nominal, and ordinal features.

## P1 — DO AFTER P0 WORKS
8. **Datetime/cyclic exact-basis experiment: one-hot ↔ full real Fourier, plus phase/rotation controls.**
9. Leakage-safe residual target encoding comparison.
10. Frequency-aware categorical preconditioning / optimizer experiments.
11. Broader model/dataset validation.

## P2 — EXPLORATORY ONLY
12. Datetime low-frequency/truncated Fourier and learned periodic representations as inductive-bias experiments.
13. Numeric-atom × categorical-level cross features.

If compute/time becomes constrained, cut P2 first, then reduce P1 breadth. Do **not** sacrifice P0.

---

# 3. Repository setup and reproducibility

Create a Day-3 experiment namespace without breaking prior code.

Suggested structure, adapted to the repository:

```text
experiments/day3/
    configs/
    run_equivalent_basis.py
    run_whitening.py
    run_invariant_regularizer.py
    run_categorical_basis.py
    run_ordinal_basis.py
    run_datetime_geometry.py
    run_block_residualization.py
    run_residual_te.py
    run_frequency_preconditioning.py
    run_cross_atoms.py
    analyze_day3.py
tests/
    test_basis_equivalence.py
    test_ordinal_equivalence.py
    test_fourier_equivalence.py
    test_whitening.py
    test_block_residualization.py
    test_invariant_regularizer.py
results/day3/
REPORT_DAY3.md
```

Use the existing project abstractions instead if equivalent infrastructure already exists.

Do not duplicate entire training pipelines.

Record:
- git commit hash;
- package versions;
- CUDA/PyTorch versions;
- dataset fingerprints;
- split IDs;
- seeds;
- exact configs.

Save raw results in machine-readable form (`parquet`, `csv`, or `jsonl`) before making plots.

---

# 4. Datasets

Use a tiered protocol.

## Tier A — Anchor/mechanism datasets

At minimum preserve the prior anchors if they are available in the repo:

- Adult
- Black Friday
- Miami
- Diamonds / Diamond

Use the exact dataset naming already used by the project.

Purpose:
- fast mechanism debugging;
- reproduce known gaps;
- inspect geometry in detail.

## Tier B — Prospective Day-2 datasets

Locate the untouched/prospective datasets used for the Day-2 breadth test.

Run them **without replacing failures**.

Purpose:
- guard against post-hoc storytelling.

## Tier C — Mixed numerical/categorical breadth

After P0 passes basic sanity checks, select a broader fixed subset from the project's existing TabArena-compatible pool with:
- both numeric and categorical columns;
- classification and regression represented;
- low- and high-cardinality categoricals;
- different sample sizes;
- different categorical frequency skews;
- at least a few datasets with declared or defensible ordinal columns;
- at least a few datasets with genuine datetime/timestamp or cyclic calendar components, if available.

Before running proposed methods, save a structured-feature audit for every selected dataset:

```text
dataset
column
raw_dtype
semantic_type = numerical | nominal | ordinal | datetime | cyclic-derived
ordinal_levels/order (if applicable)
datetime_timezone/calendar notes (if applicable)
cardinality
missing_rate
```

Selection must be based on schema/statistics only, not proposed-method performance.

Record the selection rule in the report before examining results.

If feasible, target 10–20 datasets for Day 3 screening and expand later.

---

# 5. Models

Use a staged matrix.

## Mechanism screen
Start with:
- MLP

Then validate important findings on:
- ResNet
- TabM
- FT-Transformer, if already supported cleanly by the codebase.

Avoid introducing retrieval-heavy or foundation models during the first mechanism pass unless already trivial to run; they add confounds.

## Optimizers

At minimum compare:
- the project's existing/default optimizer, likely AdamW;
- SGD + momentum for selected controlled experiments.

Reason:
adaptive optimizers may already partially compensate for frequency/curvature differences.

Keep learning-rate search budget matched.

---

# 6. EXPERIMENT A — Controlled information-equivalent basis sweep

This is the single most important Day-3 experiment.

## Goal

Demonstrate causally that changing only the conditioning of an information-equivalent representation changes optimization and possibly generalization.

## 6.1 Construct a canonical full-rank representation

For a selected encoded block `Z_train`:

1. center using training statistics;
2. compute thin SVD:

   `Z = U S V^T`

3. determine numerical rank using a documented tolerance;
4. drop only exact/numerically redundant directions;
5. construct a whitened/canonical coordinate system:

   `Z0 = Z V_r S_r^{-1}`

so the retained training covariance is proportional to identity.

Use the corresponding fixed train-fitted transform on validation/test.

Do not use validation/test data to determine the transform.

## 6.2 Controlled condition-number transform

Starting from a well-conditioned full-rank representation `Z0`, generate

`Z_kappa = Z0 A_kappa`

where `A_kappa` is invertible and has a controlled condition number.

Construct e.g.

`A = Q1 D Q2`

with random orthogonal `Q1,Q2` and diagonal singular values `D`.

Use target condition numbers:

```text
1, 3, 10, 30, 100, 300, 1000, 3000
```

where numerically stable.

Normalize `D` to geometric mean 1 so the experiment changes anisotropy rather than global scale.

Use several random `A` draws for selected kappas to ensure results are not an artifact of one rotation.

## 6.3 Blockwise first

First apply transforms **within a single raw-feature encoding block**.

Then optionally apply transforms across the entire representation as a stress test.

The blockwise result is scientifically cleaner because it preserves feature boundaries.

## 6.4 Measurements

For every run record:

### Representation geometry
- nonzero singular values;
- condition number;
- effective rank;
- log-determinant on retained subspace;
- mean feature norm;
- max/min variance;
- pairwise/coherence statistics where useful.

### Training
- train loss by epoch;
- validation metric by epoch;
- steps/epochs to a fixed loss threshold;
- first-layer gradient norm;
- first-layer weight norm;
- update/weight ratio;
- gradient covariance diagnostics if cheap.

### Performance
- validation/test task metric;
- paired delta relative to `kappa=1`.

## 6.5 Primary test

Plot:

`performance delta vs log10(kappa)`

and

`convergence speed vs log10(kappa)`.

Compute:
- per-dataset Spearman correlation;
- pooled mixed-effects or hierarchical regression if easy;
- fraction of dataset/model pairs with worsening performance as kappa increases.

This experiment provides the strongest causal evidence because the representation is deliberately information-equivalent.

---

# 7. EXPERIMENT B — Does whitening collapse the PLE-vs-identity gap?

## Goal

Test whether the prior PLE-vs-state-local identity differences are primarily geometric.

For every previously identified feature/dataset where PLE and identity-like representations are claimed to span the same relevant state functions:

1. explicitly construct both design matrices;
2. test linear reconstructability in both directions;
3. compute principal angles between their column spaces;
4. report reconstruction error;
5. report rank.

Do not merely assert equivalence.

## Variants

For each representation:

1. original;
2. centered;
3. standardized by diagonal variance only;
4. ZCA/SVD whitened;
5. whitened + orthogonal Procrustes alignment to a shared reference basis where the subspaces match.

If two representations truly span the same subspace, the aligned whitened representations should be numerically nearly identical on training rows.

Verify validation/test consistency as well.

## Key question

Does the original predictive gap shrink after canonicalization?

Report:

`gap_raw`
`gap_standardized`
`gap_whitened`
`gap_aligned`

Define:

`gap_reduction = 1 - |gap_whitened| / |gap_raw|`

when the denominator is stable.

Do not force an arbitrary success threshold, but highlight whether the gap is reduced substantially and consistently.

If whitening does not collapse the gap, investigate:
- nonlinearity after the representation;
- finite-width effects;
- initialization;
- optimizer;
- implicit regularization;
- representation sparsity/locality.

That negative result is still useful.

---

# 8. EXPERIMENT C — Representation-invariant first-layer regularization

## Goal

Test whether standard weight decay is one mechanism producing basis sensitivity.

## 8.1 Baselines

Compare:

1. standard training / standard weight decay;
2. no first-layer weight decay, normal weight decay elsewhere;
3. function-space first-layer regularizer:

   `lambda * tr(W Sigma W^T)`

   while preserving the usual regularization on later layers.

`Sigma` must be estimated from training representation only.

Use a stable centered covariance and regularization/shrinkage if needed.

## 8.2 Controlled basis sweep

Repeat a subset of Experiment A across several kappas under:
- standard WD;
- invariant first-layer regularizer.

Main metric:

> How much does the variance/worst-case spread of test performance across equivalent bases shrink?

Define a basis-sensitivity metric, for example:

`Sens_basis = std_A(metric(A))`

plus:
- max-min spread;
- worst-case drop from kappa=1.

## 8.3 Unit test

For random `Z`, invertible `A`, `W` verify numerically:

`tr(W Sigma W^T)`

equals

`tr(W' Sigma' W'^T)`

within numerical tolerance.

If the implementation fails this test, do not run the experiment.

---

# 9. EXPERIMENT D — Categorical exact-basis conditioning

This is the cleanest categorical counterpart and should be treated as P0.

## Goal

Show that categorical representations have the same basis-conditioning problem without invoking target labels.

## 9.1 Categorical contrast basis

For a categorical variable with `K` levels:

1. create one-hot matrix `C`;
2. center by training frequencies `p`;
3. work on the `K-1` dimensional nonconstant subspace;
4. construct an orthonormal contrast basis using QR/SVD/Helmert-style contrasts;
5. verify exact recovery of categorical state information.

With an intercept or retained mean component handled consistently, different full-rank contrast codings represent the same categorical information.

## 9.2 Frequency-induced covariance

For nonuniform categories, sample covariance is related to:

`diag(p) - p p^T`.

Measure its nonzero spectrum and condition number.

Stratify categorical columns by:
- cardinality;
- entropy;
- Gini/imbalance;
- min/median/max frequency;
- head/tail mass.

Test whether frequency skew predicts condition number.

## 9.3 Frequency-whitened categorical basis

Construct a train-fitted whitening transform on the centered categorical contrast subspace:

`C_white = C_centered Q Lambda^{-1/2}`

with a carefully documented epsilon/shrinkage for tiny eigenvalues.

This is an information-preserving invertible transform on the retained categorical subspace.

Compare:
1. standard one-hot/contrast;
2. centered contrast;
3. diagonal variance scaling;
4. exact/shrunk frequency whitening;
5. controlled ill-conditioned transforms with target kappas.

Feed these into the same MLP first layer.

This gives the categorical analogue of Experiment A.

## 9.4 Relationship to embeddings

Remember:

A one-hot input followed by a learned linear layer is algebraically an embedding lookup.

Therefore the contrast-basis experiment can be interpreted as a reparameterization/preconditioning of the categorical embedding problem.

Document this equivalence.

For high-cardinality categoricals, do not materialize huge dense matrices unnecessarily; use sparse matrices or derive the equivalent parameter/gradient transformation.

---


# 9A. STRUCTURED EXPERIMENT — Ordinal basis geometry

This is **P0** and should be treated as a possible core-paper experiment.

## Goal

Test whether an ordered discrete feature exhibits the same **local-vs-cumulative basis-conditioning** phenomenon as numerical PLE while keeping the represented ordinal state exactly fixed.

For genuine ordinal levels

`c1 < c2 < ... < cK`

construct several exact representations of the same state space.

## 9A.1 Local-state basis

Use centered one-hot or another full-rank `K-1` contrast representation with intercept/constant handling documented explicitly.

Do not compare rank-deficient encodings without restricting them to a common full-rank nonconstant subspace.

## 9A.2 Cumulative / thermometer basis

For `K` ordered states construct `K-1` threshold indicators such as

`T_j(c) = 1[c > c_j]`.

With appropriate intercept handling, the local contrast basis and cumulative threshold basis encode exactly the same ordinal state.

Explicitly verify:
- equal retained rank;
- reconstruction in both directions;
- near-zero reconstruction error;
- identical ability to distinguish states.

This is the ordinal analogue of local state identity versus cumulative PLE.

## 9A.3 Orthogonal / spectral ordinal bases

Construct:
1. QR/SVD-orthogonalized ordinal contrasts;
2. orthogonal polynomial contrasts and/or a path-graph Laplacian eigenbasis.

Interpret levels as a path:

`c1 -- c2 -- ... -- cK`.

The path-Laplacian eigenvectors provide low-to-high-frequency coordinates over the ordered domain.

Do not assume equal spacing unless justified. If only the ordering is known, keep that distinction explicit.

## 9A.4 Frequency whitening

Because ordinal levels can also be imbalanced, construct a sample-frequency-whitened representation on the nonconstant ordinal subspace.

Compare:
1. local centered contrast;
2. cumulative/thermometer;
3. diagonal-standardized cumulative basis;
4. orthogonal polynomial/path-spectral basis;
5. sample-whitened basis.

## 9A.5 Controlled condition-number sweep

Starting from a canonical whitened ordinal basis `O0`, apply invertible transforms with the same target condition numbers used in Experiment A.

This separates:
- the **semantics of order**;
- the **choice of coordinates inside the same ordinal state space**.

Include random orthogonal transforms at `kappa = 1`.

## 9A.6 Monotonicity warning

Do **not** assume that an ordinal feature implies a monotonic target response.

Main experiments should allow arbitrary target functions over the ordered states.

Only if domain semantics genuinely guarantee monotonicity may you add a clearly labeled monotonic-model baseline. Treat that as a separate inductive bias, not basis conditioning.

## 9A.7 Required measurements

Report:
- nonzero spectrum and condition number;
- cumulative-basis coherence/correlation;
- convergence speed;
- train/validation curves;
- test metric;
- basis sensitivity;
- gap before/after whitening;
- consistency across model families.

## 9A.8 Strong result

A strong result is:

> Cumulative ordinal thresholds are much more ill-conditioned than local/orthogonal bases, the learning gap follows that geometry under exact equivalent transforms, and whitening/canonicalization removes a substantial fraction of the difference.

If this mirrors the PLE result, elevate ordinal geometry to the paper core immediately.

---

# 9B. STRUCTURED EXPERIMENT — Datetime and cyclic topology

This is **P1**. The obvious use of sine/cosine for timestamps is established prior art; do not claim cyclical encoding itself as novel.

The useful question is:

> When a feature has known circular topology, which effects come from arbitrary basis choice, and which come from imposing a circular smoothness prior?

## 9B.1 Semantic decomposition

For each genuine datetime column, construct only justified components such as:
- absolute or relative continuous time;
- hour of day;
- day of week;
- day of year / seasonal phase where meaningful;
- month only with explicit acknowledgement of unequal month lengths;
- elapsed durations already defined by the task/data.

Fit origins/scales from training data only.

Do not create future-looking or target-relative features.

Preserve timezone/DST semantics when they matter.

## 9B.2 Exact finite-cycle one-hot ↔ full Fourier experiment

For a finite cyclic component with `K` states, such as 24 hours or 7 weekdays:

1. create centered one-hot / contrast coordinates;
2. construct a **full real Fourier basis** on the same `K-1` nonconstant subspace;
3. include every required sine/cosine frequency pair and the Nyquist component for even `K` when applicable;
4. verify exact rank and reconstruction equivalence.

The full Fourier representation is an orthogonal change of coordinates of the same finite cyclic state space.

Do **not** confuse this with the common 2D encoding

`[sin(theta), cos(theta)]`

which discards higher-frequency functions when the cycle has more than a few states.

## 9B.3 Phase-shift control

Changing the phase origin of a full Fourier basis should produce rotations inside sine/cosine frequency pairs and should leave condition number essentially unchanged.

Test multiple phase origins.

If neural training changes substantially under these exact orthogonal phase rotations, investigate optimizer coordinate dependence **beyond condition number**.

## 9B.4 Controlled ill-conditioning

Starting from the full whitened Fourier/contrast basis, apply the same controlled invertible transforms as in Experiment A.

Ask whether cyclic variables obey the same `kappa -> optimization` relationship.

## 9B.5 Truncated Fourier is a different scientific question

Only after the exact-basis experiment, compare:
- full exact Fourier basis;
- first harmonic only `[sin(theta), cos(theta)]`;
- first `m` harmonics;
- optional learned periodic/Fourier embedding already supported by the codebase.

These are **not** information-equivalent representations.

They impose a circular smoothness prior by removing high-frequency state functions.

Label them **Intervention Class C: changed inductive bias**.

## 9B.6 Absolute-time controls

For continuous timestamp trend coordinates, test:
- different time origins;
- different units;
- centering/standardization.

Be careful: if a nonlinear/data-dependent numerical encoder is applied after the affine transform, the overall pipeline may no longer be a pure fixed basis change.

## 9B.7 Keep/drop criterion

Keep datetime in the main paper only if one of these occurs:
1. exact cyclic basis changes reveal a clear optimizer/basis effect;
2. controlled nonorthogonal transforms inside the same cyclic state space reproduce the main condition-number result;
3. the exact-basis vs truncated-smoothness separation gives a particularly clean mechanistic result.

Otherwise move datetime to appendix/follow-up.

---

# 10. EXPERIMENT E — Exact block residualization: numerical vs categorical confounding

This is the rigorous replacement for the loose claim that “residual target encoding = Gram-Schmidt”.

## Goal

Test whether correlated numerical and categorical blocks create harmful cross-block collinearity, and whether an **exact information-equivalent block basis change** fixes it.

Let:
- `N` = numerical basis, e.g. PLE or canonicalized numeric features;
- `C` = categorical basis, e.g. centered one-hot/contrast features.

Fit on training data:

`B = argmin_B ||C - N B||_F^2`

using QR/SVD or ridge if rank deficient.

Then construct:

`C_perp = C - N B`.

The joint representation becomes:

`[N, C_perp]`.

Because

`C = C_perp + N B`,

the joint design is related to `[N,C]` by an invertible block-triangular transform when the full columns are retained.

Thus this is an **information/span-equivalent basis change**, unlike target encoding.

## 10.1 Apply to validation/test

Fit `B` using training data only.

Then:

`C_val_perp = C_val - N_val B`
`C_test_perp = C_test - N_test B`.

## 10.2 Verify

Report:
- `||N^T C||` before;
- `||N^T C_perp||` after;
- joint condition number before/after;
- principal-angle or canonical-correlation diagnostics;
- reconstruction equivalence.

For exact least squares and full-rank handling, cross-block correlation should collapse on the training set.

## 10.3 Diamond/Diamonds anchor

Use the Diamonds dataset as a focused case if prior results suggest:
- categorical features such as cut/color/clarity;
- strong numerical drivers such as carat;
- naive categorical treatment behaves poorly.

Do not hard-code those columns if the actual repo schema differs; inspect the dataset.

Compare:
1. `[N, C]`;
2. `[N, standardized C]`;
3. `[N, C_perp]`;
4. joint whitening of `[N,C]`;
5. optional block-whitened `N` and `C` separately.

This experiment directly tests the confounding/conditioning story.

---

# 11. EXPERIMENT F — Residual target encoding, carefully framed

This is P1, not the main novelty.

Residual/marginal target encoding already exists in prior work/software discussions, so do not present the basic idea as novel.

Use it as a diagnostic and possibly as a practical comparison.

## 11.1 Regression

Goal:

`r_i = y_i - yhat_num_i`

then encode category by a smoothed estimate of:

`E[r | category]`.

But construct it leakage-safely.

### Recommended training construction

For each outer training split:
1. create OOF predictions for the numeric-only model;
2. compute OOF residuals;
3. create cross-fitted category residual means so each encoded row excludes its own target;
4. fit final train-derived mappings for validation/test using only training targets/residuals.

If using the same folds can induce indirect leakage through upstream models, use nested folds or a three-way construction.

Document the exact procedure.

## 11.2 Classification

Use a principled residual such as:

`r = y - p_hat`

for binary classification.

Do not invent a regression-style residual on logits without justification.

## 11.3 Compare

1. standard cross-fitted target encoding on `y`;
2. residual target encoding;
3. exact unsupervised block residualization from Experiment E;
4. target encoding followed by linear residualization against `N`;
5. plain categorical embedding/one-hot baseline.

## 11.4 Interpretation

If residual TE helps but exact block residualization does not:
- likely target-aware compression/inductive bias, not merely conditioning.

If exact block residualization helps similarly:
- strong support for geometry/confounding.

If both fail:
- drop the story.

---

# 12. EXPERIMENT G — Frequency-aware categorical preconditioning

This branch must acknowledge existing frequency-aware embedding optimization literature.

There is prior work on frequency-aware SGD for embedding learning showing that token-dependent learning rates can improve convergence and that adaptive optimizers already exploit frequency information.

Therefore novelty cannot simply be:

> “rare categories occur less often, so give them larger learning rates.”

The Day-3 question is narrower:

> Does the categorical frequency distribution induce measurable curvature/optimization imbalance in tabular networks, and can a simple frequency-only preconditioner reduce basis sensitivity or improve robustness?

## 12.1 Mechanism measurement first

For each categorical feature/category `c`, estimate:
- training frequency `p_c`;
- number of updates received;
- gradient norm statistics;
- squared-gradient EMA;
- parameter update norm;
- embedding norm;
- optional diagonal empirical Fisher / GGN estimate.

Plot on log-log scales:
- update count vs `p_c`;
- gradient second moment vs `p_c`;
- convergence/error by category-frequency quantile.

Do not claim “Hessian eigenspectrum collapse” unless an actual curvature proxy supports it.

For small anchor configurations, estimate first-layer/embedding-block Hessian or generalized Gauss-Newton eigenvalues using Lanczos or an equivalent method if practical.

## 12.2 Frequency-aware methods

Test simple, interpretable variants.

### Variant A — row-wise learning-rate multiplier

For category `c`:

`m_c = clip((p_ref / (p_c + eps))^gamma, m_min, m_max)`

with small gamma grid, e.g.:

```text
0.0, 0.25, 0.5
```

Use conservative clipping such as 0.25–10 initially.

Do not jump directly to inverse frequency.

### Variant B — reparameterized / activation-scaled embedding

Explore:

`e_c = theta_c / (p_c + eps)^(gamma/2)`

or the mathematically justified equivalent.

Match initialization/output variance as a control.

### Variant C — categorical frequency whitening

Prefer the exact categorical whitening from Experiment D where feasible.

This is the conceptually strongest version.

## 12.3 Optimizer controls

Compare against:
- AdamW;
- SGD;
- AdaGrad or row-wise adaptive optimizer if already available.

If AdamW/AdaGrad removes the proposed benefit, report that clearly.

## 12.4 Rare-category evaluation

In addition to aggregate task metric, stratify test rows by training frequency of the active categorical level.

For multiple categorical fields, define a documented row statistic such as:
- minimum active-category frequency;
- geometric mean frequency;
- rarest-field quantile.

Report head/mid/tail performance.

Avoid turning this into a label-imbalance paper.

---

# 13. EXPERIMENT H — Numeric atom × categorical cross-basis injections

P2 only.

This is potentially useful, but it is not an exact linear basis change.

Treat it as deterministic nonlinear lifting / optimization shortcut.

## 13.1 Candidate construction

Use training data only.

Identify numerical atoms using the existing Day-1/2 detector if available.

Otherwise use a predeclared support rule such as:
- repeated exact numerical value;
- minimum absolute count;
- minimum train proportion.

Identify categorical levels using support thresholds only for the first pass.

Construct:

`M_{v,c}(x) = 1[x_num = v] * 1[x_cat = c]`.

Keep the original numeric and categorical features too.

## 13.2 Controls

Compare:
1. baseline model;
2. explicit cross-atoms;
3. random/support-matched crosses;
4. parameter-matched wider MLP;
5. if available, a standard feature-cross baseline.

Cross-feature methods have substantial prior literature, so a gain alone is not novel.

## 13.3 Keep/drop criterion

Keep this branch in the paper only if:
- gains replicate on multiple datasets;
- support-matched random crosses do not give the same effect;
- gains are linked to sparse joint states;
- training/convergence evidence supports an optimization-shortcut interpretation.

Otherwise mention it briefly or drop it.

---

# 14. Joint whitening and block geometry

Create a unified representation diagnostic for:

`Z = [N, C]`.

Measure:
- within-numeric condition;
- within-categorical condition;
- cross-block canonical correlations;
- joint condition;
- singular spectrum;
- effective rank.

Compare these transformations:

1. raw `[N,C]`;
2. diagonal standardization;
3. blockwise whitening:
   `[N_white, C_white]`;
4. block residualization:
   `[N, C_perp]`;
5. block residualization + within-block whitening;
6. full joint whitening.

This should reveal whether:
- within-block conditioning;
- cross-block collinearity;
- or both

drive optimization differences.

---

# 15. Additional categorical equivalent-basis stress test

This is important because it creates exact symmetry with the numerical story.

For selected categorical blocks:
1. create canonical frequency-whitened contrast basis `C0`;
2. generate invertible transforms with target kappa;
3. feed them through the same model;
4. hold information exactly fixed;
5. measure performance and convergence as in Experiment A.

If both numerical and categorical blocks show a similar monotonic sensitivity to controlled kappa, that is a highly compelling Day-3 result.

Potential unified empirical statement:

> Across both numerical and categorical representations, neural tabular learners exhibit avoidable sensitivity to the condition number of information-equivalent feature bases.

Do not write this claim unless the data actually supports it.

---

# 16. Geometry metrics to implement once and reuse

Implement a reusable module that accepts a training design matrix or sparse block and returns:

```python
{
    "rank": ...,
    "effective_rank": ...,
    "sigma_max": ...,
    "sigma_min_nonzero": ...,
    "condition_number": ...,
    "log_condition_number": ...,
    "trace_cov": ...,
    "logdet_nonzero_cov": ...,
    "max_variance": ...,
    "min_nonzero_variance": ...,
}
```

For block pairs `(A,B)`, also compute:
- normalized cross-Gram norm;
- top canonical correlation;
- mean canonical correlation if stable;
- principal angles.

Use randomized/SVD methods for large sparse matrices where needed.

Never report an infinite condition number from intentionally redundant coordinates as if it were informative; report rank and the condition number on the nonzero retained subspace.

---

# 17. Optimization diagnostics

Avoid collecting dozens of metrics that will never be analyzed.

Minimum useful set:
- train loss curves;
- validation metric curves;
- epochs/steps to fixed loss threshold;
- first-layer gradient norm;
- first-layer weight norm;
- representation covariance spectrum.

For selected anchor runs:
- empirical Fisher/GGN spectrum for first-layer or embedding block;
- gradient covariance spectrum;
- cosine similarity between early gradients under equivalent bases after mapping them back to a common function space.

The last diagnostic is particularly interesting:

Given equivalent bases `z' = Az`, map transformed gradients/weights back to the reference coordinate system and compare trajectories.

If trajectories still diverge after function-space mapping, characterize when and why.

---

# 18. Pre-registered hypotheses and decision gates

Write these hypotheses into the result metadata **before** broad runs.

## H1 — Equivalent-basis sensitivity
Increasing controlled condition number while preserving information degrades convergence and/or predictive performance.

**Strong support**:
- reproducible trend across multiple datasets and at least two model families;
- meaningful correlation with log-condition-number;
- not explained by global feature norm.

## H2 — Whitening/canonicalization explains prior PLE/identity gaps
Whitening or aligned canonicalization substantially shrinks previously observed performance differences.

**If true**:
central mechanism strengthened.

**If false**:
do not force the conditioning narrative; investigate implicit regularization/locality.

## H3 — Standard weight decay contributes to basis sensitivity
Invariant first-layer regularization reduces performance variance across equivalent basis transforms.

**Strong support**:
- lower worst-case drop and lower cross-basis spread;
- no material average-performance penalty.

## H4 — Categorical basis conditioning
Controlled categorical contrast conditioning produces the same phenomenon as controlled numerical conditioning.

This is the cleanest categorical extension.

## H5 — Cross-block collinearity
Exact residualization `[N,C] -> [N,C_perp]` reduces cross-block dependence and improves optimization or robustness on datasets with strong numerical/categorical confounding.

If condition improves but predictive behavior does not, report that.

## H6 — Residual TE
Residual target encoding helps beyond leakage-safe naive TE.

Treat this as supporting/practical evidence, not core novelty.

## H7 — Frequency preconditioning
Frequency-aware preconditioning reduces per-category optimization disparity and improves tail robustness or aggregate performance.

If adaptive optimizers already solve it, report that and deprioritize the method.

## H8 — Cross-atoms
Explicit sparse numeric×categorical crosses give broad gains beyond matched controls.

This is the easiest branch to drop.

---

# 19. Minimal experiment matrix before scaling

Do not launch the full grid blindly.

## Stage 1 — Sanity
On one anchor dataset / synthetic structured fixture:
- verify transform equivalence;
- verify whitening;
- verify condition-number targeting;
- verify invariant regularizer algebra;
- verify block residualization;
- verify ordinal local ↔ cumulative reconstruction;
- verify full finite-cycle one-hot ↔ real-Fourier reconstruction;
- verify no leakage.

## Stage 2 — Core mechanism
Run MLP on anchor datasets for:
- numeric kappa sweep;
- nominal categorical kappa sweep;
- ordinal local/cumulative/orthogonal/whitened sweep wherever genuine ordinal columns exist;
- whitening;
- block residualization;
- standard WD vs invariant regularizer.

## Stage 3 — Replication
Validate strongest results on:
- ResNet;
- TabM;
- FT-Transformer if practical.

## Stage 4 — Structured-feature extension
Run a focused datetime/cyclic experiment:
- centered one-hot vs full exact real Fourier;
- phase-shift controls;
- selected controlled-kappa transforms;
- only then low-frequency/truncated Fourier if useful.

## Stage 5 — Breadth
Run fixed Tier-B/Tier-C datasets.

## Stage 6 — Secondary methods
Only then run:
- residual TE;
- frequency-aware optimizer;
- cross-atoms;
- learned periodic datetime variants if they remain scientifically useful.

---

# 20. Fair hyperparameter protocol

Do not create a hidden advantage for the proposed representation.

Use either:

### Protocol A — fixed hyperparameters
Take previously selected baseline hyperparameters and reuse them across equivalent basis transforms.

This is ideal for measuring raw optimizer sensitivity.

### Protocol B — equal small HPO budget
Give each representation the exact same search space/budget.

Use this only as a secondary question:
“Can HPO recover the damage caused by a poor basis?”

Report fixed-config and equal-HPO results separately.

Do not mix them.

---

# 21. Important controls

For every “conditioning helps” result, include:

1. **Global scale control**
   Match average feature norm or geometric mean singular value.

2. **Random orthogonal control**
   `kappa = 1`, random rotation.
   Performance should be roughly invariant if the optimizer/initialization is isotropic.

3. **Diagonal standardization control**
   Distinguish simple variance scaling from full covariance whitening.

4. **No-weight-decay control**
   Tests whether L2 interaction is responsible.

5. **Optimizer control**
   AdamW vs SGD on selected runs.

6. **Capacity control**
   Especially for cross-atoms.

7. **Leakage control**
   Especially for target encodings.

8. **Semantic-structure control**
   For ordinal variables, arbitrary permutations may be used only as diagnostics; never relabel them as valid ordinal orderings.

9. **Cyclic phase control**
   For full Fourier bases, test multiple phase origins. Exact phase shifts should not alter information or condition number.

---

# 22. Analysis linking geometry to performance

At the end, create one table with one row per:

`dataset × model × representation × seed`.

Include:
- task metric;
- train loss;
- condition number;
- effective rank;
- cross-block correlation;
- optimizer;
- weight decay;
- representation family;
- intervention class A/B/C.

Then analyze:

### Controlled causal plots
- target kappa vs convergence;
- target kappa vs generalization.

### Natural representation plots
- PLE/identity/categorical condition vs metric.

### Intervention plots
- reduction in log-kappa vs change in metric;
- reduction in basis sensitivity under invariant regularizer.

Use controlled sweeps as causal evidence.
Treat natural cross-dataset correlations only as supporting evidence.

---

# 23. Novelty guardrails

Do not claim novelty for the following by themselves:

- residual target encoding;
- generic target encoding;
- generic whitening;
- generic feature crossing;
- generic frequency-aware learning rates for embeddings;
- generic preprocessing sensitivity;
- thermometer/cumulative ordinal encoding;
- Helmert/polynomial/ordinal contrasts by themselves;
- monotonic neural networks or monotonic embeddings by themselves;
- sine/cosine datetime encoding;
- Time2Vec, generic Fourier features, or periodic numerical embeddings by themselves.

Relevant prior-work warnings:
- residual/marginal target encoding has prior implementations/discussion;
- frequency-aware SGD for embedding learning has prior theory;
- automatic/explicit feature crossing has substantial prior literature;
- preprocessing robustness for tabular models is already an active topic;
- whitening and change-of-basis-invariant optimization are established outside tabular learning;
- ordinal contrasts and cumulative encodings are classical;
- monotonic modeling is a separate established literature;
- cyclic sine/cosine, Fourier time representations, Time2Vec, and learned periodic embeddings have prior work.

The potential novelty is the **specific tabular synthesis**:

1. exact information-equivalent schema/basis transformations;
2. systematic controlled sensitivity across modern tabular neural architectures;
3. numerical + nominal categorical + **ordinal ordered-domain** unification;
4. a clean separation between **feature topology** and **coordinate/basis geometry**;
5. a concrete geometry/regularization mechanism;
6. a remedy that reduces basis dependence without sacrificing performance;
7. optionally, cyclic datetime as an orthogonal-control/topology extension rather than a feature-engineering claim.

That is the paper to test.

---

# 24. What would constitute a strong Day-3 result?

A particularly strong result would look like:

### Result A
For the same tabular information, deliberately increasing basis condition number from ~1 to ~10^3 causes reproducible degradation in convergence and/or test performance across MLP/ResNet/TabM.

### Result B
Whitening/canonicalizing PLE and identity representations collapses a substantial fraction of their prior gap.

### Result C
The same controlled phenomenon appears for nominal categorical contrast bases under frequency skew.

### Result D
For ordinal features, local-state and cumulative-threshold bases encode the same ordered states but have sharply different spectra; whitening/orthogonalization reduces the learning gap.

### Result E
Exact block residualization removes numerical/categorical collinearity on Diamonds and improves stability/performance.

### Result F
Representation-invariant first-layer regularization makes models much less sensitive to arbitrary equivalent basis choices.

### Result G — Optional
Finite cyclic datetime states behave predictably under exact full-Fourier phase rotations, while controlled nonorthogonal transforms reproduce the condition-number effect.

If A+C+D+F hold, this is potentially a very coherent paper even if residual TE, datetime truncation, and cross-atoms fail.

If only residual TE or cross-atoms work, the paper is much weaker.

---

# 25. What would falsify the main thesis?

Be explicit.

The geometry thesis is weakened if:
- controlled kappa changes do not alter optimization once scale is controlled;
- orthogonal vs ill-conditioned equivalent transforms perform similarly;
- whitening does not reduce any representation gap;
- condition metrics fail to track convergence even in controlled sweeps;
- invariant regularization does not reduce cross-basis sensitivity;
- nominal categorical basis conditioning shows no analogous effect;
- ordinal local-vs-cumulative differences do not track conditioning or survive controlled tests.

Datetime is not required for the core thesis. A null datetime result should demote that branch rather than falsify the whole project.

If these happen, say so in the report and pivot.

Do not rescue the hypothesis with post-hoc dataset selection.

---

# 26. Report requirements

Create:

`REPORT_DAY3.md`

with the following structure.

```markdown
# Day 3 — Basis Geometry and Categorical Conditioning

## Executive verdict
- Is the ICLR thesis stronger, weaker, or unchanged?
- One paragraph only.

## 1. Reproduction of Day-2 anchors

## 2. Controlled numerical equivalent-basis sweep
### Setup
### Geometry verification
### Results
### Verdict

## 3. PLE vs identity whitening/canonicalization
### Equivalence tests
### Results
### Verdict

## 4. Representation-invariant regularization
### Algebra/unit tests
### Results
### Verdict

## 5. Controlled categorical equivalent-basis sweep
### Frequency spectra
### Results
### Verdict

## 6. Ordinal basis geometry
### Dataset/column semantic audit
### Local vs cumulative equivalence
### Orthogonal/path-spectral/whitened bases
### Controlled-kappa results
### Verdict

## 7. Datetime/cyclic topology
### Exact full-Fourier equivalence
### Phase controls
### Controlled-kappa results
### Low-frequency inductive-bias results
### Verdict / core vs appendix

## 8. Numerical-categorical block residualization
### Diamonds case study
### Broader results
### Verdict

## 9. Residual target encoding
### Leakage-safe construction
### Results
### Verdict

## 10. Frequency-aware categorical preconditioning
### Mechanism diagnostics
### Results
### Verdict

## 11. Numeric × categorical cross-atoms
### Controls
### Results
### Verdict

## 12. Unified analysis
- condition number vs convergence
- condition number vs generalization
- numerical vs categorical symmetry
- effect of optimizer and weight decay

## 13. Hypothesis table
| Hypothesis | Supported? | Evidence | Confidence |

## 14. ICLR paper decision
### Strongest defensible thesis
### What should be removed
### What Day 4 should test

## 15. Full experiment inventory
```

Every figure/table must be reproducible from saved raw result files.

---

# 27. Required plots

At minimum generate:

1. `numeric_kappa_vs_metric`
2. `numeric_kappa_vs_convergence`
3. `ple_identity_gap_before_after_whitening`
4. `basis_sensitivity_standard_vs_invariant_regularizer`
5. `categorical_frequency_spectrum`
6. `categorical_kappa_vs_metric`
7. `ordinal_local_vs_cumulative_spectrum`
8. `ordinal_basis_metric_and_convergence`
9. `ordinal_gap_before_after_whitening`
10. `cyclic_full_fourier_phase_control`
11. `cyclic_kappa_vs_metric`
12. `full_vs_truncated_fourier` if the datetime branch reaches P2
13. `joint_condition_before_after_block_residualization`
14. `diamonds_variants`
15. `frequency_vs_embedding_update_statistics`
16. `summary_geometry_vs_performance`

Use clean publication-style labels, but do not spend excessive time on cosmetics.

---

# 28. Implementation details that matter

## SVD/whitening stability
- center using train statistics;
- use float64 for geometry calculations if feasible;
- define rank tolerance explicitly;
- shrink tiny eigenvalues;
- report retained rank;
- apply transform fit on train to val/test.

## Ordinal basis integrity
- store declared ordinal level order in experiment metadata;
- unit-test local ↔ cumulative reconstruction;
- never derive order from target statistics;
- distinguish equal-spacing assumptions from order-only assumptions.

## Cyclic Fourier integrity
- implement a real orthonormal DFT/contrast basis with correct treatment of the constant component and Nyquist term for even `K`;
- unit-test orthogonality and exact reconstruction;
- record phase origin;
- distinguish full exact Fourier from truncated low-frequency encodings.

## Sparse categorical matrices
- keep one-hot matrices sparse;
- use sparse QR/SVD or algebraic shortcuts where possible;
- avoid densifying high-cardinality data.

## Determinism
- seed numpy, torch, dataloaders, and transform generation;
- store generated transform matrices or their seeds.

## Condition-number transforms
- confirm realized condition number numerically;
- store target and realized kappa.

## GPU usage
- detect available GPUs;
- parallelize independent runs safely;
- do not run two memory-heavy jobs on the same GPU if it creates OOM instability.

---

# 29. Optional theoretical note to include in report

If empirical evidence supports it, include a concise derivation:

Given equivalent features:

`z' = A z`

and first-layer parameters:

`W' = W A^{-1}`,

predictions are unchanged:

`W'z' = Wz`.

But L2 changes:

`||W'||_F^2 = tr(W A^{-1} A^{-T} W^T)`,

so standard weight decay privileges some coordinate systems.

Meanwhile:

`Sigma' = A Sigma A^T`

and:

`tr(W' Sigma' W'^T)
 = tr(W A^{-1} A Sigma A^T A^{-T} W^T)
 = tr(W Sigma W^T)`.

Thus a first-layer function-space penalty based on activation energy is basis-invariant.

Verify assumptions and do not claim invariance for nonlinear/later-layer regularization beyond what is actually proven.

---

# 30. A stronger categorical formulation than “scale the embedding table”

Do not focus the main categorical story on bounding:

`kappa(E^T E)`.

That matrix describes the learned embedding table geometry, but is not necessarily the curvature seen by SGD.

Prefer:

1. input categorical covariance spectrum;
2. sample-weighted contrast/embedding Jacobian covariance;
3. gradient/Fisher/GGN statistics;
4. exact frequency whitening or its efficient optimizer dual.

If later evidence shows `E^T E` itself is predictive, report it as an empirical diagnostic, not an assumed mechanism.

---

# 31. Final decision logic

At the end of Day 3 classify each branch:

```text
CORE
SUPPORTING
FOLLOW-UP
DROP
```

Expected prior:

- Controlled numerical equivalent-basis sweep: CORE
- Whitening/canonicalization: CORE
- Invariant regularization: CORE if it works
- Nominal categorical equivalent-basis sweep: CORE if it works
- **Ordinal local/cumulative/orthogonal basis experiment: CORE if it mirrors the numerical result**
- Datetime full-Fourier exact-basis experiment: SUPPORTING; CORE only if unusually revealing
- Datetime truncated/learned periodic encodings: FOLLOW-UP/APPENDIX
- Exact block residualization: CORE/SUPPORTING
- Residual target encoding: SUPPORTING at best
- Frequency-aware optimizer: SUPPORTING/FOLLOW-UP
- Cross-atoms: FOLLOW-UP/DROP unless unusually broad

Do not allow a weak secondary branch to dilute a strong central result.

---

# 32. End-of-run answer the coding agent must give

When finished, do not merely say “experiments completed.”

Return a concise research verdict answering:

1. **Did controlled condition number causally affect learning under information-equivalent bases?**
2. **Did whitening explain the PLE/identity phenomenon?**
3. **Did nominal categorical features show the same basis-conditioning effect?**
4. **Did ordinal local vs cumulative bases reproduce the numerical PLE geometry story, and did whitening/orthogonalization remove the gap?**
5. **Did full-Fourier cyclic controls behave as expected, and did datetime reveal anything beyond established cyclical encoding practice?**
6. **Did exact categorical-vs-numeric residualization help Diamonds?**
7. **Did invariant regularization reduce schema/basis sensitivity?**
8. **Which structured-feature ideas survived: nominal whitening, ordinal basis geometry, datetime exact Fourier, residual TE, frequency preconditioning, cross-atoms?**
9. **What is now the strongest one-sentence ICLR claim?**
10. **What single experiment should Day 4 run next?**

Also point to:
- `REPORT_DAY3.md`
- raw result directory
- key plots
- relevant commits

---

# 33. One-sentence mission reminder

**Do not build a bag of tabular tricks. Determine whether information-equivalent representations of continuous, nominal, ordinal, and cyclic tabular structure change neural learning through basis geometry; separate topology/inductive bias from coordinate choice, and test whether avoidable basis dependence can be removed.**
