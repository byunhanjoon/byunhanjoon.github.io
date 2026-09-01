# AGENT.md — ICLR 2027 Research Program: Task-Isomorphic Reparameterization in Tabular Foundation Models

## Mission

You are an autonomous research-engineering agent. Your goal is to determine whether the following research direction can support a strong ICLR 2027 submission, and if it can, produce the evidence, method, ablations, benchmark results, and paper-ready artifacts needed to make the case.

**Core research question**

> When a numerical feature is transformed by an information-preserving bijection, consistently for the entire supervised-learning task (context/train and query/test), what do modern tabular foundation models believe has changed? Which parts of a feature's marginal distribution are useful task metadata, and which are accidental coordinate choices?

The intended contribution is **not** “quantile encoding is good,” “ensembling transformations helps,” or “TFMs are sensitive to preprocessing.” Those claims are too close to existing work.

The strongest target thesis is:

> Modern tabular foundation models entangle coordinate representation with marginal-distribution metadata. This gives useful priors on ordinary tables but causes systematic prediction changes on task-isomorphic reparameterizations. A principled symmetrization/factorization of the prior or representation can reduce this failure without sacrificing clean-data accuracy.

Treat this as a hypothesis to falsify. **Do not try to manufacture a positive result.** If the hypothesis fails, say so clearly and identify the most defensible pivot.

No experiment can guarantee ICLR acceptance. Optimize for a result that would survive skeptical expert review.

---

# 0. Non-negotiable scientific rules

1. **Never tune on final test results.**
2. Maintain separate:
   - development datasets,
   - method-selection validation splits,
   - frozen confirmatory datasets/splits.
3. Once the final method and hyperparameters are frozen, record the freeze in `results/METHOD_FREEZE.md`. Do not change them after viewing confirmatory results except to fix a documented implementation bug.
4. Every transformation must be labeled as one of:
   - exact analytic bijection,
   - bijection on observed support,
   - order-preserving but lossy because of ties/finite precision,
   - non-bijective control.
5. Never call a result “invariance failure” if information was lost.
6. For data-dependent transforms, fit the transform on **training/context data only** and apply it unchanged to validation/query/test.
7. Keep labels, splits, feature identities, missingness masks, and train/test membership unchanged unless a specific control experiment says otherwise.
8. Always compare:
   - original task,
   - matched whole-task transform,
   - context-only transform,
   - query-only transform.
9. Report performance **and prediction disagreement**. A model may preserve accuracy while changing its posterior substantially.
10. Preserve all raw predictions needed to recompute metrics.
11. Use confidence intervals and paired statistics across datasets/splits.
12. Prefer simple explanations over architectural complexity.
13. Do not claim novelty without checking the novelty ledger described below.
14. Never overwrite old experiment outputs. New runs get immutable run IDs.
15. If something fails, write the failure down. Negative results are useful for deciding the paper direction.

---

# 1. Current literature guardrails

Before coding the final method, create `NOVELTY_LEDGER.md` and summarize at minimum the overlap with:

- **TabPFN-3** — current large-context TabPFN generation; arXiv:2605.13986.
- **TabICLv2** — ICML 2026 generation of TabICL; official code includes inference and pretraining recipes.
- **Mitra** — mixed synthetic priors for TFMs; arXiv:2510.21204.
- **A Mechanistic Study of Tabular Foundation Models** — arXiv:2605.21288.
- **EquiTabPFN** — target-permutation equivariance, NeurIPS 2025.
- **Tabular Numeric Stretch Transformation** — arXiv:2608.09162.
- **TFM-Retouche** — architecture-agnostic input-space adaptation; arXiv:2605.06047.
- **TabM** — ICLR 2025.
- **TabR**.
- **TabDPT / current retrieval or long-context TFM work**.
- **TabArena / BeyondArena** current benchmark code.

For every proposed claim, put it in one of these columns:

| Proposed claim | Existing closest work | Difference | Required experiment proving difference | Safe to claim? |
|---|---|---|---|---|

At minimum, explicitly exclude these weak novelty claims:

- “TFMs use preprocessing ensembles.”
- “Quantile transforms can improve tabular learning.”
- “Monotone distortions can change TFM predictions.”
- “A trainable input-space adapter can improve a frozen TFM.”
- “Feature/class/row permutation symmetry matters.”
- “Different transformed views can be ensembled.”

The project should instead focus on **matched whole-task task-isomorphisms**, the role of **marginal shape as latent metadata**, and a method that addresses the resulting tradeoff.

If internet access is available, repeat the novelty search before the final benchmark and again before paper drafting. Search for papers released after August 2026 that contain combinations of:
`TabPFN monotone invariant`, `tabular foundation reparameterization`, `marginal distribution metadata tabular`, `prior symmetrization tabular`, `rank invariant TabPFN`, `coordinate invariance tabular foundation`.

---

# 2. Repository audit and reproducibility setup

First inspect the existing repository. Do not delete or restructure working code unnecessarily.

Create or ensure the following structure:

```text
configs/
  audit/
  synthetic/
  methods/
  benchmark/
src/
  data/
  models/
  transforms/
  metrics/
  synthetic/
  methods/
  pretraining/
  analysis/
scripts/
tests/
results/
  raw/
  summaries/
  figures/
  tables/
reports/
```

Create:

- `reports/EXPERIMENT_LOG.md`
- `reports/NOVELTY_LEDGER.md`
- `reports/CLAIMS_EVIDENCE_MATRIX.md`
- `results/MANIFEST.jsonl`

Each run record must include:

- git commit,
- dirty/uncommitted diff hash if applicable,
- Python version,
- package versions,
- model checkpoint/version,
- dataset and split IDs,
- transformation and parameters,
- seed,
- device,
- wall-clock time,
- peak GPU memory if measurable,
- command/config,
- result path.

Use deterministic seeds where supported. Use multiple seeds when stochasticity materially affects conclusions.

Detect available GPUs automatically. Use up to two GPUs if available, primarily by running independent dataset/model jobs in parallel. Do not blindly place two large TFMs on one GPU.

Use AMP/BF16 where supported. Do not change numerical precision between original and transformed conditions inside a paired experiment.

Add smoke tests for:
- transformation invertibility,
- no train/test leakage,
- prediction shape,
- metric correctness,
- deterministic reload,
- cached result reload.

---

# 3. Define the scientific objects precisely

## 3.1 Task isomorphism

For a supervised task

\[
D = \{(x_i,y_i)\}_{i=1}^{n},
\]

and a featurewise bijection \(h\), define the transformed task

\[
h(D) = \{(h(x_i),y_i)\}_{i=1}^{n}.
\]

For ICL TFMs, if the original episode is:

\[
(D_c, X_q),
\]

the matched transformed episode is:

\[
(h(D_c), h(X_q)).
\]

The primary audit asks whether:

\[
p_\theta(y_q \mid D_c, X_q)
\]

is close to:

\[
p_\theta(y_q \mid h(D_c), h(X_q)).
\]

This is different from a context-only attack:

\[
p_\theta(y_q \mid h(D_c), X_q).
\]

Always keep those protocols separate.

## 3.2 Primary quantities

For classification define:

- clean predictive loss,
- transformed predictive loss,
- `isomorphism_gap = transformed_loss - clean_loss`,
- accuracy/AUC gap,
- Jensen-Shannon divergence between class-probability vectors,
- total-variation distance,
- argmax flip rate,
- Brier score,
- calibration error.

For regression define:

- RMSE / MAE / R²,
- normalized absolute prediction disagreement,
- rank correlation of predictions,
- if predictive quantiles are available, interval coverage/calibration.

Also define an aggregate **Reparameterization Robustness Score** as area under the performance-vs-severity curve, normalized per dataset. Do not use this as the only metric.

---

# 4. Transformation library

Implement a composable transformation API:

```python
class FeatureTransform:
    def fit(self, X_train, feature_metadata=None): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...
    def audit(self, X_train, X_test): ...
```

Every transform must emit metadata:
- exactness class,
- monotonicity,
- whether order is preserved,
- whether distances are preserved,
- whether it is data-dependent,
- transform severity.

## 4.1 Exact analytic numerical bijections

Implement per-feature versions of:

### A. Positive affine unit change
\[
h(x)=a x+b,\quad a>0
\]

Use several severities. Choose `b` in robust units rather than arbitrary giant constants.

### B. Order-reversing affine control
\[
h(x)=a x+b,\quad a<0
\]

This preserves information but reverses rank. Keep it separate from order-preserving results.

### C. Centered signed power
For robust center `c` and scale `s`:

\[
z=(x-c)/s
\]
\[
h_p(x)=\mathrm{sign}(z)|z|^p
\]

with a safe inverse. Use values around:
`p ∈ {0.5, 2.0, 3.0}`.

### D. Smooth saturating / expanding bijections
Use numerically safe analytic functions such as scaled `asinh`, and optionally `sinh` only in clipped domains with verified no overflow.

## 4.2 Random monotone piecewise-linear bijection

This is a core stress test.

For each feature:

1. robust-standardize using train median/IQR;
2. choose fixed knots in standardized coordinate space;
3. sample positive slopes log-uniformly;
4. integrate slopes to produce a strictly increasing piecewise-linear map;
5. use positive-slope linear tails;
6. normalize output scale so severity is controlled separately from trivial scaling.

Severity should control log-slope spread.

Important:
- transformation parameters are sampled independently of labels;
- same transformation is applied to context/train and query/test;
- store the exact map for reproducibility.

## 4.3 Monotone spline warp

Implement a strictly monotone spline or monotone Hermite transform with linear tails.

Use it as a **held-out transform family** when testing a method trained on piecewise-linear/random analytic warps.

## 4.4 Data-dependent rank / CDF transforms

Implement:
- empirical CDF / midrank,
- quantile-to-uniform,
- quantile-to-Gaussian.

These are **not** the cleanest proof of information-preserving invariance in the presence of ties. Label them accurately.

They are important baselines because existing tabular methods already use them.

## 4.5 Discrete numerical / atomic remapping

For integer-like or highly tied numerical features:

- preserve equality classes;
- preserve ordering;
- randomly alter spacing between unique observed levels;
- do not merge distinct values.

Record:
- number of unique levels,
- largest atom mass,
- entropy,
- fraction of repeated values.

## 4.6 Categorical bijection

Relabel category identities bijectively while preserving category membership exactly.

Test:
- string labels,
- integer-coded categoricals,
- feature metadata preserved,
- feature metadata removed if the model supports both.

This is a control, not necessarily the main contribution.

## 4.7 Composition

Create transforms composed of 2–3 independently sampled monotone maps.

Do not allow pathological numerical overflow.

---

# 5. Four-way protocol that must appear in the paper

For every selected numerical transform, evaluate the following four cells:

| Context / train | Query / test | Meaning |
|---|---|---|
| original | original | clean |
| transformed | transformed | **matched task-isomorphism** |
| transformed | original | context-only mismatch / attack |
| original | transformed | query-only mismatch |

For trained models such as XGBoost/TabM/MLP:
- refit on transformed train when evaluating the matched condition.

For ICL TFMs:
- pass transformed context and transformed query in the same forward inference protocol.

This experiment is one of the most important pieces of the project because it distinguishes the thesis from prior perturbation work.

---

# 6. Models

## 6.1 Mandatory current TFMs

Prioritize:

1. **TabPFN-3**
2. **TabICLv2**
3. **Mitra**

Add, if installation/checkpoints are practical:

4. TabDPT
5. another strong current open TFM represented in TabArena
6. TabPFN-2.5/2.6 as historical comparison if useful

Record exact package and checkpoint versions.

For TabPFN, evaluate both:
- a minimal/single-estimator configuration with as little transform ensembling as practical,
- the recommended/default ensemble configuration.

This is essential. We need to know whether preprocessing ensembles merely hide the underlying sensitivity.

For TabICL, similarly test a low-ensemble setting and the recommended setting if exposed by the API.

Do not disable important model components silently. Every modified inference setting must be explicitly labeled.

## 6.2 Conventional controls

Mandatory:
- CatBoost
- XGBoost
- LightGBM if practical
- Random Forest
- logistic/linear model on appropriate tasks
- MLP or RealMLP
- TabM

Optional:
- TabR
- kNN as a deliberately metric-sensitive baseline

Tree methods are scientifically useful controls because order-based split models should often be much less sensitive to smooth monotone reparameterizations than metric-dependent neural models.

## 6.3 Existing transformation/adaptation baselines

Where code is practical, compare against:
- standard z-score / robust scaling,
- quantile uniform,
- quantile Gaussian,
- power/Yeo-Johnson,
- raw + quantile duplicated features,
- Numeric Stretch-style transforms where applicable,
- TFM-Retouche where an implementation is available and fair,
- test-time transform ensemble / orbit averaging.

Transform ensembling is an **upper-bound/control**, not the novelty.

---

# 7. Dataset protocol

## 7.1 Pilot panel

Programmatically create a fixed pilot panel from TabArena/OpenML with approximately 12–16 datasets, stratified by:

- classification vs regression,
- small vs medium sample size,
- low vs high dimensionality,
- mostly numerical vs mixed categorical,
- low vs high skewness,
- low vs high fraction of tied/atomic values.

Use a fixed selection seed and save the selected names before evaluating the hypothesis.

Do not cherry-pick datasets after seeing results.

## 7.2 Development suite

Use a larger development suite, ideally 20–30 datasets.

This is where:
- transform severity,
- method hyperparameters,
- consistency coefficient,
- canonicalization choices,
- pretraining recipe,
are selected.

## 7.3 Confirmatory benchmark

After `METHOD_FREEZE.md` is written:

1. run the full current **TabArena v0.1** suite if compute permits;
2. run **BeyondArena** for IID/temporal/grouped stress testing, at minimum a fixed stratified subset and preferably the full suite if feasible.

Report both ordinary clean performance and transformed-task robustness.

Do not market the paper as “SOTA on TabArena” unless it actually is. A strong mechanistic/robustness contribution does not require rank 1 on clean IID prediction.

---

# 8. Phase I — Fast kill test

Do this before building a new method.

Use:
- TabPFN-3,
- TabICLv2,
- Mitra,
- XGBoost,
- CatBoost,
- TabM if quick enough.

On the pilot datasets evaluate:
- affine,
- signed power,
- random monotone PWL,
- held-out monotone spline,
at 3 severities.

Run matched whole-task and mismatch protocols.

Primary questions:

1. Do current TFMs show reproducible prediction disagreement under **matched** transformations?
2. Is there a practically meaningful performance gap?
3. Is the effect larger than tree controls?
4. Does the default TabPFN transform ensemble eliminate it?
5. Is the effect only present for absurdly extreme transforms?
6. Do different TFM families behave differently?

Write `reports/PHASE1_KILL_TEST.md`.

## Go/no-go rule

Proceed to the full research program only if at least one of these is true:

### Route A — performance failure
At least two strong current TFMs show a systematic matched-transform performance degradation across a nontrivial fraction of datasets and ordinary smooth transformations, with confidence intervals inconsistent with a negligible effect.

### Route B — posterior instability
Clean performance is nearly unchanged but posterior predictions/calibration change substantially and systematically, creating a defensible reliability result.

### Route C — mechanistic heterogeneity
Different TFMs show sharply different sensitivity patterns that can be tied to their learned readout/embedding mechanisms, yielding a strong scientific story.

Kill or heavily pivot if:
- only context/query mismatch causes failure;
- only clearly pathological transforms cause failure;
- default modern inference entirely removes the effect;
- the phenomenon is smaller than ordinary seed noise;
- a standard quantile transform trivially fixes everything with no tradeoff;
- recent literature already reports the exact matched whole-task experiment and remedy.

If killed, do **not** continue with expensive pretraining. Write the best pivot recommendation.

---

# 9. Phase II — Full reparameterization audit

If Phase I survives, run the complete transformation matrix.

For each model × dataset × split × transform × severity:

Store:
- raw predictions,
- clean and transformed metrics,
- disagreement metrics,
- inference time,
- model preprocessing configuration.

Create figures:

1. **Matched vs mismatch gap**
   - x-axis transform severity
   - separate curves for matched, context-only, query-only.

2. **Model-family robustness**
   - per-model aggregate robustness score with paired confidence intervals.

3. **Prediction disagreement vs loss gap**
   - identify cases where posterior changes without accuracy loss.

4. **Dataset sensitivity map**
   - rows datasets, columns transform families, values normalized gap.

5. **Clean ensemble vs single estimator**
   - specifically for TabPFN and TabICL if possible.

6. **Tree vs neural control**
   - paired dataset-level gaps.

Perform a hierarchical/bootstrap analysis over datasets and splits.

Do not use dozens of uncorrected p-values. Use effect sizes and confidence intervals; use Holm correction if formal multiple comparisons are reported.

---

# 10. Phase III — What marginal shape is the model using?

The paper becomes substantially stronger if it does not stop at robustness.

For each numerical feature compute train-only descriptors:

- skewness,
- kurtosis,
- quantiles,
- robust scale,
- number of unique values,
- largest atom mass,
- entropy of binned values,
- zero mass,
- tail-heaviness,
- spacing irregularity,
- missingness rate.

Relate these descriptors to:
- isomorphism gap,
- prediction JS divergence,
- sensitivity to each transform family.

Fit simple transparent meta-models for analysis only:
- linear/ridge,
- random forest,
to predict dataset/feature sensitivity.

Use nested or cross-dataset validation if claiming predictability.

The goal is not a better benchmark predictor; the goal is to learn which aspects of feature marginals correlate with TFM instability.

---

# 11. Phase IV — Controlled synthetic meta-tasks

Create a synthetic generator where the Bayes structure and meta-level relation between marginal shape and target function are known.

Vary:
- context size: roughly `{32, 64, 128, 256, 512, 1024}` where supported,
- number of features: `{5, 20, 50}`,
- noise,
- feature interaction strength,
- class balance.

## Family S1 — Marginal is pure nuisance

1. Sample latent coordinate `u`.
2. Define target from a function of `u`.
3. Independently draw a strictly monotone transform `h`.
4. Observe `x = h(u)`.
5. `h` is independent of the target function family.

The model should learn from context rather than use the arbitrary marginal as a shortcut.

Test interpolation and held-out transform families.

## Family S2 — Marginal shape is useful metadata

Create a meta-distribution where marginal family intentionally predicts the target-function family.

Example:
- Gaussian marginal -> one function family,
- heavy-tailed marginal -> another,
- bounded/bimodal -> another.

Do this carefully so marginal shape genuinely helps when context is small.

This demonstrates why “just rank-normalize everything” can remove useful prior information.

## Family S3 — Marginal is randomized nuisance

Use the same set of target-function families as S2, but randomize marginal family independently.

Compare TFM performance to S2.

The S2–S3 difference estimates how much the model benefits from marginal-shape metadata.

## Family S4 — Prior conflict

Train/evaluate a synthetic meta-prior where marginal shape usually predicts a function family, then flip or randomize that association at test.

This tests whether the TFM is relying on a marginal shortcut that becomes harmful under meta-distribution shift.

## Family S5 — Atomic / mixed discrete-continuous numerical features

Generate:
- zero-inflated continuous features,
- spike-and-slab,
- integer counts,
- ordinal numerical levels with irregular spacing,
- mixtures containing exact point masses plus continuous tails.

Measure whether:
- equality/atom information,
- rank,
- metric spacing,
play separable roles.

## Family S6 — Multivariate interactions

Generate functions depending on interactions of 2–5 features and independently reparameterize each feature.

This prevents the project from only studying one-dimensional threshold tasks.

---

# 12. Mechanistic analysis

Use model internals where access is straightforward. Do not spend weeks reverse-engineering inaccessible models.

## 12.1 TabICL

Because TabICL explicitly builds distribution-aware column representations, test:

- how the column embedding changes under matched monotone reparameterization;
- whether embedding distance correlates with prediction disagreement;
- whether clean and warped versions of the same feature cluster together;
- how much marginal descriptors can linearly predict the column embedding.

Use hooks only if they do not materially modify inference.

## 12.2 Similarity/readout geometry

Recent mechanistic work reports similarity-based but distinct TFM readouts.

For accessible models measure whether a warp changes:
- nearest context points in learned representation,
- attention weights over context,
- class-prototype distances,
- local neighborhood overlap.

Compute:
- top-k neighbor overlap,
- Spearman correlation of context relevance scores,
- representation CKA/cosine similarity,
- relationship between neighborhood change and prediction change.

## 12.3 Causal restoration test

For examples with strong prediction changes:

1. original task -> record internal neighborhood/readout and prediction;
2. matched warped task -> record changed readout;
3. apply the proposed canonical/symmetrized representation -> test whether the original readout structure and prediction are restored.

This is far stronger than a purely correlational claim.

---

# 13. Method ladder

Do not jump directly to a large new architecture.

Evaluate increasingly ambitious remedies.

## M0 — Base model

Unmodified TFM.

## M1 — Standard preprocessing

- z-score,
- robust scaling,
- power transform,
- quantile uniform,
- quantile Gaussian.

## M2 — Canonical rank coordinate

Map each numerical feature using a train/context empirical CDF and apply the learned map to query/test.

This is a baseline, not expected novelty.

Evaluate:
- clean accuracy,
- matched robustness,
- calibration,
- damage on S2 where marginal shape is useful.

## M3 — Raw + canonical duplicate

Provide both raw and canonicalized views where the model interface permits.

This is important because TabPFN-style systems already use raw-plus-quantile ideas.

Treat it as a strong prior-art baseline.

## M4 — Explicit numerical factorization

Prototype a representation containing separable channels:

### Relative coordinate
\[
u_{ij}=\hat F_j(x_{ij})
\]

### Atom/equality mass
\[
a_{ij}=\hat P(X_j=x_{ij})
\]

Use stable transforms such as `log(a + eps)`.

### Metric/local-spacing descriptor
Estimate local quantile spacing or a robust derivative-like quantity that tells the model how raw metric spacing differs from rank spacing.

### Robust raw residual
A bounded robust coordinate such as an `asinh`-scaled residual.

Possible per-feature channels:

```text
rank_position
log_atom_mass
robust_value
local_spacing
missing_indicator
```

For a generic TFM, the simplest prototype may expand one source feature into several derived columns. This loses explicit grouping, so do not assume it is optimal.

For a trainable architecture such as TabM/MLP, preserve source-feature grouping and implement a small grouped encoder.

Key ablations:
- rank only,
- rank + atom,
- rank + robust raw,
- rank + atom + spacing,
- full factorization.

The claim is **not** that this exact factorization must win. It is a diagnostic tool for separating useful information channels.

## M5 — Orbit averaging

For K sampled matched transforms:
- transform context and query together,
- predict,
- average probabilities/predictions.

Use `K={2,4,8}` where feasible.

This establishes how much robustness is obtainable by brute force and what compute it costs.

Do not present this as the main novelty.

## M6 — Existing adaptive input method

Include TFM-Retouche or the nearest available architecture-agnostic adaptive baseline if reproducible.

## M7 — Prior symmetrization: main high-upside method

Provisional name:
**Reparameterization-Symmetrized Prior Fitting (RSPF)**.

Using open TabICL pretraining infrastructure or another reproducible TFM training codebase:

For each synthetic training episode:
1. sample task `D ~ P`;
2. sample featurewise monotone bijection `h ~ H`;
3. with probability `p_identity`, use original coordinates;
4. otherwise train on `h(D)`.

This makes the synthetic prior less dependent on arbitrary coordinate parameterization.

### RSPF-A: augmentation only

Train with the ordinary supervised TFM objective on randomly warped episodes.

### RSPF-B: paired orbit consistency

For paired original and warped views of the same task, add:

Classification:
\[
L = L_\text{sup}(D) + L_\text{sup}(h(D))
+ \lambda \, JS(p_\theta(\cdot|D), p_\theta(\cdot|h(D)))
\]

or a stable symmetric KL variant.

Regression:
use prediction/quantile consistency with an appropriate normalized loss.

Do not compare paired and unpaired training with different total supervised examples. Match compute and data exposure.

### RSPF-C: identity/warp mixture

Tune only on the development suite:
- identity probability,
- transform severity distribution,
- consistency coefficient.

Freeze afterward.

### Critical generalization test

Do **not** train and test on exactly the same transform family only.

Example:
- pretrain augmentation: affine + signed power + random PWL,
- held-out robustness: monotone spline and different slope ranges.

This tests whether the model learned a broader invariance rather than memorizing augmentation types.

## M8 — Factorized marginal/coordinate TFM

Only pursue this if:
- S2 demonstrates that marginal shape is useful,
- M2 shows canonicalization removes useful signal,
- RSPF improves robustness but costs clean/meta-prior performance.

Then modify an open TFM so coordinate and marginal information are separable.

Concept:

```text
cell coordinate stream:
    canonical rank / robust coordinate / atom information

feature marginal stream:
    train-context marginal summary token or distribution-aware feature embedding
```

The model can use marginal metadata deliberately without forcing every cell coordinate to carry it implicitly.

Possible feature summary:
- fixed quantile sketch,
- atom histogram,
- missingness,
- robust moments.

Do not add a huge hand-designed feature vector unless ablations show it is necessary.

Compare:
- coordinate only,
- marginal only,
- entangled baseline,
- separated two-stream representation.

This is the most ambitious route and should only be built if experiments justify it.

---

# 14. Pretraining experiments

Use the open TabICL pretraining recipe if practical.

Start small.

## 14.1 Small matched-compute proof

Train reduced models with identical:
- architecture,
- parameter count,
- optimizer,
- number of supervised episodes,
- total tokens/examples,
- wall-clock budget approximately,
except for the symmetrization intervention.

At least:
- Base prior
- RSPF-A
- RSPF-B

Use 3 seeds if the reduced run is cheap enough.

Evaluate on:
- held-out synthetic ordinary tasks,
- S1–S6,
- pilot real datasets,
- matched transform audit.

### Continue only if

RSPF meaningfully reduces the isomorphism gap **without a material clean-performance collapse**.

## 14.2 Scale-up

If small-model evidence is strong:

- increase model width/depth toward the official recipe,
- increase pretraining episodes,
- preserve matched-compute baselines,
- use at least two independent training seeds if possible.

Do not spend the entire compute budget reproducing a full flagship TFM if a medium-scale model already establishes the scientific claim.

A paper can be strong if the pretraining principle is clear and verified at meaningful scale, even if the new checkpoint does not beat TabPFN-3 overall.

---

# 15. Important controls reviewers will ask for

Run all of these if the project reaches paper stage.

## Control C1 — Is it just numerical scale?

Normalize output scale after warp. Show the effect persists when transformed features have matched mean/variance or robust scale.

## C2 — Is information lost?

For exact transforms:
- apply inverse,
- verify reconstruction error is near machine precision on sampled points.

## C3 — Is it just extrapolation beyond pretraining range?

Clip/control ranges and repeat with values kept in ordinary standardized ranges.

## C4 — Is it only one feature?

Run:
- one transformed feature,
- random 25%,
- random 50%,
- all numerical features.

## C5 — Does transform severity predict the gap?

Yes/no; report full curve.

## C6 — Is it a model preprocessor bug?

Feed equivalent transformed arrays through model preprocessing and audit:
- dtype,
- missingness,
- categorical inference,
- clipping,
- quantization,
- overflow.

## C7 — Does ensembling fix it?

Single estimator vs recommended ensemble.

## C8 — Does ordinary training fix it?

Compare ICL TFMs with refitted XGBoost/CatBoost/TabM/MLP.

## C9 — Is rank normalization enough?

Compare against canonical rank and raw+rank.

## C10 — Is the new method merely more compute?

Matched forward passes, matched pretraining episodes, and matched parameter counts.

## C11 — Does marginal shape sometimes help?

Synthetic S2 must answer this.

## C12 — Does the proposed method generalize to unseen transform families?

Hold out at least one transform family during method training/tuning.

## C13 — Does it generalize across classification and regression?

Run both before claiming a generic numerical representation principle.

## C14 — Does it help on OOD datasets?

Use BeyondArena temporal/grouped tasks after method freeze.

## C15 — Does it harm naturally meaningful metric features?

Create/select tasks where raw metric spacing carries useful finite-sample inductive bias and report the tradeoff.

---

# 16. Statistical analysis

Primary unit for broad claims: **dataset**, not individual row.

Use:
- paired dataset-level bootstrap confidence intervals,
- Wilcoxon signed-rank as a secondary test when appropriate,
- Holm correction for families of multiple comparisons,
- effect sizes.

For benchmark aggregate scores:
- report mean/median rank,
- normalized score if TabArena supports it,
- Elo only if produced by the benchmark's accepted procedure.

Avoid declaring success from a single average if performance is bimodal across datasets.

Always show:
- win/tie/loss counts,
- distribution of dataset-level effects,
- worst regressions.

---

# 17. Paper-worthy figures and tables

Generate these automatically from raw results.

## Figure 1 — The conceptual counterfactual
One task, two invertibly reparameterized coordinate systems, same labels and split, different TFM predictions.

Use a real dataset example plus a schematic.

## Figure 2 — Matched vs mismatch
Performance/disagreement vs transform severity for:
- matched whole-task,
- context-only,
- query-only.

## Figure 3 — Current model landscape
Robustness across TabPFN-3, TabICLv2, Mitra, trees, TabM.

## Figure 4 — Why marginal shape matters
Synthetic S2 vs S3 vs S4:
- useful metadata,
- nuisance,
- conflict.

## Figure 5 — Mechanism
Show representation/neighborhood/readout changes under warp and restoration under the proposed method.

## Figure 6 — Method tradeoff
Clean accuracy vs reparameterization robustness Pareto plot.

## Table 1 — Main real benchmark
Clean performance and robustness on frozen TabArena suite.

## Table 2 — Controlled synthetic
S1–S6.

## Table 3 — Ablations
Rank, atom, spacing, raw, augmentation, consistency, marginal stream.

## Table 4 — Compute
Parameters, pretraining compute, inference passes, latency, peak memory.

---

# 18. Theory target

Do not force a theorem if assumptions are false.

Try to formalize:

Let `G` be a finite set or suitably measurable family of invertible coordinate transformations acting on tasks.

For a task prior `P`, define a symmetrized prior:

\[
P_\text{sym} = \frac{1}{|G|}\sum_{g\in G} g_\# P
\]

for finite `G`, or an integral when a valid invariant measure exists.

Show, under clearly stated assumptions, that the Bayes predictive induced by a symmetrized prior is invariant/equivariant to the corresponding task transformation.

Important:
- do not casually invoke Haar measure for a noncompact transformation family without checking conditions;
- finite sampled transformation sets are sufficient for a clean proposition;
- distinguish exact invariance of the ideal prior predictive from approximate invariance of a finite neural approximation.

Also formalize the tradeoff:

If marginal shape `M` contains mutual information about task/function family `F` under the meta-prior, complete canonicalization can remove useful prior information. This motivates separating:
- coordinate nuisance,
- marginal metadata.

This theoretical framing should explain the S2/S3 experiments.

---

# 19. Decision gates

## Gate G1 — Phenomenon

Pass if matched task-isomorphic transforms produce a clear, reproducible and scientifically interesting effect in current TFMs.

If not: stop the main direction.

## Gate G2 — Explanation

Pass if the effect can be connected to:
- marginal descriptors,
- learned representation/readout changes,
- or controlled synthetic prior behavior.

If the effect is merely an opaque preprocessing artifact, the paper is weaker.

## Gate G3 — Remedy

Pass if one proposed method:
- substantially reduces robustness gap,
- generalizes to unseen transforms,
- does not materially harm clean performance.

## Gate G4 — Current-model relevance

Pass if the story holds for at least two current strong TFMs or if a new pretrained model demonstrates a general principle that clearly explains why existing families differ.

## Gate G5 — Benchmark relevance

Pass if the final method is competitive on clean real data and robust on the frozen benchmark.

It does not need to be rank 1, but it cannot trade a large clean-performance loss for synthetic invariance.

---

# 20. Suggested quantitative success targets

Treat these as decision heuristics, not p-hacking targets.

A compelling result would look roughly like:

1. Existing TFMs:
   - meaningful matched-transform posterior instability on many datasets;
   - measurable loss degradation on a significant subset;
   - stronger than tree controls.

2. RSPF / final method:
   - reduces median isomorphism gap by at least ~50%;
   - strongly reduces prediction JS/flip rate;
   - loses less than ~0.5–1% aggregate clean normalized performance, ideally none;
   - works on held-out transform families.

3. Synthetic:
   - standard TFM clearly benefits from marginal metadata in S2;
   - clearly suffers when that metadata is nuisance/conflicting in S3/S4;
   - final method improves the tradeoff.

4. Real benchmark:
   - no catastrophic regressions;
   - good win/tie/loss profile;
   - improvement is not isolated to one dataset family.

Do not hide failure cases to hit these thresholds.

---

# 21. Compute scheduling

Run in this order:

### Tier 0 — minutes
- unit tests,
- 2 datasets,
- 2 transforms,
- one TFM + XGBoost.

### Tier 1 — pilot
- 12–16 datasets,
- 3 TFMs,
- 2 classical controls,
- 4 transform families,
- 3 severities.

### Tier 2 — audit
- development suite,
- full model set,
- prediction storage,
- synthetic S1–S6.

### Tier 3 — method
- M1–M7 on development suite,
- small pretraining.

### Tier 4 — scale
Only after method freeze:
- larger pretraining,
- full TabArena,
- BeyondArena,
- final ablations.

Use early cancellation:
- if a run has a reproducible implementation error, cancel and rerun;
- do not cancel merely because results look bad.

---

# 22. Implementation details for performance

- Cache dataset loading and preprocessing.
- Cache model checkpoints locally.
- For ICL models that support KV/context caching, use it only when it does not alter the paired protocol.
- Batch query predictions consistently between clean and transformed conditions.
- Store probabilities in compressed arrays/parquet-compatible format.
- Use job manifests so interrupted benchmark runs resume.
- One job should correspond to a deterministic `(model, dataset, split, transform, severity, seed)` unit.
- Use a scheduler that knows GPU memory requirements.
- Keep full logs for failed jobs.
- Make benchmark scripts idempotent.

---

# 23. Required reports

After each major phase, update `reports/EXPERIMENT_LOG.md` and write:

1. `reports/PHASE1_KILL_TEST.md`
2. `reports/REPARAM_AUDIT.md`
3. `reports/SYNTHETIC_PRIOR_REPORT.md`
4. `reports/MECHANISM_REPORT.md`
5. `reports/METHOD_ABLATIONS.md`
6. `reports/PRETRAINING_REPORT.md`
7. `reports/FINAL_BENCHMARK.md`
8. `reports/ICLR_READINESS.md`

Each report must include:

- question,
- exact protocol,
- result table,
- confidence intervals,
- plots,
- interpretation,
- alternative explanations,
- next decision,
- links/paths to raw results.

---

# 24. CLAIMS_EVIDENCE_MATRIX.md

Maintain this continuously.

Example:

| Claim | Required evidence | Status |
|---|---|---|
| Modern TFMs are not stable to matched monotone task isomorphisms | full matched protocol, current TFMs, exact transforms | TODO |
| This differs from context-only attack sensitivity | 4-way protocol | TODO |
| Marginal shape acts as useful meta-information | S2>S3 controlled synthetic | TODO |
| Marginal shape can become a shortcut | S4 prior conflict | TODO |
| RSPF improves invariance | matched-compute pretraining | TODO |
| Improvement is not transform memorization | held-out transform family | TODO |
| Clean performance is preserved | frozen TabArena | TODO |
| Mechanism involves learned geometry/readout | internal-neighborhood analysis + causal restoration | TODO |

Do not draft an abstract with a claim marked TODO.

---

# 25. ICLR readiness rubric

At the end, score 0–2 on each:

## Novelty
0 = preprocessing variant  
1 = new empirical observation  
2 = new problem framing + mechanism + principled method

## Technical depth
0 = benchmark only  
1 = careful empirical study  
2 = empirical + mechanistic/theoretical explanation

## Evidence breadth
0 = few datasets/models  
1 = broad datasets  
2 = broad datasets + current TFMs + controlled synthetic + OOD

## Method strength
0 = no remedy  
1 = remedy but tradeoff/limited  
2 = simple remedy with clean/robust gains

## Reviewer-proof controls
0 = major confounds  
1 = most addressed  
2 = matched transforms, leakage, compute, ensembling, prior-art all addressed

## Reproducibility
0 = manual scripts  
1 = configs/logs  
2 = fully reproducible manifests, raw predictions, automatic figures

A credible ICLR submission should target **10–12 / 12**.

If score is ≤7, recommend a pivot or narrower venue rather than pretending the work is ready.

---

# 26. Possible final paper narratives

Choose only after results.

## Narrative A — strongest

**Marginal Shape as Metadata: Reparameterization Symmetry in Tabular Foundation Models**

1. TFMs use marginal distribution as latent task information.
2. This helps under ordinary meta-priors.
3. It makes the predictive prior coordinate-dependent on task-isomorphic views.
4. RSPF symmetrizes the prior and improves robustness without losing clean performance.
5. Controlled tasks explain when invariance should and should not be imposed.

## Narrative B — mechanistic

**Same Task, Different Coordinates: What Tabular Foundation Models Actually Learn**

Use if the empirical/mechanistic result is strong but remedy is modest.

Must still outperform the May 2026 mechanistic paper in specificity by emphasizing:
- matched whole-task transformations,
- marginal semantics,
- clean-vs-conflict meta-priors.

## Narrative C — representation

**Separating Coordinate and Marginal Information in Tabular Learning**

Use only if the factorized architecture clearly wins and has strong ablations.

Avoid claiming this if the method is merely “rank + extra features.”

---

# 27. Things not to waste time on

Unless new evidence demands it, do not make these the main paper:

- generic HeteroBag / different preprocessing views;
- generic PLE extension;
- generic feature pooling;
- generic context retrieval;
- arbitrary transformation ensembling;
- a new encoding tested only on MLP/FT-Transformer;
- a synthetic-only “geometry” result;
- huge hyperparameter sweeps before the hypothesis is validated.

The research value must come from the **problem formulation + controlled evidence + principled remedy**.

---

# 28. Autonomous operating instructions

Do not ask the user what to do next unless an external credential/license makes progress impossible.

Instead:

1. inspect the repository;
2. run the smallest valid experiment;
3. inspect results;
4. update the decision report;
5. proceed through the gates;
6. stop expensive branches that fail;
7. preserve negative results;
8. keep the best surviving direction focused.

When there are multiple reasonable implementation choices, choose the one that:
- minimizes confounds,
- is easiest to reproduce,
- has the fairest compute comparison,
- produces evidence a skeptical reviewer would trust.

If a package cannot be installed:
- record the exact error;
- isolate it in a separate environment if dependency conflicts are the issue;
- do not downgrade the whole environment destructively;
- use the nearest scientifically valid model only as a temporary fallback.

---

# 29. First concrete actions

Execute these immediately:

1. Build `NOVELTY_LEDGER.md`.
2. Install/load TabPFN-3, TabICLv2, Mitra, XGBoost, CatBoost.
3. Implement the transformation API and invertibility tests.
4. Select and freeze the pilot dataset panel.
5. Run:
   - identity,
   - affine,
   - signed power,
   - random monotone PWL,
   - monotone spline,
   with 3 severities.
6. Run all four matched/mismatch protocols.
7. Store raw probabilities.
8. Produce the Phase I plots.
9. Write `PHASE1_KILL_TEST.md`.
10. Make a go/no-go decision before implementing RSPF.

The first question to answer is not “does our method win?”

It is:

> **Does a current TFM assign materially different predictions to two fully matched, information-equivalent coordinate representations of the same task, after ruling out preprocessing bugs, information loss, context/query mismatch, and ordinary scale effects?**

Only if the answer is scientifically interesting should the project move on to the new method.

---

# 30. Final deliverable

If the direction survives, finish with a repository that can produce:

```bash
# reproduce core phenomenon
python scripts/run_reparam_audit.py --config configs/audit/main.yaml

# controlled synthetic experiments
python scripts/run_synthetic.py --config configs/synthetic/main.yaml

# method ablations
python scripts/run_methods.py --config configs/methods/main.yaml

# pretraining
python scripts/train_rspf.py --config configs/pretraining/rspf.yaml

# frozen final benchmark
python scripts/run_final_benchmark.py --config configs/benchmark/frozen.yaml

# all paper tables and figures
python scripts/make_paper_artifacts.py
```

And a final `reports/ICLR_READINESS.md` that answers, without hype:

- What is genuinely new?
- What exact prior work is closest?
- What phenomenon was discovered?
- Why does it happen?
- What method fixes it?
- What does the method cost?
- Where does it fail?
- Does it preserve clean predictive quality?
- Does it generalize across TFMs, datasets, tasks, and unseen transformations?
- What evidence would a skeptical ICLR reviewer still demand?

If those answers are not strong, recommend a pivot rather than forcing the paper.
