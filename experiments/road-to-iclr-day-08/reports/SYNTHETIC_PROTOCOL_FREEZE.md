# Prospective controlled-synthetic protocol

Frozen: 2026-08-31, before generator implementation and before any S1–S6 outcome. Execution is conditional on the Phase II/III evidence: this protocol is used to resolve Gate G2, not to rescue a failed phenomenon by searching generators.

## Scientific estimands

The synthetic study separates three quantities:

1. coordinate sensitivity when a feature marginal is nuisance (S1);
2. benefit from a meta-prior association between marginal family and target-function family (S2 minus S3);
3. harm when that learned association conflicts at test time (S4 conflict minus S4 in-prior).

S5 separates equality/atom, rank, and spacing information. S6 checks that conclusions survive multivariate interactions. Every task has a latent representation, a fully specified Bayes rule, and a train/query split generated jointly before any observation-space transformation.

## Frozen factor grid

- Context rows: `{32, 64, 128, 256, 512, 1024}`.
- Query rows: 512.
- Features: `{5, 20, 50}`.
- Observation noise: `{low, high}`; low/high are regression standard deviations `{0.05, 0.25}` after unit-variance target scaling and classification logit noise `{0.1, 0.5}`.
- Interaction order: `{1, 3}` for general families and `{2, 5}` for S6.
- Binary positive-class targets: `{0.2, 0.5}` through a train-independent intercept calibrated from the known latent distribution.
- Ten meta-task seeds per cell: `20260831` through `20260840`.
- Model/inference seeds: `20260831` and `20260832`; task rows and model randomness are distinct streams.
- Primary models: TabICLv2 default and single, Mitra default, TabPFN-v2.5 default and single (historical corroboration), TabM, XGBoost, and a linear/logistic control.

The full factorial is allowed to be executed in prespecified blocks for compute, but cells cannot be dropped because their early results are unfavorable. If compute forces a reduction, retain all factor levels using a deterministic balanced fractional design fixed before model outcomes and record the omitted cells.

## Generator families

### S1 — marginal as pure nuisance

Draw latent coordinates `u_j ~ Uniform(-1, 1)`. Draw a target function independently from `{linear, hinge, smooth periodic, low-order polynomial, random Fourier}` with coefficients normalized to unit latent-target variance. Independently draw one increasing bijection per feature from the training transform families `{positive affine, signed power, asinh, random monotone PWL}` and observe `x_j = h_j(u_j)`. The held-out test family is monotone spline with disjoint seeds and knot-slope ranges. Matched counterfactuals reuse latent rows, labels, split, and Bayes function and change only `h_j`.

### S2 — marginal as useful metadata

Draw observed marginal family from `{Gaussian, Student-t(3), bounded beta mixture}`. Map these respectively to target-function families `{linear/hinge, smooth periodic, polynomial/Fourier mixture}`. Define each target in the marginal probability coordinate `r_j = F_j(x_j)`, making the conditional rule exact while allowing the marginal family to predict the function-family prior. Mapping, coefficients, and noise are generated before rows. Small contexts are the primary regime; larger contexts test whether evidence in the table overrides the meta-prior.

### S3 — randomized marginal nuisance

Use exactly the S2 marginal and function-family multisets, but independently permute their association at the meta-task level using the task seed. All other distributions and compute are matched. `S2 - S3` is the prespecified marginal-metadata benefit estimand.

### S4 — prior conflict

Generate an 80/20 in-prior/conflict mixture. In-prior tasks use the S2 map. Conflict tasks use the fixed cyclic map `Gaussian -> periodic`, `Student-t -> polynomial/Fourier`, `bounded mixture -> linear/hinge`; a separate randomized-association cell is retained. Report in-prior, randomized, and conflict effects separately. Do not average them into a single score.

### S5 — atomic and mixed support

Feature families are `{zero-inflated continuous, spike-and-slab, Poisson count, ordinal levels, continuous-plus-tail atom}`. For each latent task construct three matched observation variants that preserve respectively: `(rank + equality + spacing)`, `(rank + equality, randomized spacing)`, and `(rank only, atom identities broken by deterministic jitter)`. The last is explicitly non-isomorphic and is a control, not evidence about exact invariance. Report atom/equality, rank, and spacing contrasts separately.

### S6 — multivariate interactions

Sample interacting feature groups of order 2 or 5 without replacement. Functions are sums of normalized products, XOR-like smooth logits, and radial interaction terms. Independently reparameterize every participating and nuisance feature. Main effects and interactions use disjoint coefficient streams so a univariate shortcut cannot solve the task.

## Leakage and counterfactual invariants

- Transformation/generator state is sampled from the task seed and never fitted to query labels.
- Any data-dependent observation transform is fitted on context features only.
- Original and matched variants share latent rows, targets, noise realization, class intercept, and row IDs.
- Context-only and query-only variants are retained as diagnostic mismatch cells but are not pooled with matched results.
- Generator state, Bayes parameters, raw probabilities/predictions, and row IDs are serialized per task.
- Exact transforms must pass the same missingness, monotonicity, and inverse audits as real-data experiments. S5 non-bijective controls are labeled explicitly.

## Metrics and inference

Primary metrics are matched loss gap and prediction disagreement (JS/TV/flips for classification; normalized absolute prediction difference and rank correlation for regression). Also report Bayes regret, calibration, context-size curves, win/tie/loss counts across meta-tasks, and worst cells.

Confidence intervals use a hierarchical bootstrap: resample generator families/cells, then meta-task seeds within cell. Paired S2–S3 and S4 contrasts reuse aligned function/noise seeds. Dataset-row or query-row pseudo-replication is forbidden. Formal families of tests, if used, receive Holm correction.

## Gate-G2 interpretation

- Evidence for useful marginal metadata requires a positive S2–S3 performance contrast concentrated at small contexts and diminishing as context becomes informative.
- Evidence for harmful shortcut use requires worse S4 conflict performance than both S4 in-prior and S3, with paired confidence intervals excluding a negligible effect.
- S1 sensitivity alone does not explain why the prior uses marginal shape.
- A generator-specific effect that does not align with real-data descriptor or model-family patterns fails G2.
- Negative or mixed results are retained in `SYNTHETIC_PRIOR_REPORT.md`; they cannot be used to tune a new family after outcomes.
