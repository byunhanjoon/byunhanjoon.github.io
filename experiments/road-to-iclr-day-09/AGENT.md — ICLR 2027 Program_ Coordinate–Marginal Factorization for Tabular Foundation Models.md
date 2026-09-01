# AGENT.md — ICLR 2027 Program: Coordinate–Marginal Factorization for Tabular Foundation Models

## 0. Mission

You are the autonomous research engineer for an ICLR 2027 project on **what numerical information a tabular foundation model should be invariant to**.

The working thesis is:

> Numerical coordinates contain at least two conceptually different kinds of information: (1) information that is stable under a chosen reparameterization group, such as order/ties/missingness, and (2) coordinate-dependent marginal geometry, such as spacing, skewness, tail shape, and scale. A tabular foundation model should not blindly keep or blindly discard the second kind. It should use it only to the extent that the task prior makes it predictively informative.

The target contribution is **not** “monotone transforms change TFM predictions.” That neighborhood is already occupied by recent mechanistic work. The target contribution is the **invariance–information tradeoff**, a controlled demonstration of when quotienting coordinates is Bayes-optimal or Bayes-suboptimal, and a factorized model/adapter that learns a better tradeoff than raw-only or invariant-only baselines.

Your job is to run the complete program below, preserve negative results, and produce the strongest scientifically defensible paper. Do not p-hack, silently change the estimand, tune on final test data, discard inconvenient datasets, or keep changing a method until a benchmark happens to turn positive. All method changes must follow the predeclared failure branches in this file and must be logged.

The desired end state is an ICLR-ready package containing:

- a formal problem statement and at least two clean propositions/theorems;
- a synthetic benchmark in which the usefulness of marginal geometry is continuously controllable;
- an audit of current TFMs on that benchmark;
- a factorized or gated method that is near the best raw/invariant tradeoff without an oracle regime bit at inference;
- a full clean-data evaluation on TabArena;
- a matched-reparameterization robustness evaluation on the same compatible tasks;
- a beyond-IID evaluation on BeyondArena, with TabReD as an optional targeted temporal stress test;
- compute-matched ablations and simple baselines that can falsify the architectural story;
- frozen protocols, immutable raw predictions, confidence intervals, and a claims/evidence matrix.

If the method does not work, the project must still produce a correct conclusion. Do not manufacture “good results.” Instead, execute the fallback tree in Section 16.

---

## 1. Existing evidence to treat as pilot evidence, not final confirmation

The previous Day-8 audit found a real but scoped phenomenon:

- TabICLv2 default showed about 1.03% excess total-variation posterior change under matched monotone reparameterizations;
- about 0.80% of predictions flipped;
- clean-to-matched accuracy changed by about -0.21 percentage points;
- regression predictions changed by about 0.027 normalized absolute units;
- single-estimator TabICLv2 was more sensitive;
- Mitra was much closer to its refit/noise floor;
- historical TabPFN-v2.5 behaved more like TabICL than Mitra;
- exploratory descriptor models implicated atom mass, skewness, kurtosis, and robust scale, but the dataset count was too small and multiple-comparison controls were incomplete.

Treat these as hypothesis-generating results. Do not reuse their small Phase-III descriptor screen as confirmatory evidence. The new project must independently validate any mechanism.

The earlier Day-1 through Day-7 program also established several constraints that should shape this project:

1. “Numerical versus categorical” is too coarse a thesis by itself.
2. Information-equivalent numerical bases can change optimization and fitted functions, but ordinary basis diversity is not automatically better than seed diversity.
3. Extreme conditioning effects can be mechanistically real without producing a competitive method on natural data.
4. Representation/orbit diversity can reduce Monte Carlo error, but equal-compute controls can erase apparent advantages.
5. Invariance is not automatically useful: known geometry helps only when the chosen quotient matches the real task.

The present project should therefore ask a sharper question: **what information is removed by a quotient, when is that removal harmless, and how should a TFM expose both stable and coordinate-dependent information without entangling them?**

---

## 2. Novelty boundary — claims we are NOT allowed to make

Do not claim novelty for any of the following alone:

- monotone transformations can change neural predictions;
- TFMs are sensitive to feature geometry;
- rank/quantile preprocessing can improve robustness;
- transform augmentation improves robustness;
- preprocessing ensembles improve tabular performance;
- synthetic priors matter for TFMs;
- mixing synthetic priors improves TFMs;
- an input adapter can improve a frozen TFM;
- permutation or order invariance matters;
- raw plus quantile views are useful;
- TabICL and Mitra implement different in-context mechanisms.

Known nearby work that must be cited and rechecked before submission:

- **TabPFN-3: Technical Report**, arXiv:2605.13986.
- **TabICLv2: A better, faster, scalable, and open tabular foundation model**, arXiv:2602.11139.
- **A Mechanistic Study of Tabular Foundation Models**, arXiv:2605.21288.
- **Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality (O'Prior)**, arXiv:2605.18971.
- **Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models**, NeurIPS 2025 / arXiv:2510.21204.
- **TabArena: A Living Benchmark for Machine Learning on Tabular Data**, arXiv:2506.16791.
- **Beyond IID: How General Are Tabular Foundation Models, Really?**, arXiv:2606.30410.
- **TabM**, ICLR 2025.
- **Better by Default / RealMLP**.
- **TabReD**, ICLR 2025.
- the TabPFN-v2/Nature work and its preprocessing ensembles.

At the start and end of the project, create/update `reports/NOVELTY_LEDGER.md` with exact overlapping claims and the remaining defensible novelty.

The paper-worthy novelty target is the conjunction of:

1. a formal distinction between quotient-stable information and coordinate-dependent marginal information;
2. a controlled prior that tunes how much the coordinate-dependent marginal is useful for predicting the task;
3. an empirical demonstration that invariant-only and raw-only learners occupy opposite sides of a predictable tradeoff;
4. evidence that current TFMs sit at different points on this tradeoff;
5. a factorized/gated model that improves the Pareto frontier without access to an oracle regime label at inference.

---

## 3. Core formalization

For a context/query episode

\[
D_c=\{(x_i,y_i)\}_{i=1}^n,\qquad X_q,
\]

let a featurewise transformation group or family \(G\) act consistently on context and query.

For the primary paper, use **strictly increasing, information-preserving featurewise transformations** as the main group \(G_+\). Treat decreasing/orientation-reversing transformations as a secondary extension. This keeps the primary invariance claim tied to order-preserving reparameterizations and avoids conflating value geometry with orientation semantics.

Define a decomposition conceptually as:

- \(S\): information retained by the chosen quotient/canonical representation, e.g. empirical order/ranks, tie structure, missingness, feature identity, and any explicitly retained invariant metadata;
- \(M\): coordinate-dependent marginal information removed by the quotient, e.g. quantile spacing, skewness, tails, robust scale, nonlinear metric geometry;
- \(Y^*\): a query label.

The paper should establish or carefully formalize the following statements.

### Proposition/Theorem T1 — invariant Bayes predictor under an invariant/symmetrized prior

For a finite transformation group \(G\), define the symmetrized task prior

\[
P_{\mathrm{sym}} = \frac{1}{|G|}\sum_{g\in G} g_\#P.
\]

Under appropriate measurable-space and likelihood conditions, the posterior predictive under the symmetrized prior can be chosen invariant to matched whole-task transformations in \(G\).

For continuous transformation families, either provide the appropriate Haar/invariant-measure version when mathematically justified or keep the theorem finite and use finite sampled groups in experiments. Do not overclaim continuous-group theory if the assumptions are not clean.

### Proposition/Theorem T2 — cost of discarding marginal information under log loss

Under optimal log loss,

\[
R_{\log}(S)-R_{\log}(S,M)=I(Y^*;M\mid S)\ge 0.
\]

Interpretation: quotient/canonicalization is free only when discarded marginal information contains no conditional predictive information beyond \(S\).

### Desired Proposition T3 — impossibility/tradeoff statement

Show that if two task priors agree on the quotient representation \(S\) but induce different predictive distributions through \(M\), then no strictly \(G\)-invariant predictor that only observes \(S\) can match the full-information Bayes risk whenever \(I(Y^*;M\mid S)>0\).

If possible, derive a mixture-prior regret expression or lower bound. If not, state the cleanest version you can prove.

Put all formal assumptions in `theory/ASSUMPTIONS.md`. Distinguish theorem, proposition, conjecture, and empirical analogy.

---

## 4. Main synthetic benchmark: PriorDial

Build a synthetic benchmark whose central control parameter tunes how informative coordinate-dependent marginal geometry is about the task.

Call it `PriorDial` internally; rename later only if useful.

### 4.1 Latent task

Generate latent numerical variables \(z\in\mathbb R^d\), then a target from a mechanism \(f_\theta(z)\).

Use a balanced mixture of mechanism families rather than one toy equation:

1. sparse linear / logistic;
2. additive smooth random splines;
3. threshold/step functions;
4. pairwise interactions;
5. shallow random tree/partition mechanisms;
6. periodic/sinusoidal mechanisms;
7. optional heteroscedastic regression family.

Use both classification and regression. Standardize regression targets using train/context information only. For classification, control the intercept to avoid extreme class imbalance.

Vary:

- context sizes: at least \(n\in\{32,64,128,256,512\}\);
- feature counts: at least \(d\in\{4,8,16,32\}\);
- informative feature fraction;
- feature correlation via independent and Gaussian-copula settings;
- label-noise levels.

The primary figures should use a modest regime where the task is nontrivial and prior information can matter, e.g. \(n=64\) or 128 and \(d=8\) or 16, while appendices sweep the rest.

### 4.2 Coordinate-shape families

Create a library of strictly increasing bijections with numerically stable inverses:

- identity;
- positive affine unit changes;
- centered signed powers with positive exponent;
- asinh/sinh-like maps with controlled severity;
- random monotone piecewise-linear maps with linear tails;
- monotone cubic/rational-quadratic splines with linear tails;
- smooth compositions of the above.

For discrete/atomic numerical variables, add ordered spacing remaps that preserve equality classes and order. Keep atom mass/ties conceptually separate from continuous metric spacing because atom mass is itself invariant under strictly monotone bijections.

Every transform object must support:

- `fit(context_x)` if data-dependent;
- `transform(x)`;
- `inverse_transform(x)` where meaningful;
- serialization of all state;
- monotonicity audit;
- inverse reconstruction audit;
- equality/tie preservation audit;
- finite-value audit.

### 4.3 The informativeness dial

Let \(C\) denote the latent task/mechanism family and \(W\) the coordinate-shape family used to map \(z\mapsto x=h_W(z)\).

Create a one-to-one or balanced mapping \(\pi(C)\) from mechanism families to warp families. Control dependence by \(\rho\in[0,1]\):

- with probability \(\rho\), choose \(W=\pi(C)\) or sample from a mechanism-specific warp distribution;
- with probability \(1-\rho\), sample \(W\) independently from the same global warp marginal.

Use a grid such as

\[
\rho\in\{0,0.1,0.25,0.5,0.75,0.9,1.0\}.
\]

Maintain the same marginal frequency of each warp family across the whole benchmark so that increasing \(\rho\) changes dependence rather than simple class frequency.

At \(\rho=0\), coordinate-dependent marginal shape should be nuisance with respect to the mechanism prior. At high \(\rho\), marginal shape should provide useful prior evidence about the mechanism, especially at small context sizes.

Verify this empirically with an oracle diagnostic before training any proposed method:

- predict \(C\) or key \(\theta\) from marginal descriptors alone;
- estimate the extra predictive value of shape for \(Y^*\) beyond the quotient representation;
- confirm that this value increases monotonically enough with \(\rho\).

If the dial does not actually control useful information, fix the generator before proceeding. Do not compensate by tuning the model.

### 4.4 Additional nuisance transform at evaluation

At evaluation, optionally compose an independent transform \(U\) on top of the original observed coordinate:

\[
x' = U(h_W(z)).
\]

This creates a controlled nuisance reparameterization.

Important: do not promise the impossible. If informative signal is encoded only in coordinate-dependent shape, an arbitrary monotone \(U\) can destroy that signal. The paper should expose this tradeoff, not claim that a model can be exactly invariant to all such \(U\) while also preserving all shape information.

Evaluate expected risk under a specified nuisance-transform prior and compare against the corresponding oracle. The proposed method should be judged by **regret under the mixture**, not by an impossible pointwise dominance requirement.

### 4.5 Synthetic confirmatory split

Separate synthetic task families into:

- development generator seeds/families;
- held-out transform parameter ranges;
- held-out warp compositions;
- held-out mechanism hyperparameter ranges;
- at least one held-out transform family not seen during method tuning.

Freeze the confirmatory suite in `configs/prior_dial_confirmatory.yaml` before final method selection.

---

## 5. Representation decomposition to test

Do not start with a large new model. Validate the principle with progressively stronger methods.

### Stable/order channel S

For every numerical feature, build a train/context-only canonical view containing:

- tie-aware empirical CDF / midrank mapped to a fixed interval;
- missingness mask;
- tie/atom information, e.g. local equality mass or largest atom mass;
- optional unique-fraction/cardinality metadata;
- feature identity.

The primary quotient should be invariant to strictly increasing transformations up to finite-sample/tie conventions.

### Coordinate-dependent shape channel M

Expose information that the rank quotient removes, but expose it explicitly rather than letting it be entangled with arbitrary raw coordinates.

Candidate shape summaries:

- a fixed grid of empirical quantiles after robust affine standardization;
- adjacent quantile spacings;
- robust scale;
- skewness;
- excess kurtosis;
- tail ratios;
- spacing irregularity;
- zero-centered/asymmetric quantile ratios;
- a learned permutation-invariant set encoder over context values.

Do not rely only on handcrafted moments in the final method unless they clearly win. They are a diagnostic baseline. The preferred final shape encoder is a small DeepSets/SetTransformer-like encoder of the context marginal or quantile function.

Keep invariant metadata such as missingness and atom mass separable from coordinate-dependent shape where possible. This distinction may itself become an important paper result.

---

## 6. Method ladder — cheapest falsification first

Implement methods in this order. Do not jump directly to expensive pretraining.

### M0 — Raw-only baseline

Use the host/backbone's ordinary numerical preprocessing.

### M1 — Robust affine baseline

Median/IQR or median/MAD standardization only.

### M2 — Quotient-only / rank baseline

Tie-aware empirical CDF or rank canonicalization. No raw coordinate information.

### M3 — Transform augmentation baseline

Train/pretrain on random monotone transforms or average several matched transform views. Match compute carefully.

### M4 — Raw + rank fixed ensemble

Two independent predictions:

- raw/host prediction;
- rank-canonicalized prediction.

Average logits/probabilities for classification and predictions for regression. This is a very strong simple baseline because it tests whether a learned gate is needed at all.

### M5 — Prediction-level Adaptive Coordinate–Marginal Gate (fast proof of concept)

Use two experts:

- `E_raw(Dc, Xq)`;
- `E_rank(Dc, Xq)`.

Train a small gate \(g(D_c)\in[0,1]\), optionally featurewise and then aggregated, to combine them:

Classification:

\[
\ell = g\,\ell_{raw} + (1-g)\,\ell_{rank}.
\]

Regression:

\[
\hat y = g\,\hat y_{raw} + (1-g)\,\hat y_{rank}.
\]

Gate inputs may include:

- context size and feature count;
- shape-channel summaries;
- invariant metadata;
- label distribution;
- low-cost context-only feature/label association summaries such as rank correlation or cross-validated univariate scores.

Never use query labels.

Train the gate on synthetic episodes only for the main clean-transfer experiment. No oracle \(\rho\) or regime bit is allowed at inference.

Evaluate:

- fixed 50/50 mixture;
- globally tuned fixed mixture using development tasks only;
- learned gate;
- oracle per-episode gate (upper bound only).

If the learned gate cannot beat the globally tuned fixed mixture and the oracle gate has little headroom, stop: there is no method opportunity in gating.

### M6 — Single-model dual-channel backbone

If M5 shows genuine learnable headroom, build a compute-matched single model that exposes S and M separately.

Preferred architecture:

- separate small numeric encoders for the order channel and shape/raw channel;
- shared row/column/task transformer trunk;
- learned featurewise or tokenwise residual gate from shape information into the stable representation;
- categorical features use the backbone's standard treatment;
- missingness retained explicitly;
- no doubling of total hidden width without a matched-capacity baseline.

A simple fusion is:

\[
e_{ij}=e_S(s_{ij}) + g_j(D_c)\,e_M(m_{ij},m_j).
\]

Also test concatenation + projection and prediction-level mixture. The final architecture should be selected on development tasks, then frozen.

### M7 — Synthetic-prior consistency training

Because the synthetic generator knows which transformations are nuisance, add an optional consistency term only on nuisance-tagged training pairs:

\[
\mathcal L = \mathcal L_{pred} + \lambda_{inv}\,D(p(\cdot\mid D_c,X_q),p(\cdot\mid U(D_c),U(X_q))).
\]

Use JS/KL for classification and squared normalized prediction difference for regression.

This is allowed at pretraining because nuisance labels are generator metadata, but the final model may not receive them at inference. Always include an ablation without this auxiliary loss.

### M8 — Frozen-TFM adapter route

If training a full TFM is too expensive or M6 works only weakly, implement M5 as a model-agnostic frozen-host adapter around current TFMs.

Mandatory hosts:

- TabPFN-3;
- TabICLv2;
- Mitra if inference interfaces permit efficient repeated calls.

Compare the learned gate against equal-forward-pass fixed ensembling and transform ensembling. Report wall-clock and forward-pass cost.

This route is publishable only if it provides a clear cross-family tradeoff advantage; “two views are better than one” is insufficient novelty.

---

## 7. Same-compute controls

Every claimed method improvement must survive at least one fair compute/capacity control.

For learned standalone models:

- match total train tokens;
- match optimizer steps;
- report parameter counts;
- if dual-channel doubles parameters, train a widened raw-only baseline with the same parameter count;
- if dual-channel doubles FLOPs, train a raw-only baseline for the same FLOPs/tokens or explicitly show both compute-matched and parameter-matched comparisons.

For inference adapters/ensembles:

- compare equal numbers of host forward passes;
- compare learned gate vs 50/50 raw+rank at two passes;
- compare vs two independent preprocessing views;
- compare vs random transform ensemble at the same number of views;
- if using OrbitCover-style deterministic view selection, compare it with random views at equal budget and keep the estimand identical.

Never compare a two-pass proposed method with a one-pass baseline without also showing the one-pass/two-pass tradeoff.

---

## 8. Existing-model scientific audit on PriorDial

Before claiming a method, audit current models to understand where they lie on the tradeoff.

### Mandatory current TFMs

1. **TabPFN-3 default local research checkpoint** — headline current frontier model.
2. **TabPFN-3 OOD/preprocessing checkpoint** — important preprocessing control.
3. **TabICLv2 default inference**.
4. **TabICLv2 single-estimator / minimal-ensemble inference**, if exposed.
5. **Mitra default**.

### Secondary/historical TFM

- TabPFN-v2.5, only as continuity with the previous audit; do not headline it as current SOTA.

### Strong non-TFM controls

- TabM;
- RealMLP;
- XGBoost;
- CatBoost;
- LightGBM;
- xRFM if the current environment supports it reliably;
- optionally the strongest additional TabArena model family available at protocol freeze.

At protocol freeze, programmatically read the current TabArena leaderboard and record the top methods in `reports/BASELINE_FREEZE.md`. Do not keep adding baselines after seeing final outcomes unless clearly labeled post-hoc.

### Questions to answer

For each model, as a function of \(\rho\), context size, and transform severity:

1. How much clean risk is gained by access to coordinate-dependent marginal shape?
2. How much matched-transform posterior/prediction disagreement occurs?
3. How much of that disagreement exceeds identity/inference noise?
4. Does default inference ensembling reduce the tradeoff or merely average it?
5. Does TabPFN-3's OOD preprocessing shift it toward the invariant side?
6. Does Mitra remain more invariant, and if so does that ever cost performance when shape is informative?
7. Does the model track the Bayes-optimal crossover as \(\rho\) varies?

The most persuasive figure is a **phase diagram**: x-axis \(\rho\), y-axis performance/regret or robustness, curves for raw-only, rank-only, current TFMs, proposed method, and oracle.

---

## 9. Real-data benchmark protocol

The real-data section must be frozen only after the synthetic method is chosen.

### 9.1 Primary: TabArena-v0.1

Use the full current TabArena-v0.1 benchmark, subject only to predeclared compatibility rules.

At the current project date, TabArena contains 51 curated IID datasets with 9–30 evaluated splits and many current methods. Use TabArena's official dataset/task definitions and evaluation machinery.

Compatibility rules:

- include all tasks with at least one genuine numerical feature for the reparameterization robustness analysis;
- do not infer numerical status from integer dtype alone; use benchmark metadata;
- leave categorical features untouched unless a separate categorical experiment is explicitly preregistered;
- for clean predictive performance, evaluate all tasks the method natively supports, even if some do not enter the numerical-transform robustness subset;
- do not drop datasets because the proposed method loses.

Run two distinct evaluations:

**A. Clean benchmark:** ordinary TabArena scoring under official splits/protocols.

**B. Matched-reparameterization benchmark:** for every compatible task/split, transform context/train and validation/test/query consistently using the frozen transform bank.

Do not tune transform-specific hyperparameters on test splits.

### 9.2 Secondary: BeyondArena

Use BeyondArena as the external generalization check because it spans IID, temporal, and grouped tasks and broader scale/feature regimes.

Two-stage execution is allowed:

- development: a frozen stratified subset covering IID/temporal/grouped and size regimes;
- confirmatory: all compatible tasks that fit the declared resource cap, or at minimum all compatible temporal + grouped tasks plus a matched IID subset.

If full 142-dataset execution is feasible, run it. Otherwise document the exact resource-driven exclusion rule before outcomes.

The goal is not to claim the method solves distribution shift. The goal is to show that the factorization does not buy invariance by sacrificing practical generalization outside IID benchmarks.

### 9.3 Optional targeted benchmark: TabReD

Run all eight TabReD datasets if feasible, preserving their official time-based splits. This is especially useful if BeyondArena execution is partial or if reviewers may worry that the method was optimized only for academic IID tasks.

### 9.4 Optional breadth: TALENT

TALENT is secondary only. It is broad (300 basic datasets plus large tasks), but TabArena/BeyondArena should be the primary paper evidence because they provide more current curation and beyond-IID structure.

Use TALENT only if:

- the final method is lightweight enough;
- additional breadth is needed;
- no method choices are made from the final TALENT results.

---

## 10. Matched transformation bank for real data

Freeze a moderate and a stress bank.

### Primary moderate bank

These are the transformations that should carry the headline robustness claim:

- identity/refit noise;
- positive affine scale/shift (unit changes);
- mild centered signed power;
- mild asinh map;
- mild random monotone PWL;
- mild held-out monotone spline.

Use train/context-only fitting for data-dependent transforms.

Calibrate severity before final runs using transform diagnostics, not model outcomes. Record a distribution-level severity statistic such as normalized quantile displacement or Wasserstein distance after robust affine alignment.

### Stress bank

- stronger signed power;
- stronger PWL/spline;
- compositions;
- ordered spacing remaps for discrete numeric features;
- transform only a random subset of numerical columns;
- 0/4/16/32/64 irrelevant numerical columns where dataset size permits.

### Secondary orientation extension

- negative affine / order-reversing bijections.

Do not let orientation-reversing transforms define the primary paper claim unless the final architecture explicitly handles them and the semantics are justified.

---

## 11. Metrics

### Classification primary metrics

- log loss / cross entropy;
- TabArena official task metric for clean leaderboard comparison;
- posterior Jensen–Shannon divergence between clean and matched predictions;
- total variation distance;
- argmax flip rate;
- Brier score;
- calibration/ECE as secondary.

For binary tasks also report AUROC where appropriate, but do not use AUROC as the only metric because the paper is about predictive distributions and stability.

### Regression primary metrics

- official benchmark error / RMSE as appropriate;
- normalized prediction disagreement, e.g. mean absolute clean-vs-matched prediction difference divided by train/context target scale;
- normalized matched-clean loss gap;
- optional Gaussian/log-likelihood metrics only if all models provide comparable uncertainty.

### Synthetic metrics

- excess risk relative to full-information oracle where available;
- excess risk relative to quotient-only oracle;
- raw-vs-rank crossover as a function of \(\rho\);
- proposed-method regret relative to oracle gating or Bayes/meta-oracle;
- invariance disagreement under held-out nuisance transforms;
- optional gate calibration against oracle expert advantage.

### Efficiency

- model parameters;
- training tokens/steps;
- GPU hours;
- peak memory;
- inference wall-clock;
- number of TFM forward passes.

Do not collapse clean performance, robustness, and compute into one opaque score. Show Pareto plots.

---

## 12. Statistical protocol

### Real datasets

Use dataset as the main unit of generalization.

- aggregate transform seeds/settings within dataset/split first;
- use hierarchical or paired dataset-level bootstrap with at least 10,000 draws;
- report 95% confidence intervals;
- report win/tie/loss counts across datasets;
- report average rank or TabArena ELO only as secondary aggregate context;
- predefine the primary comparisons to avoid a multiple-comparison explosion.

Primary comparisons should be limited to:

1. proposed vs its raw-only host/backbone;
2. proposed vs rank-only;
3. proposed vs fixed raw+rank two-view ensemble;
4. proposed vs strongest simple transform-augmentation baseline;
5. clean performance vs the unmodified SOTA host.

Other baseline tables can be descriptive.

### Synthetic

Use independent task seeds as the unit.

- at least hundreds of independent tasks per main cell;
- enough tasks that confidence intervals are far smaller than the target effect;
- repeat training with at least 3 model seeds for development and 5 for final key models if feasible;
- hold generator seeds and model seeds separately.

### Descriptor/mechanism screens

If testing whether skewness/kurtosis/etc. predict instability:

- use grouped held-out-dataset evaluation;
- target-permutation controls;
- stability across splits;
- corrected or explicitly exploratory multiple comparisons;
- no causal language from feature importance alone.

---

## 13. Confirmatory success gates

Freeze numeric gates before the final benchmark. The exact thresholds may be adjusted once during pilot calibration, before confirmatory runs, and then locked.

Suggested gates:

### G1 — PriorDial is scientifically nontrivial

Pass if:

- rank/quotient-only is measurably better than raw-only under nuisance-heavy \(\rho\approx0\) with held-out transforms;
- raw/full-information is measurably better than quotient-only for informative \(\rho\ge0.75\) at small/moderate context;
- the crossover is replicated in both classification and regression or in two substantially different mechanism families;
- the effect persists on held-out warp families.

If there is no crossover, the central thesis is not empirically instantiated; fix the generator once using Section 16A, then rerun from scratch.

### G2 — Existing TFMs occupy meaningfully different tradeoff points

Pass if at least two current TFM families differ materially in robustness/shape-use behavior and the difference is larger than identity/inference noise.

This does not need to be universal. A TabICLv2-vs-Mitra or TabPFN-3-default-vs-OOD contrast is enough if reproducible.

### G3 — Proposed method has real headroom over trivial mixtures

Pass if the final learned method:

- beats raw-only in nuisance-heavy cells;
- beats rank-only in informative-heavy cells;
- beats or significantly improves over the globally tuned fixed raw+rank mixture on average across the \(\rho\) dial;
- captures a substantial fraction of the oracle-gate headroom;
- works on a held-out transform family and held-out mechanism hyperparameters.

A useful target is at least 20–30% reduction in regret to the oracle versus the best fixed non-oracle mixture, but use confidence intervals rather than treating this exact number as sacred.

### G4 — Real clean performance is preserved

On TabArena, the proposed method should not materially damage clean performance relative to its unmodified host/backbone.

Target:

- average normalized error degradation <= 0.5%, and/or
- paired 95% CI rules out >1% degradation,

with no catastrophic dataset subset.

If the method improves clean performance, excellent, but improvement is not required for the paper.

### G5 — Real matched-transform robustness improves

On compatible TabArena tasks, target:

- at least ~30% reduction in excess posterior disagreement / normalized prediction disagreement relative to the host for the moderate bank;
- improvement on a clear majority of datasets;
- no compensating material loss increase.

For a cross-family adapter, pass on at least two current TFM families. For a standalone model, beat its matched-compute raw-only backbone and at least one strong current neural/TFM baseline on the robustness-performance frontier.

### G6 — External validity

On BeyondArena and/or TabReD:

- no major clean-performance regression;
- robustness gains should not be confined to one IID benchmark;
- any gains under temporal/grouped shift are bonus evidence, not a required claim.

---

## 14. Experiments required for an ICLR-ready paper

### E0 — Reproduce and harden the Day-8 phenomenon

- rerun a small frozen panel with TabICLv2 default/single, Mitra, and now TabPFN-3;
- include TabPFN-3 OOD checkpoint;
- verify identity noise, transform inverse accuracy, and shared context/query transformation;
- confirm that the original phenomenon is not a software/preprocessing artifact.

Deliverable: `reports/E0_REPRODUCTION.md`.

### E1 — PriorDial oracle verification

Before training any new model:

- show that \(\rho\) actually controls mechanism information in marginal shape;
- fit descriptor-only predictors of mechanism family;
- compare full-information and rank-only simple learners/oracles;
- plot information/accuracy vs \(\rho\) and context size.

Kill the generator if the desired monotone trend does not appear.

Deliverable: `reports/E1_PRIORDIAL_VALIDATION.md`.

### E2 — Current TFM phase diagram

Run TabPFN-3 default/OOD, TabICLv2, Mitra, and selected non-TFM controls on PriorDial.

Main plot:

- x-axis \(\rho\);
- y-axis excess risk or normalized score;
- second panel matched-transform disagreement;
- curves for each TFM plus rank/raw simple baselines.

Deliverable: `reports/E2_TFM_PHASE_DIAGRAM.md`.

### E3 — Cheap method ladder M0–M5

Use a small backbone or frozen host to test:

- raw;
- rank;
- transform augmentation;
- 50/50 raw+rank;
- globally tuned fixed mixture;
- learned gate;
- oracle gate.

The critical diagnostic is **oracle headroom**. If oracle selection barely beats the fixed mixture, do not build a complicated gate.

Deliverable: `reports/E3_METHOD_KILL_TEST.md`.

### E4 — Dual-channel model M6

Only if E3 passes.

Train raw-only, rank-only, and dual-channel models with matched compute/capacity.

Ablate:

- handcrafted shape descriptors vs learned marginal encoder;
- featurewise gate vs global gate;
- embedding-level fusion vs prediction-level mixture;
- with/without raw residual;
- with/without invariant atom/missingness metadata;
- with/without synthetic consistency regularizer.

Deliverable: `reports/E4_DUAL_CHANNEL.md`.

### E5 — Held-out synthetic confirmation

Freeze the method in `METHOD_FREEZE.md`.

Evaluate on:

- new generator seeds;
- held-out transform family;
- held-out warp compositions;
- unseen context sizes;
- unseen feature counts;
- stronger and weaker \(\rho\);
- classification + regression;
- correlated features;
- missingness;
- atomic/discrete numeric features;
- irrelevant columns.

No method changes after this except bug fixes that require rerunning all affected cells.

Deliverable: `reports/E5_SYNTHETIC_CONFIRMATION.md`.

### E6 — TabArena clean benchmark

Run the frozen method under official TabArena protocol.

Compare against:

- unmodified host/backbone;
- rank-only;
- fixed raw+rank mixture;
- TabPFN-3;
- TabICLv2;
- Mitra;
- TabM;
- RealMLP;
- XGBoost/CatBoost/LightGBM;
- current cached TabArena leaderboard context.

Do not require rank #1. The target is competitive clean performance plus a new robustness property.

Deliverable: `reports/E6_TABARENA_CLEAN.md`.

### E7 — TabArena matched-reparameterization benchmark

Use the same frozen method and compatible TabArena tasks.

Run the moderate bank as primary and stress bank as secondary.

Report:

- clean vs matched loss;
- JS/TV/flips for classification;
- normalized disagreement for regression;
- identity noise;
- per-transform and per-dataset results;
- robustness-performance Pareto plot.

Deliverable: `reports/E7_TABARENA_REPARAM.md`.

### E8 — BeyondArena / TabReD

Run the frozen method on BeyondArena compatible slices; run TabReD if feasible.

Do not tune on these results.

Deliverable: `reports/E8_EXTERNAL_VALIDITY.md`.

### E9 — Mechanistic intervention

If the paper claims that current-model sensitivity is driven by marginal geometry, do at least one causal intervention rather than only descriptor correlation.

Possible interventions:

1. replace/ablate the numeric preprocessing path while holding model weights fixed;
2. feed rank-canonicalized values but restore explicit marginal descriptors to the proposed model;
3. for the proposed model, zero the shape channel and show nuisance robustness increases while informative-prior performance drops;
4. zero the stable channel and show the opposite behavior;
5. intervene on the gate to force raw or rank expert and compare with oracle expert advantage;
6. if internals are accessible, inspect representation distance/attention changes under matched transforms, but do not rely on probes alone.

Deliverable: `reports/E9_MECHANISM.md`.

### E10 — Efficiency and scaling

Measure:

- 1/2/4-view adapter costs;
- context size scaling;
- feature-count scaling;
- peak GPU memory;
- single-forward dual-channel cost;
- comparison with host default and simple two-view ensemble.

Deliverable: `reports/E10_EFFICIENCY.md`.

---

## 15. Baseline details reviewers will expect

### TabPFN-3

Use the public research checkpoint and the appropriate classifier/regressor checkpoint. Also test the OOD-preprocessing checkpoint because its preprocessing explicitly changes the robustness story. Record exact package commit, model filename, checkpoint hash, inference configuration, ensemble count, and preprocessing transforms.

### TabICLv2

Use official code/weights. Report default inference and a minimally ensembled/single-estimator configuration when available, because prior results show ensembling can suppress but not eliminate instability.

### Mitra

Use official implementation/checkpoint. It is important scientifically because its mixed-prior design and prior robustness behavior make it a contrasting TFM family.

### TabM and RealMLP

Use official or TabArena-native implementations and strong defaults. These are strong neural controls that are not in-context TFMs.

### GBDTs

At minimum:

- CatBoost;
- XGBoost;
- LightGBM.

Use TabArena-recommended/meta-tuned spaces or cached official benchmark configurations for clean comparisons. For matched-transform experiments, run the same frozen fitting protocol under original and transformed data.

### xRFM / other current strong methods

If installation is stable, include xRFM as a recent strong method, especially for regression. Also inspect the living TabArena leaderboard at protocol freeze and include any newly dominant open model family if it materially changes the SOTA context.

### AutoML

Use AutoGluon/TabArena cached leaderboard results as context for clean predictive competitiveness if appropriate. Do not spend excessive compute transforming a huge AutoML stack unless a reviewer-critical comparison requires it.

---

## 16. Pre-approved failure branches

These are the only scientifically legitimate ways to “make it work” after failures. Log every branch in `reports/DECISION_LOG.md` with the triggering evidence.

### 16A — PriorDial does not create a real information tradeoff

Symptoms:

- raw and rank are equally good for all \(\rho\);
- marginal descriptors cannot predict mechanism better as \(\rho\) increases;
- no crossover even at small context.

Allowed adjustments:

1. strengthen the mapping between mechanism hyperparameters and warp family while preserving equal marginal warp frequencies;
2. reduce context size so prior information is useful but keep a context-size sweep showing the effect weakens as data dominate the prior;
3. correlate warp family with a continuous mechanism hyperparameter rather than only a coarse class;
4. increase diversity of warp shapes;
5. simplify the latent mechanism temporarily to verify the generator mathematically, then reintroduce diversity;
6. add an oracle calculation or simulation proving that \(M\) contains information beyond \(S\).

Not allowed:

- selecting only the one mechanism family where the effect appears and calling it general;
- making train and test marginals trivially separable by a leaked ID;
- using query labels in the regime cue.

After one generator redesign, freeze it. If the tradeoff still does not exist, kill the main direction.

### 16B — Existing TFMs are already almost perfectly invariant at \(\rho=0\)

This is not fatal.

Pivot the empirical claim from “TFMs fail invariance” to:

- current TFMs implement different implicit tradeoffs;
- the central contribution is the controlled information-theoretic tradeoff and whether their priors track it;
- evaluate whether highly invariant models pay a price at high \(\rho\).

If all current TFMs are invariant and also lose nothing at high \(\rho\), inspect whether the benchmark actually makes marginal information useful. If yes, the models may infer the mechanism from context well enough; reduce context size and report the context-size phase transition rather than forcing a failure.

### 16C — Rank-only dominates raw-only everywhere

Likely causes:

- shape is not truly informative;
- context labels already identify the task;
- raw model is poorly optimized.

Allowed adjustments:

- strengthen \(\rho\) and verify oracle marginal utility;
- use smaller context sizes;
- give raw and rank models equal tuning/compute;
- use a task family where the latent mechanism remains ambiguous from sparse context.

If a full-information oracle also fails to beat rank-only, the generator is wrong. If the oracle wins but the raw learner cannot exploit it, that itself may be a representation/optimization finding, but it is a different paper.

### 16D — Raw-only dominates nuisance regime

Likely causes:

- transformations are too mild;
- the model learned internal canonicalization;
- train augmentation leaked the transform family.

Allowed adjustments:

- add held-out monotone spline/PWL families;
- widen severity within numerically realistic bounds;
- remove transform family from pretraining for the held-out test;
- compare with an explicitly invariant rank oracle to verify headroom.

Do not make the headline bank absurdly pathological. Moderate transformations must remain primary.

### 16E — Learned gate fails to beat fixed raw+rank mixture

First compute the oracle gate.

**If oracle headroom is small:** stop gating. Use the fixed mixture as a strong result/baseline and decide whether the paper is primarily theory/benchmark. Do not build a more complex gate.

**If oracle headroom is large:** improve the gate with only the following steps, in order:

1. increase gate training tasks;
2. use quantile-function/set-encoder inputs instead of fragile moments;
3. include context-label association summaries;
4. use featurewise gates plus global pooling;
5. add an auxiliary synthetic task-family/warp-predictive objective during pretraining only;
6. calibrate gate outputs with temperature/logit regularization;
7. try a mixture-of-experts loss that trains both experts and gate jointly.

If none beats the fixed mixture on held-out synthetic tasks, kill the gating method.

### 16F — Dual-channel model underperforms because it is harder to optimize

Allowed steps:

1. residual-initialize the shape branch so the model starts near the rank-only or host solution;
2. use separate layer normalization for S and M;
3. fuse late at logits rather than early embeddings;
4. pretrain experts separately, then train the gate/fusion;
5. use distillation from the better expert on each synthetic training episode, using generator knowledge only during training;
6. reduce branch width while keeping total capacity matched.

Always compare against the simple M5 prediction-level gate. If M5 wins, prefer it rather than forcing a more elegant architecture.

### 16G — Method improves robustness but hurts clean TabArena performance

Allowed steps:

1. add a raw residual/expert path;
2. initialize the gate toward the original host and learn only deviations;
3. reduce invariance regularization;
4. train on a broader mixture of informative priors;
5. use prediction-level gating rather than hard canonicalization;
6. tune one global regularization/gating hyperparameter on development datasets only.

Do not tune per TabArena test dataset.

If clean degradation remains >1% normalized error with no compensating major robustness win, the method is not ready.

### 16H — Method only wins on extreme transforms

Do not headline it.

Try:

- increasing representation-level regularization at moderate severity;
- training on a severity curriculum that includes mild transforms;
- using the TabPFN-3 OOD preprocessing as a baseline/teacher;
- focusing the method on unit/shape changes seen in the moderate bank.

If gains disappear on moderate transformations, reduce the claim to a stress-test paper or kill the method.

### 16I — TabPFN-3 OOD preprocessing already matches the proposed method

This is a major novelty warning.

The method can survive only if it still contributes something the OOD checkpoint does not:

- adapts between nuisance and informative marginal regimes;
- dominates both default and OOD preprocessing over the \(\rho\) dial;
- provides a theory-backed explanation of when each preprocessing is appropriate;
- generalizes the principle across TabICLv2/Mitra or a standalone backbone.

If it is merely another quantile/robust scaling recipe, stop.

### 16J — Descriptor correlations from Day 8 disappear

Do not rescue them.

Delete causal language about skewness/kurtosis/atom mass. Keep the synthetic controlled-prior mechanism if it works. Handcrafted descriptor failure is acceptable and may motivate a learned marginal encoder.

### 16K — BeyondArena is negative

Analyze whether the problem is:

- large sample size;
- high dimension;
- temporal/grouped split;
- categorical dominance;
- adapter compute;
- a particular host limitation.

If the method preserves TabArena robustness but hurts beyond-IID clean performance, narrow the claim to IID reparameterization reliability and report the limitation prominently. Do not hide BeyondArena.

### 16L — No method beats simple baselines, but theory + phenomenon are strong

Fallback paper:

> **Not All Marginals Are Nuisance: The Invariance–Information Tradeoff in Tabular Foundation Models**

Make it a theory + controlled benchmark + current-model audit paper. This can still be worthwhile if:

- the tradeoff is clean and previously uncharacterized;
- current TFMs differ systematically;
- simple rank/raw preprocessing choices move models along the predicted frontier;
- there is a strong negative result showing why universal canonicalization is impossible/suboptimal.

However, acceptance odds are lower than with a successful adaptive method. Strengthen the theorem, benchmark design, and mechanistic intervention.

### 16M — The entire direction fails

Stop after documenting the negative result. Do not consume the ICLR deadline trying endless variants.

Return to the strongest existing fallback project rather than manufacturing a marginal gain.

---

## 17. Reviewer-proof ablations

The final paper should contain or append the following.

### Representation ablations

- raw only;
- robust-affine only;
- rank only;
- rank + invariant metadata;
- rank + handcrafted shape;
- rank + learned shape encoder;
- raw + rank fixed average;
- raw + rank learned gate;
- full dual-channel.

### Prior ablations

- \(\rho=0\) only;
- \(\rho=1\) only;
- full \(\rho\) continuum;
- same architecture/optimizer, different prior mixtures;
- informative shape removed while holding mechanism distribution fixed;
- nuisance transform distribution changed at test.

### Transform ablations

- affine only;
- nonlinear smooth only;
- PWL/spline only;
- seen transforms;
- held-out transform family;
- mild vs stress severity.

### Data-regime ablations

- context size;
- number of features;
- correlated features;
- missingness;
- atomic/discrete numerical features;
- irrelevant numerical columns;
- classification vs regression.

### Model ablations

- single estimator vs default inference ensemble where applicable;
- TabPFN-3 default vs OOD preprocessing;
- TabICLv2 vs Mitra;
- neural vs tree controls;
- proposed gate forced to 0 / 0.5 / 1.

### Compute ablations

- 1 view vs 2 views vs 4 views;
- same-FLOP raw baseline;
- same-parameter widened baseline;
- training token curve.

---

## 18. What results would make the paper compelling

The ideal story is not “our model wins every benchmark.” The ideal story is a sequence of logically connected results:

1. **Formal:** quotienting coordinates removes exactly the predictive value \(I(Y^*;M\mid S)\) under log loss.
2. **Controlled:** PriorDial smoothly moves that quantity from approximately zero to substantial.
3. **Behavioral:** rank-only wins when shape is nuisance; raw/full-information wins when shape is informative.
4. **Current TFMs:** TabPFN-3, TabICLv2, and Mitra occupy different locations on this frontier and do not all track the Bayes-optimal crossover.
5. **Method:** the factorized/gated model tracks the crossover better without an oracle regime bit.
6. **Generalization:** the advantage persists on a held-out transform family and held-out mechanism settings.
7. **Real data:** clean TabArena performance is essentially preserved while matched-reparameterization disagreement falls substantially.
8. **External:** BeyondArena/TabReD show no major practical regression.
9. **Mechanism:** ablating the shape channel selectively hurts informative-prior tasks while improving nuisance robustness; ablating the stable channel has the opposite pattern.

The single most important figure should be a phase diagram over \(\rho\) showing raw, rank, fixed mixture, proposed method, and oracle.

The second most important figure should be a TabArena robustness-performance Pareto plot showing that the proposed method reduces disagreement without moving materially down the clean-performance axis.

---

## 19. What results are NOT enough

Do not call the project successful if only one of these occurs:

- TabICLv2 predictions change by ~1% under transforms;
- rank transform improves one model;
- the proposed method wins on 5–10 cherry-picked OpenML datasets;
- handcrafted skewness/kurtosis descriptors correlate with instability on fewer than ~10 datasets;
- a two-view ensemble beats a one-view baseline without equal-compute controls;
- the method works only on synthetic tasks with \(\rho=1\);
- the method loses clean TabArena performance materially;
- only historical TabPFN-v2.5 shows the phenomenon;
- the method cannot beat TabPFN-3's existing OOD preprocessing or a simple raw+rank average;
- results require per-dataset tuning using final test outcomes.

---

## 20. Dataset selection and anti-cherry-picking rules

### Synthetic

All task-family, context-size, feature-count, noise, and \(\rho\) grids must be specified in config files and versioned.

### TabArena

Use every compatible dataset. Compatibility must be determined only by feature metadata and model resource limits, never outcome.

### BeyondArena

If resource limits prevent the full compatible benchmark, choose a deterministic stratified rule before results, balancing:

- IID / temporal / grouped;
- small / medium / large row count;
- low / medium / high dimensionality;
- classification / regression;
- numeric-heavy / mixed feature types.

Save the selected task IDs and rule to `configs/beyondarena_confirmatory.yaml` before running the proposed method.

### TabReD

Use all eight official datasets and official time splits.

---

## 21. Engineering and reproducibility requirements

Create this structure:

```text
project/
  AGENT.md
  PROTOCOL.md
  METHOD_FREEZE.md
  environment/
    lockfile
    hardware.json
  configs/
    pilot.yaml
    prior_dial_dev.yaml
    prior_dial_confirmatory.yaml
    tabarena_clean.yaml
    tabarena_reparam.yaml
    beyondarena_confirmatory.yaml
  src/
    transforms/
    priors/
    representations/
    models/
      raw/
      rank/
      gate/
      dual_channel/
    metrics/
    stats/
  scripts/
    run_e0_reproduction.py
    run_e1_priordial.py
    run_e2_tfm_phase_diagram.py
    run_e3_method_kill_test.py
    run_e4_dual_channel.py
    run_e5_synth_confirmatory.py
    run_e6_tabarena_clean.py
    run_e7_tabarena_reparam.py
    run_e8_external.py
    run_e9_mechanism.py
    analyze_all.py
  results/
    MANIFEST.jsonl
    raw/
    processed/
  reports/
    NOVELTY_LEDGER.md
    BASELINE_FREEZE.md
    DECISION_LOG.md
    CLAIMS_EVIDENCE_MATRIX.md
    E0_REPRODUCTION.md
    ...
  figures/
  paper/
```

For every run save:

- git commit hash;
- config hash;
- package versions;
- CUDA/PyTorch version;
- GPU model;
- model/checkpoint hash;
- dataset/task/split ID;
- transform state/seed;
- model seed;
- predictions;
- metrics;
- wall-clock;
- peak memory;
- error trace if failed.

Raw result bundles are immutable. Analysis scripts may create new processed outputs but must never overwrite raw predictions.

Add unit tests for:

- transform monotonicity;
- inverse reconstruction;
- tie preservation;
- train-only fitting;
- no query-label leakage;
- shared original/transformed row alignment;
- deterministic seeds;
- rank invariance under increasing transforms;
- metric implementation.

---

## 22. Execution order and resource discipline

Run experiments in the following order and stop at gates.

### Stage A — cheap science

1. E0 reproduction.
2. E1 PriorDial validation.
3. T1/T2 proof draft and simulation checks.
4. E2 small current-TFM phase diagram.

Do not launch full real-data sweeps before PriorDial is validated.

### Stage B — method kill test

5. M0–M5 on synthetic development tasks.
6. Compute oracle gate headroom.
7. Choose one fusion design.

If the learned method cannot beat fixed mixtures with real oracle headroom, stop method development.

### Stage C — final synthetic method

8. M6/M7 only if justified.
9. Freeze method.
10. E5 held-out synthetic confirmation.

### Stage D — real confirmation

11. E6 TabArena clean.
12. E7 TabArena matched transforms.
13. E8 BeyondArena/TabReD.
14. E9 mechanism.
15. E10 efficiency.

If multiple GPUs are available, parallelize independent model/dataset jobs, but keep confirmatory configs immutable.

---

## 23. Paper-writing outputs to generate automatically

After each stage, update `reports/CLAIMS_EVIDENCE_MATRIX.md` with rows:

| Claim | Required evidence | Current evidence | Status | Figure/Table |
|---|---|---|---|---|

Claims should include:

1. quotienting can be Bayes-optimal;
2. quotienting can be Bayes-suboptimal;
3. PriorDial controls marginal utility;
4. current TFMs occupy different tradeoff points;
5. proposed method adapts better than fixed preprocessing;
6. gains survive held-out transforms;
7. clean TabArena performance is preserved;
8. matched-transform robustness improves;
9. external validity does not collapse;
10. mechanism ablations support the factorized interpretation.

Also maintain `paper/RESULTS_SENTENCES.md` containing only statements that are already supported by frozen analyses and confidence intervals. This prevents writing claims first and searching for supporting numbers later.

---

## 24. Final paper table/figure checklist

Target main-paper content:

### Figure 1 — Conceptual diagram

Raw numerical coordinate -> stable/order channel S + coordinate-dependent shape channel M -> adaptive fusion. Show nuisance vs informative prior cases.

### Figure 2 — PriorDial phase diagram

Performance/regret across \(\rho\) for raw, rank, fixed mixture, proposed, oracle.

### Figure 3 — Current TFM behavior

TabPFN-3 default/OOD, TabICLv2, Mitra across \(\rho\), with transform disagreement.

### Figure 4 — Real robustness-performance frontier

TabArena clean score vs matched-reparameterization disagreement.

### Table 1 — Synthetic held-out confirmation

Classification and regression, seen/unseen transforms, raw/rank/proposed/oracle.

### Table 2 — TabArena clean performance

Proposed, host, current TFMs, TabM/RealMLP, GBDTs; include compute.

### Table 3 — TabArena matched robustness

Loss gap, JS/TV/flips, regression disagreement, identity noise.

### Table 4 — BeyondArena/TabReD

IID/temporal/grouped or industrial time-split results.

### Figure/Table 5 — Mechanistic ablation

Force/ablate S and M channels and show the predicted opposite effects.

Appendix:

- all 51 TabArena per-dataset results;
- all compatible BeyondArena results;
- transform definitions;
- all hyperparameters;
- descriptor screens;
- additional current models;
- severity curves;
- context/feature-count sweeps;
- complete failure log.

---

## 25. Decision rule for the final ICLR submission

Recommend this project as the main ICLR submission only if all of the following are true:

1. T1/T2 are correct and clearly presented;
2. PriorDial creates a replicated nuisance-vs-informative crossover;
3. the final method beats raw-only, rank-only, and fixed raw+rank on held-out synthetic tasks by a meaningful margin or tracks the oracle frontier substantially better;
4. the result survives at least one held-out transformation family;
5. TabArena clean performance is essentially preserved;
6. TabArena matched-transform robustness improves materially;
7. the novelty ledger shows that the final contribution is not reducible to prior rank-warp, quantile preprocessing, mixed-prior, or transform-ensemble work;
8. external evaluation does not reveal a severe practical regression.

If 1–2 pass but 3 fails, consider a theory/benchmark paper only if the current-model audit is unusually strong.

If 4–6 fail, do not market the method as a general tabular solution.

If the novelty ledger fails, stop even if the numbers are good.

---

## 26. Final instruction

Optimize for a paper whose central sentence can truthfully be:

> **A tabular foundation model should not be universally invariant to numerical reparameterization: invariance is optimal only when the discarded coordinate-dependent marginal carries no task information. We formalize this tradeoff, build a controllable benchmark that exposes it, show that current TFMs occupy different points on the frontier, and introduce a factorized model that adapts between stable order information and useful marginal geometry while preserving real-data performance.**

Everything you run should either strengthen, falsify, or narrow that sentence.

If an experiment does neither, do not spend compute on it.