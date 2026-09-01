# Decision log

## 2026-08-31 — initial freeze

- The Day-09 directory initially contained only the user-provided program.
- Work is isolated to this directory because the surrounding repository has unrelated
  modified and untracked user artifacts.
- Conda base is the declared runtime; `/usr/bin/python` is Python 2.7 and is invalid.
- Root storage has roughly 12 GiB free, so the approximately 100-GB-per-method TabArena
  raw cache is prohibited there. This is a resource rule, not an outcome-based exclusion.
- Mitra is not installed. E0 must record an unavailable result unless an official,
  reproducible interface is installed without displacing existing user data.
- No failure branch has been triggered yet.

## 2026-08-31 — pilot calibration, pre-confirmation

- Pilot v2 established the label-free half of E1: marginal descriptors predicted the
  six mechanism families at chance near rho=0 and 90% at rho=1 while the empirical warp
  marginal stayed fixed. Ordinary raw/rank linear learners did not establish predictive
  utility, so E1 was not passed.
- Following failure branch 16A(2/5/6), a cross-fitted diagnostic was added: compare a
  mechanism mixture selected from invariant context-label summaries against the same
  mixture selected from invariant summaries plus label-free marginal shape. This uses no
  query labels for selection.
- The initial classification logit scale of 1 left mechanism experts dominated by label
  noise. Before any confirmatory run it was calibrated once to 2.5, preserving balanced
  intercepts and the declared noise grid. Pilot v5 then showed positive high-rho
  marginal query utility in both task types and approximately zero utility at rho=0.
- This is the single permitted generator calibration. `prior_dial_v1_1` is now frozen;
  further failures kill or narrow the direction rather than changing the generator.

## 2026-09-01 — E0 panel freeze

- E0 deterministically selects the first regression task (`wine_quality`) and first
  binary task (`churn`) from the Day-8 frozen panel; this is not outcome-based selection.
- Context/query caps are 512/256. Identity, positive affine, random PWL, and monotone
  spline are run at three seeds under the original paired-fit four-way protocol.
- TabPFN-3 default and OOD names remain mandatory in the config. The official local
  client/checkpoints and credentials are absent, so the hardened runner should emit
  immutable `unavailable` records rather than substituting v2.5.
- TabICLv2 single/default and the isolated official Mitra integration are runnable.

## 2026-09-01 — E2 small-phase freeze

- E2 uses exactly one task from each of six mechanism families per rho and task type.
  It is an exploratory current-model phase diagram, not a confirmatory estimate.
- Every episode is evaluated clean, after a held nuisance PWL reparameterization, and
  under an independent identity refit. Disagreement is reported after subtracting that
  model/episode identity floor.
- TabPFN-3 remains in the frozen model list but cannot run locally. Available E2 hosts
  are TabICLv2 single/default and Mitra.

## 2026-09-01 — E2 gate decision

- All 252 available model episodes completed without failure. Derived classification
  probabilities are renormalized to correct maximum floating-point row-sum drift of
  `5.96e-8`; immutable raw bundles are preserved.
- TabICLv2 single/default exceeded Mitra's identity-floor-subtracted disagreement in all
  42 paired episodes in both task types. Paired mean differences exclude zero and the
  average ratios range from 9.43x to 14.27x.
- G2 passes only in its scoped TabICLv2-vs-Mitra form. No TabPFN-3 or universal-TFM claim
  is authorized, and the six tasks per rho do not authorize a rho-dependent performance
  ranking.
- Stage B M0–M5 is now authorized. M6 remains prohibited until E3 shows meaningful oracle
  headroom and a learned non-oracle improvement over the development-tuned fixed mixture.

## 2026-09-01 — E3 kill-test freeze

- A one-pass screen selected an 11-neighbor distance-weighted KNN as the cheap frozen
  backbone. Raw means context-fitted z-score coordinates; M1 uses median/IQR; M2 uses the
  train-only tie-aware ECDF. No screen result is confirmatory evidence.
- M3 averages two raw-backbone views (identity and random monotone PWL). M4 and M5 reuse
  the exact same cached raw and rank predictions, so fixed, learned, and oracle mixtures
  have identical expert fit compute.
- Independent train/development/test generator seeds contain 240/120/420 tasks per rho
  and task type. The fixed mixing coefficient and learned-gate shrinkage are selected on
  development tasks only; the final comparison is on untouched test tasks.
- Gate inputs are context-only invariant association summaries plus label-free marginal
  descriptors. Rho, mechanism, warp identity, coupling bit, and query labels are
  prohibited at gate inference. Query labels define oracle targets only on training
  episodes and the diagnostic oracle only on development/test episodes.

## 2026-09-01 — E3 failure branch 16E, step 1

- The first untouched E3 test showed clear oracle headroom but M5 captured only 0.67% of
  it in classification and 2.63% in regression. Its fixed-mixture improvements were
  positive but only `3.1e-5` log loss and `2.22e-4` MSE, far below the G3 target.
- The program therefore permits increasing gate-training tasks as the first 16E remedy.
  A second config changes only split sizes and all generator seeds: 840 train, 240
  development, and 420 test tasks per rho/task. Backbone, descriptors, gate class, and
  tuning grids remain fixed.
- The second test is fresh because the original test was inspected. If more data does not
  materially increase held-out oracle-headroom capture, the next allowed diagnostic is a
  featurewise gate; M6 remains prohibited throughout this failure branch.

## 2026-09-01 — E3 failure branch 16E, step 4

- Increasing training from 1,680 to 5,880 episodes per task did not improve scientific
  headroom capture: the fresh classification estimate was 0.20% with a CI spanning zero,
  and regression was 2.67%. Data volume is not the primary bottleneck.
- Steps 2 and 3 were already present before this outcome: the aggregate descriptor uses a
  robust quantile grid and context-label rank associations. The next run therefore uses
  step 4, retaining aligned per-feature shape/association blocks before permutation-
  invariant global mean/std/min/max, association-weighted, and alignment pooling.
- Training volume, backbone, estimator, and tuning remain unchanged; all three split seeds
  are fresh. If this featurewise diagnostic does not approach substantial oracle-headroom
  capture, no M6 work is authorized without first exhausting later 16E steps.

## 2026-09-01 — E3 failure branch 16E, step 5

- Featurewise pooling captured only 0.82% of classification and 4.09% of regression
  oracle headroom on fresh test tasks. This remains substantially below G3.
- The next frozen run adds only the permitted auxiliary synthetic mechanism/warp-
  predictive objective. The gate receives 12 predicted class probabilities, not true
  labels. Training episodes use five-fold out-of-fold probabilities; development/test
  probabilities come from auxiliary models fitted only on training episodes.
- Rho, coupling bits, and query labels remain unavailable at inference. All split seeds
  are fresh, while backbone, training volume, featurewise descriptor, main gate, and
  tuning grids remain fixed.

## 2026-09-01 — E3 failure branch 16E, step 6

- Auxiliary probabilities are informative but insufficient: fresh-test oracle-headroom
  capture was 0.64% for classification and 6.18% for regression. Classification again
  had an interval spanning zero improvement over fixed.
- The step-6 run adds only development-selected logit temperature, bias, and shrinkage
  calibration to the predicted raw-expert weight. A predeclared 4x5x5 grid is selected
  independently by task type; the fresh test is not consulted.
- All prior step-1/2/3/4/5 choices are retained and all split seeds are fresh. M6 remains
  prohibited unless this produces a substantial, held-out fraction of oracle headroom.

## 2026-09-01 — final E3 kill decision

- Development calibration selected identity temperature/bias for classification and a
  sharper temperature of 0.5 for regression. On fresh test tasks, classification M5 did
  not distinguish itself from fixed; regression captured 16.96% of oracle headroom.
- G3 fails: classification capture is 0.27%, regression remains below the 20–30% guide,
  and rank dominates ordinary raw/robust experts rather than showing the required
  nuisance/informative crossover.
- A jointly trained neural expert/gate would be a materially more expensive hypothesis,
  not validation of the cheap M5 opportunity. Following Stage B's stop rule, M6, method
  freeze, E5, and E6–E10 are not launched. The negative result, immutable runs, and
  scoped G1/G2 evidence are retained without outcome-based deletion.

## 2026-09-01 — theory/benchmark fallback freeze, before outcomes

- The user requested continued Day-09 experimentation after the G3 method kill. This
  does not reopen M6 or any method-dependent E4–E10 stage.
- Following failure branch 16L, the continuation is limited to strengthening the
  theory/benchmark result: an exact population information calibration for PriorDial
  and an independent replication of explicit marginal-shape utility.
- `fallback_dial_replication.yaml` freezes generator seed 92001, 630 tasks per rho/task,
  context size 96, 12 features, 256 queries, and the unchanged `prior_dial_v1_1`
  generator. These settings differ from the development result and were recorded before
  outcomes. This run is not called E5 or method confirmation.
- Literature rechecking was started before interpreting the replication. Existing work
  already covers monotone perturbations, quantile/rank preprocessing, marginally
  realistic synthetic priors, and generic TFM ensembling; none of those is available as
  a standalone novelty claim here.

## 2026-09-01 — fallback replication result

- The 8,820-episode replication completed without failure. Mechanism information
  replicated in both tasks, and regression marginal-shape utility replicated with a
  rho=1 MSE reduction of 0.24339 [0.21664, 0.27044].
- Classification contradicted development: rho=1 shape routing increased log loss by
  0.00402 [0.00152, 0.00653], despite 99.2% mechanism selection accuracy.
- The diagnostic matched-family route was also harmful for classification and beneficial
  for regression. This triggers a claim narrowing, not a new method branch: task-family
  identification and predictive routing are not interchangeable objectives under expert
  misspecification/ensemble calibration.
- No generator, expert, or method was changed after this result. M6 and E4–E10 remain
  prohibited. Any future loss-aligned routing study requires a new protocol.

## 2026-09-01 — routing-axis diagnostic freeze, before outcomes

- The independent replication changed context size and feature count together. To avoid
  attributing the classification reversal to either axis without evidence, two post-hoc
  diagnostic cells are frozen: `(n=96,d=8,seed=93001)` and
  `(n=64,d=12,seed=94001)`, each with 420 tasks per rho/task and 256 queries.
- These cells use the unchanged generator and analysis. They isolate one axis at a time,
  are explicitly exploratory, and cannot promote the killed method or count as E5.

## 2026-09-01 — routing-axis diagnostic result

- At eight features, raising context size from 64 to 96 preserved classification routing
  utility; the contrast was -0.00104 [-0.00546, 0.00335].
- At context size 64, raising dimension from 8 to 12 reduced classification utility by
  0.00532 [-0.00937, -0.00129]. The `(96,12)` corner remained negative. Higher dimension
  is therefore the supported weakening axis, while the corner interaction remains
  uncertain under independent cell seeds.
- Regression routing stayed strongly beneficial in all four cells. No outcome triggers
  a method branch; the diagnostic only narrows the benchmark mechanism.

## 2026-09-01 — mechanism decomposition freeze, before outcomes

- The dimensional weakening will be decomposed over all six frozen mechanism families
  at rho=1, with no family selection or deletion. Primary diagnostics are shape-routing
  gain, matched-family gain, and the d=12 minus d=8 contrast at n=64, separately by task.
- This is a post-hoc explanation audit on immutable predictions. It cannot alter the
  generator, experts, gates, or method decision.

## 2026-09-01 — mechanism decomposition result

- Classification routing at `(n=64,d=8)` is positive for interaction, periodic, and
  linear mechanisms but negative for additive, threshold, and partition mechanisms.
- The d=8→12 weakening is concentrated in interaction (-0.01985, CI
  [-0.03057, -0.00926]) and periodic (-0.01652, [-0.02552, -0.00769]); linear improves
  and the other family contrasts include zero.
- The result supports expert-specialization misalignment rather than a universal
  dimension effect. No family is removed and no aggregate is recomputed on a favorable
  subset.

## 2026-09-01 — loss-aligned routing protocol freeze, before implementation outcomes

- A new fallback protocol tests context-only three-fold predictive competence as the
  routing target, rather than generator mechanism identity. It reuses all six frozen
  experts and cannot reopen M6/E4–E10.
- Development/test generator seed families are 95001/105001, with 120/240 tasks per
  rho/task/regime over five rhos and all four `(n,d)` cells. Temperature, shrinkage, and
  one fixed global mixture are selected on development episodes only.
- The primary comparator is the development-tuned fixed six-expert mixture. The test is
  immutable and has a single stop decision; no feature, expert, or grid refinement is
  permitted after outcomes.
- Generic cross-validated stacking or competence weighting is not claimed novel. The
  test targets the controlled distinction between identifiable generator metadata and
  usable predictive information.

## 2026-09-01 — loss-aligned routing untouched-test decision

- The frozen development/test runs completed 4,800/9,600 episodes and
  115,200/230,400 expert fits. The test was not inspected until all cells were written;
  analysis used the prewritten equal-cell, 10,000-draw paired bootstrap.
- Competence routing beats the development-tuned fixed mixture by 0.005465
  [0.004797, 0.006129] classification log loss and 0.254185
  [0.246516, 0.262313] regression MSE. It captures 29.51% and 99.66% of the declared
  best-individual headroom, so both tasks pass the frozen opportunity gate.
- The difficult 12-feature, rho>=0.75 classification slice also passes at 0.004811
  [0.003364, 0.006253]. Only two of 40 regime/rho/task cells have negative point
  estimates, both low-rho classification cells at `(n=64,d=12)`.
- Keep this as a successful fallback target-alignment result. It does not reopen the
  killed M6/E4–E10 path, and generic cross-validated competence routing remains explicitly
  outside the novelty claim.

## 2026-09-01 — soft-versus-hard competence diagnostic freeze, before outcomes

- The next diagnostic reuses immutable loss-router predictions without refitting any
  expert. It asks whether context CV works by selecting one expert or by supporting a
  calibrated mixture.
- Hard selection is the deterministic context-CV argmin; soft weights, fixed weights,
  metrics, cells, and 10,000-draw paired bootstrap are inherited unchanged. Recomputed
  parent losses must agree within `1e-6`.
- This is a post-result mechanism diagnostic, not independent confirmation. Any method
  informed by it requires fresh seeds before a performance claim.

## 2026-09-01 — soft-versus-hard competence diagnostic result

- Parent fixed/soft losses recomputed to `2.22e-16`, validating exact data and metric
  parity. Soft CV weighting beats hard CV selection by 0.023797
  [0.022615, 0.024993] classification log loss and 0.024655
  [0.022748, 0.026604] regression MSE.
- Hard selection is 0.018333 [0.016796, 0.019878] worse than fixed in classification,
  despite using the same predictive loss signal. In regression it is strongly better
  than fixed but still worse than soft weighting.
- The soft advantage concentrates at small CV margins and when the hard choice is wrong.
  The parent performance result is therefore an aggregation/calibration effect, not
  evidence that context CV reliably identifies a single best expert.

## 2026-09-01 — numeric real-panel competence transfer freeze, before outcomes

- Seven cached Day-8 datasets are fixed: adult/churn/higgs-small classification and
  california/diamond/house/black-friday regression. Otto is excluded because the frozen
  expert API is binary; numeric-only input matches the continuous expert scope.
- Each dataset receives 120 fresh stratified/random `(96 context, 256 query)` episodes.
  Synthetic-development fixed weights, temperature, and shrinkage transfer unchanged;
  no real-data tuning or dataset-identity routing is allowed.
- The primary interval uses a hierarchical paired bootstrap over datasets and episodes.
  Strong transfer requires both task intervals positive; scoped transfer requires one
  positive task and no material harm in the other. Dataset count limits breadth claims.

## 2026-09-01 — numeric real-panel competence transfer result

- The 840-episode run completed 20,160 fits in 108.10 seconds with checksums for every
  cached source. Competence-minus-fixed point gains are +0.003461 classification and
  +0.236997 regression, but hierarchical CIs include zero in both tasks.
- Four datasets favor competence, two are inconclusive, and Black Friday is harmed by
  0.01499 [0.00847, 0.02218] standardized MSE. Neither the strong nor scoped external
  gate passes.
- Soft competence remains better than hard CV selection in both task panels, supporting
  the aggregation diagnostic but not fixed-mixture transfer. The supported next check is
  breadth expansion to all previously unseen compatible identities from the already
  frozen Day-8 OpenML panel, with no method change or favorable dataset selection.

## 2026-09-01 — unseen-identity OpenML breadth freeze, before outcomes

- All 13 compatible identities in the pre-existing Day-8 OpenML panel that were not
  inspected in the seven-dataset check are included: six binary and seven regression
  tasks. Churn/diamonds are excluded solely because their identities were already seen.
- Official task repeat-0/fold-0 splits, numeric-only train preprocessing, first-32 source
  ordered numeric features, 40 fresh `(96,128)` episodes, and seed 135001 are frozen.
  The dimension cap follows a label-free structural audit and bounds quadratic compute.
- The synthetic router and hierarchical dataset/episode gate are unchanged. No failed
  or unfavorable dataset may be removed or replaced.

## 2026-09-01 — unseen-identity OpenML breadth result

- All 13 identities completed 520 episodes / 12,480 fits in 76.39 seconds. Regression
  competence beats fixed by 0.29103 [0.02525, 0.94647] across seven datasets; five
  dataset intervals are positive and a sixth point estimate is positive.
- Classification competence is worse than fixed by 0.002130
  [0.000256, 0.004652] across six datasets, although the loss stays within the frozen
  0.005 material-harm margin and soft weighting remains better than hard selection.
- Strong transfer fails; scoped regression transfer passes. The task asymmetry motivates
  a frozen leave-one-dataset-out calibration diagnostic on the complete real panel,
  without refitting experts or using held-out-dataset labels for parameter selection.

## 2026-09-01 — dataset-cross-fitted calibration freeze, before outcomes

- Combine all 9 classification and 11 regression real identities. For each held-out
  dataset, choose temperature and shrinkage-to-fixed on every other dataset only, using
  the frozen 8x5 grid and dataset-balanced loss.
- Immutable predictions and CV losses are reused. Parent synthetic/fixed losses must
  reproduce within `1e-6`; hierarchical dataset/episode inference and harm thresholds
  remain unchanged.
- This is a task-library calibration diagnostic, not independent confirmation. A new
  dataset-identity panel is required for any confirmatory claim about the calibrated rule.

## 2026-09-01 — dataset-cross-fit precision failure and v2 freeze, before outcomes

- V1 stopped before candidate selection: parent cells used float64 predictions while raw
  bundles store float32, giving maximum baseline error `2.864e-6` versus the frozen
  `1e-6` guard. No result artifact was written and v1 is invalid.
- V2 changes only that audit tolerance to `1e-5`. This accommodates the known storage
  quantization; the candidate grid, cross-fitting, bootstrap, and gates are untouched.

## 2026-09-01 — dataset-cross-fitted calibration result

- Leave-one-dataset-out shrinkage gives classification +0.000824
  [-0.000563, 0.002310] versus fixed: a favorable point estimate but no positive
  dataset-level interval. Regression remains +0.24113 [0.01642, 0.65643] versus fixed.
- The cross-fitted regression choice is 0.03025 [0.00141, 0.12041] worse than the
  original synthetic-tuned router. Calibration does not provide a clean improvement and
  is not promoted; the original regression rule remains the confirmation target.

## 2026-09-01 — independent regression confirmation freeze, before outcomes

- A deterministic task-ID rule selects the first five unseen cached 361xxx regression
  identities after excluding prior datasets: Abalone, Auction Verification,
  Geographical Origin of Music, Solar Flare, and Naval Propulsion.
- The original synthetic-tuned router is evaluated without tuning on 60 fresh episodes
  per dataset, official splits, numeric first-32 features, and seed 145001. The primary
  10,000-draw hierarchical interval must be positive to confirm transfer.

## 2026-09-01 — regression confirmation structural failure and v2 freeze

- V1 aborted without artifacts after two buffered summaries: one official test fold has
  only 106 rows, below the 128-query request. The partial run is invalid.
- A label-free split audit gives test sizes 418/205/106/107/1194. V2 uses 96 queries for
  all five datasets; identities, seeds, repeats, methods, feature cap, and gate are
  unchanged.

## 2026-09-01 — independent regression confirmation result

- The valid v2 run completed 300 episodes / 7,200 fits. Competence beats fixed by
  0.100451 standardized MSE with dataset-hierarchical CI [0.001731, 0.250552], passing
  the frozen confirmation gate.
- Four of five dataset intervals are positive. Auction Verification is a preserved
  negative at -0.006629 [-0.011584, -0.001815]. No dataset was removed.
- Together with the disjoint seven-dataset breadth result (+0.29103
  [0.02525, 0.94647]), this supports scoped real numeric regression transfer. Binary
  classification remains explicitly unsupported.

## 2026-09-01 — real regression context-scaling freeze, before outcomes

- The five confirmation identities receive 30 fresh nested repeats at context sizes
  32/64/96/192 and seed 155001, with a query shared across sizes. No method or dataset
  changes are made.
- Primary evidence is the dataset-balanced slope of fixed-minus-competence gain against
  log2 context size, with the same hierarchical dataset/episode bootstrap. This is a
  mechanism diagnostic, not another independent-identity confirmation.

## 2026-09-01 — real regression context-scaling result

- Gains are already positive at 32 rows (+0.08257 [0.00891, 0.21011]) and remain
  positive at 96/192; the 64-row interval narrowly includes zero.
- The gain slope per doubling is +0.00665 [-0.02456, 0.04618], so the positive-slope
  gate fails. The confirmed effect is a level effect over 32–192 rows, not a supported
  context-evidence scaling law.

## 2026-09-01 — expert-assignment permutation control freeze, before outcomes

- Preserve each episode's competence-weight spectrum but cyclically misassign its six CV
  losses to experts using all five nonidentity rotations. Compare their mean loss with
  aligned routing on synthetic and all nonduplicate real identities.
- Equal-cell synthetic and hierarchical real bootstraps are frozen. A positive interval
  isolates expert-specific assignment from generic dynamic weight concentration.

## 2026-09-01 — expert-assignment permutation control result

- Correct assignment beats the five cyclic misassignments on synthetic classification
  (+0.02574 [0.02485, 0.02665]) and regression (+0.55553 [0.54046, 0.57051]).
- The same control is positive across 9 real classification datasets (+0.00983
  [0.00278, 0.02056]) and 16 real regression datasets (+0.56642
  [0.09606, 1.25180]). All four intervals exclude zero.
- Performance requires expert-specific loss alignment, not merely an episode-varying
  concentrated weight vector. Together with the hard-selection failure, this isolates
  calibrated aligned aggregation as the operative mechanism.

## 2026-09-01 — real regression CV-budget freeze, before outcomes

- On all five confirmation identities and 40 fresh seed-165001 repeats, compare 2/3/5
  context-CV folds with shared contexts, queries, and full expert predictions.
- Two-fold uses 18 fits versus three-fold's 24, a 25% reduction. It passes only if its
  fixed-mixture gain interval is positive and its upper harm bound versus three-fold is
  at most 0.01 MSE. Temperature remains 0.1 for every arm.

## 2026-09-01 — real regression CV-budget result

- Two-, three-, and five-fold routers all beat fixed with positive dataset-level
  intervals. Their gains are 0.09089, 0.09838, and 0.10121 MSE respectively.
- Two-fold minus three-fold loss is +0.00748 [-0.00295, 0.02572]. The upper bound exceeds
  the frozen 0.01 noninferiority margin, so the 25%-fit-reduction gate fails.
- Five-fold's small point gain does not justify 50% more fits than three-fold. Retain
  three-fold as the supported default and do not promote the cheaper variant.

## 2026-09-01 — complete real-panel synthesis freeze

- Retrospectively pool every completed real numeric panel with no dataset exclusion:
  9 classification and 16 regression identities. This is a synthesis of known component
  results, not new confirmation evidence.
- Give each dataset equal weight. Freeze a 50,000-draw dataset bootstrap plus median,
  sign count, 10% trimmed mean, and leave-one-dataset-out sensitivity checks. Regression
  and classification are judged separately.

## 2026-09-01 — complete real-panel synthesis result

- Regression has dataset-balanced gain +0.217965 [0.041682, 0.459988], with 14/16
  positive datasets, median +0.023386, 10%-trimmed mean +0.134626, and all leave-one-out
  means positive (+0.124652 to +0.233495). The robustness gate passes.
- Classification is -0.000266 [-0.002976, 0.003167], with negative median and only 3/9
  positive datasets. Its gate fails.
- This consolidates a scoped numeric-regression transfer result while rejecting a
  task-general or classification-transfer claim. Because component outcomes were
  already known, the independent five-dataset confirmation remains the evidential anchor.

## 2026-09-01 — real-classification ranking/calibration diagnostic

- Freeze an immutable-prediction comparison of log loss, Brier, AUC, calibration bias,
  ECE, and sharpness over all nine binary identities. No expert or weight is refit.
- Aggregate AUC shift is +0.000285 [-0.003372, 0.004642]; Brier and global calibration
  summaries are mixed and inconclusive. Neither ranking collapse nor uniform
  miscalibration explains the classification result.
- A post-result tail addendum finds that, on the six unseen breadth datasets, the bottom
  90% NLL gain is +0.000469 [-0.000620, 0.001497], but the worst-decile gain is -0.025122
  [-0.045252, -0.008502] and the NLL>2 rate rises by 0.001497
  [0.000228, 0.002930]. Preserve classification as a rare-error-amplification failure.

## 2026-09-01 — regression tail-risk contrast freeze and result

- After observing the classification tail result, freeze the analogous squared-error
  decomposition across all 16 regression identities. This is retrospective and reuses
  immutable predictions; total MSE remains the endpoint.
- Competence reduces worst-decile squared error by 1.88738 [0.22577, 5.03685], improves
  bottom-90% squared error by 0.02878 [0.01127, 0.05024], and lowers the SE>4 rate by
  0.004831 [0.000910, 0.010452]. The effects cover 14/16, 13/16, and 12/16 datasets.
- The performance asymmetry is therefore tail-localized with opposite signs: regression
  catastrophes are suppressed while rare classification catastrophes are amplified.
  Treat this as a mechanism hypothesis, not a causal result or a tuned tail method.

## 2026-09-01 — weight-shift/tail diagnostic numerical invalidation

- Freeze within-dataset associations between competence-to-fixed weight KL and tail
  gain, plus high-versus-low KL quintiles, on six classification and 16 regression
  identities.
- V1 is invalid: exact-zero competence weights made a literal `0*log(0)` KL term NaN, so
  no valid regression correlation existed. V2 changes only the computation to the
  defining zero-mass KL convention; all analysis choices and seeds remain frozen.

## 2026-09-01 — weight-shift/tail diagnostic result

- Classification has mean within-dataset Spearman -0.3524
  [-0.5157, -0.1923] between competence-to-fixed KL and tail gain; every dataset is
  negative. Highest-minus-lowest KL quintile tail gain is -0.07868
  [-0.12097, -0.03483]. Stronger movement consistently amplifies high-NLL errors.
- Regression Spearman is +0.0124 [-0.0505, +0.0798], rejecting a general monotone
  relationship. The positive high-minus-low average is carried by a few large-headroom
  datasets while ten datasets have small negative contrasts.
- Do not infer a universal shrinkage or adaptation threshold. Dataset-specific reducible
  tail headroom, not movement magnitude alone, separates the regression wins.

## 2026-09-01 — synthetic-to-real shrinkage path result

- Freeze the 0.0–1.0 fixed-to-competence prediction path on immutable synthetic test and
  all initial real panels. This is retrospective and pointwise intervals are descriptive.
- Synthetic classification peaks at lambda=0.7; real classification peaks at 0.5 and
  declines to a negative point gain at 1.0. Small lambda=0.1/0.2 steps remain favorable.
- Synthetic and real regression curves both improve monotonically to lambda=1; all 20
  synthetic cells prefer full adaptation. Routing strength transfers for regression but
  not classification.

## 2026-09-01 — independent lambda=0.1 classification freeze, before outcomes

- Deterministically select the first five compatible, identity-unseen OpenML-CC18 tasks
  under the frozen ascending-ID rule: breast-w, credit-approval, credit-g, spambase, and
  electricity. Use official splits, numeric-only inputs, 80 fresh n96/q64 episodes, and
  seed 225001; no replacement is allowed.
- The sole candidate is 0.9 fixed + 0.1 competence, chosen on the earlier real panels.
  Confirmation requires a positive hierarchical log-loss interval and at least 3/5
  positive dataset point estimates.

## 2026-09-01 — independent lambda=0.1 classification result

- The 400-episode / 9,600-fit run completed in 61.95 seconds. Candidate-minus-fixed
  improvement is +0.000600 log loss [0.000038, 0.001471], with 3/5 datasets positive;
  the frozen confirmation gate passes.
- Brier improves +0.000207 [0.000001, 0.000527], while AUC is unchanged and the NLL>2
  rate does not improve. Full routing remains heterogeneous and inconclusive.
- Authorize only a modest numeric-binary performance claim for the real-development-tuned
  shrinkage rule. Do not relabel it synthetic-only transfer or algorithmic novelty.
- Post-result sensitivity for overlapping query samples remains positive when only the
  five dataset means are bootstrapped [0.000051, 0.001455]; all leave-one-dataset-out
  means are positive, minimum +0.000181.

## 2026-09-01 — context-rescaled robustness freeze, before outcomes

- A hostile audit notes that all original OpenML real panels fit affine feature and
  regression-target normalization on the complete official training fold before
  sampling 96-row contexts. This was protocol-compliant and used no query labels, but
  weakens a strict few-shot interpretation.
- Freeze fresh seeds on both five-dataset confirmation panels, refitting feature scaling
  and regression target scaling within each context. Outer-train label-free feature
  schema and imputation remain. Both task intervals must be positive for a joint pass.

## 2026-09-01 — context-rescaled robustness result

- Classification 10% shrinkage improves +0.000732 [0.000142, 0.001680], with all 5/5
  datasets positive; its robustness gate passes.
- Regression full competence retains +0.092896 and 4/5 positives, but its interval
  [-0.001456, 0.238104] narrowly crosses zero; its robustness gate and the joint gate fail.
- Preserve the original regression result within its frozen outer-fold-normalized scope.
  Treat context-rescaled regression as favorable but unconfirmed; do not omit Auction.
