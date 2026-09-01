# Day-6 research log

All times Asia/Seoul on 2026-08-28 unless noted.

- ~23:21: audited Days 1–5, final closure, idle compute, and prior failures.
- ~23:28: froze H1 Semantic Arithmetic Amplification before outcomes.
- ~23:30: first H1 smoke shows fp32 FT orbit MSE growing from ~`3e-15` to
  ~`6e-3`; IEA64 path remains bitwise closed.
- ~23:36: completed H1 pilot, 18 bundles; all four frozen gates pass.
- ~23:42: completed H1 fixed six-seed confirmation; 72 total bundles, 576
  paths; 9/9 exact IEA64 closure cells and architecture-specific fp32 effect.
- ~23:43: froze H2 Precision-Delay Law.
- ~23:48: completed 18 H2 bundles, 216 paths; H2 fails exact precision order
  because float16 separates earlier than bfloat16 in all FT datasets.  H1
  controls survive.  No post-hoc H2b confirmation launched.
- ~23:46: froze and began H3 all-row, 200-epoch persistence matrix.  First MLP
  bundle establishes feasibility and 14.3% IEA64 timing overhead.
- 23:54: H3 moved to detached resumable `tmux` session `day6_h3`, two H100
  shards.  Frozen size: 36 bundles, 288 paths.
- ~23:58: literature audit identifies JMLR 2024 numerical-sensitivity work as
  a close collision, narrowing H1 novelty to semantic conjugacy/localization
  and exact interface closure.
- ~23:58: froze H4 Semantic Shadowing Forecast before outcomes.  H4 waits for
  H3 to avoid contaminating the H3 timing gate.
- 2026-08-29 00:00: first completed H3 evidence uses 27,126 training rows.
  Bank/MLP fp32 orbit MSE grows from `9.31e-17` at epoch 0 to `6.13e-12` at
  epoch 200 (still inside the frozen stable boundary); IEA64 remains exactly
  zero at all nine checkpoints.  One cell is not used to adjudicate broad H3.
- 2026-08-29 ~00:03: implementation audit verifies H3 permutation direction,
  parameter conjugacy, class alignment, and common random tape.  The timing
  arm order is fixed rather than randomized, so its gate is retained as an
  engineering check rather than a precise benchmark.
- 2026-08-29 ~00:05: froze H5 Cross-Perturbation Fragility Transfer before
  any H4/H5 artifact existed.  It prospectively reuses H4 tensors to test
  whether a two-epoch matched schema shadow forecasts epoch-20 independent-seed
  prediction fragility; no new fit or outcome-dependent threshold is added.
- 2026-08-29 ~00:07: second H3 bundle completes.  Bank/ResNet fp32 mean orbit
  MSE moves from `3.27e-13` at epoch 20 to `8.15e-4` at epoch 50 and `2.32e-2`
  at epoch 200, while IEA64 stays exactly closed.  This is a prospective
  delayed-instability counterexample to a universal 20-epoch architecture
  boundary; the 36-bundle H3 gates remain unadjudicated.
- 2026-08-29 ~00:09: pre-run H4 statistical correction, still with zero H4
  artifacts: pooled correlation/AUROC scores are now ranked within dataset
  before concatenation so dataset MSE scale cannot create a spurious pooled
  association.  The matrix, material threshold, per-dataset gate, and numeric
  pass thresholds are unchanged.
- 2026-08-29 00:12: froze H6 Semantic Lyapunov Screen after exactly three
  observed development bundles (Bank MLP/ResNet seed 8101 and Credit MLP seed
  8101).  Those stems are excluded.  The remaining 33 not-yet-created H3
  bundles prospectively test whether an epoch-5/10/20 log-orbit slope predicts
  material epoch-200 divergence better than the epoch-20 level alone.
- 2026-08-29 ~00:17: added a direct one-update conjugacy audit for all three
  architectures on CPU.  IEA64 gives bitwise-conjugate parameters and AdamW
  moments after the matched update; 8/8 total unit tests pass.  Long-run claims
  are relabeled precisely as checkpoint-prediction closure because every
  intermediate GPU state is not stored.
- 2026-08-29 ~00:18: projected H3 operational critical path at roughly 6.2
  hours from observed row-scaled path times; this is not a scientific timing
  result.  Installed a detached post-H3 chain that runs H6, then H4/H5, audits,
  and tests only after H3's complete analyzer releases the GPUs.
- 2026-08-29 ~00:20: froze the final idea-ranking rubric after the three
  explicitly declared H3/H6 development bundles but before all remaining H3
  and H4–H6 test outcomes.  It compares Day-6 directions against OrbitCover on novelty,
  theory, prospective evidence, utility, and feasibility, with explicit caps
  for three-dataset scope, close collisions, post-hoc evidence, and failed
  gates.
- 2026-08-29 ~00:22: first untouched H6 test bundle completes (Bank /
  FT-Transformer / 8101).  The frozen screen correctly predicts material
  epoch-200 fp32 orbit MSE (`4.91e-2`), but epoch-20 MSE is already material, so
  this case cannot establish improvement over the raw early level.  IEA64 is
  exactly prediction-closed; FT median timing ratio is 1.013 in this bundle.
  Canonical IEA64 test loss is 8.0% lower for this one seed and is retained only
  as path-selection evidence.
- 2026-08-29 ~00:26: formalized the canonicalization dominance baseline.
  A known schema permutation can be undone by a bitwise gather, yielding
  unconditional closure without float64; 9/9 tests pass.  IEA64 is therefore
  framed as a causal/local fallback, not universally preferred over carrying
  semantic metadata to preprocessing.
- 2026-08-29 00:28: Credit/FT/8101 falsifies universal long-horizon IEA64
  exactness: all views are exact through epoch 20, one becomes material by 50,
  and two others first become nonzero/material at checkpoint 200.  Integrity
  passes.  Froze successor H7 Rounding-Cell Survival on the 31 not-yet-created
  H3 bundles, interpreting precision as conditional hazard suppression rather
  than an exactness guarantee.
- 2026-08-29 ~00:31: first untouched H7 bundle (Credit/ResNet/8101) delays all
  three fp32 material crossings by ~100 checkpoint epochs and lowers all three
  final MSEs, but one IEA64 view is material at 200.  This second nonexact cell
  makes H3's ≥8/9 exact gate impossible; Bank and Credit ResNet also make its
  ≥5/6 stable-control gate impossible.  H3 is logically falsified but continues
  unchanged for the 33-bundle H6 and 31-bundle H7 prospective tests.  H6's
  fixed slope screen correctly flags the delayed Credit ResNet case despite
  raw epoch-20 orbit MSE near `4.8e-14`.
- 2026-08-29 ~00:35: added and ran the strict Day-6 completion audit.  It
  passes 11/19 requirements and remains incomplete by construction: the clock
  is before 06:21, H3/H4/H5/H6/H7 matrices are unfinished, final integrity
  counts are unavailable, and H3–H7 plus overall verdict reports do not yet
  exist.  Frozen protocol, 3×3 coverage, H1/H2 evidence, and theory/novelty
  artifacts pass.
- 2026-08-29 ~00:43: a refreshed primary-source novelty search found DynaTab
  (PMLR 2026), which explicitly learns dynamic feature order and evaluates 36
  datasets.  It is adjacent rather than a direct collision because it changes
  representations/objectives instead of comparing exactly conjugated,
  real-function-matched training paths.  The audit now explicitly forbids a
  broad “feature order newly matters” claim and keeps Day 6 on causal numerical
  semantics.  Both H3 GPU shards and the post-H3 watcher remain healthy at
  6/36 completed bundles.
- 2026-08-29 ~00:44: FreMTPL/MLP/8101 closes as H3 bundle 7/36 and is the
  first prospective H6 false positive.  Its frozen three-point score is
  `-4.684` (material prediction), while actual epoch-200 mean fp32 orbit MSE is
  only `1.92e-9` (`log10=-8.716`).  The curve grows strongly but sublinearly
  relative to the extrapolation, so this is a real curvature failure.  IEA64
  is exact through epoch 50, becomes nonzero at 100, and ends near `4.91e-12`,
  adding a third nonexact H3 cell without a material H7 failure.  Partial H7
  now has 2/31 bundles: its IEA64 material-failure fraction improves to 1/6,
  while all three currently eligible fp32-material paths retain later hits and
  strict final-MSE reductions.  Integrity remains clean.
- 2026-08-29 00:47:10: froze H8 Level-or-Acceleration Semantic Screen after
  exactly the seven completed H3 bundles.  All seven are named development
  exclusions; the remaining 29 bundles are prospective.  Proposition 10 shows
  that a positive exponential modal mixture has convex log energy, motivating
  a fixed level-or-acceleration rule that handles already-material FT paths and
  delayed ResNet mode takeover without H6's long extrapolation.  H8 must also
  improve fixed-decision accuracy over unchanged H6 by at least 0.10.  Its
  first analyzer run sees exactly 0/29 prospective artifacts, and 11/11 tests
  pass.
- 2026-08-29 ~00:55: FreMTPL/ResNet/8101 closes as H3 bundle 8/36 and the
  first untouched H8 test.  It is a true positive but not a delayed case:
  epoch-20 log orbit MSE is `-2.649`, the finite-difference slope increase is
  `0.940` log10-MSE per epoch, and final mean
  MSE is `0.159`; both H8 branches and H6 flag it.  H7 strengthens
  provisionally: FreMTPL's eligible-path median delay is 181 epochs, all six
  currently eligible paths across Credit/FreMTPL are later and lower under
  IEA64, and only 1/9 prospective IEA64 paths is material.  These are partial
  bundle-level results, not final dataset replication.
- 2026-08-29 ~00:57: corrected H8's *comparator implementation* to compute
  H6 with H6's own frozen `1e-30` floor, checkpoint menu, horizon, and material
  threshold instead of H8's `1e-18` curvature floor.  H8's rule/gates are
  unchanged.  The authoritative H6 and H8 comparator scores now agree exactly
  on the common prospective bundle (maximum absolute difference `0.0`), and
  11/11 tests still pass.
- 2026-08-29 ~01:01: Bank/MLP/8202 closes as H3 bundle 9/36 and H8's first
  prospective negative.  H8 correctly rejects it (`A=0.00455`, final mean MSE
  `2.57e-12`), so partial sensitivity, specificity, and balanced accuracy are
  all 1.0 over one positive and one negative.  H6 also gets this case right,
  raising its partial specificity to 0.5; H8 still has zero test-set accuracy
  improvement over H6 and no prospective delayed positive, so its defining
  gates remain untested.  H7's partial IEA64 material-failure fraction is now
  1/12, with eligible delay/reduction fractions still 1.0.
- 2026-08-29 ~01:02: corrected an operational ETA interpretation.  Each seed
  has nine dataset-model jobs, so modulo-two shard parity flips between seeds;
  the expensive FreMTPL/FT workload alternates GPUs across the four seeds.
  The full matrix is therefore balanced despite the seed-8101 snapshot, and
  the earlier roughly 6.2-hour H3 projection remains plausible.  No dynamic
  work stealing or schedule mutation is used.
- 2026-08-29 ~01:04: refreshed H8's primary-source novelty subtraction.  No
  exact prior surfaced for the fixed tabular schema-orbit slope-increase rule.
  Frankle et al.'s *Butterfly Effect* is nevertheless a close warning that
  separation rates vary and need not follow simple dynamics, and Amarel et al.
  (PMLR 2026) make symmetry-orbit gradient alignment an adjacent established
  diagnostic.  H8 is therefore capped as a narrow prospective numerical
  diagnostic, not a general training-dynamics or orbit-analysis novelty.
- 2026-08-29 ~01:05: completed a non-evidentiary H4/H5 CPU pipeline smoke in
  a temporary directory using one Bank/MLP configuration and all three frozen
  seeds.  Nine paths serialized, both analyzers ran end to end, and the H4
  hash/menu/shape/finiteness audit passed with zero errors; the temporary files
  were removed and no metric entered the scientific ledger.  The permanent H4
  result directory remains empty.  Full tests remain 11/11.
- 2026-08-29 01:07:37: removed a pre-outcome H4 device confound.  The original
  modulo-two enumeration perfectly assigned batch 128 to GPU 0 and batch 256
  to GPU 1.  With zero permanent H4 artifacts, replaced it by a balanced
  parity assignment over batch, weight-decay, and learning-rate indices.  In
  every seed/dataset/model cell each GPU now receives 6/12 configs: 3 of each
  batch, 3 of each weight decay, and 2 of each learning rate.  The 324 frozen
  jobs, models, seeds, hyperparameters, and gates are unchanged; partition
  coverage is disjoint/exhaustive and 12/12 tests pass.
- 2026-08-29 ~01:10: audited the strongest prior-day incumbent from Day 5's
  authoritative `results.md` before any Day-6 ranking.  The resulting
  `ORBITCOVER_INCUMBENT_AUDIT.md` records its 12-source/144-neural-cell breadth,
  55.9% coupled-estimator variance reduction, weak independent-repeat result,
  small downstream test-regret improvement, target-shift caveat, and closest
  classical controls.  This is an evidence-extraction audit only: no numeric
  Day-6 ranking or post-outcome rubric change was made.
- 2026-08-29 ~01:13: corrected H5's top-quartile tie handling before any
  permanent H4/H5 artifact existed.  The frozen protocol requires average
  ranks, but the analyzer had used first-occurrence ranks.  It now uses average
  ranks and has a direct tie-case regression test.  No estimand, threshold,
  model, dataset, seed, or optimizer configuration changed.
- 2026-08-29 ~01:16: completed a pre-outcome reporting-path audit.  Added
  fixed visualizations for H8's level/acceleration rule and H4/H5's two
  forecast relationships, and made the detached chain regenerate figures
  after H4/H5 analysis.  Plotting reads frozen outputs and changes no gate.
- 2026-08-29 ~01:19: found and removed a pre-outcome undefined-control failure
  in H4/H5.  Epoch-zero shadows are necessarily constant across optimizer
  configurations, so their Spearman association is mathematically undefined;
  the original code would have made both improvement gates fail via `NaN`.
  With zero permanent H4 artifacts, froze the standard predictive convention
  that a constant score has zero association.  Nonconstant comparisons retain
  ordinary average-tie Spearman, and all thresholds/matrices remain unchanged.
- 2026-08-29 ~01:21: directly verified the negative-control premise with two
  temporary Bank/MLP/seed-9101 CPU bundles at opposite optimizer settings.
  Their complete canonical-plus-two-view epoch-zero prediction tensors were
  bitwise identical (`array_equal=True`, maximum gap zero).  The temporary
  bundles were excluded from evidence and removed; permanent H4 remains empty.
- 2026-08-29 ~01:23: froze `STATISTICAL_SCOPE_AUDIT.md` before the remaining
  outcomes.  It declares datasets—not seeds, views, or configuration repeats—
  as the scientific replication scope, treats pooled screen metrics as panel
  diagnostics, preserves successor development exclusions, and forbids the
  final report from converting repeated paths into generalization confidence.
- 2026-08-29 ~01:27: Bank/FT-Transformer/8202 closes as H3 bundle 10/36 with
  zero integrity errors.  FP32 is material by epoch 5 and ends at mean
  validation orbit MSE `4.483e-2`; all three IEA64 views stay bitwise closed
  through epoch 200.  This is an untouched H6/H8 true positive and adds three
  strict H7 delay/final-reduction wins (paired censored delay 196).  H6's
  extrapolated log10 score is an absurd `21.001`, so it is retained only as a
  correct binary classification, not a calibrated magnitude forecast.  Partial
  states are H6 7/33 (sensitivity 1, specificity .5), H7 5/31 / 15 paths (all
  current gates except completeness pass), and H8 3/29 (2 TP, 1 TN; no delayed
  case or improvement over H6 yet).
- 2026-08-29 ~01:32: brought H7 reporting into line with its frozen scope
  statement by adding a dataset-level descriptive table for eligible counts,
  later-hit fraction, median delay, exact-early fraction, final-win fraction,
  and IEA64 material-failure fraction.  No gate or pooled calculation changed.
- 2026-08-29 ~01:35: Credit/ResNet/8202 closes as clean H3 bundle 11/36 and
  supplies H8's first untouched delayed positive.  Mean FP32 orbit MSE is only
  about `8.8e-14` at epoch 20, its late-minus-early log-slope is `0.05875`, and
  it reaches `2.861e-2` at epoch 200; H8's acceleration branch correctly fires.
  H6 also predicts it, so H8's defining improvement remains zero.  Conversely,
  all three IEA64 paths are already nonexact before epoch 20 and hit material
  MSE at epoch 100, tied with FP32, though their final MSEs are lower.  H7's
  partial later-hit fraction falls to `.75`, exact-early survival to `.833`,
  and IEA64 material-failure fraction rises to `.222`; completion is still
  required before adjudication.
- 2026-08-29 ~01:37: added a fixed H7 survival figure faceted by dataset so
  Credit's current 0.5 later-hit fraction cannot be obscured by the pooled
  Bank/FreMTPL paths.  This is descriptive reporting only; gates are unchanged.
- 2026-08-29 01:38: froze H9 Post-Breach Arithmetic Attenuation after all 11
  currently observed H3 bundles and before the remaining 25.  The new rule
  distinguishes final paired damage from H7's failed partial delay behavior:
  it requires 80% final wins, dataset median ratio <=0.5 in 2/3 datasets, 50%
  material rescue, <=10% twofold worsening, canonical loss change within 1%,
  and complete integrity.  Proposition 11 gives only a restrictive linear
  covariance-order motivation.  H7 is unchanged, all development stems are
  excluded verbatim, and the ranking rubric/weights remain frozen.
- 2026-08-29 ~01:46: before H9's first artifact, made its split limitation
  explicit: completion order is runtime-driven rather than randomized, so the
  11 development and 25 prospective bundles are composition-unbalanced.  This
  changes no gate and requires a new randomized panel after any positive H9.
- 2026-08-29 ~01:49: FreMTPL/MLP/8202 closes as clean H3 bundle 12/36 and
  H9's first untouched bundle.  FP32 final mean orbit MSE is `1.383e-9`, so all
  three pairs are correctly excluded from H9's post-breach estimand; IEA64 is
  exactly zero.  H6 score `-5.413` and H8's level/acceleration rule both
  correctly predict nonmaterial, leaving H8 improvement over H6 at zero.
  Partial states: H6 9/33 with specificity `.667`, H7 7/31 / 21 paths, H8
  5/29 with 3 TP / 2 TN, and H9 1/25 with zero eligible pairs.
- 2026-08-29 ~01:50: recorded H9's development calibration transparently.
  Across the 11 excluded bundles, 21 pairs are eligible, with 100% IEA64 final
  wins, 66.7% material rescue, zero twofold worsening, and median-ratio success
  in 2/3 datasets; the prospective thresholds are explicitly relaxed from
  these observations rather than presented as theorem-derived constants.
- 2026-08-29 ~01:56: FreMTPL/FT-Transformer/8101 closes as H3 bundle 13/36.
  FP32 is material by epoch 1 and ends at mean orbit MSE `.279`; H6 and H8's
  level branch correctly predict it.  IEA64 delays every path, but one becomes
  material by epoch 100 and ends at `.193` while two remain exact.  H7's
  partial later-hit fraction returns exactly to `.80`, exact-early survival is
  still only `.875`, final-reduction wins remain 1.0, and IEA64 material
  failures are `.208`.  H8 is 4 TP / 2 TN but still ties H6 on all its tests.
- 2026-08-29 ~01:59: froze a final-report evidence checklist mapping H3–H9
  reports to their machine-readable summaries/CSVs and mandatory scope/failure
  statements.  It explicitly prevents partial gates, repeated paths, selected
  successors, or checkpoint equality from being overstated at finalization.
- 2026-08-29 ~02:07: Bank/ResNet/8202 closes as clean H3 bundle 14/36 and a
  delayed positive: epoch-20 FP32 MSE is about `1.29e-12`, acceleration `.124`,
  and final MSE `.0272`.  H6/H8 both predict it.  All three IEA64 paths remain
  exact and give H7 delays of 151.  Refreshing H9 now includes this bundle and
  the post-freeze FreMTPL/FT/8101 bundle: across 6 eligible pairs, IEA64 wins
  6/6, rescues 5/6, and is final-exact in 5/6.  Its partial canonical loss
  change is `-2.07%`, outside the frozen ±1% neutrality interval despite being
  favorable, so that gate fails unchanged.  H6 rank rho drops to `.647`; H8
  is 5 TP / 2 TN with two delayed positives but still no accuracy gain over H6.
- 2026-08-29 ~02:10: Credit/MLP/8202 closes as H3 bundle 15/36.  It is a
  correctly predicted stable control (final FP32 mean MSE `1.762e-12`), with
  IEA64 at or near the numerical floor.  H6 specificity returns to `.75` but
  rank rho `.667` and zero AUROC improvement still fail.  H7 exact-early
  survival reaches `.90`, making all current scientific gates provisionally
  pass.  H8 is 5 TP / 3 TN but still ties H6.  The noneligible H9 bundle pulls
  its partial equal-dataset canonical loss change from `-2.07%` to `-1.38%`,
  still outside the frozen interval.
- 2026-08-29 ~02:12: froze an outcome-contingent external confirmation
  roadmap.  H7/H9 success routes to a randomized >=12-dataset, multi-hardware,
  canonical-gather/compensated-sum comparison; H4/H5 success routes to new
  datasets/seeds and equal-cost forecast baselines; failure of both routes
  returns the lead recommendation to OrbitCover rather than creating another
  post-hoc H3 rule.  This is future design, not evidence or ranking.
- 2026-08-29 ~02:14: relabeled the literature audit's original two-bundle H3
  claim as a historical pre-H3 freeze.  Final reporting is required to use the
  complete successor summaries, preventing that stale snapshot from being
  mistaken for current evidence.
- 2026-08-29 ~02:28: Credit/FT-Transformer/8202 closes as clean H3 bundle
  16/36.  Unlike the retained seed-8101 IEA64 failure, all three untouched
  IEA64 paths stay exact through epoch 200 while FP32 ends near `.0631` and is
  material by epochs 2–5.  H7 reaches later-hit `.857`, exact-early `.909`,
  final wins 1.0, and IEA64 failures `.152`; all scientific gates provisionally
  pass.  H9 reaches 9/9 eligible wins, 8/9 rescues/exact finals, passes all 3
  dataset median-ratio checks, and returns inside loss neutrality at `-.834%`.
  H6 now passes every partial gate except its defining AUROC gain (still zero
  because raw epoch-20 AUROC is 1.0).  H8 is 6 TP / 3 TN but still ties H6.
- 2026-08-29 ~02:30: reran H6–H9 analysis at the fixed 16-bundle snapshot.
  SHA-256 hashes of all four summaries and their primary prospective/dataset
  CSVs were byte-identical before and after.  This validates deterministic
  analysis plumbing at the partial snapshot; it is repeated after completion.
- 2026-08-29 ~02:34: automated the final analysis-reproducibility proof.  The
  post-H4 chain now reruns H3/H3-dynamics/H4/H5/H6/H7/H8/H9 and SHA-256 checks
  28 declared JSON/CSV outputs byte-for-byte.  The strict completion audit and
  a regression test require this pass; no scientific estimand changed.
- 2026-08-29 ~02:35: extended the strict completion audit to require nonempty
  PNG/PDF outputs for all eight declared final figures.  Existence is followed
  by manual visual inspection after H4/H5; no metric or gate changed.
- 2026-08-29 ~02:38: completed a targeted H9 collision search.  ICML 2015 and
  AISTATS 2025 already cover rounding-sensitive and stochastic-rounding
  low-precision training, while ICML 2025 formalizes floating-point networks as
  discrete function classes.  No searched source directly tests an
  interface-only precision intervention after an exactly schema-conjugate orbit
  has breached.  The literature audit records that bounded result as no direct
  collision found, not proof of novelty, and denies H9 credit for the generic
  benefits of higher precision or rounding control.
- 2026-08-29 ~02:50: an exact-construction collision search found two important
  conceptual predecessors: JSS 2011 already treats attribute/class
  permutations as supervised metamorphic relations, and NeurIPS 2024 uses
  topological conjugacy to identify equivalent neural training dynamics.  The
  audit now subtracts both.  Day 6's remaining distinction is a fixed algebraic
  parameter conjugacy whose finite-precision pathwise failure is measured and
  prospectively reused; this reinforces rather than relaxes the novelty cap.
- 2026-08-29 ~02:55: FreMTPL/ResNet/8202 completed as H3 bundle 17/36 with
  zero integrity errors.  FP32 is material by epoch 20 and ends near `.1955`;
  IEA64 stays exact for two views and ends at `1.21e-12` for the third.  This
  removes FreMTPL/ResNet from H3's exact-cell count (now 2/9) while strongly
  supporting the narrower intervention.  On untouched data H7 moves to
  later-hit `.875`, exact-early `.917`, final wins 1.0, and IEA64 material
  failures `.139`.  H9 is 12/12 eligible final wins with 11/12 rescues and
  zero twofold worsenings.  H6's within-dataset pooled rank rho falls to `.697`
  and now fails alongside its unchanged zero incremental AUROC; H8 remains
  perfect on 10 tests but still offers zero improvement over H6.
- 2026-08-29 ~03:01: Bank/MLP/8303 and FreMTPL/FT-Transformer/8202 bring H3
  to 19/36 with 152 paths and zero integrity errors.  The former stays
  nonmaterial.  In the latter, FP32 final view MSEs are `.324`–`.362`; IEA64
  first becomes material at epoch 100 and ends at `.216` for all three views.
  Thus H9 gets three strict final wins with moderate ratios `.597`–`.668`, but
  no rescues: its pooled rescue rate moves from 11/12 to 11/15 while all frozen
  scientific gates still pass.  H7's same three paths have 99-epoch delays;
  pooled later-hit is `.889`, exact-early `.929`, final wins 1.0, and IEA64
  material failures `.190`.  H6 rank rho falls further to `.661`; H8 remains
  12/12 correct but exactly ties H6.  Only 1/9 H3 IEA64 cells is still exact.
- 2026-08-29 ~03:16: Bank/ResNet/8303 and Credit/MLP/8303 bring H3 to 21/36
  with 168 paths and zero integrity errors.  Bank ResNet FP32 is material at
  epoch 50; IEA64 delays one view to epoch 200 with final ratio `.629` and
  keeps two views exact.  H9 therefore reaches 18/18 strict wins and 13/18
  rescues, while equal-dataset canonical loss moves to `+.072%`.  H7 reaches
  later-hit `.900`, exact-early `.938`, final wins 1.0, and IEA64 failures
  `.188`.  Credit MLP remains nonmaterial.  H6 rank rho returns above gate at
  `.733` but still has zero incremental AUROC; H8 is 14/14 correct and still
  tied.  Every H3 dataset/model cell now contains at least one nonzero IEA64
  checkpoint path, making exact-cell closure 0/9 on the partial matrix.
- 2026-08-29 ~03:29: Bank/FT-Transformer/8303 closes H3 bundle 22/36.  FP32
  ends at `.0436`–`.0455`; IEA64 hits at epoch 100 and ends at `.0265`–`.0275`,
  giving 99-epoch delays and strict H9 ratios `.583`–`.631` but no rescues.
  H7's later-hit and exact-early rates improve to `.909` and `.941`, but its
  IEA64 material-failure rate rises to `.235`, close to the frozen `.25`
  maximum.  H9 is 21/21 strict wins and 13/21 rescues, while its equal-dataset
  canonical loss moves to `+.726%`, close to the ±1% gate.  These narrowing
  margins are retained rather than redefined; integrity remains error-free.
- 2026-08-29 ~03:35: Credit/ResNet/8303 and Credit/FT-Transformer/8303 bring
  H3 to 24/36 and 192 paths with no integrity error.  ResNet IEA64 is material
  in all three views at epoch 200, with final ratios `.606`–`.780`; FT IEA64
  rescues two views exactly and ends at ratio `.695` in the third.  H7 improves
  on four gates (later-hit `.923`, exact-early `.947`, final wins 1.0, all
  dataset delays pass) but its all-path IEA64 material-failure rate becomes
  `.281`, provisionally failing the frozen `.25` maximum.  H9 remains 27/27
  strict wins, with 15/27 rescues, no twofold worsening, all three dataset
  medians passing, and canonical loss `+.613%`.  H6/H8 still add no value over
  the perfect raw-level/fixed-decision comparators.
- 2026-08-29 ~03:50: FreMTPL/MLP/8303 closes H3 bundle 25/36.  Its FP32 final
  mean orbit MSE is `3.20e-9`, below material; H6 extrapolates `-5.130` and H8
  activates neither branch, so both correctly call it stable.  This contrasts
  with H6's retained seed-8101 false positive and shows seed-specific early
  slopes rather than a deterministic model label.  The noneligible bundle
  improves H7's denominator (exact-early `.950`, IEA64 failure `.267`) without
  changing H9's 27 eligible pairs; H9 canonical loss is now `+.715%`.
- 2026-08-29 ~03:52: corrected a purely algebraic normalization typo in the
  still-unscored frozen ranking rubric.  A weighted mean of 0–5 scores must be
  multiplied by 20, not 5, to produce the declared 0–100 total.  Weights,
  dimension scores, caps, tie-breaks, candidate menu, and relative ordering are
  unchanged; no candidate had yet received a score.
- 2026-08-29 ~04:03: FreMTPL/ResNet/8303 closes as bundle 26.  Its three FP32
  paths end at `.147`–`.161`, whereas IEA64 keeps two exact and one at
  `2.41e-12`; H9 adds three rescues and reaches 18/30.  H7's failure rate
  improves from `.267` to `.254`, still just outside its gate.
- 2026-08-29 ~04:08: Bank/MLP/8404 closes nonmaterial as bundle 27, correctly
  called stable by both H6 and H8.  Its three exact IEA64 paths expand H7's
  denominator enough to move failure rate to `.242`, barely back inside the
  `.25` gate; later-hit is `.929`, exact-early `.955`, and final wins 1.0.
  H9 remains 30/30 strict wins and 18/30 rescues; canonical loss improves to
  `+.571%`.  All results remain provisional and integrity remains error-free.
- ~04:36: Bank/FT/8404 completed, taking H3 to 28/36 bundles and 224 paths.
  The new seed is material in FP32 but exactly closed under IEA64 at epoch 200.
  H7 therefore moves to 23/31 prospective bundles with later-hit `.933`,
  exact-early `.957`, final-win `1.0`, and IEA64 material-failure `.232`.
  H9 moves to 17/25 bundles: 33/33 eligible final wins, 21/33 rescues, all
  three dataset median-ratio gates, and equal-dataset canonical loss `+.640%`.
  H6 and H8 remain perfect classifiers on observed tests but still provide
  exactly zero improvement over their simpler comparators.  Integrity passes
  on all 28 H3 bundles with zero errors.
- ~04:43: Credit/ResNet/8404 completed, taking H3 to 29/36 and 232 paths.  It
  is a delayed FP32-positive but all three IEA64 views remain exactly closed
  at epoch 200.  H7 improves to 24/31 bundles: later-hit `.938`, exact-early
  `.958`, final-win `1.0`, and material-failure `.222`.  H9 improves to 36/36
  strict wins and 24/36 rescues, with the equal-dataset loss shift returning
  to `+.535%`.  H6/H8 remain exact on their observed binary targets without
  incremental value; the 29-bundle integrity audit has zero errors.
- ~04:57: FreMTPL/MLP/8404 completed, taking H3 to 30/36 and 240 paths.  It is
  nonmaterial at epoch 200 under both arms but produces a second H6/H8 false
  positive; both screens still tie, so their defining incremental gates remain
  at zero.  H7 improves to 25/31 bundles with exact-early `.960` and IEA64
  material-failure `.213`; H9 gains no eligible pair and remains 36/36 wins,
  24/36 rescues, with equal-dataset canonical loss `+.589%`.  Integrity has
  zero errors across all 30 H3 bundles.
- ~05:02: FreMTPL/FT/8303 completed, taking H3 to 31/36 and 248 paths.  All
  three eligible FP32-material views are exact under IEA64 at epoch 200.  H7
  moves to later-hit `.941`, exact-early `.962`, final-win `1.0`, and material-
  failure `.205`; H9 reaches 39/39 strict wins and 27/39 rescues.  Its loss
  shift rises to `+.705%`, still inside the frozen ±1% gate.  H6 rank rho
  falls to `.747` but passes, while its zero AUROC improvement remains fatal;
  H8 continues to tie its comparator.  Integrity is error-free at 31 bundles.
- ~05:13: Bank/ResNet/8404 completed, taking H3 to 32/36 and 256 paths.  It is
  another delayed FP32-positive whose three IEA64 views remain exact at epoch
  200.  H7 improves to later-hit `.944`, exact-early `.963`, final-win `1.0`,
  and material-failure `.198`.  H9 is now 42/42 strict wins and 30/42 rescues;
  equal-dataset loss rises to `+.746%` but remains inside the gate.  H6/H8
  still share the same two/one false positives on their different frozen test
  splits and both have zero improvement.  Integrity remains error-free.
- ~05:16: Credit/MLP/8404 completed, taking H3 to 33/36 and 264 paths.  It is
  nonmaterial and exact under IEA64.  H7 improves mechanically to 28/31
  bundles, exact-early `.964`, and material-failure `.190`; H9 has no new
  eligible pair and its equal-dataset loss eases to `+.725%`.  H6 specificity
  returns to `.800`, H8 balanced accuracy is `.944`, and both still fail their
  zero-improvement gates.  Integrity remains error-free.
- ~05:35: Credit/FT/8404 completed, taking H3 to 34/36 and 272 paths.  Two of
  its three IEA64 paths remain material at epoch 200, but all three have lower
  final MSE than FP32.  H7 is now 29/31 bundles with later-hit `.947`, exact-
  early `.966`, final-win `1.0`, and material-failure `.207`; the final six
  FreMTPL paths can still decide its ≤.25 gate.  H9 reaches 45/45 strict
  wins and 31/45 rescues, while canonical loss rises to `+.840%`, close to the
  frozen +1% limit.  H6/H8 remain nonincremental and integrity is error-free.
- ~06:02: FreMTPL/ResNet/8404 completed, taking H3 to 35/36 and 280 paths.  It
  is FP32-material and all three IEA64 paths are exact at epoch 200.  H7 moves
  to 30/31 bundles, later-hit `.950`, exact-early `.967`, final-win `1.0`, and
  material-failure `.200`; even the worst possible final bundle cannot now
  break any scientific threshold, so only completeness/integrity remains.
  H9 reaches 48/48 wins and 34/48 rescues; all non-loss gates are secure, while
  equal-dataset canonical loss `+.870%` remains outcome-sensitive.  H6/H8 are
  still nonincremental and integrity has zero errors.
- 06:08:54: H3 completed all 36 bundles / 288 paths with 12.402 summed
  fit-hours.  Final H3 is falsified: exact IEA64 cells 0/9 and stable
  MLP/ResNet cells 3/6 fail; material FT datasets 3/3, timing models 3/3, and
  equal-dataset canonical loss `+.282%` pass.  H6 is falsified despite AUROC
  1.0 because improvement over the equally perfect raw epoch-20 level is zero.
  H7 is supported on 31 prospective bundles / 93 paths: later-hit `.952`, all
  three dataset median delays at least 101 epochs, exact-early `.968`, final
  wins `1.0`, and material failures `.204`.  H8 is falsified because its
  balanced accuracy `.944` ties H6 exactly.  H9 is supported on 25 prospective
  bundles / 75 paths: 51/51 eligible wins, 36/51 rescues, zero twofold
  worsenings, all three ratio gates, 34 exact final eligible pairs, and
  canonical loss `+.760%`.  H4 begins only after these summaries are written.
- ~06:28: H4 completed all 324 bundles / 972 paths.  H4 is falsified: FT
  epoch-2/final rho is `.650`, `.413`, and `.119` (0/3 reach .70); pooled
  improvement over the constant epoch-zero control is `.394` and material
  AUROC is `.916`, but the stable-control fraction `.861` also misses .90.
  H5 is falsified on the same tensors: FT rho `.364`, `.441`, `.322`, pooled
  rho `.375`, and top-quartile AUROC `.704`; only the epoch-zero improvement
  and completeness/matching gates pass.  The exact reanalysis audit reruns all
  eight H3–H9 analyzers and byte-matches all 28 JSON/CSV outputs with no
  mismatches.
- 2026-08-29 ~04:11: an intervention-specific collision search confirmed that
  wider training accumulators (ICLR 2019) and conditioning-guided selective
  higher-precision accumulation (2025) are established.  The literature audit
  now explicitly denies IEA64 novelty credit for either ingredient.  Its
  residual distinction is selecting the schema-facing interface from a known
  semantic group action and measuring commutation of exactly conjugate paths;
  no gate or ranking cap changed.

The requested seven-hour research horizon begins no later than the initial
23:21 audit.  Do not issue the final idea ranking before 2026-08-29 06:21+09:00
and before all active frozen evidence is analyzed.
- ~06:31: after the seven-hour horizon and complete evidence, the frozen rubric
  ranks OrbitCover 65/100 (lead), Semantic Arithmetic 58/100 (alternative),
  Semantic Shadowing 40/100 (discarded), and Quantization Coalescence 29/100
  (mechanism clue only).  The final integrity audit covers 450 bundles / 2,052
  paths with zero errors; 28/28 analysis outputs reproduce byte-for-byte and
  16/16 tests pass.  All eight PNG/PDF figure pairs were regenerated and
  visually inspected for titles, axes, legends, threshold lines, panel labels,
  clipping, and blank output.
- 06:32:15: strict completion audit passes 25/25 checks, including the seven-
  hour wall-clock horizon, all H1–H9 summaries, 3-dataset × 3-model configs,
  all reports, 16 figure files, integrity, byte-reproduction, and foundational
  theory/literature/ranking artifacts.  Day 6 is complete.
