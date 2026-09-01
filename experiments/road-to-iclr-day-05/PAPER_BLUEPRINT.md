# Paper blueprint after Day 5

Working title: **OrbitCover: Designed Nuisance Quotients for Stable Tabular
Evaluation and Model Selection**

Status: evidence-weighted blueprint. Confirmatory, prospective, post-gate, and
failed experiments are explicitly separated.

## One-sentence thesis

Exact representation equivalences and ordinary training randomness form a
joint nuisance product for a complete learning pipeline; aligned
prediction-space fANOVA measures its proper-loss cost, randomized
strength-matched covers estimate the nuisance quotient more efficiently than
equal-fit IID and structured controls, and independent-cover U-statistics turn
that gain into an unbiased quotient-risk criterion, while disjoint packing
reduces its finite-budget covariance and resolvable schedules close exactly.

## Continuity from Days 1--4

1. Day 1 found a large Adult gain from exposing numerical identities, but no
   broadly reliable multiview method.
2. Day 2 localized that effect and exposed genuine interactions on Black
   Friday; utility selectors correctly abstained elsewhere.
3. Day 3 showed causally that equivalent numeric, nominal, and ordinal bases
   can alter finite neural training; local equivariance fixes worked, but broad
   predictive remedies did not.
4. Day 4 concluded that complete schema-orbit auditing was more defensible than
   another universal encoder. Heterogeneous representation bags improved raw
   performance, but their semantic-specificity control was still missing.
5. Day 5 measures the full schema×seed field, discovers that pair and higher
   interactions make marginal actions inadequate, and converts that diagnosis
   into an equal-compute quotient estimator and model-selection score.

## Proposed contributions and novelty boundaries

### C1. Complete-pipeline nuisance quotient

Declare a finite product of exact schema symmetries and random seed, pass every
action through fitting and any selection path, align outputs semantically, and
define the quotient prediction as the product average. For Brier/MSE, the
member-to-quotient loss gap is exactly prediction-field Hilbert variance.

Not new: metamorphic permutations, multiplicity, group averaging, or
proper-score variance. Candidate contribution: the end-to-end stochastic
pipeline estimand, semantic alignment, and retained product tensor used for
both evaluation and selection.

### C2. Exact prediction-space fANOVA as an action diagnostic

Decompose vector predictions—not scalar leaderboard scores—over the complete
product. Persistent schema effects, conditional effects, and interactions are
then exact and reconstruct the quotient risk. The decomposition predicts when
a proposed cover helps and gives explicit adverse regions when higher-order
energy dominates.

Not new: functional ANOVA. Candidate contribution: its exact aligned
proper-risk role, finite-product covariance audit, and use to falsify actions.

### C3. Randomized strength-matched quotient covers

At 16 fits, randomize a mixed-level strength-2 orthogonal array over feature
order, category IDs, target IDs, and seed. Every row is marginally uniform, so
the estimator is unbiased for the quotient; pair balance annihilates all main
and pair fANOVA components. A nested 4/16/64 construction supplies
strength-1/2/3 checkpoints without discarding earlier fits.

Not new: orthogonal arrays, randomized-OA integration, reduced group
averaging, QMC, or ANOVA filtering. Candidate contribution: exact pipeline
equivalences plus seeds, black-box retraining, mixed-level finite coefficients,
semantic prediction alignment, modern equal-budget controls, and supervised
selection consequences.

### C4. Cover-stabilized model selection

Evaluate every candidate at the same nuisance-cover budget and select using
validation labels only. Cover residual bounds quotient-loss error, selection
regret, and pairwise ranking inversions. Candidate rankings—not only the final
winner—improve across all studied panels.

Not new: multi-seed evaluation, common random numbers, low-variance selection,
or generic stable selection. Candidate contribution: a declared
equivalence×seed quotient target, strength/computation hierarchy, independent
candidate randomization control, and exact separation of nuisance-selection
error from validation-to-test target shift.

### C5. Unbiased cross-quotient risk

Ordinary loss of a finite unbiased prediction average is upward-biased by its
cover residual. Two independent covers give the cross-score

`<Y-Q_hat_A, Y-Q_hat_B>`,

which is exactly unbiased for quotient Brier/MSE. A complete U-statistic over
independent cover blocks gives higher-budget checkpoints. Its variance is
`2<r,Cr> + tr(C²)`, tying selection-score efficiency to the same cover-error
covariance operator.

Not new: U-statistics or unbiased quadratic-risk estimation. Candidate
contribution: combining them with designed nuisance-cover blocks to estimate
and select an abstract pipeline quotient, with an IID member-U statistic as
the equal-compute comparator.

### C6. Antithetic cover packing and resolvable closure

Build a regular disjointness graph over strength-matched covers. Sampling a
uniform edge preserves quotient-unbiased endpoints while reducing the cover
covariance; crossing independently sampled packed blocks retains exact
quadratic-risk unbiasedness. When the mixed design is resolvable, sampling
cosets without replacement obeys an exact finite-population risk law and full
resolution enumerates the quotient.

Not new: antithetic variates, sliced/resolvable OAs, negative dependence, or
finite-population correction. Candidate contribution: the regular graph over
complete-pipeline nuisance covers, its interaction-spectrum operator, the
independent outer risk score, and the empirical 16/32/64/128 chain. The exact
128-cell control is mandatory: randomized pack-cross128 is not optimal once
  the full product can be enumerated at equal cost.
- Packing's non-smooth metric audit passes for probabilistic/ranking metrics:
  Brier and log RMSE improve on 23/23 non-exhaustive candidates and AUC on
  22/23 at pair32 and pack64. Accuracy improves 15/23 then 17/23, so the frozen
  universal-metric label remains rejected.
- On the four late untouched sources, Brier/log repeat in 11/12 candidates and
  4/4 source means. AUC improves in every source mean but only 8/12 candidates,
  missing the frozen 9/12 gate; accuracy is 7/12. This preserves a
  probabilistic-score claim but narrows broad ranking scope.
- In source block B, even the nonlinear extension narrows: Brier wins 10/12,
  clipped log 8/12, AUC 5/12 or 6/12, and accuracy 3/12, despite all 4/4 source
  means remaining favorable. Quadratic quotient risk—not universal metric
  improvement—must remain the headline scope.
  Across the eight untouched source identities, Brier and clipped log are
  favorable 8/8 with source-bootstrap intervals above zero for pair32 and
  pack64; AUC/accuracy are 7/8 plus one exact tie. This supports source-stable
  mean scope, not the failed universal candidate claim.
- Systems scope is explicitly statistical: none of 177 manifests records
  end-to-end timing. Generating 1,024 pack actions takes about 36 ms locally,
  but the paper cannot claim latency, throughput, energy, or dollar speedup
  from that audit alone. A later separate refit supplies local telemetry for
  12 cells/1,536 fits: all artifacts reproduce exactly, with 11.1 s median
  end-to-end per 128-fit cell under four-process concurrency (2.82/14.0/23.1 s
  model-family medians for linear/forest/MLP). This is reproducibility evidence,
  not a portable systems benchmark.

## Empirical spine

### Exact quotient-risk evidence

- Frozen confirmation: 25/25 validation-screened held-out cells beat IID-16,
  four independent strength-1 blocks, and four seed blocks; all 6/6 source
  groups beat every control.
- Mean residual reductions were 68.9% versus IID, 59.1% versus strength-1, and
  59.5% versus seed blocks. Source-cluster intervals remain strictly positive;
  the equal-source IID reduction is 58.4% `[46.9%,78.2%]`.
- Changed nuisance menu: 10/10 material cells and 5/5 represented sources beat
  all controls. Changed data subsample: 13/14 cells and 6/6 source means.
- Prospective external binary OpenML: 12/12 material cells beat IID and
  strength-1, 11/12 beat seed blocks, and 7/8 source groups beat all.
- Post-gate task-balanced OpenML: 16/16 material cells and 8/8 sources beat all
  three controls. The non-enumerating subset remains 10/10 cells across five
  sources, with 56.5% pooled reduction versus IID.
- Multiclass scope: 4/4 material cells beat IID and strength-1; 3/4 beat seed
  blocks. This is only two sources and therefore supporting evidence.
- Descriptive roll-up: 57/57 validation-screened cells beat IID and
  strength-1, 55/57 beat seed blocks, and 23/24 panel×source groups beat all.
  These counts are dependent and are not an IID meta-analysis.
- Against the stronger finite-population SRSWOR baseline, strength-2 reduces
  pooled residual by 51.0% on confirmation and 55.8% externally. It also beats
  LHS in 23/25 and Sobol in 19/25 confirmation cells, with stable pooled
  reductions of 37.4% and 28.8%; the originally stricter all-cell QMC gate
  remains failed.

### Strength, compute, and scope

- One literal nested schedule beats equal-budget IID at all 4/16/64
  checkpoints. Confirmation pooled reductions rise from 23.9% to 68.9% to
  94.6%; task-balanced reductions rise from 22.2% to 84.4% to 95.6%.
- Exact strength-3 on confirmation is tied or lower in all 25 material cells,
  strictly lower in 21, and wins all 6/6 source groups.
- The empirical phase diagram places all 57 material cells in the favorable
  regions. It also retains exact counterexamples: pure triple energy makes
  strength-2 16/9 times IID risk, and pure four-way energy makes strength-3
  64/27 times IID risk.
- A 16-fit confirmation cover has a finite median IID-equivalent budget of
  42.4 fits; 10 additional cells have numerical-zero residual. This is
  prediction-risk equivalence, not wall-clock speedup.
- Field-wise high-dimensional strength-2 failed against a sophisticated
  marginal control. A prospective mixed strength-3 OA-128 then beat all
  controls in 6/9 cells and 2/3 datasets, cutting risk 49.3% versus four
  strength-2 blocks and 39.4% versus IID. Its advantage over the marginal
  control is weak: 9.7% point reduction, bootstrap probability only 59.8%, and
  mean Brier is worse. The higher-order-vs-lower-strength claim survives; a
  universal high-dimensional advantage does not.

### Model selection

- Original confirmation: strength-2 has 95.6% quotient-winner agreement versus
  91.1% for IID and lower realized held-out loss on 10/11 datasets. Changed
  menu and changed subsample repeats pass their frozen gates.
- Ranking fidelity improves on all five panels: both mean Spearman correlation
  and pairwise-order accuracy exceed IID. Mean pairwise inversion rates fall
  versus IID on every binary panel; the analytic bound covers every observed
  pair but is often vacuous for small margins.
- Independent nuisance coordinates per candidate preserve the conclusion. On
  task-balanced OpenML they lower realized selected-test loss on 7/8 sources
  and by `1.12e-3` on average versus independent IID actions.
- Prospective external model selection is a required failure: despite 95.8%
  agreement versus 90.5% for IID, strength-2 improves held-out loss on only
  3/8 sources and is worse by `2.93e-3` on average. The exact validation winner
  equals the test winner on only 2/8 datasets; the target-shift floor is 0.047.
- A frozen conditional repeated-held-out audit attributes most of that result
  to finite evaluation-sample instability rather than demonstrated population
  shift. Across 4,096 repartitions, external winner agreement averages 60.4%,
  seven original floors lie within their 95% intervals, and Sonar is the sole
  upper-tail exception. Task-balanced classification is also only 49.5--66.6%
  stable despite perfect original alignment; regression is nearly stable.
- Task-balanced selection is the complementary success: agreement is 97.3%
  versus 92.7%, all 8/8 sources improve versus IID, the paired source-bootstrap
  interval excludes zero, and classification and regression both contribute.
  Validation and test quotient winners align on 8/8 sources. All five
  non-enumerating sources beat every core control.
- The multiclass selection addendum fails by ceiling: all methods already have
  100% agreement and zero validation regret on both sources. Strength-2 has the
  lowest realized test loss, but there is no selection-stability headroom.

### Unbiased cross-quotient selection

- At 32 fits, two independent strength-2 covers beat the stronger IID-U32 risk
  estimator in winner agreement, validation regret, and mean selected quotient
  test loss on all five panels; the frozen gate passes 5/5 for every clause.
- Candidate score calibration matches the theorem: mean bias is `1.10e-6` for
  cover cross-score and `5.50e-6` for IID-U, while ordinary finite-ensemble
  losses retain upward biases of `2.85e-5` and `1.21e-4`.
- A four-stream real-tensor audit calibrates Proposition 19 directly: 100% of
  141 nondegenerate IID cells and 98.6% of 73 cover cells lie within 2.58
  combined Monte Carlo standard errors; all ten panel/method geometric
  predicted/observed variance ratios lie in `[0.988,1.042]`.
- Componentwise, cover/IID ratios are 0.048--0.147 for `2<r,Cr>` and
  0.032--0.140 for `tr(C²)` across panels. All ten equal-source difference
  intervals exclude zero favorably, so the gain is not only a covariance
  orientation accident.
- Cover cross-score has lower candidate-level quotient-loss RMSE in 158/171
  cells (92.4%) and lower panel mean in 5/5. Every panel's source-bootstrap
  RMSE-difference interval excludes zero; RMSE is 27--38% of IID-U by panel.
- A frozen multiclass addendum lowers RMSE in all 6/6 candidates on 4-class
  Vehicle and 7-class Segment (source-mean reductions 37.4% and 18.7%), while
  selection remains at the acknowledged 100% ceiling.
- An approximate two-block log-quotient jackknife passes RMSE,
  validation-regret, and bias-reduction clauses on all 6/6 classification
  panels; cover/IID RMSE ratios are 0.149--0.691. This is classical
  second-order jackknife correction, not exact cross-score unbiasedness.
- Its four-block 64-fit extension lowers RMSE from 32 fits and versus IID-64 on
  all 6/6 panels; every source is favorable and all twelve source intervals
  exclude zero. Bias is not monotone, preserving the approximate boundary.
- Independent action streams for every candidate preserve the 32-fit result:
  cover agreement is higher and validation regret lower than IID-U in all 5/5
  panels. Common nuisance coordinates are not carrying the conclusion.
- A 64-fit U-statistic over four independent cover blocks passes all frontier
  clauses in 5/5 panels: score RMSE remains lower than IID-U64 and agreement
  and regret improve or tie the 32-fit cover checkpoint. Candidate-level RMSE
  source-bootstrap intervals exclude zero in every panel.
- A two-replicate 64-fit stability set passes validation-winner coverage,
  smaller-set, and wrong-singleton clauses on all 5/5 panel means versus the
  analogous IID-U union. Confirmation coverage is 99.61% at mean size 1.037;
  task-balanced is 99.83% at size 1.021. Many source intervals touch zero at
  ceiling, and this diagnostic is not a confidence set.
- With independent nuisance actions for every candidate, the stability-set
  coverage, size, and wrong-singleton advantages are strict on all 5/5 panels.
  External test-winner inclusion remains lower, preserving the held-out
  sampling boundary.
- A regular disjoint-cover graph gives an antithetic refinement. At 32 fits it
  passes all panel clauses and is exactly the quotient for 126/171 candidates
  whose products have at most 32 cells. On the 45 non-partition candidates,
  RMSE and direct residual still fall 15.3% and 25.5%; source intervals favor
  packing in 5/5 panels. Crossing two independent packed pairs at 64 fits is
  exactly unbiased and beats independent block-U64 on all four clauses in
  5/5 panels. Candidate-independent actions preserve agreement/regret gains in
  5/5, ruling out common nuisance coordinates.
- A mutually disjoint four-cover pack closes 148/171 candidates exactly at 64
  fits and improves all 23/23 remaining full-product candidates. Its controlled
  pure-component covariance ratios versus two disjoint pairs are 0.624--0.644
  for triples and 0.837 for four-way structure. The source-level addendum fails
  strictly because only one to three non-exact sources remain per panel. A
  first frozen four-source extension remedies this scope deficit: 12 complete
  128-cell tensors give 11/12 literal all-cell and 8/8 material-cell
  strength-2 wins, with all 4/4 source means favorable.
  A second four-source block fails its own strength-2 gate (9/12 cells and 3/4
  source means) but retains all 4/4 packing source wins. After clustering both
  blocks and repeats, pair32, pack64, and unbiased pair-cross64 improve on
  11/11 unique sources with bootstrap intervals bounded above zero; pack64's
  equal-source reduction is 19.72% `[18.49%,21.06%]`. Selection remains at
  ceiling, so this enlarges risk scope rather than selection evidence.
  A labeled post-hoc deletion audit leaves every one-source-deleted mean
  positive: pair32 7.49--8.15%, pack64 19.28--20.04%, and unbiased
  pair-cross64 7.13--7.79%; no single source carries the combined result.
  A frozen added-family audit on the same eight late sources contributes 16
  native-HistGB/CatBoost tensors. Strength-2 passes (15/16 cells, 14/14
  material cells, 8/8 sources), whereas the frozen 13/16 strict packing clause
  fails: eight HistGB cells are strength-2 exact ties. All eight nondegenerate
  CatBoost cells improve and none loses. A labeled post-outcome recomputation
  with five late-source families keeps all 11/11 sources favorable and pack64
  at 19.74% `[18.56%,21.00%]`.
  Three frozen alternate split seeds add 48 tensors: strength-2 passes on
  35/43 material cells and 24/24 dataset×split means; every packed estimator
  improves all 24 nondegenerate CatBoost cells and 24/24 means. Exact
  validation/test winner agreement is only 21/24 on these splits and 28/32
  including the original. This strengthens nuisance-risk transport while
  independently retaining the data-partition boundary.
  Proposition 34 separates approximate-validation error from partition shift.
  Empirically, exact gap movement is median 55.3 times the quadrature unbiased
  pair-cross64 RMSE scale, and 100.6 times among the four winner flips. A
  post-primary metric repeat gives 24/24 Brier wins at both budgets; clipped
  log has one `7.2e-13` pair32 loss and none for pack64, while AUC/accuracy are
  sparse but otherwise nonadverse.
  A final prospective mixed-schema block adds 20 tensors and 2,560 fits across
  four entirely new sources and all five model families. Every frozen gate
  passes: strength-2 wins 19/20 literal cells, 13/13 material cells, and 4/4
  source means; all three packing actions win 17/17 nondegenerate cells with
  no loss and 4/4 source means. Validation/test winners agree 4/4. Combined
  source effects are positive on 15/15 unique sources, with equal-source
  reductions 7.71%, 19.39%, and 7.34% for pair32, pack64, and unbiased
  pair-cross64; the aggregation is labeled post-outcome while the new-source
  block itself is prospective.
  A conditionally frozen alternate split repeats all 20 source-C tensors:
  strength-2 wins 20/20 literal and 12/12 material cells, and each packing
  action wins 16/16 nondegenerate cells. Validation/test winners nevertheless
  fall from 4/4 agreement to 2/4, independently sharpening the partition-
  shift boundary.
  The sole material strength-1 loss is vote/Adam-MLP, whose nuisance energy is
  61.7% higher-order. Across the 16 material extension cells, pair-to-higher
  energy ratio correlates 0.832 with the strength-2 log advantage
  (`p=1.70e-4` permutation), a post-failure confirmation of the interaction
  boundary rather than a new gate.
- On the two multiclass sources, pair32 strictly improves both nondegenerate
  MLP tensors and ties four already invariant candidates; pack64 is exact for
  all 6/6. The predeclared all-six strict-win gate therefore fails, and
  selection remains at ceiling.
- A new controlled `4^4` construction replaces graph enumeration with 16
  affine `GF(4)` cosets. The resolvable pack follows the exact risk ratio
  `(16-K)/15` at `K=1,2,4,8,16` and reaches the quotient at full resolution.
- The frozen nonlinear scope test transfers packing to log loss. Disjoint
  pair32 lowers score RMSE and absolute bias on all 6/6 classification panels;
  four-pack64 also wins all RMSE, bias, agreement, and regret clauses. The
  117 and 127 exact-partition candidates close with zero numerical error. A
  frozen support audit finds the `1e-12` floor active in 10/150 candidates
  (five with exact zeros), so the empirical statement is explicitly for
  clipped log loss. Proposition 30's smooth bias argument applies only away
  from the floor; the clipped map retains a global but potentially vacuous
  Lipschitz MSE bound. Uniform smoothing at `1e-6`, `1e-4`, and `1e-2`
  restores explicit interior support; pair32 and pack64 retain all frozen
  RMSE gates at every level, and pack64 retains all 6/6 regret gates. A
  92-cell Taylor audit finds second-order relative error below 10% everywhere
  (median `6e-5`), while also showing that the global epsilon-bound is orders
  of magnitude too loose for quantitative prediction.
- The observed mixed `4x4x2x4` product is itself resolvable into eight cosets,
  giving the exact ratios `1,6/7,4/7,0`. Its stronger graph-comparison gate
  fails: the graph pack is better on 21/23 real full-product candidates and
  all four represented panels. The operator explains why—graph sampling has
  smaller coefficients for all four triples, while resolution is better only
  for the four-way component.
- A symmetry-orbit linear program over 32,827 valid pack templates fails its
  conservative improvement gate. Its sparse five-template optimum is 0.16%
  worse in worst normalized coefficient and wins 0/23 real candidates. This
  supports the graph law's empirical Pareto-edge status but is not a global
  optimality claim.
- Crossing two independently randomized four-packs at 128 fits is exactly
  unbiased and beats the complete U-statistic over eight independent covers:
  RMSE, agreement, and regret pass 5/5 panels, all 23 full-product candidates
  have lower RMSE, and 148 smaller candidates close at `2.8e-17`.
- A fresh 65,536-pack operator audit predicts the cross128 score variance:
  22/23 candidates fall within 2.58 combined SE and panel geometric ratios are
  0.991--1.053. More than 98.7% of panel-mean variance comes from the
  residual-alignment term, sharpening the covariance mechanism.
- On 43 non-exhaustive candidate pairs, injected-gap ranking power improves in
  all 344 pair×gap×coupling cells. At a one-control-SD gap, inversion falls
  from 15.4--16.4% to 7.0--9.2%; deterministic independent candidate streams
  preserve all comparisons.
- The unbiased cross-score frontier is strictly monotone on all 23/23
  full-product candidates at both 32→64 and 64→128 fits. Panel RMSE ratios are
  0.639--0.650 then 0.514--0.567; median effective decay exponents are 0.62 and
  0.84, steeper than the independent-Monte-Carlo `B^-1/2` reference.
- A required equal-fit stronger control demotes the randomized 128 checkpoint:
  the eight-coset resolution enumerates all 128 cells once, giving zero RMSE
  and exact selection on 23/23. Pack-cross128 is relevant below product closure,
  on larger products, or when no resolution is available—not optimal at `B=N`.
- Across 1,024 fresh evaluation repartitions on 15 exact-closure sources,
  packed selection reaches zero validation regret in both panels but slightly
  worsens complement-test regret (`~1e-5`); both source intervals cross zero.
  This validation-only pass preserves the target-shift boundary at 128 fits.
- An independent cheap-screen/precise-deployment allocator passes its
  post-failure gate: it matches a paired all-candidate U64 selector in 4/5
  panels while saving 25--35% where there are four or five candidates. The
  confirmation exception is localized to one unstable source, and the
  three-candidate panel saves only 8.3%.
- The unbiased-score selection-regret bound is smaller for the cover on every
  dataset in all five panels, with source intervals excluding zero, but remains
  tens to hundreds of times the observed regret. It grounds the mechanism and
  is not a tight performance certificate.
- The external validation-fidelity improvement is source-stable, but its mean
  selected-test improvement is not: only 1/8 sources favors it and the
  bootstrap interval crosses zero. Task-balanced held-out transfer is stable.
  Cross-scoring solves quotient-risk estimation, not evaluation-sample rank
  instability. The conditional audit does not identify population shift.
- In 1,024 paired repartitions per source, cover-U again lowers validation
  regret with favorable source intervals in both OpenML panels. Test-regret
  intervals cross zero in both: external is `-1.13e-4` (4/8 favorable, one
  tie), task-balanced is `+6.51e-6` (1/8 favorable, four ties). This frozen
  validation-only pass is the empirical boundary formalized by Proposition 22.

### Predictive metrics

- Exact expected absolute Brier/MSE gains are small even when relative
  robustness gains are large: about `2.21e-4` versus IID-16 on confirmation.
- In a 2,048-action high-precision audit, Brier and log-loss favor strength-2
  versus IID in all 23 binary cells (20/23 Monte Carlo intervals exclude zero),
  ROC AUC favors it in 19/23 (15/23 intervals), and regression MSE/R² favor it
  in both cells. Accuracy is mixed (13/23 favorable; 9 intervals).
- The claim is therefore improved probabilistic prediction, ranking, and
  selection efficiency—not universal accuracy or predictive SOTA.

## Failed and demoted directions that stay in the paper

- leave-one-dataset-out Cartesian marginalization: only 2/18 material actions
  beat both IID and seed controls;
- deterministic canonicalization: exact zero declared schema risk, but only
  6/20 equal-compute wins and +0.30% mean loss;
- quotient-HPO: lower development risk but +0.436% Brier; failed predictive
  gate;
- HPO cover selection: reduced nuisance estimation but failed the downstream
  selection gate;
- aggressive prediction-dependent adaptive stopping: failed two of three
  panels and loses the fixed-budget unbiasedness guarantee;
- row-wise/high-dimensional strength-2: failed against marginal balance;
- HeteroBag: strong raw result, but Phase-2 coordinate placebo and untouched
  semantic-specificity gates failed. It supports generic ensemble diversity,
  not semantic Q-PLE superiority;
- external held-out model-selection transfer and multiclass selection
  headroom failures described above.

## Theorem package

1. aligned Brier/MSE ambiguity identity;
2. product-Hilbert fANOVA orthogonality;
3. persistent/conditional schema-risk law;
4. quotient-HPO and selection-path covariance decompositions;
5. exact randomized-design covariance operator and component multipliers;
6. strength-`t` cancellation through order `t`;
7. exact componentwise cover residual;
8. cover risk equals expected proper-loss overhead and IID-equivalent budget;
9. row order as an exact data symmetry;
10. finite quotient-selection regret bound;
11. mixed-level finite-field construction;
12. exact SRSWOR baseline;
13. favorable/adverse interaction regions;
14. nested 4/16/64 construction;
15. prediction-dependent stopping bias boundary;
16. pairwise ranking-inversion bound;
17. exact target-shift versus nuisance-selection decomposition;
18. unbiased independent-cover cross-score and IID-U comparator;
19. exact cross-score variance operator;
20. unbiased multi-cover block-U schedule;
21. cross-score variance bound for quotient-selection regret;
22. exact pooled/validation/complement loss identity and rank-reversal boundary;
23. exact two-replicate union coverage, size, and wrong-singleton identities;
24. second-order smooth-score bias cancellation by a two-block jackknife.
25. non-monotone strength-2 selection boundary from interaction-order aliasing.
26. independent screening regret/cost decomposition.
27. regular disjoint-cover graph covariance and unbiased packed-pair cross-score.
28. group-equivariant disjoint-pack unbiasedness and exact-exhaustion frontier.
29. resolvable-coset block finite-population risk law.
30. smooth nonlinear-score RMSE and bias bounds from prediction packing.
31. interaction-spectrum boundary between valid pack distributions.
32. convex orbit-design program and finite-library support certificate.
33. margin-conditioned accuracy-disagreement bound and near-tie boundary.
34. deterministic test-regret decomposition into validation-score error and
    validation-to-test partition shift.
35. exact separation between partial finite-population packing and zero-sum
    maximal `K`-antithesis.

The paper must frame the OA, fANOVA, U-statistic, and proper-score ingredients
as classical. Antithetic prediction-error estimation is also directly occupied
by Liu, Panigrahi, and Soloff (JRSS-B 2026), and general optimality language is
further occupied by Chattopadhyay, Liu, and Panigrahi (August 2026). The defensible novelty lies in the
finite nuisance-quotient estimand, its mixed-level operator composition, and
the empirical chain from prediction covariance to selection—not in negative
dependence itself.

## Main-paper figures

1. Pipeline and estimand diagram.
2. Representative prediction-field fANOVA stacks.
3. Equal-budget residual ratios with IID, SRSWOR, seed, strength-1, and QMC.
4. Exact interaction phase diagram including adverse corners.
5. Nested strength/computation frontier.
6. Cross-score RMSE and quotient-winner agreement versus IID-U.
7. Validation-to-test target-shift decomposition.
8. Conditional repeated-held-out rank instability.
9. High-dimensional success/failure boundary.
10. Controlled selection inversion phase across pure interaction orders.

## Claims to avoid

- discovery of representation or seed sensitivity;
- invention of OAs, randomized-OA integration, fANOVA, group averaging,
  U-statistics, or proper-score decompositions;
- exact invariance from one realized randomized cover;
- broad predictive or leaderboard SOTA;
- independence of transformed cells, architectures, query rows, or repeated
  source panels;
- universal strength-2 advantage when higher-order interactions dominate;
- validation fidelity as a guarantee of held-out performance;
- semantic superiority of HeteroBag.

## ICLR verdict

This is plausibly ICLR-level as a robustness, evaluation, and model-selection
paper—not as a tabular SOTA method. The strongest novelty is now the coherent
chain

`declared nuisance quotient → exact prediction fANOVA → strength-matched cover
→ unbiased cross-quotient risk → stable model selection`,

backed by frozen confirmation, changed-menu/subsample repeats, prospective
external tensors, task-balanced classification/regression scope, strong
finite-population and QMC controls, and explicit adverse cases. The absolute
predictive gains are small and external test transfer can fail; those facts
must be central, not footnotes. A submission would still benefit from an even
broader preregistered repeated-split source suite and an expert novelty review in statistical
design and ensemble-risk estimation—especially against antithetic Gaussian
cross-validation—but the Day-5 evidence is strong enough to justify developing
the paper rather than abandoning the direction.
