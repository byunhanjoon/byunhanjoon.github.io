# Day-5 final report: from schema sensitivity to nuisance-quotient selection

## Executive verdict

The best Day-5 direction is **OrbitCover**, not another tabular encoder and not
HeteroBag.

The paper-level idea is to treat exact schema representations and training
seeds as a declared finite nuisance product for the *complete* learning
pipeline. The full product average is a quotient prediction. Exact aligned
prediction fANOVA says which nuisance interactions matter; randomized
strength-matched covers estimate the quotient with much lower finite-fit risk;
and independent-cover U-statistics provide an unbiased validation criterion
for selecting models against that quotient.

The strongest late refinement is to make successive cover blocks negatively
dependent. Regular disjoint-cover graphs and resolvable cosets give a
16/32/64/128 schedule; crossing independent packed blocks preserves unbiased
quadratic risk below closure, while exhaustive resolution is exact at closure.

A final literature refresh found a close 2026 boundary: Liu, Panigrahi, and
Soloff's JRSS-B antithetic Gaussian cross-validation already establishes
negative-dependence variance reduction for prediction-error estimation and
model selection. OrbitCover therefore cannot claim the broad antithetic-risk
idea. Its remaining novelty is the finite exact-pipeline nuisance quotient,
the OA/fANOVA operator and independent-block cross-score composition, and the
end-to-end covariance-to-selection audit.

An August 2026 companion preprint by Chattopadhyay, Liu, and Panigrahi goes
further, proving optimality properties for equicorrelated antithetic Gaussian
CV in a normal-means/vanishing-bias regime. This rules out general optimality
language as well. OrbitCover's defensible distinction is finite, discrete, and
interaction-aware; even its own orbit-law LP fails to beat the graph sampler,
so no minimax claim is made.

The final operator audit makes that distinction exact. A partial pack drawn
without replacement from an `H=16` cover resolution has off-diagonal
coefficient `-1/15` for every `K`, whereas a zero-sum Gaussian antithetic
`K`-tuple has `-1/(K-1)`. They coincide only at `K=H=16`. Exhaustive finite
enumeration and 100,000 Gaussian replicates verify both laws. OrbitCover's
pre-closure gain is therefore interaction-aware finite-product cancellation,
not the maximal-antithesis result already established for Gaussian CV.

This is plausibly ICLR-level as an evaluation, robustness, and model-selection
paper. It is not predictive SOTA. Relative nuisance-risk reductions are large,
ranking and selection results are strong, and the theory/experiment chain is
coherent. Absolute Brier/MSE changes are usually `1e-4` to `1e-3`, accuracy is
mixed, and finite validation-to-test rank instability can overwhelm the
method. Those limits are central to the result.

The strongest prospective breadth check was frozen last and uses four entirely
new mixed-schema OpenML sources and all five model families. All 20 complete
tensors (2,560 fits) ran without replacement or schema fallback. Every frozen
strength and packing gate passes, including 19/20 literal strength-2 wins,
13/13 material wins, and zero packing losses across 17/17 nondegenerate cells.

## What survived

### 1. The nuisance field is real and interaction-heavy

The retained object is the aligned prediction tensor over exact feature-order,
category-ID, target-ID, and model-seed actions. Brier/MSE member-to-quotient
excess loss is exactly its Hilbert variance, and the product fANOVA
reconstructs that variance to numerical precision.

This continues Days 1--4 cleanly: rather than asking whether one representation
is best, it asks how much a finite pipeline depends on arbitrary equivalent
representatives and how to integrate that dependence under a fit budget.
Material pair and higher-order interactions explain why the earlier Cartesian
marginal action was weak.

### 2. Strength-2 covers are a robust finite-compute action

The frozen confirmation is unusually clean:

- 25/25 validation-screened held-out cells beat IID-16, four independent
  strength-1 blocks, and four seed blocks;
- all 6/6 source groups beat every core control;
- mean residual reductions are 68.9% versus IID, 59.1% versus strength-1, and
  59.5% versus seed blocks;
- equal-source reduction versus IID is 58.4%, with source-bootstrap interval
  `[46.9%, 78.2%]`;
- the finite median IID-equivalent budget is 42.4 fits for a 16-fit cover, with
  10 additional numerical-zero cells.

The result survives changed nuisance generators/model seeds (10/10 cells), a
changed data subsample (13/14 cells and 6/6 source means), a prospective
eight-source binary OpenML panel (12/12 versus IID/strength-1), and a
task-balanced classification/regression panel (16/16 cells, 8/8 sources).
Removing six full-enumeration regression cells leaves 10/10 non-enumerating
cells across five sources beating all controls.

The descriptive exact roll-up is 57/57 material cells versus IID and
strength-1, 55/57 versus seed blocks, and 23/24 panel×source groups versus all.
These are dependent counts, not an IID significance calculation.

Stronger baselines matter. Strength-2 reduces risk 51.0% and 55.8% versus
simple random sampling without replacement on confirmation and external
panels. It beats LHS in 23/25 and Sobol in 19/25 confirmation cells, with
stable pooled reductions of 37.4% and 28.8%. The originally frozen all-cell
QMC gate remains failed and is reported as such.

### 3. A real strength/computation hierarchy exists

A literal nested array gives 4-, 16-, and 64-fit prefixes of strength 1, 2,
and 3. Confirmation reductions versus equal-budget IID rise from 23.9% to
68.9% to 94.6%; task-balanced reductions rise from 22.2% to 84.4% to 95.6%.
Strength-3 is tied or lower in all 25 confirmation cells, strictly lower in 21,
and wins all six source groups.

The theory also supplies the failure boundary. For the standard four-factor
shape, strength-2 versus IID-16 has risk ratio
`16(E3/9 + E4/27)`, and strength-3 versus IID-64 has ratio `64 E4/27`.
All 57 empirical material cells fall in both favorable regions, but exact pure
triple and pure four-way tensors are adverse. This prevents a universal claim.

The high-dimensional experiments confirm the boundary. Field-wise and
row-wise strength-2 fail against sophisticated marginal balance. A prospective
mixed strength-3 OA-128 beats lower-strength and IID controls in 6/9 cells and
2/3 datasets, but its advantage over the marginal control is uncertain and its
mean Brier is worse. The defensible claim is “match strength to interaction
order,” not “OA always wins.”

### 4. Model rankings and validation selection improve

At equal 16-fit budgets, strength-2 improves quotient-winner agreement and
ranking fidelity on the original, changed-menu, changed-subsample, external,
and task-balanced panels. Spearman ranking correlation and pairwise order
accuracy exceed IID on all five. Mean pairwise inversion rates are lower on
all five binary panels; on confirmation they fall from 1.53% to 0.72%, and on
external OpenML from 3.19% to 0.77%.

The analytic inversion bound covers every observed candidate pair and becomes
nonvacuous more often under the cover, but remains loose on small-margin
pairs. It is valid mechanism support, not a sharp certificate.

Common random numbers are not essential. With independent nuisance actions
per candidate, strength-2 still lowers realized selected-test loss on 7/8
task-balanced sources and by `1.12e-3` on average versus IID.

The task-balanced prospective selection panel is the strongest held-out
performance result:

- 97.3% agreement versus 92.7% for IID;
- lower realized test loss on 8/8 sources;
- mean difference `-1.19e-3`, source-bootstrap interval
  `[-3.07e-3, -1.54e-4]`;
- both four-source classification and four-source regression strata improve;
- all five non-enumerating sources beat every core control.

### 5. Independent-cover cross-scores add the most promising novelty

An unbiased prediction estimate does not yield an unbiased squared-loss
estimate: ordinary loss has upward bias equal to the estimator residual.
Given independent unbiased cover predictions `Q_hat_A,Q_hat_B`, however,

`<Y-Q_hat_A, Y-Q_hat_B>`

is exactly unbiased for full quotient Brier/MSE. Its exact variance is
`2<r,Cr> + tr(C²)`, so the same covariance operator that fANOVA and the cover
control also governs selection-score efficiency. A complete U-statistic over
independent cover blocks gives higher-budget checkpoints.

A disjoint-stream calibration checks that equation on every real candidate.
All 141 nondegenerate IID cells and 72/73 nondegenerate cover cells are within
2.58 combined Monte Carlo standard errors. Every panel/method geometric mean
predicted/observed variance ratio lies in `[0.988,1.042]`. This is unusually
clean evidence that the operator mechanism, not merely unbiasedness, describes
the measured score variance.

The component audit is equally sharp. Cover/IID panel ratios are 0.048--0.147
for the residual-aligned term and 0.032--0.140 for covariance
self-interaction; all ten equal-source intervals exclude zero favorably. The
first term dominates absolute variance, but the cover reduces both. This
closes the empirical mechanism chain from strength balance to covariance to
score RMSE to selection stability.

A frozen controlled selection phase makes the adverse boundary operational.
Across 65,536 actions per cell and four positive quotient-risk gaps,
strength-2 cross-selection has zero winner inversions for pure first- and
second-order nuisance fields. It is worse than IID-U32 on three of four
pure-triple gaps, yet better on all four pure-four gaps. Proposition 25
explains this non-monotonicity through the `16/9` versus `16/27` equal-budget
variance multipliers. This is a deliberately constructed counterexample, not
an estimate of interaction prevalence in real data.

At 32 fits, two strength-2 covers beat the stronger IID-U32 comparator on all
five frozen panel means for quotient-winner agreement, validation regret, and
selected quotient test loss. Calibration supports the identity:

- cover cross-score mean bias: `1.10e-6`;
- IID-U mean bias: `5.50e-6`;
- ordinary cover-mean upward bias: `2.85e-5`;
- ordinary IID-mean upward bias: `1.21e-4`.

The cover cross-score has lower candidate quotient-loss RMSE in 158/171 cells
(92.4%). Its panel mean RMSE is only 27--38% of IID-U, and every panel's
source-bootstrap RMSE-difference interval excludes zero. It also beats
cross-scored SRSWOR, seed blocks, and LHS on all five panels, and strength-1
and Sobol on four of five.

A frozen multiclass scope addendum also passes: cross-score RMSE is lower in
all 6/6 candidates on the 4-class Vehicle and 7-class Segment tensors, cutting
source means by 37.4% and 18.7%. Selection remains at 100% ceiling for every
method, so this supports vector-valued score efficiency rather than a new
selection-performance claim.

The exact theory is quadratic, but an approximate nonlinear extension is
promising. A frozen two-block log-loss jackknife passes RMSE, validation-regret,
and bias-correction clauses on all 6/6 classification panels. Cover/IID
jackknife RMSE ratios range from 0.149 to 0.691, and mean absolute bias falls
versus ordinary cover log loss everywhere. Proposition 24 shows second-order
Taylor-bias cancellation when class probabilities stay away from zero. This
must be framed as classical jackknife correction and approximate evidence—not
as an exact proper-score theorem. External test log loss still worsens for the
more faithful validation selector.

At 64 fits, the four-block generalization passes the entire frontier on all
6/6 panels. Cover RMSE drops another 29--32% from 32 fits, is below IID-64 for
every source, and all twelve equal-source RMSE intervals exclude zero
favorably. Bias is not monotone everywhere, which is consistent with the
approximate Taylor statement. This makes log loss a credible secondary scope
result, while quadratic scoring remains the exact core.

The candidate-independent coupling repeat passes agreement and regret on all
5/5 panels. At 64 fits, a U-statistic over four independent cover blocks has
lower RMSE than IID-U64 and nondecreasing agreement/nonincreasing regret from
32 fits on all 5/5 panels. Confirmation agreement reaches 98.54% versus 94.94%
for IID-U64; task-balanced reaches 99.71% versus 97.05%.

Unbiasedness is not automatically optimal in every finite cell: the ordinary
biased cover-mean score has slightly lower RMSE in many near-invariant cells,
and the cross-score has lower panel mean RMSE than it on four of five panels.
This bias/variance tradeoff should be reported rather than hidden.

The strongest late extension is a regular disjoint-cover packing. The full
`4 x 4 x 2 x 4` cover family has 1,728 distinct vertices and exactly 485
disjoint partners per cover. Averaging a uniform cover with a uniform graph
neighbor preserves quotient unbiasedness while reducing the exact surviving
fANOVA covariance multipliers to 41.9--47.7% of one cover. On 32-cell products,
the sole neighbor is the set complement, so the packed 32-fit prediction is
the exact full quotient.

The frozen 32-fit gate passes every clause in 5/5 panels. Packed-pair ordinary
loss RMSE is 15--58% of an independent two-cover mean, and all 126/171
candidates on products of at most 32 cells close to the quotient with exactly
zero numerical error. Equal-source RMSE and direct-residual intervals favor
packing on all 5/5 panels. Removing every exact partition leaves 45 candidates:
mean RMSE is still 15.3% lower and direct prediction residual 25.5% lower,
with 33/45 candidate wins.

Candidate-independent graph actions preserve the selection result on all 5/5
panels. Confirmation agreement is 97.36% versus 97.13% for independent pairs;
external and task-balanced packed selection are exact while their controls are
95.56% and 99.01%. Thus common nuisance coordinates are not carrying the
packed-pair gain.

At 64 fits, crossing two independent packed-pair averages restores exact
quotient-risk unbiasedness on arbitrary product sizes. It beats the complete
U-statistic over four independent covers on score RMSE, agreement, regret, and
prediction residual in all 5/5 panels. Proposition 27 gives the regular-graph
covariance identity. Sliced/resolvable designs and antithetic sampling are
classical; the claimed contribution is the nuisance-cover graph plus the
independent outer quotient cross-score. External selected quotient test loss
worsens when validation selection becomes exact, again exposing rather than
solving the target-shift failure.

Mutually disjoint four-cover packing extends the same idea to 64 fits and
passes every frozen panel clause. It exactly closes 148/171 candidates on
products of at most 64 cells. Relative to two already-antithetic disjoint
pairs, score RMSE and direct residual fall strictly in all 5/5 panels;
confirmation agreement rises from 98.53% to 99.16%. On the remaining 23
full 128-cell candidates, all 23 favor the four-pack, with 21.8% lower RMSE
and 35.0% lower residual. The controlled operator audit predicts this:
four-pack covariance is 62.4--64.4% of the two-pair value for pure triples and
83.7% for the pure four-way field, with all five 99% endpoints favorable.

The source addendum deliberately fails its stricter gate: after exact closures,
only one to three nontrivial sources remain per panel, and every equal-source
interval touches zero despite favorable means. Proposition 28 proves
group-equivariant pack unbiasedness and exact closure when the pack exhausts
the product; the non-enumerating 128-cell advantage still needs a larger new
source panel.

That limitation was addressed in a late, source-frozen extension rather than
by reusing more tensors. Four previously unused OpenML IDs contributed 12 new
complete 128-cell tensors (1,536 fits). Two categorical-only datasets first
failed the inherited loader and were recovered with a disclosed post-failure
zero-numeric-column adapter; no source was replaced. The predeclared primary
gate passes literally: 11/12 all-cell comparisons beat IID and strength-1,
although three are near numerical invariance; substantively, all 8/8 material
cells beat IID, strength-1, and seed blocks, and all 4/4 source means
beat IID/strength-1. Pair32, pack64, and unbiased pair-cross64 each lower RMSE
on 11/12 candidates and 4/4 sources; selection is uninformative at a 100%
agreement ceiling. Exact validation/test winners agree on 3/4 sources, with
mean test regret `2.46e-5`.

A second source-frozen block adds another 12 tensors/1,536 fits and supplies a
real counterexample to broad strength-2 language. Its primary gate fails:
9/12 all cells, 7/8 material cells, and only 3/4 source means beat IID and
strength-1 as required. Yet pair32, pack64, and unbiased pair-cross64 each win
10/12 candidates and all 4/4 source means. Exact validation/test winners agree
only 2/4 times and the validation winner's mean test regret is `0.0126`, a
large independent repeat of the evaluation-sample boundary.

The post-failure fANOVA diagnostic localizes the base-cover miss. The sole
material cell that loses to strength-1 is vote/Adam-MLP, with 61.7% higher-order
energy but only 19.5% pair energy. Across all 16 material cells in the two
blocks, pair-to-higher energy ratio correlates `0.832` with strength-2's log
advantage over strength-1 (100,000-permutation `p=1.70e-4`). This is
post-failure evidence, but it independently matches Proposition 31's
interaction-spectrum boundary rather than suggesting an unexplained dataset
exception.

Combining both blocks with the earlier full-product tensors and collapsing
repeated panels by dataset identity yields eleven unique non-exhaustive sources.
All 11/11 favor every comparison. Equal-source RMSE reductions are 7.73% for
pair32 (95% bootstrap `[6.52%,8.83%]`), 19.72% for pack64
(`[18.49%,21.06%]`), and 7.36% for unbiased pair-cross64
(`[6.08%,8.55%]`); each sign-test `p=0.0009766`. This post-gate addendum
supersedes the initial sparse-source interval failure for these 32/64-fit
comparisons, while remaining a conditional eleven-source result.

A post-hoc source-concentration audit finds that this average is not carried by
one favorable dataset. Every individual effect is positive; after deleting
any one source, the equal-source mean remains in `[7.49%,8.15%]` for pair32,
`[19.28%,20.04%]` for pack64, and `[7.13%,7.79%]` for unbiased
pair-cross64. Because this diagnostic was conceived after seeing the result,
it supports robustness of interpretation but does not upgrade the evidence
grade.

Model-family breadth was then tested without relabeling the sources as new.
A config frozen before either added-family outcome contributes 16 complete
tensors (2,048 fits) for native HistGB and CatBoost on the same eight late
OpenML datasets. Strength-2 passes its declared transport gate: 15/16 cells
beat IID/strength-1, all 14/14 material cells beat every control, and all 8/8
source means pass. The packing gate fails literally—only 8/16 strict wins
versus 13 required—but for an interpretable reason: every HistGB tensor has
only strength-2-removable energy, so both action and control have numerical-zero
score RMSE. All 8/8 nondegenerate CatBoost cells improve and no candidate
loses; mean nondegenerate reductions are 9.26% (pair32), 19.86% (pack64), and
10.60% (unbiased pair-cross64). This is a qualified transport result, not a
retroactive pass.

Folding the two added families into the late-source means leaves all three
comparisons favorable on 11/11 sources. Pack64's post-outcome sensitivity mean
is 19.74% with bootstrap interval `[18.56%,21.00%]`; pair32 and unbiased
pair-cross64 are 7.86% and 7.52%. The essentially unchanged source result
reduces concern that the original three-family menu carried the effect, but it
does not add prospective-source evidence.

Three previously unused stratified split seeds then test the more important
partition axis, adding 48 tensors and 6,144 fits. Every frozen nuisance-
transport criterion passes: strength-2 beats all controls on 35/43 material
cells (81.4%, threshold 80%) and all 24/24 dataset×split means; pair32,
pack64, and unbiased pair-cross64 improve all 24/24 nondegenerate CatBoost
cells and all 24/24 dataset×split means with zero adverse cells. Mean
nondegenerate reductions are 8.56%, 20.52%, and 8.99%, respectively.

The same experiment preserves the orthogonal limitation. Exact validation and
test quotient winners agree on 21/24 alternate dataset×split pairs, with one
miss each on credit-approval, sick, and vote and mean test regret `0.00112`.
Including the original split gives 28/32 agreement and mean regret `0.00148`.
Finite nuisance Monte Carlo is essentially solved in this two-model panel, yet
partition rank reversals remain; OrbitCover should therefore be paired with
repeated data splits rather than advertised as a substitute for them.

Proposition 34 formalizes that separation: held-out test regret is bounded by
twice the uniform validation score error plus twice the maximum
validation-to-test quotient-loss shift. A post-outcome scale audit makes the
distinction concrete. Across the 32 dataset×split pairs, absolute candidate-gap
movement is at least 2.51 times, and has median 55.3 times, the quadrature
pair-cross64 score-RMSE scale; the four rank flips have median ratio 100.6.
The RMSE ratio is descriptive rather than a confidence bound, but it shows why
optimizing nuisance actions further is not the main remaining selection fix.

A post-primary repeated metric audit also keeps the headline narrow. Brier
improves on all 24 nondegenerate cells for pair32 and pack64. Pack64 improves
clipped log on all 25 nondegenerate cells; pair32 has 24 wins and one tiny
`7.2e-13` RMSE loss on HistGB/Mushroom. AUC and accuracy have fewer
nondegenerate cells (20 and 9--10) but no adverse cases, and all metric/source/
split means are no higher. This supports smooth-score transport while leaving
quadratic quotient risk as the exact primary theorem.

A final source block was frozen before downloading or observing outcomes on
Adult, Bank Marketing, Titanic, and Churn. It crosses all four sources with
one-hot linear, ordinal forest, native HistGB, CatBoost, and Adam-MLP models,
adding 20 complete `4x4x2x4` tensors and 2,560 fits without a failed-cell
replacement or schema adapter. The full frozen gate passes: strength-2 beats
IID-16 and strength-1 in 19/20 cells (threshold 16), wins all three controls in
13/13 material cells, and wins all 4/4 source means.

The three packing actions each improve 17/17 nondegenerate candidates, have
zero adverse candidates, and win 4/4 source means. Their mean nondegenerate
score-RMSE reductions are 8.13% for pair32, 19.47% for pack64, and 7.65% for
unbiased pair-cross64. Exact validation/test winners agree on 4/4 sources with
zero observed regret. The frozen secondary metric repeat also passes: pair32
and pack64 improve Brier/log/AUC on 17/20, 17/20, and 14/20 cells, respectively,
and all four source means are non-higher; accuracy improves 11/20.

Appending these four prospective sources to the prior clustered audit gives
15/15 positive source effects for every packing comparison. Equal-source mean
reductions are 7.71% for pair32 (bootstrap 95% `[6.86%,8.50%]`), 19.39% for
pack64 (`[18.43%,20.43%]`), and 7.34% for unbiased pair-cross64
(`[6.25%,8.37%]`); each two-sided sign-test is `p=6.10e-5`. This combined
interval is a post-outcome synthesis, while the four-source gate itself is
prospective.

A final post-outcome mechanism check projects each new tensor onto the exact
graph/fANOVA operator. It predicts packed-to-independent-pair prediction-
residual ratios of 0.837--0.856 on the 13 nondegenerate cells; the original
1,024-draw estimates differ by 0.0116 on average and 0.0289 at worst. The
across-cell triple-share correlation is essentially null (`rho=0.067`,
`p=0.828`) because the predicted range is narrow relative to draw noise. This
calibrates the operator scale but does not support a cell-ranking claim.

One alternate split of the entire source-C panel was then frozen after the
original outcomes. All 20 tensors and 2,560 fits completed. Nuisance transport
is even cleaner: strength-2 wins 20/20 literal and 12/12 material cells, and
all three packing methods win 16/16 nondegenerate cells with zero losses and
4/4 source means. Yet exact validation/test winner agreement falls from 4/4
to 2/4: Titanic and Churn flip, with mean validation-selected test regret
`0.001125`. This conditional repeat sharply reinforces the paper's central
separation between solved nuisance integration and unresolved partition shift.
Across both source-C partitions, the descriptive roll-up is 39/40 literal and
25/25 material strength-2 wins, 33/33 nondegenerate wins for every packing
action, and 8/8 dataset×split source means. Exact winner transfer is only 6/8,
with equal-split mean regret `0.000563`; source identity, not these eight
dependent pairs, remains the external-inference unit.

The frozen multiclass packing addendum fails its overly strict `6/6` pair-win
clause but shows no adverse candidate: four linear/forest tensors are already
nuisance-invariant and tie at zero residual, while packed pair32 improves both
nondegenerate MLP tensors. Multiclass mean RMSE falls 19.1% and residual 34.0%,
and pack64 closes all 6/6 vector-Brier quotients exactly. Selection remains at
the acknowledged 100% ceiling. This is qualified vector scope, not a new
selection claim.

Packing also scales without constructing a cover graph. On a new controlled
`4^4` product, the 16-row strength-2 base is a linear `GF(4)` subspace and its
16 affine cosets partition all 256 cells. Proposition 29 gives the exact
without-replacement cover-block ratio `(16-K)/15` versus `K` independent
covers. The frozen audit verifies ratios `1, 0.933, 0.8, 0.533, 0` at
`K=1,2,4,8,16` to `2.2e-16` and closes at 256 fits. This is a scalable
resolvable schedule, although its design ingredients remain classical.

The frozen nonlinear scope test shows that this packing gain is not an artifact
of squared loss. Under exactly the prior `1e-12` clipping convention, pair32
has lower log-score RMSE and absolute bias on all 6/6 binary/multiclass panels,
with winner agreement and validation regret no worse on 5/6. Four-pack64 wins
all five reported clauses on all 6/6 panels. All 117 candidates whose products
close at 32 fits and all 127 that close at 64 fits have exactly zero numerical
log-score error. Proposition 30 transfers prediction MSE to smooth-score RMSE
and bias bounds; it does not claim unbiased finite log loss. A frozen support
audit also prevents overextending that theorem: 10/150 exact classification
candidates touch the `1e-12` floor, five contain literal zeros, and 13 of
240,267 exact true-class probabilities are clipped. The empirical packing
result is therefore a result for the declared clipped log score. Its global
Lipschitz MSE bound remains valid, but the smooth Taylor bias explanation is
conditional on interior support and is not universal across this panel. The lone pair32
selection reversal is a `0.04%` near-ceiling subsample effect and is retained.

Uniform prediction smoothing supplies a clean repair rather than an assumed
floor. For `alpha=1e-6,1e-4,1e-2`, every true-class probability is at least
`alpha/C`. Across all three frozen sensitivity levels, pair32 lowers RMSE on
6/6 panels and has no-higher regret on 5/6; fourpack64 lowers RMSE and has
no-higher regret on 6/6. The 117/127 exact closures remain accurate to
`1.3e-15`. Proposition 30 therefore has a directly supported interior-score
instance, while the changed smoothed-log estimand is stated explicitly.

The accompanying Taylor calibration separates a useful local explanation from
a loose theorem constant. Across 23 non-exhaustive candidates and four
methods, all 92 cells have second-order relative approximation RMSE below 10%
at every smoothing level; the median is approximately `6e-5`, and 88/92 have
first-order correlation above `0.99`. The four exceptions all come from one
Adam-MLP/credit-g tensor and still have second-order error `0.017--0.057`.
The global bound has no violations but is very conservative: at 1% smoothing,
median actual/bound MSE is `6.5e-8` and the maximum is `7.3e-7`.

The same coset argument resolves the actually observed mixed
`4 x 4 x 2 x 4` product into eight strength-2 covers. Its exact packed-to-IID
cover-risk ratios are `1, 6/7, 4/7, 0` at 16, 32, 64, and 128 fits. The
predeclared stronger comparison does not pass: at 64 fits, the sequential graph
pack has lower residual on 21/23 full-product candidates and every represented
panel. Proposition 31 turns this failure into a design boundary. The graph law
filters each triple component more strongly (`0.01465--0.01497` versus
`1/63`), while the resolution filters the pure four-way component more strongly
(`1/189` versus `0.007385`). Real tensors are mostly on the triple-dominated
side; a four-way-dominated field prefers the resolution. “Mutually disjoint”
therefore does not uniquely determine the best finite pack distribution.

A final covariance-design search asks whether this boundary can be beaten by a
mixture. The frozen linear program spans 32,827 mutually disjoint pack
templates, symmetrizes each under the full level-permutation orbit, and
minimizes the worst graph-normalized surviving coefficient. Its strict gate
fails: the sparse five-template optimum has minimax ratio `1.00164`, beats none
of the five graph point coefficients or 99% lower endpoints, and improves
0/23 real full-product risks. Proposition 32 gives the convex-hull program and
the six-template support bound. This is only finite-library evidence that the
graph sampler is near a Pareto edge, not proof of global optimality.

At 128 fits, two independently randomized graph four-packs can be crossed to
recover exact quotient-risk unbiasedness without giving up their within-pack
negative dependence. The frozen control is the complete U-statistic over eight
independent strength-2 covers, using all 56 ordered block pairs. Pack-cross128
has lower score RMSE, no lower agreement, and no higher regret on all 5/5
panels. It wins RMSE on all 23 non-exhaustive full-product candidates; the
remaining 148 candidates close at `2.8e-17`. Overall score biases are
`-2.99e-7` and `+2.57e-7`, consistent with Monte Carlo error. This is the
strongest randomized-estimator result: the gain survives an unbiased,
complete-pair, equal-fit sampling control. The exhaustive equal-fit control
below is stronger at product closure.

After removing all exact closures, the source-cluster addendum is favorable on
all four represented panels and every contributing source. This is not strong
generalization evidence: confirmation has only two full-product sources and
the other three panels have one each, so their bootstrap intervals are
degenerate. The 23/23 candidate result is exact for the stored tensor/action
distribution; new-source scope remains limited.

The independently estimated 65,536-pack covariance operator also predicts the
observed pack-cross128 variance. Twenty-two of 23 full-product candidates lie
within 2.58 combined standard errors; panel geometric predicted/observed ratios
range from 0.991 to 1.053. The quadratic `tr(C^2)` term contributes only
0.08--1.25% on panel means, so almost all score variance here is the
`2<r,Cr>` residual-alignment term. Proposition 19 therefore explains this late
frontier quantitatively, not just directionally.

Because real 128-fit agreement is near ceiling, a frozen injected-gap audit
tests ranking power on all 43 within-dataset pairs from the non-exhaustive
tensors. Pack-cross128 has lower inversion in all 32 panel×gap×coupling clauses
and all 344 individual cells. At a one-control-standard-deviation gap,
inversion falls from 15.4--16.4% to 7.0--9.2%; at two standard deviations it
falls from 2.2--2.5% to 0.3--0.5%. Independently permuting every candidate's
action stream preserves 172/172 comparisons. This is a calibrated score-power
diagnostic, not a claim about naturally occurring held-out gaps.

Assembling only unbiased scores on the same 23 full-product candidates gives a
clean compute frontier: independent cover-cross32, disjoint-pair-cross64, and
disjoint-four-pack-cross128. Both doublings reduce RMSE on 23/23 candidates and
every panel mean. The panel ratios are 0.639--0.650 and then 0.514--0.567;
median effective exponents `-log2(RMSE_2B/RMSE_B)` are 0.62 and 0.84. The
steeper-than-`B^-1/2` empirical decay comes from progressively stronger negative
dependence, not from claiming an asymptotic rate beyond the finite resolution.

That randomized frontier is not compute-optimal at the final point. The
required equal-fit stronger control evaluates the eight mixed-resolution cosets
once each: at 128 fits it enumerates all 128 cells, has zero score RMSE on
23/23, and selects the exact quotient winner. It strictly dominates
pack-cross128 at product closure. The randomized cross128 result remains a
mechanism/control for settings with `N>B` or no available resolution; the paper
must recommend exhaustive resolution whenever `B=N`.

Packing also transfers beyond its quadratic design metric, with the same
boundary seen earlier. At pair32 and pack64, Brier/log quotient-metric RMSE
improves on 23/23 non-exhaustive binary candidates and ROC-AUC RMSE on 22/23;
all four metric panel means are no higher on 5/5 panels. Accuracy improves only
15/23 at pair32 and 17/23 at pack64. The probabilistic/ranking scope gate
passes, but the universal-metric interpretation is rejected because pair32
misses its predeclared 70% accuracy threshold.

The late-source metric repeat sharpens that boundary. Pair32 and pack64 both
improve Brier and log RMSE in 11/12 candidates and all 4/4 source means. AUC
mean RMSE falls 6.6% and 19.6% and all source means improve, but only 8/12
candidates are strict wins, missing the frozen 9/12 ranking gate. Accuracy is
7/12 at both budgets. Thus the new sources confirm probabilistic-score scope,
but the strict broader ranking gate fails and the accuracy boundary persists.

Block B narrows nonlinear scope further. Brier wins 10/12 candidates at both
budgets, but clipped log wins only 8/12, AUC 5/12 (pair) and 6/12 (pack), and
accuracy 3/12. All four source means are still no higher for every metric and
mean RMSE ratios remain favorable, but the frozen candidate gates fail. The
headline performance claim should therefore stay with quadratic quotient risk;
log and ranking results are supporting, explicitly non-universal scope.

Clustering the two blocks by their eight untouched dataset identities resolves
the tie-versus-source distinction. Brier and clipped log improve on 8/8
sources for both comparisons. Equal-source reductions are 7.13%/6.94% for
pair32 and 19.24%/19.03% for pack64; all four bootstrap intervals exclude zero
and each sign-test is `p=0.0078125`. AUC and accuracy are favorable on 7/8 plus
one exact zero-error tie, but remain descriptive because their candidate gates
failed. Nonlinear mean gains are source-stable here, not candidate-universal.

The systems audit prevents a different overclaim. None of the 177 principal
manifests contains fit-level timing telemetry. Local design overhead is small
(about 36 ms per 1,024 graph-pack actions, 0.55 ms per mixed resolution, and
0.32 s for a cold full graph build), but this does not measure training or
scheduling. Every reported efficiency number is therefore in equivalent fits,
not latency, throughput, energy, or dollar cost.

A subsequent timed late-panel refit now provides local telemetry for 12 cells
and 1,536 fits. With at most four one-thread processes, all 12 regenerated
tensors match the originals exactly. Median end-to-end time per complete
128-fit cell is 11.1 s; model-family medians are 2.82 s for linear, 14.0 s for
forest, and 23.1 s for MLP, with a 38.4 s maximum. This repairs reproducibility
reporting for one panel, but remains machine/data specific and does not support
a portable latency, energy, or dollar-speedup claim.

Proposition 33 clarifies why accuracy remains different: an argmax flip is
bounded by `min(1,2||e_i||^2/gamma_i^2)` using the quotient class margin
`gamma_i`, so aggregate MSE gives no useful guarantee near ties. Pack64
accuracy non-wins have more near-tie mass at all five thresholds, but the
23-candidate permutation evidence is only suggestive (`p=0.078--0.104`);
pair32 correlations are inconsistent. The exact bound is retained and the
empirical explanation remains unresolved.

Finally, 1,024 fresh evaluation repartitions on the 15 sources where packing
closes exactly separate nuisance estimation from held-out transfer one more
time. Exact packed selection has zero validation regret versus `2.43e-6` and
`3.26e-6` for cover-U128, so the method gate passes in both panels. Yet its
complement-test regret is slightly worse by `1.12e-5` externally and `1.04e-5`
task-balanced; only 1/8 and 1/7 sources favor it and both source intervals cross
zero. The transfer gate fails. Even a perfect quotient selector cannot make a
finite validation winner equal the complementary test winner.

At the 64-fit budget, a deliberately simple stability output uses two
independent unbiased 32-fit selectors and returns the union of their winners.
Against the analogous IID-U union it passes exact-winner coverage, smaller-set,
and wrong-singleton clauses on all 5/5 panel means. Confirmation coverage rises
from 97.66% to 99.61% while size falls from 1.080 to 1.037; task-balanced rises
from 98.54% to 99.83% while size falls from 1.078 to 1.021. Proposition 23
gives exact coverage/size/wrong-singleton identities from one-replicate choice
probabilities. This is a stability diagnostic, not a confidence set; many
source intervals touch zero at ceiling, and external test-winner inclusion
does not improve.

The candidate-independent action repeat is stronger: coverage, size, and
wrong-singleton advantages are strict on all 5/5 panels. Confirmation is
99.54% versus 97.66% coverage at sizes 1.033 versus 1.081; task-balanced is
99.85% versus 98.90% at 1.025 versus 1.087. External test-winner inclusion is
still lower, so common random numbers are ruled out without blurring the
validation/test limitation.

A compute-allocation attempt clarifies where precision is needed. Screening
all candidates with cross-32 and redeploying fresh cross-32 scores to the top
two saves 16.7--30% and includes the exact winner at least 99.88% in every
panel, but fails its performance gate on 3/5 panels: final deployment noise,
not screening, is the bottleneck. A frozen post-failure repair uses one
strength-2 cover (16 fits) for screening and four fresh cover blocks (U64) for
the top two. It passes: relative to an exactly paired all-candidate U64
control, agreement/regret match in 4/5 panels while saving 35% on the
five-candidate panels and 25% on the four-candidate external panel. The
three-candidate panel matches but saves only 8.3%.

The exception is scientifically useful. On `kdd17_stock_direction`, the cheap
pilot includes the exact winner only 83.1% of the time, lowering confirmation
agreement from 98.57% to 97.36% and increasing its already tiny mean regret
from `2.56e-6` to `4.75e-6`. Proposition 26 separates this pilot-miss term from
fresh deployment noise. The allocation result is therefore promising compute
evidence, not a universal dominance claim.

## What failed, and why it matters

### External test transfer

The prospective external selection gate fails despite better validation
fidelity. Strength-2 agreement is 95.8% versus 90.5% for IID, but held-out loss
improves on only 3/8 sources and is worse by `2.93e-3` on average. The exact
validation winner equals the exact test winner on only 2/8 datasets.

The exact decomposition is

`test regret = validation→test target-shift floor + nuisance-selection term`.

The external floor is 0.047. A noisier selector can accidentally move toward
the test-preferred candidate, so more faithful validation selection can look
worse on test. Cross-scoring robustly improves external validation agreement
and regret, but its mean selected quotient test benefit is favorable on only
1/8 sources and its source interval crosses zero.

A frozen 4,096-draw repeated-held-out audit pools only the two untouched
evaluation sets and restores the original validation size. It supports a
finite-partition explanation: external winner agreement averages 60.4%, seven
of eight original floors lie inside their conditional 95% intervals, and the
mean repeated-partition floor is 0.00754. Sonar is the sole upper-tail
exception; its 0.290 floor contributes 77% of the total original external
floor. Conversely, the task-balanced panel's perfect original alignment was
fortunate: its four classification sources reproduce only 49.5--66.6%
agreement, while its large regression sources are nearly deterministic. The
method solves nuisance estimation, not held-out sampling noise; this audit
does not establish population shift.

A second frozen experiment pairs 1,024 fresh partitions with fresh 32-fit
cover-U/IID-U scores. It is a validation-only pass. Cover-U cuts validation
regret in both panels with favorable source intervals, but test-regret source
intervals cross zero in both. External mean test regret is slightly favorable
(`-1.13e-4`, 4/8 sources and one tie); task-balanced is effectively tied
(`+6.51e-6`, 1/8 favorable and four ties). For any fixed pooled evaluation
sample, Proposition 22 shows exactly that

`T_j-T_k = (N/m)(R_j-R_k) - (n/m)(V_j-V_k)`.

When pooled candidate gaps are small, a sampled validation advantage can
reverse on its complement. Lower nuisance variance estimates `V` more
faithfully; it does not remove the sampling deviation `V-R`.

### Predictive metrics

In a 2,048-action audit, Brier and log-loss favor the cover versus IID in all
23 binary cells, with 20/23 Monte Carlo intervals excluding zero. AUC favors it
in 19/23 (15 intervals), and both regression cells improve in MSE/R². Accuracy
is mixed: 13/23 favorable, only nine intervals. The simulation calibrates to
exact Brier/MSE within 2.94 standard errors at worst.

### Other rejected actions

- Cartesian marginalization: 2/18 material wins;
- deterministic canonicalization: exact declared invariance but weak
  predictive performance;
- quotient-HPO and HPO selection: nuisance-risk improvements do not pass the
  predictive/selection gates;
- aggressive prediction-dependent stopping: fails two of three panels and
  breaks fixed-budget unbiasedness;
- multiclass model selection: frozen gate fails by 100% agreement ceiling;
- HeteroBag: raw performance is large, but coordinate placebo and genuinely
  untouched semantic-specificity tests fail. It supports generic diversity,
  not semantic Q-PLE superiority.

## Novelty assessment against recent work

The paper cannot claim OAs, randomized-OA integration, fANOVA, group averaging,
U-statistics, proper-score variance, or stable selection. Close neighbors
include Frame Averaging, learned probabilistic symmetrization, orthogonal Monte
Carlo, randomized-OA integration, OA/uniform-design HPO, low-variance model
selection, stable set-valued selection, and finite-ensemble risk corrections.

The defensible composition not found in the audit is:

1. a declared product of *exact full-data pipeline equivalences* and seed;
2. semantic alignment and an exact vector-prediction quotient/fANOVA tensor;
3. a mixed-level randomized strength cover with finite component coefficients;
4. equal-fit SRSWOR, seed, lower-strength, Sobol, and LHS controls;
5. an independent-cover U-statistic targeting quotient validation risk;
6. an end-to-end link to full rankings, model selection, exact partition-shift
   decomposition, and conditional repeated-held-out diagnosis.

That is enough for a serious ICLR submission if the classical boundaries are
stated as prominently as the new pipeline application.

## What I would run next

The highest-value next experiment is a genuinely untouched, preregistered
20--30-source panel with repeated train/validation/test splits. It should make
the 32- and 64-fit unbiased cover-U selector primary, include IID-U, SRSWOR-U,
LHS/Sobol, and lower-strength block-U controls, and treat both validation
quotient regret and held-out transfer as co-primary axes. Repeated training
splits and new sources are more important now than adding
more action draws: the dominant unresolved failure is evaluation-sample rank
instability, not nuisance Monte Carlo error.

Second, test a larger declared nuisance product with controlled synthetic
triple/four-way injection plus real per-field and row-order factors. The goal is
to predict the required strength from a genuinely independent pilot, not to
reuse outcome-dependent stopping. The pilot and deployment cover must be
independent to preserve unbiasedness.

Third, sharpen Propositions 19, 22, and 23 into finite concentration and
selection-set guarantees, and compare directly with stable set-valued
selection and modern
finite-ensemble risk estimation. This is the main theoretical gap between the
current elementary exact identities and a mature theory paper.

Fourth, package the design generator, semantic aligner, tensor auditor, and
block-U scorer as a small library with wall-clock and parallel scheduling
benchmarks. Current “equivalent fits” are risk equivalences, not latency
claims.

I would not spend the next cycle optimizing accuracy, reviving semantic
HeteroBag, or adding another encoding. The strongest paper is about making
evaluation and selection respect a declared nuisance quotient under finite
compute.

## Reproducibility status

The final integrity audit verifies 305 complete tensors representing 25,008
product-cell fits, finite and semantically valid probabilities, consistent
labels/factor shapes, 151 parseable top-level summaries, and zero
validation-screening mismatches across the 57 headline test cells. The full
algebra/construction suite passes 94/94 tests, and 21 paper figures regenerate
successfully. Commands are recorded in the directory README.
