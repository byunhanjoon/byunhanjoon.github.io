# AGENT.md — GEOMETRY TRANSFER LAW: FINAL THEORY + EXPERIMENT PROGRAM

## Mission

Develop and rigorously test a simple theoretical explanation for:

> **When does externally supplied feature geometry improve prediction on unseen tabular states, and when does it hurt?**

The project must NOT introduce another tokenizer, router, or large engineering system unless explicitly required to test the theory.

The central object is not MPE.

The central object is:

```text
cold-state prediction using a target-independent geometry
```

The proposed core insight is:

> Geometry helps when the amount of conditional target signal that transfers through the geometry exceeds the estimation noise introduced by estimating that signal from observed states.

The project should derive this statement exactly under squared loss, derive useful corollaries and impossibility results, validate them in controlled synthetic experiments, test them retrospectively on the existing real metric-field benchmark, and then — only if the retrospective theory test succeeds — run a frozen prospective experiment on new data sources.

Do not optimize for positive results.

Do not rescue a failed theory by inventing a more complicated model.

The final mandatory deliverable is:

```text
results.md
```

The agent must finish the entire program and write `results.md`.

Do not stop after:

```text
theory derivation
synthetic experiments
retrospective experiments
data acquisition
prospective training
```

unless a later stage is explicitly gated on a failed earlier scientific criterion.

When a gate fails, finish the required analyses for that failed stage, write the negative conclusion into `results.md`, and do not invent a replacement theory post hoc.

---

# 0. Read the existing evidence

Before doing anything else, read all relevant existing artifacts.

At minimum:

```text
results.md
FINAL_REPORT.md
FINAL_AUDIT.md
THEORY.md
THEORY_FREEZE.md
PROTOCOL_FREEZE.md
```

and the existing MPE / metric-field results covering:

```text
ACS occupation
ACS industry
NYC TLC pickup/dropoff
Citi Bike stations
BTS airports
string/medical metric tasks
correct-vs-corrupted geometry
support-distance analysis
similarity encoding
Nyström/RBF
hierarchy encodings
MPE
```

Preserve these established observations:

1. Correct externally supplied geometry often contains real predictive information.

2. Correct geometry beat corrupted geometry broadly in the completed MPE benchmark.

3. MPE itself did not outperform established ways to expose the same metric.

4. Support distance alone was a weak predictor of MPE benefit.

5. A valid semantic metric can still be irrelevant or harmful for a particular prediction target.

6. Hierarchy-specific, kernel, graph, coordinate, and similarity methods can outperform a generic tokenizer.

These observations motivate the new theory.

Do NOT rerun the full MPE neural matrix.

---

# 1. Research question

The paper should answer one question:

> Given a tabular field with an externally defined geometry and test states that were unseen during training, what determines whether transferring information through that geometry decreases prediction risk?

The goal is to replace vague statements such as:

```text
nearby states should have similar outcomes
```

or:

```text
correct geometry should help
```

with an exact risk decomposition.

---

# 2. Formal problem setup

Let:

```text
S = semantic state of one tabular field
Z = all remaining tabular covariates
Y = target
```

Examples of `S`:

```text
occupation
industry
airport
station
taxi zone
diagnosis
product category
string-valued category
```

Let the field have an externally supplied metric or geometry:

```text
(X, d)
```

where `d` is constructed independently of the prediction target.

Examples:

```text
hierarchy shortest-path distance
graph shortest-path distance
geographic distance
string distance
ontology distance
```

---

# 3. Residualize the ordinary tabular information

Train a base predictor:

```text
b(Z)
```

that excludes the metric-aware state representation of `S`.

Use cross-fitting so residuals are out-of-fold.

Define:

```text
R = Y - b(Z)
```

and the conditional state residual signal:

```text
mu_s = E[R | S=s]
```

Interpretation:

> `mu_s` is exactly the remaining predictable contribution associated with state `s` after ordinary non-state tabular covariates have been accounted for.

This distinction is fundamental.

Do not analyze raw target smoothness alone.

The geometry only needs to explain:

```text
conditional residual state signal
```

not the entire target.

---

# 4. Observed and unseen states

Let:

```text
T = set of observed training states
U = set of unseen evaluation states
```

with:

```text
T ∩ U = empty
```

For training states estimate:

```text
mu_hat_T
```

from cross-fitted residual means.

Assume:

```text
mu_hat_T = mu_T + epsilon
```

with:

```text
E[epsilon] = 0
Cov(epsilon) = Sigma
```

Estimate `Sigma` from within-state residual uncertainty.

Do not assume equal variance unless testing a simplified theoretical special case.

---

# 5. Geometry operator

Represent any geometry-based transfer method as an operator:

```text
A_UT
```

mapping training-state residual effects into predictions for unseen states:

```text
mu_hat_U_geometry = A_UT mu_hat_T
```

This should encompass many existing methods.

Implement operators corresponding to:

```text
metric kNN
normalized similarity/kernel interpolation
RBF smoothing
Nyström-derived interpolation
graph harmonic extension
Laplacian regularization
hierarchy/path interpolation
coordinate kernel interpolation
MPE partition weights
```

Not every method must literally be a fixed linear operator if its natural implementation is nonlinear.

However, the PRIMARY theory experiments must use methods expressible as a fixed or training-only-estimated linear operator.

Keep nonlinear methods as secondary comparisons.

---

# 6. Baseline

The geometry-free fallback predicts no additional state residual for unseen states:

```text
mu_hat_U_fallback = 0
```

so:

```text
Y_hat_fallback = b(Z)
```

and:

```text
Y_hat_geometry = b(Z) + A_UT mu_hat_T
```

The theory should compare these two.

---

# 7. THEOREM 1 — EXACT GEOMETRY TRANSFER IDENTITY

Derive this carefully from squared prediction risk.

Let:

```text
Q
```

be a diagonal weighting matrix over unseen states corresponding to the desired state distribution.

For state-balanced evaluation:

```text
Q_ss = 1 / |U|
```

For deployment-weighted evaluation, use the appropriate state probabilities.

Show that:

```text
R_fallback - E[R_geometry]
```

equals:

```text
||mu_U||_Q^2
-
||mu_U - A mu_T||_Q^2
-
tr(Q A Sigma A^T)
```

where:

```text
||x||_Q^2 = x^T Q x
```

Equivalently:

```text
Delta
=
2 mu_U^T Q A mu_T
-
mu_T^T A^T Q A mu_T
-
tr(Q A Sigma A^T)
```

Call:

```text
G_transfer
=
||mu_U||_Q^2
-
||mu_U - A mu_T||_Q^2
```

the:

```text
TRANSFERABLE GEOMETRIC SIGNAL
```

and:

```text
C_noise
=
tr(Q A Sigma A^T)
```

the:

```text
ESTIMATION NOISE COST
```

Then:

```text
Delta = G_transfer - C_noise
```

and:

```text
geometry improves expected squared risk
iff
G_transfer > C_noise
```

This is the central theorem.

Verify every algebraic step symbolically and numerically.

Do not publish an incorrect cross term.

---

# 8. Irreducible outcome noise

Explicitly include:

```text
R = mu_S + eta
```

where:

```text
E[eta | S] = 0
```

Show that the test outcome-noise variance cancels when comparing geometry to fallback.

Therefore the geometry decision depends on:

```text
conditional state signal
training-state estimation noise
transfer operator
```

rather than irreducible test noise.

State assumptions clearly.

---

# 9. GEOMETRY TRANSFER RATIO

Define:

```text
GTR
=
G_transfer / C_noise
```

when:

```text
C_noise > 0
```

Interpretation:

```text
GTR < 0
geometry systematically transfers signal in the wrong direction

0 <= GTR < 1
some transferable geometric signal exists, but not enough to overcome estimation noise

GTR = 1
theoretical break-even point

GTR > 1
geometry should improve expected squared risk

GTR >> 1
strong geometry-transfer regime
```

Do not overpromote the name if the empirical results do not support its usefulness.

Also retain the signed quantity:

```text
Delta_theory
```

as the primary theoretical prediction.

---

# 10. THEOREM 2 — NO METRIC-ONLY DECISION RULE

Prove a simple no-free-lunch result.

Fix:

```text
the metric d
training states T
unseen states U
support distances
cover radius
geometry operator A
state sample sizes
noise covariance Sigma
```

Construct two possible conditional state signals:

```text
mu^(good)
mu^(bad)
```

on the SAME geometry.

For the first:

```text
Delta > 0
```

For the second:

```text
Delta < 0
```

An easy construction is acceptable if rigorous.

For example choose:

```text
mu_U = A mu_T
```

for a geometry-aligned target.

Then construct a target whose unseen effects are anti-aligned or orthogonal to:

```text
A mu_T
```

while leaving geometry and support distances unchanged.

Conclude:

> No statistic depending only on the feature metric, state positions, support distance, graph topology, covering radius, or state cardinality can universally determine whether geometry improves prediction.

This theorem should formally explain why previous support-distance heuristics can fail.

---

# 11. THEOREM 3 — SPECTRAL SPECIAL CASE

Derive a transparent spectral corollary.

Use a simplified symmetric setting where the geometry estimator is a graph/kernel smoother:

```text
H = V diag(h_1,...,h_m) V^T
```

with:

```text
0 <= h_k <= 1
```

or whatever valid range follows from the chosen smoother.

Write the state signal as:

```text
mu = V a
```

and assume for this corollary:

```text
Cov(epsilon) = sigma^2 I
```

Derive:

```text
Delta
=
sum_k [
    (2 h_k - h_k^2) a_k^2
    -
    sigma^2 h_k^2
]
```

Verify the derivation carefully.

Then each mode contributes:

```text
Delta_k
=
h_k(2-h_k)a_k^2
-
sigma^2 h_k^2
```

For:

```text
h_k > 0
```

derive the break-even signal-to-noise criterion:

```text
a_k^2 / sigma^2
>
h_k / (2-h_k)
```

subject to the exact assumptions.

Interpretation:

> Geometry helps when target residual signal has sufficient SNR in modes transmitted by the geometry operator.

Do not claim this exact spectral form for arbitrary rectangular cold-state transfer operators.

Label it:

```text
symmetric / transductive special case
```

used for insight.

---

# 12. THEOREM 4 — STATE-HELD-OUT RISK ESTIMATION

Derive a result connecting the theory to observable data.

Assume semantic states are sampled exchangeably from a state distribution:

```text
S ~ P_S
```

with rows conditionally sampled within each state.

Define cold-state risk:

```text
R_cold(f)
=
E_{S_new ~ P_S}
E[L(f(X),Y) | S_new]
```

Show that leave-state-group-out or K-fold state-group cross-validation estimates the risk of applying the same geometry-transfer procedure to a new state.

State exactly what needs to be retrained inside each fold:

```text
base predictor
state residual estimates
geometry transfer operator if data-dependent
hyperparameters
```

Do not leak held-out state outcomes into operator tuning.

If a full unbiasedness theorem is not valid under the exact procedure, prove consistency or state an approximation result instead.

Do not force an incorrect theorem.

---

# 13. OPTIONAL THEOREM 5 — FINITE-STATE CONCENTRATION

If clean, derive a finite-library concentration bound showing that estimating geometry-transfer benefit depends primarily on:

```text
number of held-out semantic states
```

rather than the raw number of rows.

A possible target form:

```text
|Delta_hat - Delta|
=
O_p( sqrt(log K / n_states) )
```

for a fixed finite collection of `K` operators under bounded/sub-Gaussian assumptions.

Only include if rigorous.

This is optional.

---

# 14. PROPOSITION — WHY MPE DID NOT BEAT SIMILARITY FEATURES

Preserve the useful negative MPE result mathematically.

Let normalized similarity coordinates be:

```text
w(x) in R^m
```

MPE computes:

```text
z(x) = w(x) V
```

and suppose the next backbone operation is linear:

```text
u(x) = z(x) W + b
```

Then:

```text
u(x)
=
w(x) V W + b
=
w(x) B + b
```

where:

```text
B = V W
```

Therefore MPE followed immediately by an unconstrained linear stem is simply a factorized linear map of the same normalized similarity vector.

Consequences:

```text
no new metric information is introduced
rank constraints / optimization / regularization are the only possible distinction
```

Validate numerically with matched models.

This should be a supporting proposition, not the paper's main contribution.

---

# 15. THEORY FILES

Create:

```text
experiments/geometry_transfer/THEORY.md
experiments/geometry_transfer/THEORY_CHECKS.md
```

`THEORY.md` must contain:

```text
definitions
assumptions
Theorem 1
Theorem 2
Theorem 3
Theorem 4
optional theorem if retained
MPE proposition
proofs
scope limitations
```

Use symbolic algebra or numerical simulations to check formulas.

---

# 16. EXPERIMENT STAGE 1 — EXACT SYNTHETIC VALIDATION

The first experiment should make Theorem 1 almost impossible to misunderstand.

Generate synthetic semantic states with known:

```text
mu_T
mu_U
Sigma
A
```

Generate repeated noisy estimates:

```text
mu_hat_T
```

and repeated test outcomes.

For each configuration compute:

```text
Delta_theory
Delta_empirical
```

where:

```text
Delta_empirical
=
MSE_fallback - MSE_geometry
```

averaged over many replicates.

Sweep:

```text
signal magnitude
training-state sample size
training-state noise
geometry accuracy
operator strength
number of states
```

Plot:

```text
x = exact theoretical Delta
y = empirical Monte Carlo Delta
```

Target:

```text
near-perfect diagonal agreement
```

The purpose is theorem validation, not benchmark performance.

---

# 17. SYNTHETIC PHASE TRANSITION

Use a fixed geometry and fixed operator.

Vary only:

```text
signal magnitude
noise magnitude
```

Create a 2-D grid.

Plot:

```text
x = transferable signal
y = estimation noise cost
```

or:

```text
x = GTR
```

against measured risk difference.

The theoretical boundary:

```text
GTR = 1
```

should coincide with:

```text
Delta_empirical = 0
```

within Monte Carlo uncertainty.

This should become a central paper figure if successful.

---

# 18. SYNTHETIC SPECTRAL EXPERIMENT

Generate a graph or kernel geometry.

Compute:

```text
V, h_k
```

for a chosen smoother.

Generate target signals with energy concentrated in:

```text
lowest-frequency modes
middle modes
highest-frequency modes
mixed modes
```

Sweep noise.

For every mode configuration compare:

```text
predicted spectral Delta
actual measured Delta
```

Test the threshold:

```text
a_k^2 / sigma^2
>
h_k / (2-h_k)
```

Create a heatmap showing:

```text
beneficial
neutral
harmful
```

regions.

---

# 19. SYNTHETIC NO-FREE-LUNCH EXPERIMENT

Use exactly:

```text
same metric
same train states
same unseen states
same support distances
same sample sizes
same noise
same A
```

Construct:

### Target A

geometry-aligned.

### Target B

geometry-misaligned.

Require:

```text
geometry helps Target A
geometry hurts Target B
```

Then show that these geometry-only quantities are identical:

```text
nearest support distance
median support distance
cover radius
state degree distribution
metric diameter
landmark coverage
metric dimension
```

This is the experimental illustration of Theorem 2.

---

# 20. SYNTHETIC OPERATOR COMPARISON

On the same signal compare:

```text
kNN
RBF
normalized similarity
graph harmonic
Laplacian smoothing
```

Compute each operator's:

```text
G_transfer
C_noise
Delta_theory
```

Show that the law explains why one geometry method can outperform another despite using the same metric.

This is important:

> the metric is not sufficient; the transfer operator also matters.

---

# 21. EXPERIMENT STAGE 2 — RETROSPECTIVE REAL-DATA TEST

Reuse the existing metric benchmark rather than launching another huge model sweep.

Primary existing source families:

```text
ACS
NYC TLC
Citi Bike
BTS
string / medical benchmark
```

Use every valid state-disjoint task available.

Where multiple tasks share a source, retain source clustering.

Examples:

```text
ACS occupation
ACS industry
TLC pickup
TLC dropoff
BTS origin
BTS destination
Citi Bike station
Employee Salaries
Medical Charges
```

Do not remove cells because the theory performs poorly.

---

# 22. Base predictor construction

For each task train a strong ordinary predictor:

```text
b(Z)
```

excluding the semantic-state geometry representation.

Use:

```text
CatBoost or strong GBDT
```

and:

```text
MLP/ResNet
```

as two base-model families if computationally manageable.

The THEORY should not depend on architecture.

Primary retrospective analysis may use one frozen high-quality base learner for simplicity.

Also run a second family as robustness.

---

# 23. Cross-fitting

For each outer train split:

1. split rows/state groups appropriately;
2. train `b(Z)` out-of-fold;
3. obtain cross-fitted residuals;
4. compute state residual means:

```text
mu_hat_s
```

5. estimate state residual uncertainty.

Do not compute training-state residual means using in-sample predictions from an overfit base learner.

This is mandatory.

---

# 24. Estimating Sigma

Primary:

```text
diagonal Sigma
```

using per-state standard errors of cross-fitted residual means:

```text
Sigma_ss
=
Var(R | S=s) / n_s
```

Use finite-sample corrections.

Secondary:

if dependencies justify it, test:

```text
cluster/bootstrap Sigma
```

Do not invent off-diagonal covariance without evidence.

---

# 25. Geometry operators to test

For every applicable field use multiple operators built from the SAME external metric.

Mandatory generic operators:

```text
1-nearest metric neighbor
kNN with k in {3,5,10}
normalized Gaussian similarity
normalized Laplacian/RBF weights
Nyström/kernel interpolation
```

Hierarchy tasks additionally:

```text
parent/sibling averaging
path-weighted interpolation
graph harmonic extension
Laplacian regularization
```

Geographic tasks additionally:

```text
coordinate RBF
geographic kNN
graph/network interpolation where available
```

String tasks:

```text
trigram similarity interpolation
string kNN
RBF over string distance
```

Do not require MPE.

MPE may be included as a historical baseline/operator.

---

# 26. Real-data oracle decomposition

On the RETROSPECTIVE benchmark, evaluation labels are available.

Therefore estimate:

```text
mu_U
```

using held-out test residuals.

This is allowed because the purpose here is to test the theoretical identity after outcomes are known.

For every:

```text
source
task
state split
operator
```

calculate:

```text
A mu_T
||mu_U||_Q^2
||mu_U - A mu_T||_Q^2
G_transfer
C_noise
Delta_theory
Delta_actual
GTR
```

where:

```text
Delta_actual
=
actual fallback test MSE
-
actual geometry test MSE
```

Use state-balanced evaluation as primary.

---

# 27. Primary retrospective test

Create a scatter plot:

```text
x = Delta_theory
y = Delta_actual
```

over all:

```text
task × state split × geometry operator
```

cells.

Report:

```text
Pearson
Spearman
R^2
calibration slope
mean absolute prediction error
sign accuracy
```

Sign accuracy:

```text
sign(Delta_theory)
==
sign(Delta_actual)
```

is particularly important.

The ideal result is:

```text
theory predicts both helps and harms
```

not merely a positive correlation.

---

# 28. Compare to simpler heuristics

For every cell compute:

```text
nearest support distance
mean support distance
cover radius
metric diameter
state cardinality
mean train-state frequency
raw target smoothness
conditional residual smoothness
Dirichlet energy
Moran-like autocorrelation
metric corruption advantage
```

Then ask which predicts:

```text
Delta_actual
```

Compare them against:

```text
Delta_theory
GTR
```

Primary evaluation:

```text
leave-one-source-out
```

Do not evaluate only in-sample correlations.

---

# 29. Leave-one-source-out prediction

For each source:

1. fit any required calibration using all OTHER sources;
2. predict geometry benefit on held-out source;
3. compare predicted vs actual benefit.

Keep any calibration extremely simple.

Preferred:

```text
identity prediction from exact theory
```

or:

```text
one scalar calibration slope
```

Do not use a complex machine-learning predictor.

The paper should demonstrate the law, not learn a new router.

---

# 30. Decomposition plots

For every source produce a decomposition:

```text
Total possible conditional state signal
Transfer approximation error
Noise cost
Net geometry gain
```

A useful visualization:

```text
fallback state signal = ||mu_U||²

minus:
untransferred/mis-transferred signal

minus:
estimation noise

equals:
net gain
```

This should make harmful geometry intuitively obvious.

---

# 31. Test the "correct metric" paradox

Use existing correct-vs-corrupt experiments.

For each field compare:

```text
correct A
corrupted A
```

Ask:

```text
Does correct geometry increase G_transfer?
Does it also increase C_noise?
Does the theorem explain cases where correct geometry still fails to improve final prediction?
```

This is a central empirical question.

The paper should distinguish:

```text
metric contains target-relevant structure
```

from:

```text
using metric improves total risk
```

---

# 32. Why support distance fails

Within the same dataset find pairs of unseen states with similar:

```text
d(s,T)
```

but very different:

```text
Delta_state
```

where possible.

Explain them through:

```text
local transfer alignment
local state signal
local uncertainty
```

Create case studies.

Do not overdo anecdotal examples.

---

# 33. State-level decomposition

Derive/compute per-unseen-state contributions where possible.

For state `u`:

```text
Delta_u
=
mu_u^2
-
(mu_u - a_u^T mu_T)^2
-
a_u^T Sigma a_u
```

Report distributions.

This lets you ask:

> Why does the same metric help one unseen state but hurt another?

Potentially one of the clearest figures.

---

# 34. Retrospective scientific gate

Before spending compute on new prospective datasets, require the theory to satisfy ALL of these:

### Gate R1

```text
Spearman(Delta_theory, Delta_actual) >= 0.70
```

across the full cell set.

### Gate R2

```text
sign accuracy >= 75%
```

for whether geometry helps or harms.

### Gate R3

The theory must outperform:

```text
support distance
cover radius
raw smoothness
conditional smoothness
```

in leave-one-source-out prediction.

### Gate R4

The relationship must not be driven solely by one source.

Leave-one-source-out correlations/sign accuracy must remain materially positive.

### Gate R5

The exact decomposition must explain at least some real harmful-geometry cases as:

```text
low/negative transferable signal
or
noise cost > transferable gain
```

If these fail badly:

```text
STOP THE GEOMETRY-TRANSFER PAPER
```

Finish `results.md` with a negative verdict.

Do NOT invent another theory.

---

# 35. EXPERIMENT STAGE 3 — PROSPECTIVE CONFIRMATION

Only run this stage if retrospective Gate R passes.

Before obtaining final outcomes, create:

```text
experiments/geometry_transfer/PROSPECTIVE_PROTOCOL.md
experiments/geometry_transfer/PROSPECTIVE_CONFIG.json
experiments/geometry_transfer/PROSPECTIVE_HASH.txt
```

Freeze:

```text
sources
targets
field geometry
state splits
base models
operators
hyperparameters
theory equations
Sigma estimator
metrics
success criteria
```

Do not modify after outcome inspection.

---

# 36. Prospective source requirements

Use at least:

```text
4 genuinely new source families
```

preferred:

```text
5
```

They must not be sources used to develop the theory.

Include at least:

```text
2 geographic/spatial
1 hierarchy/taxonomy
1 graph/network or another irregular metric
```

and, if feasible:

```text
1 weak/irrelevant geometry case
```

The panel should contain both helpful and harmful geometry if reality provides them.

Do not manufacture harm using target information.

---

# 37. Recommended prospective source 1 — NOAA GHCN

Use public weather station data.

State:

```text
weather station
```

Geometry:

```text
geodesic station distance
```

Potential targets:

```text
daily maximum temperature
daily minimum temperature
precipitation
```

Choose one primary target prospectively.

Use ordinary covariates:

```text
date
season
elevation if metadata
regional/time features
```

Hold out entire stations.

Prefer geographically structured state splits.

---

# 38. Recommended prospective source 2 — EPA AQS

State:

```text
air-quality monitor/site
```

Geometry:

```text
geographic distance
```

Possible target:

```text
daily PM2.5
or
ozone
```

Choose one prospectively based on data completeness, not performance.

Hold out complete monitoring sites.

---

# 39. Recommended prospective source 3 — DIVVY / OTHER BIKE SYSTEM

Use a bike system not previously used in method development.

Examples:

```text
Chicago Divvy
London Santander Cycles
Bay Wheels
```

State:

```text
station
```

Geometry:

```text
station coordinates
```

Task:

```text
trip duration
demand
or another fixed tabular prediction task
```

Hold out stations.

Do not use NYC Citi Bike again as the prospective confirmation.

---

# 40. Recommended prospective source 4 — NEW HIERARCHY

Use a hierarchy not previously used for the main results.

Possible families:

```text
BLS occupation/industry statistics
another official occupational taxonomy
product taxonomy with reliable hierarchy
biological taxonomy
```

The geometry must be declared independently of the target.

Hold out leaf states.

---

# 41. Optional prospective source 5 — SENSOR NETWORK

If easily accessible and reproducible use:

```text
PEMS
traffic sensor network
power/grid sensor network
```

State:

```text
sensor
```

Geometry:

```text
road/network distance
or physical distance
```

Hold out sensors.

Do not block the mandatory program if acquisition becomes impossible.

---

# 42. Prospective prediction target

The theory must predict geometry benefit BEFORE seeing outer-test outcomes.

For each outer train/test state split:

1. use only training states;
2. perform nested state-held-out CV inside the training states;
3. estimate:

```text
Delta_hat_cold(A)
```

for each operator;
4. freeze that prediction;
5. reveal outer unseen-state outcomes;
6. compute:

```text
Delta_actual_outer
```

The primary test is:

```text
Does the theory correctly predict whether geometry helps?
```

Not:

```text
Can we select the best encoder?
```

---

# 43. Prospective operator set

Keep it small.

Use:

```text
metric kNN
normalized similarity/RBF
one stronger domain-specific operator
geometry-free fallback
```

Optionally Nyström.

Do NOT benchmark 20 representation methods.

The paper is about the law.

---

# 44. Prospective success criteria

Freeze these before outcomes.

### P1 — Benefit prediction

Across prospective:

```text
source × task × operator
```

aggregates:

```text
Spearman >= 0.60
```

between predicted and actual benefit.

### P2 — Sign prediction

Correctly predict:

```text
geometry helps
vs
geometry hurts
```

in:

```text
>= 75%
```

of source/operator aggregates.

### P3 — Better than metric-only heuristics

Geometry Transfer Law must outperform:

```text
support distance
cover radius
raw smoothness
```

on prospective prediction.

### P4 — Cross-source breadth

At least:

```text
3 independent source families
```

must individually show qualitatively correct theory behavior.

### P5 — Harmful geometry

If at least one prospectively evaluated geometry/operator is harmful, the theory should identify it as:

```text
Delta_hat <= 0
```

or near break-even.

Do not require artificial harmful examples in the primary prospective set.

---

# 45. Robustness to base predictor

On a representative subset compare:

```text
GBDT residualization
MLP residualization
```

The decomposition should remain informative.

If conclusions change radically with `b(Z)`, analyze why.

This could itself be useful:

> geometry usefulness is conditional on what the rest of the model has already explained.

That is theoretically expected.

---

# 46. Important derived experiment — conditioning matters

For several datasets construct:

### Base predictor A

weak:

```text
intercept / small feature set
```

### Base predictor B

medium.

### Base predictor C

strong full-covariate model.

For each compute:

```text
mu_s
Delta_theory
Delta_actual
```

Question:

> As ordinary covariates explain more of the target, does the residual geometry signal shrink or change sign?

This is potentially a very important insight:

```text
geometry is not intrinsically predictive;
its value is conditional on the rest of the feature set.
```

---

# 47. Metric perturbation experiment

For selected retrospective datasets gradually corrupt the metric:

```text
0%
10%
25%
50%
100%
```

For each corruption calculate:

```text
G_transfer
C_noise
Delta_theory
Delta_actual
```

Test whether:

```text
Delta_theory
```

tracks actual degradation better than raw corruption level.

This directly connects the theory to the earlier corrupted-geometry evidence.

---

# 48. Sample-size experiment

Hold geometry and signal fixed.

Vary rows per training state:

```text
5
10
20
50
100
500
```

or feasible equivalents.

Theory predicts:

```text
Sigma decreases as n_state increases
```

so a geometry that is harmful at low sample size may become beneficial when:

```text
C_noise
```

drops below:

```text
G_transfer.
```

Test this exact phase transition.

Do this:

```text
synthetically
```

and on at least:

```text
2 real datasets
```

through controlled subsampling.

This is another simple, useful insight.

---

# 49. Number-of-states experiment

Hold total rows approximately fixed while changing:

```text
number of training states
rows per state
```

Test whether geometry benefit is controlled more strongly by:

```text
coverage/transfer quality
```

versus:

```text
per-state estimation noise.
```

Use theory to predict the tradeoff.

---

# 50. Why more data can hurt or help differently

Explicitly decompose changes from increased data into:

```text
better mu_T estimation -> lower C_noise

additional training states -> different A / better support

stronger base predictor -> different residual signal mu
```

Do not summarize all of these as simply:

```text
more data helps.
```

---

# 51. The main empirical table

Create a table with columns:

```text
Source
Task
Operator
Actual gain
Predicted Delta
Transferable signal
Noise cost
GTR
Support distance
Residual smoothness
Sign correct?
```

This should become the central result table.

---

# 52. Main figures

Generate at minimum:

## Figure 1 — Theory diagram

```text
ordinary tabular prediction b(Z)
        |
residual state signal mu_s
        |
geometry operator A
        |
transferable signal
minus
estimation noise
        |
help or harm
```

---

## Figure 2 — Exact synthetic identity

```text
x = predicted Delta
y = empirical Delta
```

with diagonal.

---

## Figure 3 — Phase transition

```text
x = GTR
y = actual risk improvement
```

with vertical:

```text
GTR = 1
```

---

## Figure 4 — Spectral phase diagram

Signal frequency × SNR.

Show predicted help/harm.

---

## Figure 5 — Same metric, opposite target

Identical geometry statistics but:

```text
Target A: geometry helps
Target B: geometry hurts
```

---

## Figure 6 — Real retrospective law

```text
x = Delta_theory
y = Delta_actual
```

colored by source/operator.

---

## Figure 7 — Theory vs heuristics

Compare leave-one-source-out prediction accuracy/correlation for:

```text
support distance
cover radius
smoothness
Dirichlet energy
Geometry Transfer Law
```

---

## Figure 8 — Real risk decomposition

Stacked or waterfall plots:

```text
available state signal
untransferred signal
noise cost
net gain
```

for representative helpful/harmful cases.

---

## Figure 9 — State-level effects

```text
x = state support distance
y = state Delta
```

colored by:

```text
local transferable signal / noise
```

---

## Figure 10 — Prospective confirmation

```text
predicted vs realized geometry benefit
```

on completely new sources.

Only generate as primary if prospective stage is reached.

---

## Figure 11 — Sample-size threshold

Rows per state vs:

```text
GTR
actual improvement
```

showing predicted sign change if present.

---

# 53. Heuristic baselines for theory prediction

The theoretical law must be compared with:

```text
nearest support distance
average support distance
cover radius
metric diameter
state frequency
target/state raw smoothness
conditional residual smoothness
Dirichlet energy
Moran-like statistic
corrupted-vs-correct metric distance
```

These are not representation baselines.

They are competing EXPLANATIONS for when geometry helps.

---

# 54. Representation methods are secondary

Do not spend most of the compute comparing architecture performance.

Use enough geometry methods to demonstrate that the law applies across operators.

The paper should NOT become:

```text
MPE vs kNN vs RBF vs Nyström leaderboard
```

The methods are examples of:

```text
A
```

inside the theory.

---

# 55. Statistical unit

Primary independent unit:

```text
source family
```

Secondary:

```text
task/state split
```

Do not treat rows as independent evidence for cross-source claims.

Use:

```text
source bootstrap
leave-one-source-out analysis
paired state-split comparisons
```

State-level analyses may use state bootstrap within a task.

---

# 56. Uncertainty on Delta_theory

Estimate uncertainty from:

```text
mu_T
mu_U in retrospective diagnostics
Sigma
finite held-out states
```

Use bootstrap over states.

For prospective prediction:

```text
mu_U outer test
```

is unavailable, so predictions must come exclusively from nested state-held-out estimates.

Report uncertainty intervals for:

```text
Delta_hat
GTR_hat
```

Do not convert the paper into a binary confidence-gated router.

Uncertainty is for scientific calibration.

---

# 57. Calibration experiment

Bin predicted:

```text
Delta_hat
```

into:

```text
strong harm
weak harm
near zero
weak benefit
strong benefit
```

Compare average realized:

```text
Delta_actual
```

in each bin.

This tests whether the law provides quantitative insight rather than only rank ordering.

---

# 58. Alternative loss functions

Primary theory:

```text
squared error
```

because the decomposition is exact.

For classification, optionally apply the theory to:

```text
Brier score
```

using vector probability residuals if the extension is mathematically correct.

Do NOT claim the exact formula applies automatically to:

```text
log loss
accuracy
AUROC
```

These may be secondary empirical metrics only.

---

# 59. Vector-valued extension

If clean, derive the Hilbert/vector-valued form for:

```text
multi-output regression
multiclass probability vectors
```

Replace scalar products with:

```text
Q-weighted Frobenius / Hilbert norms.
```

This is optional but likely straightforward.

Keep the scalar theorem primary.

---

# 60. Connection to prior MPE theory

Retain useful MPE results only as context:

```text
chart invariance
interpolation bound
equality-metric impossibility
```

But do not make them central contributions.

The new paper should explain:

> Even if an encoding perfectly respects geometry, prediction improves only when the target residual signal transfers through that geometry with sufficient SNR.

This is the more fundamental result.

---

# 61. Literature audit

Create:

```text
experiments/geometry_transfer/LITERATURE_AUDIT.md
```

Search through 2026 for work on:

```text
graph signal reconstruction
graph signal denoising
kernel task alignment
kernel target alignment
spectral bias
cold-start prediction
unseen category generalization
hierarchical generalization
graph semi-supervised learning
harmonic extension
bias-variance graph smoothing
multi-task relatedness
domain adaptation across groups
random effects / hierarchical models
state-level cross-validation
spatial prediction / kriging
```

For each close paper record:

```text
setting
main theorem
whether there are arbitrary tabular covariates Z
whether state geometry is externally supplied
whether unseen state prediction is the target
whether residualization/conditional state signal is explicit
whether it derives an exact geometry-help-vs-harm risk law
whether empirical validation spans heterogeneous tabular field geometries
```

Do NOT claim generic spectral bias-variance or kernel alignment is novel.

Potential novelty should be the:

```text
conditional cold-state tabular formulation
+
exact transfer/noise decomposition
+
no-metric-only-decision result
+
cross-geometry empirical validation
```

if literature supports that distinction.

---

# 62. Scientific claims that are forbidden unless proven

Do NOT claim:

```text
correct geometry always helps
```

Do NOT claim:

```text
smooth target implies geometry helps
```

without specifying estimation noise/operator assumptions.

Do NOT claim:

```text
support distance predicts geometry usefulness
```

unless data supports it.

Do NOT claim:

```text
GTR is universally observable
```

The oracle GTR uses unseen state signal in retrospective analysis.

Do NOT claim:

```text
MPE is a novel superior tokenizer.
```

Do NOT claim:

```text
graph spectral bias-variance is new.
```

Do NOT claim:

```text
cold-state cross-validation itself is novel.
```

---

# 63. Expected useful insights

The paper is strongest if experiments support several simple conclusions:

### Insight 1

```text
A semantically correct metric is not sufficient for predictive benefit.
```

### Insight 2

```text
Geometry acts on conditional residual state information,
not raw target smoothness.
```

### Insight 3

```text
Geometry helps when transferable signal exceeds state-estimation noise.
```

### Insight 4

```text
Support distance and metric quality alone cannot determine the sign of the gain.
```

### Insight 5

```text
The same geometry can help at high per-state sample size
and hurt at low per-state sample size.
```

### Insight 6

```text
A stronger ordinary tabular predictor can reduce or alter
the residual signal available to geometry.
```

These are more important than outperforming every representation baseline.

---

# 64. Execution discipline

Use an experiment registry.

Key each artifact by:

```text
source
task
outer state split
inner fold
base predictor
geometry metric
operator
operator hyperparameters
sample-size condition
seed
```

Reuse existing predictions/results whenever mathematically valid.

Do not rerun identical jobs.

Do not stop because one dependency fails.

Fix environment issues where feasible.

If a source becomes unavailable:

```text
document it
do not replace it after outcome inspection
```

For the prospective panel, replacement rules must be frozen BEFORE results.

---

# 65. Staged execution

Proceed automatically through stages.

## Stage 0

Read prior results.

## Stage 1

Theory derivation and symbolic/numerical checks.

## Stage 2

Synthetic identity + phase-transition experiments.

## Stage 3

Retrospective real-data decomposition.

## Stage 4

Heuristic comparison and leave-one-source-out validation.

## Stage 5

Apply retrospective scientific gate.

If failed:

```text
do not run prospective panel
```

but still complete negative analysis and `results.md`.

If passed:

```text
freeze prospective protocol
```

and continue.

## Stage 6

Acquire and process prospective data.

## Stage 7

Run prospective nested state-held-out prediction.

## Stage 8

Final theory validation.

## Stage 9

Integrity audit.

## Stage 10

Write `results.md`.

---

# 66. Completion requirement

The project is NOT complete until:

```text
THEORY.md
THEORY_CHECKS.md
LITERATURE_AUDIT.md
retrospective raw results
synthetic raw results
all main figures
all main tables
RETROSPECTIVE_AUDIT.md
```

exist.

If Gate R passes, also require:

```text
PROSPECTIVE_PROTOCOL.md
PROSPECTIVE_CONFIG.json
PROSPECTIVE_HASH.txt
prospective raw results
PROSPECTIVE_AUDIT.md
```

Finally require:

```text
FINAL_AUDIT.md
results.md
```

---

# 67. Final integrity audit

Create:

```text
experiments/geometry_transfer/FINAL_AUDIT.md
```

Verify:

```text
Theorem formulas reproduce synthetic Monte Carlo
no state overlap in cold-state splits
cross-fitting is genuine
no test outcomes enter prospective predictions
metric construction is target independent
Sigma estimator uses training information only
all retrospective cells are retained
all prospective cells are retained
no unfavorable source is dropped
no outcome-driven hyperparameter changes
all figures regenerate
all tables regenerate
all headline statistics regenerate
```

Run all unit tests.

Report:

```text
tests passed / total
```

---

# 68. results.md structure

At the very end create:

```text
results.md
```

Use this structure exactly.

---

# RESULTS — GEOMETRY TRANSFER LAW

## 1. Executive verdict

Choose exactly:

```text
SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED
```

State in 10–15 sentences:

```text
whether the exact theory is correct
whether it predicts real geometry benefit/harm
whether it beats simpler heuristics
whether prospective confirmation succeeded
whether the paper is worth writing
```

---

## 2. Core theoretical result

State the final verified Geometry Transfer Identity.

Define:

```text
transferable signal
noise cost
Delta
GTR
```

Keep it understandable.

---

## 3. Theorems and proof status

Table:

| Result | Status | Main statement | Empirical validation |
|---|---|---|---|

Include:

```text
Theorem 1 transfer identity
Theorem 2 no metric-only decision
Theorem 3 spectral corollary
Theorem 4 state-held-out risk
optional theorem
MPE factorization proposition
```

---

## 4. Synthetic exact validation

Report:

```text
correlation predicted vs empirical Delta
calibration slope
maximum identity discrepancy
sign accuracy
```

Include phase-transition results.

---

## 5. Spectral experiment

Report:

```text
low-frequency
mid-frequency
high-frequency
```

target behavior across SNR.

State whether the predicted threshold matches observations.

---

## 6. Same metric, opposite target

Show the no-free-lunch synthetic experiment.

Explicitly verify that:

```text
metric
support distances
coverage
sample sizes
```

are unchanged while geometry switches from helpful to harmful.

---

## 7. Retrospective dataset panel

Table:

```text
source
task
field
metric
train states
test states
rows
operators
```

---

## 8. Main retrospective law test

Report:

```text
Pearson
Spearman
R²
calibration slope
MAE
sign accuracy
```

for:

```text
Delta_theory vs Delta_actual
```

Include confidence intervals.

---

## 9. Transferable signal vs noise

Report average distributions of:

```text
G_transfer
C_noise
Delta
GTR
```

by source.

Identify:

```text
strong-help
weak-help
neutral
harm
```

regimes.

---

## 10. Why valid geometry sometimes hurts

Give several real examples.

For each identify whether the problem is:

```text
misaligned transfer
weak residual state signal
large estimation noise
operator oversmoothing
poor support
```

Do not hand-wave.

Use the decomposition.

---

## 11. Comparison with heuristics

Table:

| Predictor | LO-source Spearman | Sign accuracy | MAE |
|---|---:|---:|---:|

Include:

```text
support distance
cover radius
raw smoothness
conditional smoothness
Dirichlet energy
Geometry Transfer Law
```

This is one of the central sections.

---

## 12. State-level analysis

Report whether per-state:

```text
Delta_u
```

explains why nearby unseen states can have opposite outcomes.

---

## 13. Correct vs corrupted metric

Report how:

```text
G_transfer
C_noise
Delta
```

change under metric corruption.

State whether the theory explains the previous correct-vs-corrupt results.

---

## 14. Sample-size phase transition

Report whether reducing:

```text
Sigma
```

through more rows per state moves tasks across:

```text
GTR = 1
```

and whether actual risk changes accordingly.

---

## 15. Conditioning on other tabular features

Compare weak and strong:

```text
b(Z)
```

models.

State whether stronger base prediction changes:

```text
mu
G_transfer
Delta
```

as expected.

This section should answer:

> Is geometry usefulness conditional on what the rest of the table already knows?

---

## 16. Prospective protocol

If Gate R failed:

write:

```text
NOT RUN — RETROSPECTIVE THEORY GATE FAILED
```

and explain.

If Gate R passed:

describe frozen new-source protocol and hash.

---

## 17. Prospective results

If run, report:

```text
predicted Delta
actual Delta
sign
source
operator
```

for every prospective aggregate.

Report:

```text
Spearman
sign accuracy
MAE
calibration
```

Do not hide failures.

---

## 18. Prospective comparison with simple heuristics

Same comparison as retrospective but on untouched sources.

This is the highest-value empirical confirmation.

---

## 19. MPE reinterpretation

Explain the MPE negative result through:

```text
similarity-factorization proposition
```

and through the new geometry-transfer theory.

Answer:

> Was MPE solving the wrong problem?

---

## 20. What is genuinely new after literature subtraction

Discuss:

```text
graph smoothing
kernel alignment
spatial prediction
cold-start
similarity encoding
hierarchical prediction
```

and state the narrowest defensible novelty.

Do not exaggerate.

---

## 21. Failed hypotheses

List every hypothesis that failed.

Examples:

```text
support distance predicts benefit
semantic correctness guarantees improvement
low-frequency signal always helps
noise term is negligible
theory transfers prospectively
```

Only include those actually tested.

---

## 22. Main useful insights

Write 3–6 simple statements that a practitioner/researcher can understand.

For example, only if supported:

```text
1. A correct feature metric can still hurt prediction.
2. Geometry should be judged on the residual state signal after other features.
3. Transferable signal must exceed state-estimation noise.
4. The same metric can switch from harmful to useful as per-state sample size grows.
5. Metric-only diagnostics cannot universally decide usefulness.
```

---

## 23. ICLR readiness

Score 1–5:

```text
conceptual novelty
theoretical novelty
theorem strength
synthetic validation
real-world explanation
prospective validation
dataset breadth
clarity
baseline/heuristic strength
reproducibility
```

Choose exactly:

```text
READY TO WRITE ICLR
ONE THEORETICAL GAP REMAINS
ONE EMPIRICAL GAP REMAINS
PIVOT
STOP DIRECTION
```

---

## 24. Reviewer simulation

Write the five strongest likely rejection arguments.

For each:

```text
objection
evidence supporting it
evidence against it
remaining weakness
best honest response
```

---

## 25. Final paper thesis

Choose exactly one:

### Thesis A

```text
Geometry helps unseen tabular states when transferable conditional
state signal exceeds estimation noise; an exact risk decomposition
predicts this help/harm boundary.
```

### Thesis B

```text
The exact decomposition is correct but not reliably estimable
before unseen-state outcomes, limiting its practical/scientific value.
```

### Thesis C

```text
Geometry usefulness cannot be predicted sufficiently well even
with the proposed decomposition; the direction should stop.
```

---

## 26. Best paper titles

Give five ranked titles.

Candidate style if Thesis A survives:

```text
When Does Feature Geometry Help?
A Risk Decomposition for Unseen Tabular States
```

```text
Valid Geometry, Wrong Prediction:
When Structure Transfers to Unseen States
```

```text
Geometry Is Not Enough:
Signal, Noise, and Cold-State Tabular Prediction
```

---

## 27. Final recommendation

Choose exactly:

```text
COMMIT TO THEORY PAPER
CONTINUE ONE SMALL GAP
RETURN TO ORBITCOVER
STOP THIS DIRECTION
```

Explain in one paragraph.

---

# 69. Final success criterion

This project should NOT be judged by whether a new representation achieves state-of-the-art accuracy.

The project succeeds if it produces a simple law that:

```text
1. is mathematically exact under clearly stated assumptions;

2. explains why the same valid geometry can help or hurt;

3. explains existing real negative and positive geometry-transfer outcomes;

4. substantially outperforms simple metric-only heuristics at predicting the sign/magnitude of geometry benefit;

5. prospectively predicts help/harm on completely new state-disjoint sources.
```

If all five survive, this is potentially a much cleaner ICLR paper than MPE.

If the exact decomposition is mathematically correct but cannot predict real or prospective outcomes because its required quantities are not reliably estimable, state that clearly.

Do not invent another encoder or router to rescue it.

The purpose of this project is to determine whether there is a **simple, general scientific law of when feature geometry transfers to unseen tabular states**.