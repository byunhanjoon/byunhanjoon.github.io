# A Feature Is a Domain

**Road to ICLR, Day 4 / 30 — August 27, 2026**

## The answer first

The idea is simple:

> Do not hand a neural network an arbitrary vector for a scalar feature. Treat
> the feature as a small function space, measure that space using the training
> distribution, and add only the geometry that the field's meaning licenses.

This is a plausible secondary ICLR direction, but it is not yet an ICLR-ready paper. It
has a clear mathematical object, a chart-covariance property, a control that
isolates semantic geometry, and a strong but selected repeated California
Housing result.
It does not yet have broad wins or a recovered-official-configuration temporal
win, and its current selection rule is not reliable enough.

The portfolio decision is unchanged: **OrbitANOVA remains the primary paper**
from the adjacent idea-search record. FieldRiesz is a conditional method
direction or a targeted covariant intervention inside that paper. Day 4 makes
its mathematics and controls sharper; the negative audits do not justify
replacing the primary schema-quotient audit with this method.

OrbitANOVA's idea is simpler than its name. Refit the same complete tabular
pipeline under every schema spelling declared equivalent—for example, reorder
features or relabel category IDs—then align the predictions. If those
predictions differ, their variance is an exact label-free tax under squared or
Brier loss. A product functional ANOVA attributes that tax to individual
schema choices, their interactions, and training randomness; the attribution
then tells us which nuisance to pool, cover, or make covariant. The identities
are established mathematics. The plausible novelty is this complete
measurement-to-action chain for the schema quotient.

I call the construction **FieldRiesz**. The paper-level framing is **fields,
not features**: a table schema should specify domains and admissible function
priors, not merely storage types and coordinates.

## How Days 1–3 led here

Day 1 found that a numerical column can have categorical states hiding inside
it. On Adult, the useful exact numerical identities were concentrated in
capital gain and loss. Day 2 found the sharper boundary: rare values already
seen in training benefited, unseen values did not. Day 3 showed that even an
invertible change of coordinates can change optimization and predictions. A
matched initial function removed most of that effect, but generic
canonicalization and orbit ensembling did not become competitive methods.

Those results point to one missing object. A scalar field is not its current
coordinate vector. It is a set of functions of that field, together with a
notion of which functions are large and which are rough.

## The mathematical object

For field `j`, choose a finite function space with a centered coordinate chart

```text
phi_j(x_j) in R^{d_j},        E_train phi_j(X_j) = 0.
```

For an ordered numerical field, I use continuous piecewise-linear functions
on train-only support nodes. The nodes combine quantiles with individually
repeated values that clear a fixed excess-count threshold. This is a
deterministic construction rule, not a statistical significance test. The
chart has two bilinear forms:

```text
M_j = E_train[phi_j(X_j) phi_j(X_j)^T]       empirical mass
S_j = semantic Dirichlet/stiffness form      declared geometry
K_j(tau_j) = M_j + tau_j S_j                 Riesz operator
```

`M_j` measures a field function under the observed covariate distribution.
`S_j` penalizes variation between semantically adjacent states. Ordered values
form a path, cyclic values a ring, and nominal values get no invented
adjacency.

Solving

```text
S_j v_jk = lambda_jk M_j v_jk,
v_jk^T M_j v_jl = 1{k=l}
```

gives the rendered coordinates

```text
psi_j,tau(x) = [
  (1 + tau_j lambda_jk)^(-1/2)
  v_jk^T (phi_j(x) - mean_j)
]_k.
```

At `tau=0`, this is empirical `L2` normalization of the field-function space.
At `tau>0`, it adds a semantic smoothness prior. Every finite `tau` is an
invertible basis of the same piecewise-linear space. It supplies no labels and
no new representable functions.

Why should it change learning? If the first layer uses Euclidean weight decay,
then in the original nodal chart the penalty becomes

```text
||u_j||^2 = a_j^T (M_j + tau_j S_j) a_j.
```

The representation therefore turns ordinary initialization and regularization
into an explicit prior on field functions.

## The Day 3 repair becomes a proposition

Suppose the same field functions are written in another invertible chart,
`phi'_j = B_j phi_j`. Then

```text
M'_j = B_j M_j B_j^T,
S'_j = B_j S_j B_j^T,
K'_j = B_j K_j B_j^T.
```

Transporting the first-layer block as `W'_j = W_j B_j^{-1}` preserves:

```text
the function:       W'_j phi'_j = W_j phi_j
the penalty:        tr(W'_j K'_j W'_j^T) = tr(W_j K_j W_j^T)
the metric update:  G_j K_j^{-1}
```

This is deliberately within-field covariance. A cross-field rotation destroys
schema meaning and is not an admissible rewrite.

## Controls before celebration

The decisive comparison keeps the function space, dimension, parameter count,
and training budget fixed:

1. quantile PLE;
2. support PLE without normalization;
3. mass only, `tau=0`;
4. correct ordered stiffness, `tau>0`;
5. the same stiffness after permuting the support nodes.

If mass alone wins, the effect is conditioning. If correct and permuted
stiffness tie, there is no evidence that semantic order helped. The latter is
the most important control in Day 4.

## What survived

### 1. Mass normalization transfers on Adult and Black Friday

In a parameter-matched, fixed-budget, one-seed panel, support-aware mass
normalization improved test loss over quantile PLE in all eight Adult and Black
Friday architecture cells:

| Dataset | MLP | ResNet | TabM | field-token Transformer |
| --- | ---: | ---: | ---: | ---: |
| Adult | +3.43% | +4.13% | +4.32% | +3.03% |
| Black Friday | +8.61% | +1.27% | +1.59% | +0.75% |

This is evidence that the Day 1–2 phenomenon was not an MLP accident. But a
broad one-seed screen was harsh: among 12 additional MLP/ResNet cells, it won
one, tied four because the detector made no change, and lost seven. Fractional
mass powers softened some failures but did not solve selection.

A separate three-seed test on UCI Bike Sharing changed only the Hour field and
used a chronological 60/20/20 split. Mass improved mean test loss by 3.69% for
MLP and 6.43% for ResNet. This is the cleanest separate-dataset mass
replication so far, but it was added during the same exploratory research loop
and is not a preregistered confirmatory result.

### 2. California isolates semantic geometry

California Housing is the mechanism-bearing result. Over three held-out
seeds:

| Model | quantile PLE | mass only | wrong order | correct order |
| --- | ---: | ---: | ---: | ---: |
| MLP, mean test MSE | 0.18291 | 0.17387 | 0.17265 | **0.16460** |
| gain vs PLE | — | +4.92% | +5.60% | **+10.01%** |
| ResNet, mean test MSE | 0.18077 | 0.17730 | 0.18249 | **0.17128** |
| gain vs PLE | — | +1.89% | −0.98% | **+5.24%** |

Correct order beats both mass alone and deliberately wrong order. A one-seed
TabM cell has the same ranking: +15.71% for correct stiffness, +13.16% for
mass, and +8.91% for wrong stiffness.

The single-field intervention makes the result much harder to dismiss:

| Only changed field | gain vs unchanged PLE |
| --- | ---: |
| MedInc | −1.95% |
| HouseAge | −0.19% |
| AveBedrms | −0.46% |
| Latitude | **+9.92%** |
| Longitude | **+10.40%** |

For Latitude and Longitude, the correct path also beats the permuted path.
The improvement localizes exactly where smooth spatial order is plausible.

## What failed

The current method is not universal.

- A field-token Transformer lost 5.78% with mass normalization on California,
  and the Riesz version also lost. Architecture-universal transfer is false.
- Churn rejects every mass strength tested.
- On official temporal splits, all 16 non-baseline Weather, Cooking Time, and
  Delivery ETA cells lost to their paired PLE baselines. The best loss was
  still −1.51%.
- House and Microsoft were clear failures in the broad screen.
- A validation improvement is not yet a trustworthy selector: Diamond MLP
  selected mass on validation but lost 2.45% on test.
- UCI Bike Sharing did not replicate semantic geometry: the correct 24-hour
  ring tied a permuted ring for MLP and separated only for ResNet, while
  validation usually selected mass or a path instead.

These failures change the proposal. FieldRiesz should not silently assume
geometry for every numeric column. The semantic family must come from schema
metadata, and the system needs a conservative fallback to ordinary PLE.

## Novelty after the literature survey

The surrounding territory is busy. Numerical PLE is established; GGPL learns
supervised breakpoints; TabR and ModernNCA occupy row-neighborhood learning;
TabM sets a strong efficient-ensemble baseline; TabReD makes temporal splits
mandatory; TabICL and TabDPT define the foundation-model path.

Mass matrices, stiffness matrices, splines, generalized eigenfunctions, and
Riesz maps are classical mathematics. The possible novelty is their specific
tabular composition:

```text
declared field semantics
  + train-measured support
  + mass/stiffness function prior
  + within-field chart covariance
  + transport across modern tabular backbones.
```

I found no recent top-conference tabular paper that makes this exact
composition. That is a search result, not proof of novelty. The correct claim
is that the direction is differentiated enough to investigate, not that the
ingredients were invented here.

The residual equation has an even sharper boundary. At `tau=0`, `M^dagger c`
is ordinary least-squares/Galerkin estimation of a conditional residual mean;
adding `tau S` is classical smoothness regularization. RieszNet/ForestRiesz and
conditional kernel mean methods also occupy Riesz-learning language. The
residual solve is a performance bridge, not the novelty by itself.

## The performance bridge I almost missed

There is already a stronger empirical asset in the workspace. RAPLE uses an
out-of-fold LightGBM anchor, smoothed target-response curves, and residual-
selected pair features. Its checked-in TabReD report records raw neural wins on
11/16 full-budget dataset–model means across MLP, ResNet, TabM, and TabR; its
validation-selected three-way hybrid wins 16/16. These are three-seed
engineering means, not significance claims, and the gated system is an
ensemble rather than a pure encoding comparison. FieldRiesz should build on
this result, not pretend that its current unsupervised rendering has stronger
benchmark evidence.

The synthesis has one clean equation. For out-of-fold anchor residual `r`, let

```text
c_j = E_train[phi_j(X_j) r],
g_j = (M_j + tau_j S_j)^(-1) c_j,
h_j(x_j) = c_j^T (M_j + tau_j S_j)^(-1) phi_j(x_j).
```

`h_j` is the minimum-Riesz-norm field function aligned with what the anchor
still misses. Under `phi'_j=B_j phi_j`, it is unchanged exactly because
`c'_j=B_jc_j` and `g'_j=B_j^(-T)g_j`. This turns RAPLE-like response curves
into leakage-safe, smooth, chart-independent residual features.

It also gives a strength-free diagnostic. In generalized modes,

```text
E_j(tau)=c_j^T(M_j+tau S_j)^(-1)c_j
        =sum_k q_jk^2/(1+tau lambda_jk),
(-1)^m E_j^(m)(tau) >= 0.
```

The normalized curve `E_j(tau)/E_j(0)` measures how long residual signal
survives as declared roughness becomes expensive. A correct-versus-random gap
over the entire curve avoids reporting only the most favorable strength. An
`M`-isospectral rotation preserves every `lambda_jk` while changing the modes,
so it isolates whether semantics assigned smoothness to the right functions.
These spectral facts are classical; using them as a preregistered tabular
schema test is the proposed contribution.

There is also an exact reference law. Under a Haar-random orientation with the
same spectrum, the squared normalized residual coordinates follow
`Dirichlet(1/2,...,1/2)`, so

```text
E[R_iso(tau)] = mean_k (1+tau lambda_k)^(-1).
```

The variance is analytic, and thousands of whole-curve controls require only
normalized Gaussian vectors—not thousands of eigendecompositions. This score
ranks California Latitude and Longitude first and King County latitude second,
but no field survives a within-dataset 10% BH screen. It is an isospectral
reference percentile, not a p-value without a random-orientation null.

This **residual Riesz representer** is the clearest performance-bearing
hypothesis. I ran the decisive direct control: every row shares the exact RAPLE
encoder, out-of-fold LightGBM anchor and residuals, data split, seed, and
approximately matched neural parameter budget. The complete panel contains
five datasets, MLP, ResNet, and TabM, and three seeds: 45 paired seed-cells.

| Dataset | wins vs raw RAPLE | mean gain vs RAPLE | wins vs node-permuted | mean semantic gap |
| --- | ---: | ---: | ---: | ---: |
| California | 8/9 | +1.64% | 8/9 | +0.88% |
| TabReD Weather | 5/9 | +0.22% | 9/9 | +1.40% |
| TabReD Cooking Time | 3/9 | -0.42% | 4/9 | +0.20% |
| TabReD Delivery ETA | 0/9 | -1.40% | 4/9 | -0.39% |
| TabReD Maps Routing | 1/9 | -0.78% | 8/9 | +0.76% |

Pooled across all 45 cells, semantic Riesz beats PLE in 30 (+4.05% mean,
dominated by California), raw RAPLE in only 17/45 (-0.15%), anchor-only in 23
(-0.11%), mass in 29 (+0.24%), and one node-permuted operator in 33 (+0.57%).
At the dataset--model aggregate level it wins 5/15 against raw RAPLE and 11/15
against the node control. The method is therefore not a broad performance win;
the more defensible result is geometry sensitivity.

That mechanism signal survives several harder audits. Across five fixed-seed
exploratory node permutations on California and Weather MLP/ResNet, correct geometry wins
52/60 control comparisons (+1.05% mean), or 11/12 unique cells after averaging
the five controls within cell. More importantly, a node permutation does
not preserve the generalized spectrum relative to empirical mass. I therefore
added an `M`-isospectral control: rotate the semantic operator in mass-whitened
coordinates while preserving every generalized eigenvalue exactly. Across five
such rotations and all three backbones, correct geometry wins 80/90 control
comparisons (+0.90% mean): 25/30 for MLP, 27/30 for ResNet, and 28/30 for
TabM. Those are stress comparisons, not independent fits—the same 18 semantic
models are reused. After averaging the five randomized controls within each
dataset--model--seed cell, semantic geometry wins 18/18 (+0.90%). This is the
strongest mechanism result: mode assignment matters even after the full
generalized spectrum is fixed.
It is not a confirmatory 18-cell test: California and Weather were selected
after the broad pilot was inspected, and seeds/models within a dataset do not
replace independent dataset replication.

The strength audit is a warning, not a victory lap. At `tau=0.3`, correct
geometry wins 8/12 California/Weather MLP/ResNet cells against both node and
exact isospectral controls. At `tau=3`, the exact-control count falls to 5/12.
So `tau=1` is not a universal constant; a paper method needs nested selection
inside the declared semantic family and must report the full retention curve.

I then ran the missing predeclared replication. King County house sales has
known Latitude/Longitude fields and sale dates, so I fixed those two fields
before training and used a chronological 60/20/20 split. It did not reproduce
California: across MLP, ResNet, TabM, and three seeds, semantic Riesz wins only
2/9 cells against raw RAPLE (+0.07%). After averaging five randomized controls
inside each cell, it wins 6/9 against node controls (-0.27%) and 7/9 against
exact isospectral controls (-0.03%). This is not a second geometry win; it is
evidence that metadata declaring order is necessary but not sufficient.

The failure suggested one last, more faithful extension: Latitude/Longitude is
a two-dimensional field group, not two marginal paths. The first pilot formed
a centered tensor-product basis with joint empirical mass and product
stiffness `S_lat⊗M_long + M_lat⊗S_long`. Its California residual surface is
genuinely useful—8/9 wins versus raw RAPLE (+2.12%) and 7/9 versus the anchor
(+1.18%). But it loses to the wrong product geometry in all 9 cells. King
County is negative on average and also fails the exact isospectral control.

Auditing that extension exposed two missing details. First, marginal centering
does not make a pure interaction when the two coordinates are dependent; the
tensor chart must be projected, under empirical joint mass, off the constant
and both marginal spaces. Second, `tau` is not comparable until the group
stiffness is normalized by a joint generalized-frequency scale. Across legacy
8-, 12-, and 16-knot spaces, the gain versus raw RAPLE persists, but the exact-
isospectral verdict flips from 0/9 to 6/9 to 7/9 wins. I therefore added an
empirical-ANOVA projection and a train-only haversine support-graph Dirichlet
form. With both corrections frozen, the California product form wins 8/9
versus raw RAPLE (+2.63%), 7/9 versus mass (+0.37%), and 8/9 versus wrong and
finite-mass-spectrum controls (+0.23% and +0.55%). The haversine graph wins 8/9
versus its finite-support control but only 5/9 versus mass. Applied unchanged to
chronological King County, the product is +0.25% versus raw but -0.24% versus
the finite-support control; the graph is -0.03% versus raw and -0.57% versus its
control. There is also a decisive algebraic caveat: purification leaves the
California 169-column chart with empirical mass rank 69. Its semantic operator
has rank 144, while the original randomized control has rank only 68--69, so
the +0.55% gap is not a pure orientation test. I repaired this with
`M_rho=(1-rho)M_emp+rho M_ref`, using tensor trapezoid reference mass on the
declared support nodes. It completes all 144 nonredundant modes and matches the
full generalized spectrum to about `1e-12`. The honest sensitivity result is
mixed: California's semantic gain versus the completed control is -0.05%,
+0.19%, and -0.02% at `rho=0.001,0.01,0.1` (7/9, 8/9, 7/9 wins), while King
County also has no stable hierarchy. Reference mass repairs the audit but does
not rescue a semantic product-geometry claim. A selection-free average of the
three completed representers retains the California gain versus raw RAPLE
(8/9, +1.21%) but loses its matched isospectral mixture on average (5/9,
-0.08%); King County is effectively tied (+0.02%). The surface is promising for
performance; its current geometry and discretization are not yet reliably why.
The initially favorable `rho=0.01` control is also orientation-sensitive. Over
five full-space rotations, California's mean gap ranges from -0.06% to +0.19%;
after averaging rotations within each cell it is only +0.04% (7/9). King County
is +0.19% (6/9) against its averaged rotation while still failing wrong
geometry. A tiny isospectral gap is therefore not mechanism-specific.

The reference completion does pass a controlled sanity check. I generated a
smooth residual function on `[0,1]`, removed every training point in
`(0.4,0.6)`, and repeated the noisy experiment 200 times. Empirical mass has
rank 21--22/25; the reference mass restores 25/25. Inside the unseen gap,
completed correct geometry beats empirical correct geometry in 192/200 runs
and reduces MSE by 18.3% on average; it beats wrong and full-spectrum controls
in 200/200. This only shows that the code recovers a prior deliberately built
into the data-generating process. It does not offset the mixed real-data result.

Validation can still use the family pragmatically. Selecting among raw RAPLE,
anchor, all mass completions, and all semantic completions using validation
only wins 8/9 California cells (+1.17% versus raw) and 7/9 King County cells
(+0.81%), with about 0.2% oracle regret. But choosing only the semantic `rho`
does not restore the mechanism on King County (4/9 versus the matched control,
-0.09%). This is a performance selector lead, not semantic replication.
The original finite-support California exact-control gap is
directionally stable at calibrated strengths 0.3, 1, and 3 (7/9, 8/9, and 8/9
wins; +0.30%, +0.55%, and +0.43%). A validation-only choice among raw RAPLE,
anchor, mass, and the correct surface wins 8/9 California cells against always
using raw (+2.38%), but only 5/9 King County cells (+0.91%, 0.77% mean oracle
regret). That is encouraging gating behavior, not a no-harm guarantee.

The group surface can also be rendered as one token instead of concatenated as
a flat scalar. In a small parameter-matched field-token Transformer it beats
raw RAPLE in all 3 California seeds (+3.81%) and 2/3 King County seeds (+1.97%).
But California loses the wrong-geometry token in all 3 (-0.14%), while King
wins the controls but loses anchor-only on average (-0.44%) with an unstable
seed. This is a useful architecture adapter, not consistent semantic evidence,
and it is not an official FT-Transformer configuration.

Sparse selection remains unsolved. Dense Maps Riesz loses raw RAPLE by 1.07%.
A topology-neutral top-24 OOF screen reaches a practical tie (+0.09%), but
fixed top-k is unstable. A strict multiplicity screen abstains to the exact
anchor on California, Cooking, Weather, and Maps; on Delivery it selects two
fields whose semantic control fails. The approximate BH screen is useful
exploration, not a finite-sample FDR guarantee: it selects zero fields on
California, Cooking, and Maps, six on Delivery, and one on Weather. It improves
anchor-only for both Delivery backbones but still loses raw RAPLE; on Weather
it hurts anchor-only for both backbones. These failures separate two
hypotheses: the operator can be semantically meaningful without being a
reliably deployable residual feature.

The spectral curve does contain one sharper clue. Ranking fields by semantic-
minus-isospectral retention puts California Latitude and Longitude first—the
same pair found by the manual single-field intervention. Using only those two
residual representers beats raw RAPLE in 5/6 MLP/ResNet cells (+2.30% mean),
anchor-only in 4/6 (+0.82%), and is tied with the dense semantic variant (3/6,
+0.08%). This is an automatic recovery of the spatial mechanism. It is not a
general selector: the analogous positive top-8 Weather rule loses both raw
RAPLE and anchor-only in both backbone means. Because the score itself uses the
semantic operator, it also cannot be reused as unbiased evidence that semantics
beat the control; it is a performance localizer only.

## ICLR verdict

There is enough differentiated novelty for a serious secondary ICLR project,
but not enough evidence for a standalone submission. OrbitANOVA remains the
stronger primary paper.

What raises it above an encoding tweak is the combination of:

- a precise object, `K_j=M_j+tau_j S_j`;
- an exact chart-covariance statement;
- node-permuted and exactly `M`-isospectral falsification controls;
- a completely monotone semantic spectral-retention diagnostic;
- a repeated gain that localizes to semantically appropriate fields; and
- an obvious interface to MLPs, ResNets, TabM, token models, retrieval models,
  and foundation-model field tokens;
- a chart-invariant residual representer that can build directly on RAPLE's
  existing TabReD gains.

What blocks a paper today is equally clear:

- plain FieldRiesz has no temporal win; dense Residual-Riesz loses/ties raw
  RAPLE over the complete fixed-pilot panel despite California/Weather gains;
- only one repeated geometry-bearing dataset;
- the corrected spatial product result is sensitive to reference mass and
  control orientation, and wrong geometry does not replicate the hierarchy;
- mixed architecture transport;
- no reliable neural or temporal selector; the additive i.i.d. calibration
  certificate does not cover downstream neural optimization or drift;
- no full-system comparison yet against GGPL, tuned RealMLP, TabR, ModernNCA,
  or the recovered official RAPLE configurations and gated hybrid.

The next high-value **FieldRiesz** experiment is not another wide sweep. It is
a registered panel of datasets with declared spatial, temporal, cyclic,
ordinal, and nominal
fields. For each field: compare mass, correct geometry, and a permuted geometry
over multiple seeds; tune only the strength within the declared semantic
family; report the selection regret. A second dataset where correct geometry
beats both controls would make standalone promotion credible. Until then,
FieldRiesz belongs as a targeted intervention/control in OrbitANOVA.

The portfolio-level decisive experiment is different: on matched development
budgets, compare actions selected by pairwise metamorphic violations,
single-knob preprocessing sensitivity, total prediction variance, and the full
OrbitANOVA interaction/tuning-path audit. Freeze each audit-to-action rule and
give every audit the same library of abstention, iid seed/schema ensembles,
factor covers, pooled HPO, and available covariant closures. Then score it
first on disjoint nuisance levels, then on unseen dataset/model cases
without test-outcome access. OrbitANOVA earns the extra framework only if its
structured attribution changes the chosen repair and improves held-out
residual schema risk or the proper-risk/resource frontier.
For cross-case evaluation, keep all model families from one dataset in the
same outer fold and aggregate once per dataset; models are repeated
measurements, not independent replications.

## Reproducibility

The full theory is in [`THEORY_FIELDRIESZ.md`](THEORY_FIELDRIESZ.md), the
novelty survey in [`LITERATURE_MAP.md`](LITERATURE_MAP.md), and the compact
generated tables in [`results/`](results/). Run:

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/analyze_day4.py
```

All pilots estimate support and mass from training covariates only. The
California repeat panel uses three paired seed runs with held-out test splits.
Architecture transport and
the broad screen are pilots with one seed and should not be read as final
benchmark estimates.
