# Day 4 ICLR direction freeze

## Working title

**Fields, Not Features: Riesz Geometry for Tabular Neural Networks**

Alternative public title: **A Feature Is a Domain**.

## The idea in one sentence

Represent each scalar table field as a finite function space whose empirical
mass, declared semantic geometry, and cross-fitted residual signal determine
the neural prior—independently of the arbitrary coordinates used to encode the
field.

## Day 4 decision

There is enough differentiated structure for a serious secondary research
program, but not enough novelty-plus-evidence for an ICLR submission today.
The operator, chart covariance, spectral-retention diagnostic, and exact
isospectral control form a coherent mechanism package. The complete direct
benchmark, rank audit, completion sensitivity, and failed semantic replication
do not support the required method claim. Continue only under the promotion
gates at the end of this note.

## Portfolio placement

This note does **not** replace the decision in the adjacent
`road-to-iclr-idea-search` record. **OrbitANOVA remains the primary ICLR paper**:
it has the stronger Days 1--3 evidence, an attributable schema-quotient risk
estimand, and a clearer measurement-to-action contribution. FieldRiesz remains
the high-risk secondary method or an OrbitANOVA intervention. Promote it to a
standalone paper only after preregistered real-data semantic replication,
competitive reference loss, and architecture/temporal transfer. The new Day 4
audits move it farther from promotion despite making its mathematics cleaner.

| Decision axis | OrbitANOVA | FieldRiesz |
| --- | --- | --- |
| Paper role | **Primary** | Secondary method / intervention |
| Novelty carrier | Schema quotient + risk-valued attribution + selection-path audit + targeted action | Declared field operator + chart covariance + residual and false-geometry protocol |
| Strongest evidence | Multi-dataset sensitivity and HPO-path pilots with exact attribution identities | Selective California/Weather control gaps and exact operator identities |
| Main unresolved risk | Broad modern-family prevalence and held-out action transfer | No robust semantic replication, temporal gain, or reliable selector |
| Current decision | Execute the frozen 12--15 dataset audit | Keep only under explicit promotion gates |

## The simplest explanation

A neural model should know two things about a column before it learns:

1. where the training data put probability mass; and
2. which values are genuinely neighbors according to the schema.

Ordinary encodings mix those choices with arbitrary coordinates. FieldRiesz
separates them. Residual-Riesz then asks which smooth field function explains
what a cross-fitted anchor still gets wrong.

## Core construction

For centered field coordinates `phi_j(x)`:

```text
M_j = E_train[phi_j(X_j) phi_j(X_j)^T]
S_j = declared field stiffness
K_j(tau_j) = M_j + tau_j S_j
psi_j,tau(x) = K_j(tau_j)^(-1/2) phi_j(x)
```

These inverses and square roots are taken on the nonredundant centered field
quotient; implementations use the Moore--Penrose inverse only when evaluation
and residual covectors stay in the operator range. Empirical-null product
directions require the reference-mass completion described in the theory note.

On numerical fields, `phi_j` is a nodal finite-element chart on train-only
quantiles plus support spikes that clear a fixed excess-count threshold.
This is a deterministic construction rule, not a significance test. The
current allocator takes a five-level local median `m_l` of unique-value counts
`n_l`, defines `e_l=max(n_l-max(m_l,1),0)`, and retains a spike only when
`e_l >= max(2, 5e-4 n_train)`, subject to the fixed node budget. This heuristic
must be frozen or treated as a tunable preprocessing choice in any broad run.
Ordered, cyclic, and ordinal schemas determine `S_j`; nominal fields receive
no invented adjacency.

For an out-of-fold anchor residual `r`:

```text
c_j = E_train[phi_j(X_j) r]
g_j = K_j(tau_j)^(-1)c_j
h_j(x) = c_j^T K_j(tau_j)^(-1)phi_j(x)
```

`h_j` is the residual Riesz representer. It is the minimum-`K_j`-norm field
function aligned with the anchor residual.

The chart-invariant diagnostic

```text
E_j(tau)=c_j^T K_j(tau)^(-1)c_j
        =sup_{g != 0} <c_j,g>^2/(g^T K_j(tau)g)
```

measures smooth residual energy. Its gap against preregistered permuted
operators is the proposed mechanism score; an independent held-out lower
confidence bound can abstain to the anchor when the score does not translate
into positive residual-loss reduction.

In generalized eigenmodes, this energy has the exact profile

```text
E_j(tau) = sum_k q_jk^2 / (1 + tau lambda_jk),
(-1)^m E_j^(m)(tau) >= 0.
```

The normalized curve `E_j(tau)/E_j(0)` records how quickly residual signal is
removed as schema-declared roughness becomes expensive. A node-permuted path
is a false-adjacency control, but it does not preserve the spectrum relative to
`M_j`. The harder control rotates the operator in `M_j`-whitened coordinates:

```text
A_j = M_j^(-1/2) S_j M_j^(-1/2) = V Lambda V^T,
A_j,iso = Q Lambda Q^T.
```

This preserves the generalized eigenvalues exactly and tests whether semantic
geometry assigns them to the right field-function modes.

For a Haar-random exact-spectrum orientation, normalized modal weights obey

```text
(U_1^2,...,U_r^2) ~ Dirichlet(1/2,...,1/2),
R_Q(tau)=sum_k U_k^2/(1+tau lambda_k).
```

Hence the random-control curve has analytic mean and variance and can be
sampled with normalized Gaussian vectors rather than repeated matrix
factorizations. This creates a cheap, strength-integrated semantic-orientation
reference score. It is not a literal p-value unless random orientation is the
null, and it must be nested if used for feature selection.

## Formal spine

Under any invertible within-field chart change `phi'_j=B_j phi_j`:

```text
M'_j = B_j M_j B_j^T
S'_j = B_j S_j B_j^T
K'_j = B_j K_j B_j^T
c'_j = B_j c_j
```

Consequently:

```text
W'_j = W_j B_j^(-1)              preserves the first-layer function
g'_j = B_j^(-T)g_j               preserves h_j(x)
tr(W'_j K'_j W'_j^T)             preserves the Riesz penalty
G'_j K'_j^(-1)                   preserves the metric-gradient update
```

The theorem is deliberately local to a field. Cross-field rotations destroy
schema identity and are not admissible symmetries.
It preserves the function, penalty, and stated metric-gradient step; it does
not assert that ordinary coordinatewise Adam/AdamW has an identical paired
trajectory under an orthogonal change of rendered coordinates.

## Paper claim to pursue

> Declared field geometry and cross-fitted residual Riesz representers provide
> a chart-covariant interface between tabular schema semantics and modern
> neural backbones; chart-invariant residual energy and a held-out gain
> certificate decide when to activate the interface and otherwise abstain to
> the anchor or ordinary PLE.

The concentration certificate is exact only for the additive predictor under
i.i.d. calibration. A reliable neural and temporal fallback remains
aspirational.

## Current evidence

- Adult and Black Friday: support-aware mass normalization improves a one-seed
  parameter-matched panel of MLP, ResNet, TabM, and a field-token Transformer.
- UCI Bike Sharing: changing only Hour, mass improves three-seed chronological
  mean test loss by 3.69% for MLP and 6.43% for ResNet.
- California Housing: correct ordered stiffness improves three-seed mean test
  loss by 10.01% for MLP and 5.24% for ResNet; it beats mass and permuted
  stiffness. A one-seed TabM cell has the same hierarchy.
- California single-field controls: the gain localizes to Latitude and
  Longitude; three nonspatial fields do not improve.
- Negative: the cyclic ring does not consistently replicate on Bike; the
  California token Transformer fails; Churn, House, Microsoft, Weather,
  Cooking Time, and Delivery ETA reject unconditional application.
- Existing internal RAPLE evidence: raw response/relation features win 11/16
  full-budget TabReD dataset–model means across MLP, ResNet, TabM, and TabR;
  the validation-selected three-way hybrid wins 16/16. These are three-seed
  engineering means, not significance claims, and the gated system is an
  ensemble. Residual-Riesz is a mathematical build-on to
  this result, not a claim to replace its full system.
- Complete direct shared-anchor panel: over California and four official-split
  TabReD datasets, MLP, ResNet, TabM, and three seeds (45 seed-cells), dense
  semantic Riesz beats PLE in 30/45 (+4.05% mean, California-dominated), raw
  RAPLE in only 17/45 (-0.15%), anchor-only in 23/45 (-0.11%), mass in 29/45
  (+0.24%), and one node-permuted operator in 33/45 (+0.57%). At the more
  defensible dataset--model aggregate level it beats raw RAPLE in 5/15 and the
  node-permuted control in 11/15. This is a semantic-mechanism signal, not
  broad performance superiority.
- The positive direct result is concentrated in California and Weather.
  Correct Riesz beats raw RAPLE for all California MLP/ResNet seeds and all
  Weather MLP seeds, but loses every Delivery and Maps MLP/ResNet seed. TabM is
  mixed: 5/15 wins versus raw RAPLE and essentially zero mean gain.
- Node-control robustness: across five fixed-seed node permutations on
  California and Weather MLP/ResNet, correct geometry wins 52/60 control
  comparisons (+1.05% mean), or 11/12 unique cells after averaging the five
  controls within cell. This repairs dependence on one lucky permutation, but
  those controls do not preserve the generalized `(S,M)` spectrum.
- Harder control: across five exact `M`-isospectral rotations on California
  and Weather, correct geometry wins 80/90 control comparisons (+0.90% mean)
  over MLP, ResNet, and TabM. Because each semantic model is reused across the
  five rotations, repeated controls must first be collapsed to the 18 unique
  dataset--model--seed cells: against each cell's mean randomized control it
  wins 18/18 (+0.90%).
  The stress counts are 25/30 MLP, 27/30 ResNet, and 28/30 TabM. This controls
  the complete generalized spectrum while randomizing which field functions
  carry its smoothness costs.
  California and Weather were selected after the broad pilot looked positive,
  so 18/18 is mechanism stress evidence, not a confirmatory p-value or 18
  independent task replications.
- Strength robustness is incomplete. At `tau=0.3`, correct geometry wins 8/12
  cells against both node and exact isospectral controls; at `tau=3`, it wins
  8/12 against node controls but only 5/12 against isospectral controls. The
  method therefore needs nested, semantic-family strength selection; `tau=1`
  cannot be presented as a universal constant.
- A genuinely predeclared spatial replication did not clear the gate. On
  OpenML King County house sales, only Latitude and Longitude receive
  representers and rows are split chronologically 60/20/20. Across three seeds
  and MLP/ResNet/TabM, semantic Riesz wins only 2/9 cells against raw RAPLE
  (+0.07% mean). After averaging five controls within each unique cell it wins
  6/9 against node controls (-0.27%) and 7/9 against exact isospectral controls
  (-0.03%). The California result therefore lacks an independent performance
  replication.
- The natural field-group extension is mathematically cleaner but does not
  rescue semantics yet. For `G=(Latitude,Longitude)`, project the tensor chart
  off the empirical constant and marginal spaces, use joint empirical mass,
  and test either product stiffness `S_G=S_lat⊗M_long+M_lat⊗S_long` or a
  train-only haversine support-graph form. The first unprojected,
  uncalibrated California surface beats raw RAPLE in 8/9 cells
  (+2.12%) and anchor-only in 7/9 (+1.18%), but loses to the wrong product
  geometry in 9/9 (-0.53%). On King County it wins 6/9 versus raw RAPLE but is
  negative on average (-0.49%) and loses the exact isospectral control overall.
  The interaction representation may be useful; the product stiffness has not
  earned a semantic claim. Changing the legacy product resolution from 8 to
  12 to 16 knots keeps the raw-RAPLE gain but changes exact-isospectral wins
  from 0/9 to 6/9 to 7/9. Joint generalized-frequency calibration and empirical
  ANOVA projection are now mandatory, not optional implementation details.
  With both repairs frozen, the California product form wins 8/9 versus raw
  RAPLE (+2.63%), 7/9 versus mass (+0.37%), and 8/9 versus both wrong and exact
  finite-mass-spectrum controls (+0.23% and +0.55%). A train-only haversine support-graph
  form also wins 8/9 versus its finite-support control but only 5/9 versus mass. Applied
  unchanged to chronological King County, neither earns replication: the
  corrected product is +0.25% versus raw but -0.24% versus that control;
  the graph is -0.03% versus raw and -0.57% versus its control. The
  correction sharpens the California lead without clearing the two-dataset
  promotion gate. Its California control gap is directionally stable at
  calibrated strengths 0.3, 1, and 3: 7/9, 8/9, and 8/9 versus finite-support
  isospectral controls, with +0.30%, +0.55%, and +0.43% mean gaps. Validation
  selection among raw RAPLE, anchor, mass, and correct surface wins 8/9
  California cells versus always using raw (+2.38%) but only 5/9 King County
  cells (+0.91%, 0.77% mean oracle regret). This is pilot gating, not a
  certified fallback.
- Product-control audit: the purified California chart has dimension 169 and
  empirical mass rank 69; the semantic operator has rank 144 while the original
  control has rank only 68--69. The +0.55% finite-support control gap is therefore
  confounded. A chart-covariant reference mass
  `M_rho=(1-rho)M_emp+rho M_ref` completes all 144 nonredundant modes and matches
  the full spectrum to about `1e-12`. In a predeclared sweep, California's mean
  semantic gain versus the completed isospectral control is -0.05%, +0.19%, and
  -0.02% at `rho=0.001,0.01,0.1` (7/9, 8/9, 7/9 wins); King County is also
  inconsistent. Freeze reference-mass completion as an audit contribution, but
  reject a robust product-geometry claim until `rho` is selected without test
  access and the hierarchy replicates. A selection-free mixture over all three
  `rho` values also fails: it retains California's raw gain but loses its
  matched isospectral mixture on average (5/9, -0.08%).
  Repeating the favorable `rho=0.01` full-space control over five orientations
  leaves only +0.04% after within-cell averaging on California (7/9); King
  County is +0.19% (6/9) despite failing wrong geometry. This control is not
  mechanism-specific at such small effect sizes.
  A 200-repetition synthetic missing-interval sanity check does validate the
  intended extrapolation: reference mass restores rank 25/25 and completed
  correct geometry lowers gap MSE by 18.3% versus empirical correct geometry
  (192/200 wins). Treat this only as implementation validation.
  Validation-only selection across raw, anchor, mass, semantic, and all three
  `rho` values wins 8/9 California cells (+1.17% versus raw) and 7/9 King
  County cells (+0.81%), with about 0.2% oracle regret. Retain this as a
  performance-routing hypothesis, not semantic evidence.
- Token transport is promising but inconsistent. Treating the purified surface
  as one extra group token beats raw RAPLE in 3/3 California seeds (+3.81%) and
  2/3 chronological King County seeds (+1.97%) in the small parameter-matched
  field-token Transformer. California nevertheless loses the wrong-geometry
  token in 3/3 (-0.14%); King wins both controls but loses anchor-only on
  average (-0.44%) and contains an unstable seed. This is not an official
  FT-Transformer result and does not establish architecture-independent
  semantics.
- Field localization: the semantic-minus-isospectral retention score ranks
  California Latitude and Longitude first. Using only those two representers
  beats raw RAPLE in 5/6 MLP/ResNet cells (+2.30% mean), anchor-only in 4/6
  (+0.82%), and is essentially tied with dense semantic Riesz (3/6, +0.08%).
  This automatically recovers the earlier manual spatial ablation. The same
  topology-specific top-8 rule fails on Weather, losing raw RAPLE and anchor-
  only for both backbone means; it is not yet a general selector.
- Sparse-selection attempts have not solved deployment. On 984-field Maps,
  dense Riesz loses raw RAPLE by 1.07%; the best topology-neutral top-24 OOF
  screen is a near tie (+0.09%) and the selected semantic operator is not
  reliably better than its false controls. A strict multiplicity screen
  abstains to the exact anchor on California, Cooking, Weather, and Maps; it
  selects two Delivery fields, where the semantic control fails. The
  approximate BH screen selects zero fields on California, Cooking, and Maps,
  six on Delivery, and one on Weather. It improves anchor-only for both
  Delivery backbones but still loses raw RAPLE; on Weather it hurts anchor-only
  for both backbones. It is exploratory, not a finite-sample FDR guarantee.

## What must not be claimed

- Mass/stiffness matrices, Riesz maps, splines, graph Laplacians, whitening,
  or target smoothing were invented here.
- `M^dagger c` or smooth residual regression is new; it is ordinary Galerkin/
  conditional-mean estimation, and Riesz learning is established elsewhere.
- FieldRiesz is currently universal, SOTA, or temporally robust.
- Validation can infer whether a topology is semantically true.
- A node-permuted stiffness has the same generalized spectrum as the semantic
  operator. It preserves the ordinary stiffness spectrum only; the separate
  `M`-isospectral control is required for the stronger claim.
- Complete monotonicity of the energy curve or isospectral rotation is new
  mathematics. They are classical facts used here to sharpen the tabular
  mechanism test.
- Target-consistent embedding pretraining is new. The ICML 2026 uncertainty
  analysis already trains LRLR embeddings with a whole-row triplet objective;
  FieldRiesz must earn its distinct per-field semantic and residual estimand.
- The California result proves spatial smoothness is the cause beyond the
  specified wrong-order and single-field controls.
- The residual Riesz pilot proves improvement over the full RAPLE system, recovered official
  model configurations, or a broad benchmark. The complete fixed-pilot direct
  panel is a tie/loss against raw RAPLE overall, despite geometry-control wins.

## Model interfaces

| Model family | Proposed interface |
| --- | --- |
| MLP / ResNet / RealMLP | Concatenate `psi_j` and optional cross-fitted `h_j`; preserve per-field blocks for the first-layer penalty. |
| TabM | Share `(M_j,S_j)` across ensemble members; let members vary ordinary parameters, not schema semantics. |
| FT-Transformer | Project each `psi_j` into one field token and each declared `psi_G,h_G` into one group token; preserve token count in every control. |
| TabR / ModernNCA | Use the invariant distance `d_j(x,z)^2=(phi_j(x)-phi_j(z))^T K_j^dagger(phi_j(x)-phi_j(z))` in keys/queries while holding retrieval memory fixed. |
| Tabular foundation models | Supply domain type/operator spectrum with each field token, or sample synthetic task functions from the invariant kernel `k_j(x,z)=phi_j(x)^T K_j^dagger phi_j(z)`. |

## Registered next matrix

1. Recover official model configurations and compare raw RAPLE response bins,
   `tau=0`, correct `S`, node-permuted `S`, `M`-isospectral `S`, and no residual
   feature at equal compute.
2. Separately register fields with known path, ring, spatial, and nominal
   semantics; tune strength only inside the declared semantic family.
3. Use nested cross-fitting or an independent calibration split for field/group
   activation. Report the neural result separately from the certified additive
   anchor-plus-representer fallback.
4. Run at least three seeds on all four unrestricted official TabReD datasets
   for MLP, ResNet, TabM, and TabR using recovered official configurations.
5. Report raw neural gain, validation-selected gain, abstention rate, selection
   regret, node-control gap, isospectral gap, and the entire preregistered
   spectral-retention curve rather than the best `tau`.
6. Compare GGPL, numerical PLE, LRLR-triplet, tuned RealMLP, TabR, ModernNCA,
   and the full validation-gated RAPLE system.

## Promotion criteria

Promote to an ICLR submission only if all are true:

- correct geometry beats mass and permuted geometry on at least two independent
  real datasets over repeated seeds;
- Residual-Riesz improves raw neural means on at least 10/16 full-budget
  TabReD dataset–model cells or provides a statistically clearer robustness
  benefit than ordinary RAPLE;
- validation abstains on harmful cases with low test selection regret;
- at least three architecture families benefit without changing the theorem or
  semantic declaration rule; and
- the GGPL/LRLR-triplet/RAPLE/RealMLP/TabR controls leave a nontrivial residual
  contribution.

If these gates fail, retain FieldRiesz as a diagnostic/theory contribution or
an ablation inside RAPLE rather than forcing a standalone performance paper.
