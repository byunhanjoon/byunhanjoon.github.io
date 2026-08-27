# Day 4 candidate matrix

The bar is not "interesting."  A survivor needs a crisp estimand, a theorem or
formal property, a mechanism-specific diagnostic, broad paired gains, and a
credible path into both ordinary backbones and tabular foundation models.

| Candidate | Mathematical object | Collision / falsification | Current decision |
| --- | --- | --- | --- |
| Unstructured schema averaging as a universal method | Orbit over equivalent schemas | Closest work already studies representation sensitivity; Day 3 equal-compute orbit averaging lost by 0.521%. | Reject as a generic performance method. Retain the structured quotient-risk audit and audit-guided action as the primary OrbitANOVA paper. |
| Measured-support FieldRiesz + residual representer | Per-field finite-element space with `K=M+tau S`, cross-fitted `h(x)=c^T K^-1 phi(x)`, and a semantic spectral-retention curve | Mass/stiffness, smoothing, complete monotonicity, and isospectral rotations are classical; `tau=0`, node-permuted, exact `M`-isospectral, full RAPLE, and GGPL controls are mandatory. | **Primary technical direction within Day 4; secondary in the ICLR portfolio; not yet a paper.** The full direct panel ties/loses raw RAPLE (17/45 wins), while node and isospectral controls support a selective California/Weather geometry mechanism. Selection remains unsolved. |
| Global common/innovation coordinates | Orthogonal PCA split `x=UU^T x+(I-UU^T)x` | Day 4 pilot: mixed MLP/ResNet result; minimal FT-Transformer worsened on Diamond, HIGGS, and TabReD Weather. | Falsified as a universal method. Keep as an ablation or optional continuous block. |
| Support-adaptive local residual memory | Leverage/effective-support gate plus neighbor residuals | Direct collision with TabR, ModernNCA, and LoCalPFN. | Reject as headline. |
| Function-prior TabM ensemble | Distribution over initial functions | Randomized prior functions, function-space empirical Bayes, TabM extensions, and hyperparameter ensembles crowd the claim. | Reject unless a mixed-measure prior creates a unique prediction. |
| Boosted TabM | Sequential residual learners in one shared network | Classical boosting and recent neural residual boosting crowd the idea; no link to Day 1--3 mechanism. | Reject. |
| Measure-adaptive direct sum | Atomic/non-atomic decomposition of each numerical marginal, with smooth and shrunk atomic fields | Classical semiparametric/random-effects models are a serious collision; the expanded support diagnostic selected only Adult and Black Friday. | Demoted: useful motivation and atom diagnostic, not a broad standalone method. |

## Promoted hypothesis: measured-support FieldRiesz

For each scalar field, construct a train-only piecewise-linear function space
on quantile nodes plus support spikes that clear a fixed excess-count
threshold. This is a construction rule, not a significance test. In any
centered chart `phi_j`, define

```text
M_j = E_train[phi_j(X_j) phi_j(X_j)^T],
S_j = sum_edges c_e (e_left-e_right)(e_left-e_right)^T,
K_j(tau_j) = M_j + tau_j S_j.
```

The generalized eigenproblem

```text
S_j v_jk = lambda_jk M_j v_jk,
v_jk^T M_j v_jl = 1{k=l}
```

yields the rendering

```text
psi_j,tau(x) = [(1 + tau_j lambda_jk)^(-1/2)
                 v_jk^T(phi_j(x)-mean_j)]_k.
```

At `tau=0`, this is empirical `L2(mu_hat_j)` normalization.  At `tau>0`,
ordinary Euclidean first-layer weight decay becomes the field-function penalty
`a_j^T(M_j+tau_j S_j)a_j`.  Every finite-`tau` representation spans the same
piecewise-linear function space: performance differences test the function
prior, not extra information.

Under an invertible within-field chart change `phi'_j=B_j phi_j`, both forms
transform by congruence and `K'_j=B_j K_j B_j^T`.  Transporting the first-layer
block as `W'_j=W_jB_j^{-1}` preserves the function, its Riesz penalty, and a
`G_jK_j^{-1}` metric-gradient update exactly.  This is the clean mathematical
bridge from Day 3's basis sensitivity.

### Current falsification evidence

- Exact per-field whitening explains the apparent advantage of the unweighted
  support spectral basis on Adult; the two are effectively tied.  The paper
  must not claim that Laplacian eigenvectors alone are superior features.
- Parameter-matched support/mass coordinates improve Adult for MLP, ResNet,
  TabM, and a field-token Transformer, and improve Black Friday in the same
  four architecture families in the current single-seed transport panel.
- On California Housing, the correct ordered Riesz operator beats quantile PLE
  by 10.01% mean test loss over three MLP seeds and 5.24% over three ResNet
  seeds.  The node-permuted stiffness improves by 5.60% for MLP and worsens by
  0.98% for ResNet; the geometry-specific remainder is therefore nonzero in
  this pilot.
- Single-field interventions localize the California gain to Latitude and
  Longitude (about 10% each); MedInc, HouseAge, and AveBedrms do not improve.
- Churn and official temporal splits for TabReD Weather, Cooking Time, and
  Delivery ETA reject the intervention on validation and are worse on test.  A
  deployable method needs conservative strength selection or a reliable
  fallback; unconditional whitening is falsified.
- On California, TabM benefits strongly while the current field-token
  Transformer does not.  Architecture-universal improvement is not supported.
- On a three-seed chronological UCI Bike Sharing test, changing only Hour gives
  mass gains for MLP (+3.69%) and ResNet (+6.43%), but the correct 24-hour ring
  does not consistently beat a permuted ring.  This is a mass replication and
  a failed cyclic-geometry replication.
- In the complete shared-anchor residual panel (five datasets, MLP/ResNet/TabM,
  three seeds), correct Riesz wins only 17/45 cells against raw RAPLE (-0.15%
  mean) but 33/45 against one node-permuted operator (+0.57%).  The current
  contribution is a mechanism hypothesis, not broad benchmark improvement.
- On the positive California/Weather MLP/ResNet subset, correct geometry wins
  52/60 comparisons across five node permutations (+1.05%), or 11/12 unique
  cells after within-cell averaging. Across five exact
  `M`-isospectral rotations and all three backbones it wins 80/90 (+0.90%),
  eliminating the generalized-spectrum confound. The result is 25/30 for MLP,
  27/30 for ResNet, and 28/30 for TabM. Since the semantic fits are reused,
  averaging the five controls within each unique cell gives 18/18 wins. Because
  these two datasets were selected after inspection, this is mechanism stress
  evidence rather than a confirmatory test; independent datasets are the unit.
- Strength is a real degree of freedom: the exact-control win count is 8/12 at
  `tau=0.3` but 5/12 at `tau=3` on California/Weather MLP/ResNet. Nested
  strength selection remains required.
- Predeclared King County Latitude/Longitude on a chronological split fails to
  replicate California: 2/9 wins versus raw RAPLE (+0.07%), 6/9 versus the
  five-node-control mean (-0.27%), and 7/9 versus the five-isospectral-control
  mean (-0.03%).
- The first tensor-product Latitude/Longitude residual surface improves
  California over raw RAPLE in 8/9 cells (+2.12%), but wrong geometry wins
  9/9. Auditing it finds that marginal centering is not a pure interaction
  under dependence and that the group stiffness lacked joint spectral
  calibration. With empirical-ANOVA purification and calibration, California
  improves to 8/9 wins versus raw, wrong, and finite-support-isospectral controls
  (+2.63%, +0.23%, +0.55%). The frozen King County transfer is only +0.25%
  versus raw and -0.24% versus that control. Retain purified field-group
  interactions as a stronger performance/mechanism lead, not replicated
  semantic evidence. The purified California mass has rank 69 in a 169-column
  chart, while the semantic operator has rank 144 and the original control only
  68--69, so the +0.55% control gap is confounded. A chart-covariant reference
  mass completes all 144 functional modes, but the resulting mean semantic gap
  is -0.05%, +0.19%, and -0.02% at `rho=0.001,0.01,0.1`. Retain completion as
  an audit contribution; a selection-free three-rho mixture also loses its
  matched isospectral mixture on average (-0.08%). Reject a robust
  product-geometry claim.
- A one-token version beats raw RAPLE in 3/3 California seeds (+3.81%) and 2/3
  King County seeds (+1.97%), but its semantic controls conflict across the two
  datasets. Retain the group-token interface; do not call it FT-Transformer
  validation or architecture-independent mechanism evidence.
- Top-k, OOF-alignment, strict multiplicity, and approximate BH screens have
  not produced a reliable neural selector.  In particular, strict screening
  abstains on four datasets and selects two Delivery fields where the semantic
  control fails.
- Semantic-minus-isospectral retention does rank California Latitude and
  Longitude first; their two-field model beats raw RAPLE in 5/6 MLP/ResNet
  cells (+2.30%) and ties dense Riesz.  The same top-8 rule fails Weather, and
  topology-specific selection cannot be used as an unbiased mechanism test.

### Decisive next hypotheses

- **H1 (mass):** `tau=0` beats both raw PLE and global/all-field whitening only
  when the field support diagnostic activates; inactive fields are exactly the
  baseline.
- **H2 (geometry):** for declared ordered fields, selected `tau>0` beats
  `tau=0` and a permuted-node stiffness across seeds.  A tie means there is no
  evidence for semantic order.
- **H3 (selection):** a validation-margin rule chooses the baseline on Churn,
  Weather, and no-change controls while retaining Adult/Black Friday/
  California gains without material selection regret.
- **H4 (transport):** the same field operator improves MLP, ResNet, TabM, and
  token-native Transformer cells; TabR is a separate retrieval interaction,
  not assumed to inherit the result.
- **H5 (temporal):** at least one official TabReD temporal dataset improves.
  Plain rendering fails Weather, Cooking Time, and Delivery ETA.  Residual-
  Riesz is positive beyond anchor-only on a three-seed Weather pilot and beats
  raw RAPLE for MLP, but not ResNet; recovered official configurations and the
  validation-gated RAPLE hybrid are still required.

The full derivation, covariance proposition, and kill controls are in
`THEORY_FIELDRIESZ.md`.

## Demoted hypothesis: MeasureLift

For numerical feature `X_j` with marginal law `mu_j`, define its unique
atomic/non-atomic split

```text
mu_j^a = sum_{z: mu_j({z}) > 0} mu_j({z}) delta_z,
mu_j^c = mu_j - mu_j^a.
```

The corresponding square-integrable field has the orthogonal direct sum

```text
L2(mu_j) = L2(mu_j^a) direct-sum L2(mu_j^c).
```

The first model family to test is

```text
f(x) = B_theta( smooth_j(x_j), atom_j(x_j) for j=1,...,d ),

atom_j(z) = 1{z in A_j} * [ n_j(z) / (n_j(z) + alpha_j) ] * e_{j,z}.
```

`smooth_j` is a lossless PLE/quantile coordinate.  `A_j` and the shrinkage are
estimated from training data only.  The atom embedding is zero for unseen
values, so it cannot hallucinate a memorized state off support.  A cross-fitted
residual-energy statistic decides whether a field's atomic block is enabled.

This first expression is deliberately simple.  A stronger version should use
an empirical-Bayes penalty

```text
Omega_atom = sum_j sum_{z in A_j} (alpha_j / n_j(z)) ||e_{j,z}||^2,
```

or its posterior-mean equivalent, rather than freezing a target residual as an
input.  That makes the method task-trained and architecture-compatible while
retaining Day 2's frequency law.

## Formal claims worth proving

1. **Orthogonality.** The atomic and nonatomic subspaces have zero inner
   product under `mu_j`; their approximation errors add under squared loss.
2. **Safe fallback.** Unseen values and fields rejected by the train-only
   diagnostic receive exactly the baseline smooth representation.
3. **Empirical atom recovery.** Under a threshold sequence `tau_n -> 0` and
   `n tau_n -> infinity`, population atoms above the threshold are eventually
   retained, while exact collisions from a nonatomic law vanish almost surely.
   Rounding must be modeled separately because it intentionally creates atoms.
4. **Shrinkage risk.** In a Gaussian random-effect model, the posterior factor
   `n_z/(n_z+alpha)` has lower Bayes risk than an unshrunk per-state estimate;
   its gain increases with repeat support and between-state residual variance.
5. **No universal-improvement claim.** If the atomic residual energy is zero,
   the oracle method is the baseline.  The contribution must include a
   leakage-safe selector and selection-regret accounting.

## Predeclared hypotheses

- **H1:** Cross-fitted atomic residual energy and repeated-support coverage
  predict the gain from the atomic block across datasets.
- **H2:** The largest gains occur for mixed numerical fields (nontrivial atom
  mass plus a high-cardinality remainder), not for purely categorical or
  nearly continuous columns.
- **H3:** Frequency shrinkage beats unregularized identity embeddings on rare
  atoms and maps unseen values to the smooth branch.
- **H4:** A selected MeasureLift block improves PLE for MLP, ResNet, and TabM;
  a token-native implementation is required for FT-Transformer.
- **H5:** On TabReD temporal splits, atoms whose effects drift should be
  downweighted by validation evidence; support alone is insufficient.

## Kill criteria

Stop calling this an ICLR method if any of these holds after the decisive
pilot:

- the selector still chooses only Adult-like census data;
- gains disappear outside MLP or under parameter-matched controls;
- a plain categorical cast or TabR gives the same gain at equal compute;
- a classical random-effect/GAM baseline subsumes both the formulation and the
  result; or
- validation-only selection cannot beat always-PILE/PLE after accounting for
  selection regret.
