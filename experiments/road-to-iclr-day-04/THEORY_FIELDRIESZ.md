# Day 4 theory: fields are function spaces, not feature coordinates

## One-sentence idea

A scalar table field should specify a finite function space together with a
data-mass form and a semantic smoothness form; the neural network should see a
Riesz-normalized rendering of that field, so equivalent encodings induce the
same prior while numerical order, cyclicity, or nominal identity remain
deliberate inductive biases.

This is the earlier **FieldRiesz** idea made concrete on measured numerical
support.  `SupportHeat` was a useful experimental route to it, not a separate
headline contribution.

## 1. A field, its charts, and two bilinear forms

For field `j`, choose a finite-dimensional function space `H_j` and a centered
chart

```text
phi_j(x_j) in R^{d_j},       E_train[phi_j(X_j)] = 0.
```

For a numerical field, `H_j` is the space of continuous piecewise-linear
functions on ordered support nodes.  For an ordinal field it can be the full
function space on a path; for a cyclic field, on a ring; and for a nominal
field no adjacency should be invented.

The chart carries two positive-semidefinite forms:

```text
M_j = E_train[phi_j(X_j) phi_j(X_j)^T]        (mass),
S_j = semantic Dirichlet/stiffness form       (geometry).
```

The proposed Riesz map is

```text
K_j(tau_j) = M_j + tau_j S_j,       tau_j >= 0.
```

`M_j` says which field functions are large under the observed covariate law.
`S_j` says which field functions are rough under admissible field semantics.
They answer different questions and must not be conflated.

## 2. Measured-support finite elements

Let the train-only ordered nodes be

```text
z_1 < ... < z_m.
```

They contain quantile nodes plus repeated-value spikes that clear a fixed
excess-count threshold. This is a deterministic construction rule, not a
statistical significance test. The rule must threshold each excess count before
summing; otherwise float rounding turns many one-count fluctuations into a
false atomic mass. Concretely, for ordered unique levels with counts `n_l`, let
`m_l` be the five-level local median (nearest-value padding at the endpoints),
set

```text
e_l = max(n_l - max(m_l, 1), 0),
```

and retain a spike only when `e_l >= max(2, 5e-4 n_train)`, subject to the
declared node budget. This frozen heuristic is an allocator, not a latent-atom
estimator. Let `b_r(x)` be
the nodal hat functions, so `sum_r b_r(x)=1`.  On the centered rank-`m-1`
subspace,

```text
M = (1/n) sum_i (b(x_i)-b_bar)(b(x_i)-b_bar)^T,

S = sum_{r=1}^{m-1} c_r (e_r-e_{r+1})(e_r-e_{r+1})^T,
c_r = 1 / (z_{r+1}-z_r).
```

The generalized eigenproblem

```text
S v_k = lambda_k M v_k,
v_k^T M v_l = 1{k=l}
```

simultaneously diagonalizes empirical mass and numerical roughness.  Normalize
positive `lambda_k` by their field-wise median to remove affine measurement
units.  The rendered coordinates are

```text
psi_tau(x)
  = [ (1 + tau lambda_k)^(-1/2) v_k^T (b(x)-b_bar) ]_k.
```

The earlier heat rendering replaces `(1+tau lambda)^(-1/2)` by
`exp(-tau lambda/2)`.  The rational Riesz form is the primary version because
its induced quadratic penalty is exactly `M + tau S`.

## 3. Why this is not merely another embedding

For every finite `tau`, `psi_tau` is an invertible chart of the same centered
piecewise-linear function space.  It adds no labels and no representable
functions.  With a Euclidean first-layer penalty in the rendered chart,

```text
||u||_2^2 = a^T (M + tau S) a,
```

where `a` is the coefficient vector of the same field function in the original
chart.  Likewise isotropic initialization of `u` induces a Gaussian function
prior with coefficient covariance `(M+tau S)^(-1)`.

Thus:

- `tau=0` is empirical function-space whitening;
- `tau>0` is a deliberate smoothness prior, not a larger hypothesis class;
- quantile PLE, local PLE, nodal hats, and the generalized eigenbasis are
  implementation charts rather than different semantic claims.

This is the exact bridge from Day 3: ordinary AdamW silently makes an
arbitrary coordinate chart into a prior.  FieldRiesz makes the prior explicit
in field-function space.

This does not make coordinatewise Adam or AdamW trajectory-covariant. Under an
equivalent chart, two Riesz renderings differ by an orthogonal transform, so
isotropic initialization and the Euclidean penalty induce the same function
prior in distribution. Elementwise adaptive moments need not commute with that
rotation. Exact paired trajectory closure requires the transported metric step
in Section 5 or an appropriate block/vector-equivariant optimizer.

## 4. Performance extension: a residual Riesz representer

The workspace's earlier RAPLE study shows that cross-fitted target-response
curves and anchor residuals can improve modern neural backbones on TabReD.  A
direct mathematical extension of FieldRiesz is to turn that empirical signal
into a chart-independent field function.

Let `r_i = y_i - a_{-fold(i)}(x_i)` be a leakage-safe out-of-fold anchor
residual, and define the residual covector

```text
c_j = E_train[phi_j(X_j) r].
```

The **residual Riesz representer** is

```text
g_j = K_j(tau_j)^(-1) c_j,
h_j(x_j) = c_j^T K_j(tau_j)^(-1) phi_j(x_j).
```

Equivalently, `g_j` solves the variational problem

```text
argmin_g  (1/2) g^T K_j g - E_train[r g^T phi_j(X_j)].
```

This is the minimum-`K_j`-norm field function aligned with what the anchor
still misses.  Training rows must receive out-of-fold values; validation and
test rows use a representer fitted on all training data.

Its associated **smooth residual energy** is

```text
E_j(tau) = c_j^T K_j(tau)^(-1)c_j
         = sup_{g != 0} <c_j,g>^2 / (g^T K_j(tau)g).
```

This scalar measures how much anchor residual is expressible by a unit-smooth
field function.  It is also the squared `K_j` norm of the representer.  Unlike
raw coefficient norms, it is chart invariant, making it a natural pre-neural
field diagnostic.  A semantic energy gap

```text
D_j(tau) = E_j,declared(tau) - average_pi E_j,permuted-pi(tau)
```

tests whether the declared adjacency aligns with residual signal more strongly
than information-equivalent false adjacencies.  At `tau=0`, this reduces to
classical projected conditional-residual energy; neither the Rayleigh quotient
nor the quadratic form is a standalone novelty claim.

There is a sharper, strength-free view.  Let `(lambda_jk,v_jk)` be the
generalized eigenpairs of `(S_j,M_j)` and let `q_jk=v_jk^T c_j`.  Then

```text
E_j(tau) = sum_k q_jk^2 / (1 + tau lambda_jk),
(-1)^m d^m E_j(tau)/d tau^m
  = m! sum_k q_jk^2 lambda_jk^m / (1 + tau lambda_jk)^(m+1) >= 0.
```

Residual energy is therefore a completely monotone **semantic spectral
retention curve**.  Its first derivative has the direct form

```text
-E'_j(tau) = g_j(tau)^T S_j g_j(tau).
```

Energy disappears quickly when residual signal lies in modes declared rough,
and slowly when it lies in modes declared smooth.  The normalized curve
`R_j(tau)=E_j(tau)/E_j(0)` removes residual scale.

This observation also repairs the negative control.  Permuting path nodes
preserves the ordinary eigenvalues of the stiffness matrix but generally does
**not** preserve the generalized spectrum relative to empirical mass.  It is
still a useful false-adjacency control, but it cannot isolate mode orientation
from spectral shrinkage.  A harder control works in `M_j`-whitened coordinates:

```text
A_j = M_j^(-1/2) S_j M_j^(-1/2) = V Lambda V^T,
A_j,iso = Q Lambda Q^T,                 Q orthogonal/Haar,
S_j,iso = M_j^(1/2) A_j,iso M_j^(1/2).
```

`S_j,iso` has exactly the same generalized eigenvalues as `S_j` but randomizes
which field functions receive them.  A gap between semantic and
`M`-isospectral retention curves therefore tests whether residual signal is
assigned to the *right modes*, not merely whether one operator has a more
favorable spectrum.  A scale-integrated diagnostic such as

```text
Q_j = integral_[log tau_min]^[log tau_max]
        (R_j,semantic(exp(u)) - E_Q R_j,iso(exp(u))) du
```

avoids selecting one favorable `tau`.  Complete monotonicity and isospectral
rotation are classical linear algebra; the candidate contribution is their
schema-semantic tabular use, cross-fitted residual covector, and preregistered
mechanism protocol.

### 4.1 The Haar-isospectral reference law

The random control has a closed-form reference distribution. Let `r` be the
rank of `M_j`, put the residual covector in mass-whitened coordinates, and
normalize it to unit length. For a Haar rotation `Q`, its coordinates
`U=Q^T q/||q||` are uniform on the sphere and

```text
(U_1^2,...,U_r^2) ~ Dirichlet(1/2,...,1/2).
```

Writing `d_k(tau)=(1+tau lambda_k)^(-1)`, the normalized random-control
retention is

```text
R_Q(tau) = sum_k d_k(tau) U_k^2,
E_Q[R_Q(tau)] = (1/r) sum_k d_k(tau),
Var_Q[R_Q(tau)]
  = 2/[r(r+2)] * [sum_k d_k(tau)^2 - (sum_k d_k(tau))^2/r].
```

The same law applies to a log-strength-integrated statistic after replacing
each `d_k(tau)` by its prespecified grid average. This yields an inexpensive
**semantic orientation score**: compare the observed curve or integrated area
to thousands of exact-spectrum orientations by sampling normalized Gaussian
vectors, with no repeated eigendecompositions. It is invariant to field-chart
changes, residual scale, and the generalized eigenvalue profile.

This is an isospectral *reference law*, not automatically a frequentist null:
the declared schema was not generated by Haar randomization. Its upper-tail
fraction becomes a p-value only under an explicit random-orientation null.
Using the same labels to select fields and prove their semantics is also
circular. In the current audit, the reference score ranks California Latitude
and Longitude first, and King County latitude second, but no field survives a
within-dataset 10% BH screen. The right uses are preregistered diagnostics or
nested selection, not a post-hoc theorem of semantic truth.

### 4.2 Declared field groups and product geometry

Some semantics belong to a group, not a marginal column. Latitude/longitude,
time/location, and dose/duration are examples. For a declared two-field group
`G=(j,k)`, start from the tensor chart

```text
P_G(x_j,x_k) = phi_j(x_j) tensor phi_k(x_k).
```

Marginal centering alone is not enough when `X_j` and `X_k` are dependent:
`P_G` can still contain additive main effects. Let `A_G` be the sample-space
span of the constant and both marginal charts and define the empirical ANOVA
interaction

```text
Phi_G = (I - Pi_A_G) P_G,
M_G = E[Phi_G Phi_G^T],
S_G = S_j tensor M_k + M_j tensor S_k,
K_G = M_G + tau_G S_G.
```

**Proposition (purification commutes with field charts).** Write the training
evaluation matrices row-wise as `P` for the tensor chart and `A` for the
constant-plus-marginal chart. Let `H_A=A A^dagger` and
`Phi=(I-H_A)P`. Under invertible factor charts, there are invertible matrices
`B_G` and `C_G` such that `P'=P B_G^T` and `A'=A C_G^T`. Column spans are
unchanged, hence `H_A'=H_A` and

```text
Phi' = (I-H_A')P' = Phi B_G^T,
A^T Phi = 0.
```

Therefore the purified interaction space is independent of the coordinates
used for its tensor and marginal charts, and its empirical mass transforms by
congruence. A graph form constructed from row differences of `Phi` does too.
This is the finite-sample linear-algebra version of functional-ANOVA
purification; its role here is to make the field-group interface correct, not
to claim a new decomposition.

Crucially, `M_G` is the empirical **joint** mass, not generally `M_j tensor
M_k`; it retains dependence between the coordinates. The Kronecker-sum
stiffness is the discrete product-Laplacian penalty. The residual surface is

```text
c_G = E[Phi_G r_oof],
h_G(x_j,x_k) = c_G^T K_G^dagger Phi_G(x_j,x_k).
```

Under factorwise chart changes, the purified chart changes by the induced
tensor matrix; its mass, any declared bilinear coefficient form, and the
covector transform by congruence, so the representer and energy remain
invariant. Exact `M_G`-isospectral rotation supplies the same mode-orientation
control as in one dimension. The empirical projection, rather than marginal
centering, makes this orthogonal to the observed additive subspace.

The separable `S_G` is only one declared geometry. A second, nonseparable
candidate builds a train-only graph from the declared group metric (haversine
distance for latitude/longitude) and projects its Dirichlet form into the same
chart:

```text
S_G,graph = (1 / |E|) sum_(a,b in E)
  w_ab [Phi_G(x_a)-Phi_G(x_b)][Phi_G(x_a)-Phi_G(x_b)]^T.
```

This respects empirical joint support without pretending that the joint mass
factorizes. Under `Phi'_G=B_G Phi_G`, it still transforms by congruence. Before
choosing a common strength grid, every group stiffness is divided by the
median positive generalized eigenvalue relative to `M_G`; the first product
pilot omitted this joint calibration and is retained only as a labeled legacy
ablation.

There is a further rank boundary. After purification, California's 169-column
chart has empirical mass rank 69, but the semantic product operator has rank
144 while its original isospectral control has rank only 68--69. The apparent
control win is therefore confounded by 75 empirically unseen functional modes,
not merely by mode orientation. King County has rank 144 for empirical mass,
semantic operator, and control, and its semantic control result is negative.
The principled completion is

```text
M_G,rho = (1-rho) M_G,emp + rho M_G,ref,       0 < rho << 1,
M_G,ref = integral Phi_G(x) Phi_G(x)^T d nu_G(x),
K_G,rho = M_G,rho + tau S_G,
```

where `nu_G` is a declared full-support reference measure or quadrature rule,
not Euclidean coordinate ridge. `M_ref` transforms by congruence, makes unseen
function modes measurable, and lets an isospectral rotation preserve the full
finite generalized pencil. This **reference-mass completion** connects the Day
2 seen/unseen boundary to the operator theory.

**Proposition (covariant rank completion).** Let `M_emp` and `M_ref` be
positive semidefinite coefficient forms and let `0<rho<1`. Then

```text
null(M_rho) = null(M_emp) intersect null(M_ref).
```

Consequently, if their only shared null directions are exact chart
redundancies, `M_rho` is positive definite on the represented function space.
Under any invertible chart change `Phi'=B Phi`,

```text
M'_rho = B M_rho B^T,  S'=B S B^T,  c'=B c,
K'_rho = B K_rho B^T,
```

and both `c^T K_rho^dagger Phi(x)` and the generalized spectrum on the quotient
space are unchanged. Hence a Haar rotation in `M_rho`-whitened coordinates is
a full-function-space orientation control.

*Proof.* For positive semidefinite matrices, `v^T M_rho v=0` iff both
nonnegative summands vanish; this gives the null-space identity. Congruence of
the two mass terms follows by substituting `Phi'=B Phi` into their defining
expectations/integrals. The stiffness and residual covector transform the same
way, after which the representer identity follows by inverse congruence on the
quotient. The generalized eigenvalues are likewise invariant under a strict
equivalence of matrix pencils. ∎

The implemented pilot uses tensor trapezoid quadrature on the declared support
nodes and sweeps `rho in {0.001, 0.01, 0.1}`. For California, `M_ref` and every
completed operator have rank 144, and the semantic/control spectra agree to
at most `3.6e-12`. The semantic surface beats the full-space isospectral
control in 7/9, 8/9, and 7/9 cells, but its mean gain is respectively -0.05%,
+0.19%, and -0.02%. Against wrong geometry the corresponding results are 6/9
(+0.06%), 9/9 (+0.07%), and 7/9 (+0.01%). King County has no stable hierarchy.
Thus completion repairs the experiment but also removes the robust semantic
claim: the sign depends on `rho`, which must be nested-selected or integrated
out in a future method.

The favorable `rho=0.01` rotation is itself unstable. Four additional
full-space isospectral orientations give California mean gaps of -0.06%,
-0.02%, +0.04%, and +0.04%, versus +0.19% for the first. Across all 45 stress
comparisons semantic wins 30 (+0.04%); after averaging the five controls within
each unique cell it wins 7/9 (+0.04%). King County is 6/9 and +0.19% against its
within-cell control mean despite failing wrong geometry. Therefore a small
isospectral gap is not specific evidence for the declared product geometry;
both wrong-geometry and repeated-orientation controls are mandatory.

A controlled off-support sanity check confirms the algebra under its own
assumptions. Training inputs deliberately omit `(0.4,0.6)`, while the residual
truth is a smooth sum of two sinusoids in the declared path geometry. Across
200 noise repetitions, empirical mass has rank 21--22/25 and reference mass
completes it to 25/25. In the unseen gap, completed correct geometry beats the
empirical-mass correct estimator in 192/200 repetitions and reduces MSE by
18.3% on average; it beats completed mass in 192/200 and wrong/isospectral
completed controls in 200/200. This is an implementation sanity check in a
data-generating process constructed to favor the declared prior, not benchmark
evidence or a novelty claim.

A selection-free alternative is an operator mixture

```text
h_pi(x) = integral c^T [M_rho + tau S]^dagger Phi(x) d pi(rho),
```

for a declared log-uniform prior `pi` over a fixed `rho` interval. Each
integrand and therefore the mixture is chart invariant. I tested the discrete
mixture over `{0.001,0.01,0.1}`. It retains California's gain versus raw RAPLE
in 8/9 cells (+1.21%) but beats its matched isospectral mixture in only 5/9
(-0.08% mean); King County is 6/9 and +0.02%, effectively a tie. The robust
mixture therefore fails to rescue the semantic claim. Kernel mixtures
themselves are classical and are not claimed as new mathematics.

Validation-only selection offers a different conclusion. Choosing `rho` only
among semantic surfaces yields a near-tie against the matched full-space
control on California (7/9, +0.03%) and a loss on King County (4/9, -0.09%).
Choosing among raw RAPLE, anchor, all three mass completions, and all three
semantic completions wins 8/9 California cells (+1.17% versus raw) and 7/9
King County cells (+0.81%), with 0.19% and 0.20% mean oracle regret. This is a
small-sample deployment lead: it improves prediction without validating the
semantic operator, and it still needs many datasets and nested evaluation.

Tensor-product splines, Kronecker-sum Laplacians, and product finite elements
are classical. Recent scalable tensor-product spline and polynomial-additive
models also occupy efficient higher-order interactions. The candidate tabular
contribution is therefore the narrower composition **schema-declared field
groups + train-only empirical joint mass + cross-fitted anchor-residual
surfaces + backbone transport + exact false-geometry controls**. This is a more
faithful extension of the Day 1--3 multifeature motivation than treating
related coordinates as unrelated marginal paths, but the tensor construction
is not itself a novelty claim.

The first, unprojected and uncalibrated pilot separates those pieces. On California, the product residual
surface beats raw RAPLE in 8/9 MLP/ResNet/TabM cells (+2.12%) and anchor-only in
7/9 (+1.18%), but the wrong product geometry beats the declared one in all 9
cells (-0.53% semantic gain). On chronological King County, it wins 6/9 versus
raw RAPLE but loses by 0.49% on average; it also loses its exact isospectral
control overall. Product interactions remain a performance lead, not evidence
for the current product-Laplacian prior. A knot-resolution audit reinforces
the warning: the raw-performance gain persists at 8, 12, and 16 knots, while
the exact-isospectral control verdict changes from 0/9 to 6/9 to 7/9 wins. The
empirical-ANOVA and calibrated-graph runs are therefore corrective hypotheses,
not a retroactive rescue of that result. With the corrections frozen, the
California product form wins 8/9 cells versus raw RAPLE (+2.63%), 7/9 versus
mass (+0.37%), and 8/9 versus both wrong and finite-mass-spectrum controls
(+0.23% and +0.55%). The haversine graph wins 8/9 versus its finite-support
control but only 5/9
versus mass. On chronological King County, the corrected product is +0.25%
versus raw yet -0.24% versus that control; the graph is -0.03% versus raw
and -0.57% versus its control. Because the product operators can differ in
empirical-null directions, these are not pure full-pencil orientation tests.
Reference-mass completion makes the full-space test exact but its mean
isospectral gap changes sign over `rho={0.001,0.01,0.1}`. This is a cleaner
California performance lead, not robust semantic evidence or an independent
replication. The original finite-support California product gap remains positive
over calibrated strengths 0.3, 1, and 3 (7/9, 8/9, 8/9 finite-support-control wins;
+0.30%, +0.55%, +0.43%), so it is not tied to one isolated `tau`. A validation
choice among raw RAPLE, anchor, mass, and semantic surface wins 8/9 California
cells versus always using raw (+2.38%), but only 5/9 on King County (+0.91%)
with 0.77% mean oracle regret. It is an empirical gate, not the independent-
calibration certificate in Section 4.4.

### 4.3 One operator, four backbone interfaces

The transferable object is the field kernel

```text
k_j(x,z) = phi_j(x)^T K_j^dagger phi_j(z),
d_j(x,z)^2 = k_j(x,x) + k_j(z,z) - 2 k_j(x,z).
```

Both are chart-invariant. This gives concrete, falsifiable adapters rather than
the vague claim that the method "works with any backbone":

- **MLP, ResNet, RealMLP, TabM:** append the scalar cross-fitted `h_j`, or use
  `K_j^(-1/2)phi_j` as a field block. TabM shares the operator across members;
  ensemble diversity does not redefine schema semantics.
- **FT-Transformer:** render one token per field and one token per declared
  group, for example `t_G=e_G+P_G psi_G(x_G)+u_G h_G(x_G)`. The equal-compute
  control replaces only the group scalar/operator, not token count.
- **TabR and ModernNCA:** replace or augment ordinary numerical distance in
  keys/queries with `sum_j d_j(x_j,z_j)^2`. Correct, mass-only, and exact-
  isospectral metrics must retrieve from the same memory so geometry is the
  only intervention.
- **Tabular foundation models:** condition a field token on a compact spectrum
  and domain type, or sample synthetic pretraining functions with covariance
  `k_j`. The latter makes the operator a prior over tasks rather than a wide
  input vector.

Only the flat adapters and a small field-token Transformer pilot are evaluated
on Day 4. The retrieval and foundation-prior versions are registered research
hypotheses, not evidence. Their advantage is conceptual economy: every adapter
uses the same invariant kernel and admits the same mass/wrong/isospectral
controls.

The token pilot illustrates why all three controls remain necessary. A
purified group token beats raw RAPLE in 3/3 California seeds (+3.81%) and 2/3
chronological King County seeds (+1.97%). Yet California loses wrong geometry
in 3/3 (-0.14%), while King wins wrong and finite-spectrum controls in 3/3 but
loses anchor-only on average (-0.44%) with one unstable seed. This supports a
group-token performance interface, not shared semantic causality. The pilot is
also smaller than, and not a recovered official configuration of,
FT-Transformer.

For sparse field screening, in-sample `E_j` must not be confused with held-out
utility. Let `h_j^oof` be cross-fitted. For a direction-preserving,
nonnegative scalar coefficient, define the signed screening score

```text
A_j = sign(<r,h_j^oof>) <r,h_j^oof>^2 / <h_j^oof,h_j^oof>.
```

For positive alignment this is the squared-loss improvement attained by the
optimal scalar coefficient. For negative alignment, the constrained optimum
is the null coefficient and has zero gain; `A_j` deliberately keeps a negative
sign so the direction is rejected rather than silently flipped. `A_j` is
invariant both to a field chart change and to positive rescaling of the
rendered feature. A topology-neutral screen ranks fields by
the `tau=0` mass version of `A_j`, then compares correct and permuted stiffness
on the exact same selected set.  This avoids choosing fields because they make
the declared topology look favorable.

It inherits the chart property exactly.  Under `phi'_j=B_j phi_j`,
`c'_j=B_jc_j` and

```text
g'_j = K'_j^(-1)c'_j = B_j^(-T)g_j,
h'_j(x_j) = h_j(x_j).
```

The energy is unchanged for the same reason:

```text
c_j'^T K_j'^(-1)c_j' = c_j^T K_j^(-1)c_j.
```

This is the cleanest bridge between the new mathematical idea and RAPLE's
existing performance: replace heuristic binned response summaries with one or
more cross-fitted Riesz representers, then test whether semantic stiffness
reduces variance and temporal selection regret.  Classical target encoding,
smoothing, and Riesz representation remain prior art; the candidate novelty is
the chart-covariant, schema-semantic residual construction and its transport
to modern tabular backbones.

A complete direct three-seed panel now exists on California Housing and four
official-split TabReD datasets for MLP, ResNet, and TabM.  Every variant uses
the same RAPLE encoder, out-of-fold LightGBM anchor, residuals, split, seed, and
approximately matched neural budget.  Across 45 seed-cells, semantic Riesz
beats PLE in 30 (+4.05% mean, California-dominated), raw RAPLE in only 17
(-0.15%), anchor-only in 23 (-0.11%), mass in 29 (+0.24%), and one node-
permuted representer in 33 (+0.57%).  At the dataset--model level it wins 5/15
against raw RAPLE and 11/15 against the node control.  It is a selective
geometry signal, not a broad predictive improvement.

The node result is robust on the two positive datasets: over five random
permutations, California/Weather MLP/ResNet correct geometry wins 52/60 cells
(+1.05%), or 11/12 unique cells after within-cell control averaging. The harder
exact `M`-isospectral control preserves the full
generalized spectrum and changes only mode orientation. Across five rotations
and all three backbones, correct geometry wins 80/90 control comparisons
(+0.90%): 25/30 for MLP, 27/30 for ResNet, and 28/30 for TabM. These are not 90
independent fits: each of 18 semantic dataset--model--seed cells is reused
against five randomized controls. Averaging the five controls inside each cell
gives 18/18 semantic wins (+0.90%); the backbone means are +0.91% MLP, +1.39%
ResNet, and +0.41% TabM. This eliminates generalized-spectrum confounding and
shows that mode assignment, not only the eigenvalue profile, affects learning
on this selected two-dataset subset. It is not a confirmatory 18-cell test:
California and Weather were selected after inspection, and the dataset—not a
seed or model—is the independent replication unit.

The effect is not strength-free. At `tau=0.3`, correct geometry wins 8/12
California/Weather MLP/ResNet cells against both node and exact isospectral
controls (+0.68% and +0.53%). At `tau=3`, it wins 8/12 against node controls
but only 5/12 against isospectral controls (+0.34% and +0.27%). These are
robustness probes rather than tuned comparisons: they make nested strength
selection part of the proposed method and prevent treating `tau=1` as a
universal constant.

A predeclared spatial replication is negative/mixed. OpenML King County house
sales is sorted by sale date into 60/20/20 splits, and only Latitude and
Longitude are allowed representers. Over MLP, ResNet, TabM, and three seeds,
semantic Riesz wins 2/9 cells against raw RAPLE (+0.07% mean). Against the
within-cell mean of five node controls it wins 6/9 (-0.27%); against five exact
isospectral controls it wins 7/9 (-0.03%). The Haar reference ranks latitude
second among 17 fields, but longitude eighth and no field survives the 10% BH
reference screen. This is useful falsification: knowing that coordinates are
ordered is not sufficient for predictive gain.

Sparse selection has not rescued the failing datasets.  On 984-field Maps,
the best topology-neutral top-24 OOF screen is essentially tied with raw RAPLE
(+0.09%), while a topology-specific screen fails.  A strict multiplicity
screen abstains to the exact anchor on California, Cooking, Weather, and Maps;
it selects two Delivery fields whose correct topology loses its control.  A BH
screen over approximate OOF z-scores is exploratory rather than a valid finite-
sample FDR guarantee.  It selects zero fields on California, Cooking, and Maps,
six on Delivery, and one on Weather.  It improves anchor-only for both Delivery
backbones but still loses raw RAPLE; it hurts anchor-only for both Weather
backbones.  This separates semantic mechanism from deployable selection and
keeps the residual extension far from a universal rescue.

The semantic-minus-isospectral retention score does recover the California
mechanism locally: it ranks Latitude and Longitude first.  Using only those two
representers beats raw RAPLE in 5/6 MLP/ResNet cells (+2.30%), anchor-only in
4/6 (+0.82%), and ties dense semantic Riesz (3/6, +0.08%).  The corresponding
positive top-8 Weather rule loses both raw RAPLE and anchor-only at the backbone
mean.  Moreover, a set selected by semantic-vs-isospectral gap is biased toward
the semantic operator and cannot be used for its own mechanism test.  It is a
candidate performance localizer; topology-neutral selection or nested
evaluation remains required for a valid semantic comparison.

### 4.4 A certified fallback for additive residual prediction

The failed temporal interventions make fallback part of the method, not an
engineering footnote.  Let `Theta` be a finite, preregistered family containing
the null intervention `h_0=0`, mass, declared stiffness strengths, and
permuted-topology controls.  Reserve an independent calibration set `C`, fit
every `h_theta` without using `C`, and measure its per-row squared-loss
improvement over the
anchor:

```text
Z_i(theta) = r_i^2 - (r_i-h_theta(x_i))^2
           = 2 r_i h_theta(x_i) - h_theta(x_i)^2,
Delta_hat_C(theta) = |C|^(-1) sum_{i in C} Z_i(theta).
```

If `Z_i(theta)` lies in a declared interval `[a_theta,b_theta]` (obtained, for
example, by clipping the anchor residual and intervention), Hoeffding plus a
union bound gives, simultaneously for every `theta`,

```text
Delta(theta) >= LCB(theta)
 = Delta_hat_C(theta)
   - (b_theta-a_theta) sqrt(log(|Theta|/delta)/(2|C|))
```

with probability at least `1-delta`.  Select the non-null candidate with the
largest positive lower bound, otherwise select `h_0`.  On an i.i.d. deployment
distribution this gives a finite-sample no-harm certificate for the *additive
anchor-plus-representer predictor*.  It does not certify a downstream neural
optimizer, and it does not survive arbitrary temporal distribution shift.
Those limitations must be stated rather than hidden.

The same construction yields a stricter semantic certificate: declare geometry
supported only when the correct topology's lower bound exceeds the upper bound
of every preregistered permutation control.  This converts the wrong-geometry
ablation into a decision rule.  Concentration-based model selection is
classical; the possible contribution is its use to gate schema-semantic field
operators and to report abstention and selection regret systematically.

## 5. Chart-covariance proposition

Let an equivalent chart be `phi'_j = B_j phi_j` for invertible `B_j`.  Then

```text
M'_j = B_j M_j B_j^T,
S'_j = B_j S_j B_j^T,
K'_j = B_j K_j B_j^T.
```

For a first-layer block `W_j`, transport it as

```text
W'_j = W_j B_j^{-1}.
```

Three objects are then chart-independent:

```text
W'_j phi'_j = W_j phi_j,
tr(W'_j K'_j W'_j^T) = tr(W_j K_j W_j^T),
W'_j - eta G'_j K'^{-1}_j
  = (W_j - eta G_j K_j^{-1}) B_j^{-1}.
```

The last identity uses `G'_j=G_j B_j^T`.  Therefore matched initialization,
the Riesz penalty, and the corresponding metric-gradient step commute with
every admissible within-field chart change.  They do not claim invariance to
cross-field rotations, which erase tabular schema semantics.

This statement uses an inverse on a nonredundant field chart. With a singular
operator, the Moore--Penrose expression gives the same scalar representer under
congruence when both the residual covector and evaluation covector lie in the
operator range; arbitrary off-range evaluation is not chart-invariant. The
centered marginal hat chart has only the known constant redundancy and stays
inside that quotient. Sparse purified product charts can have many additional
empirical-null modes, which is why Section 4.2 requires reference-mass
completion before making a global off-support covariance claim.

## 6. Decisive hypotheses

1. **Support allocation.** Support-aware nodes improve over quantile nodes
   when repeated states collapse quantiles; the gain should survive matched
   parameter counts and disappear on fields with no threshold-clearing spikes.
2. **Mass normalization.** `tau=0` support whitening should explain any gain
   caused only by conditioning.  A spectral basis at `tau=0` must not beat it
   systematically across seeds, because the two differ only by an orthogonal
   chart.
3. **Semantic stiffness.** A nonzero `tau` earns the geometry claim only if it
   beats `tau=0`, several node-permuted stiffness controls, and several exact
   `M`-isospectral controls at equal function space, dimension, parameters, and
   compute. The whole registered retention curve is reported, not its best
   post-hoc strength.
4. **Architecture transport.** The selected operator improves or safely
   abstains for MLP, ResNet, FT-Transformer, TabM, and eventually TabR; it must
   not be an MLP-only preprocessing accident.
5. **Industrial transfer.** On official TabReD temporal splits, train-only
   mass/support estimation remains useful under drift, or validation selects
   the baseline.  Random-split academic gains alone are insufficient.
6. **Foundation-model path.** For a tabular foundation model, `(M,S)` should
   be supplied as field-token geometry or used in the synthetic-table prior;
   concatenating a wide flat vector is not the claimed transfer mechanism.
7. **Residual representer.** Cross-fitted `h_j` should outperform both RAPLE-
   style unsmoothed response bins and the same representer with `tau=0` or
   permuted `S_j`, especially under sparse support and temporal drift.

## 7. Controls that can kill the claim

- quantile PLE with equal parameters;
- support PLE without normalization;
- per-coordinate standardization;
- exact per-field whitening (`tau=0`);
- permuted-node stiffness;
- exact generalized-spectrum-preserving `M`-isospectral rotations;
- multiple seeds with validation-only selection;
- equal-compute seed/checkpoint ensembles;
- GGPL or another adaptive-knot baseline;
- RAPLE response summaries and the same out-of-fold anchor without Riesz
  representers;
- TabR/ModernNCA where local support memory is a plausible explanation;
- no-change fields and continuously distributed synthetic nulls.

If whitening explains the gains, the result is an engineering note about
support discretization.  If the correct stiffness does not beat the permuted
one, there is no evidence for semantic geometry.  If gains do not transfer
beyond MLP/ResNet or survive TabReD, this is not yet an ICLR method paper.

## 8. What the Day 4 pilots actually say

The controls split the original idea into two effects.

**Mass is real but selective.**  At matched parameter counts, support-aware
`tau=0` normalization improved Adult and Black Friday for MLP, ResNet, TabM,
and a field-token Transformer in the one-seed transport panel.  On HIGGS the
support diagnostic activated no field and returned the exact baseline, which
is the intended structural abstention.  It hurt Churn, House, Microsoft, and
three official TabReD temporal checks.  Full normalization is therefore not a
universal preprocessing recipe.

**Stiffness earned a geometry-specific signal on California Housing.** Over
three paired seed runs with held-out test splits, correct numerical order improved mean test MSE over
quantile PLE by 10.01% for MLP and 5.24% for ResNet.  Mass alone improved by
4.92% and 1.89%; a node-permuted stiffness improved by 5.60% and -0.98%.
Changing only Latitude or Longitude reproduced about a 10% gain, whereas
changing MedInc, HouseAge, or AveBedrms did not.  For those two spatial fields,
the correct order also beat the permuted order.  TabM showed the same ordering
in its one-seed cell: 15.71% for correct stiffness, 13.16% for mass alone, and
8.91% for wrong stiffness.

The field-token Transformer did not reproduce the California result, and all
unconditional support/mass/stiffness interventions tested on Weather, Cooking
Time, and Delivery ETA were worse than the baseline.  The residual extension
later recovers a small repeated Weather gain.  The current outcome is therefore
a mechanism-bearing research direction, not a finished performance paper.  It
needs a conservative validation rule, more repeated datasets with declared
field geometry, and comparisons with GGPL, RealMLP, TabR, and stronger tuned
baselines.

An exploratory, separate-dataset three-seed UCI Bike Sharing test used a chronological 60/20/20
split and changed only the declared Hour field.  Mass normalization improved
mean test loss by 3.69% for MLP and 6.43% for ResNet.  The semantic result did
not replicate: the correct 24-hour ring and a permuted ring were tied for MLP;
the correct ring separated for ResNet, but validation did not reliably choose
it. This strengthens the selective mass claim and leaves California as the
only current geometry-bearing success. Because Bike was added during the same
research loop, it is not a preregistered confirmatory replication.

## 9. Honest novelty boundary

Mass and stiffness matrices, generalized eigenfunctions, Riesz maps,
smoothing penalties, and Laplacian priors are classical.  Recent work also
studies symmetry-aware models and topology-matched regularization.  Numerical
feature embeddings and adaptive PLE breakpoints are established in tabular
deep learning.

The residual extension has an especially strict boundary.  At `tau=0`,
`M^dagger c` is the ordinary Galerkin/least-squares projection of the residual
conditional mean into `H_j`; adding `tau S` is classical smoothness-
regularized regression.  Riesz representer learning is also established in
automatic debiased machine learning and conditional kernel mean estimation.
Neither the closed-form solve nor the word "Riesz" is a novelty claim.

The plausible new composition is narrower:

```text
declared tabular field semantics
  + measured-support finite elements
  + within-field chart covariance
  + one transferable initialization/penalty/update construction
  + broad neural and temporal-benchmark evidence.
```

That is a credible ICLR hypothesis, not yet a novelty certificate.  The wrong-
geometry, whitening, architecture, and TabReD tests decide whether it deserves
promotion.
