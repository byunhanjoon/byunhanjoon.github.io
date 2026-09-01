# THEORY — RETRIEVAL RISK GEOMETRY

## Setup and assumptions

Fix a query covariate $x$, candidate covariates
$X=(X_1,\ldots,X_n)$, and weights $w=w(x,X)$ measurable with respect to
those covariates. Assume $w_i\geq0$ and $\mathbf 1^T w=1$ unless stated
otherwise. Write

\[
Y_i=m_i+\varepsilon_i,
\quad m_i=\mathbb E[Y_i\mid X_i],
\quad \mathbb E[\varepsilon_i\mid X]=0,
\]

and $m_x=\mathbb E[Y\mid X=x]$. The default scalar-result assumption is
conditional independence of candidate noises, with
$\operatorname{Var}(\varepsilon_i\mid X)=\sigma_i^2$. A fresh outcome at
the query is not the estimand: if prediction risk against a fresh $Y_x$ is
used, $\operatorname{Var}(Y_x\mid x)$ is an additional irreducible term
common to all candidate-selection methods.

## Theorem A1 — exact retrieval risk identity (proved)

Let $d_i=m_i-m_x$, $\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,
\sigma_n^2)$, and $\widehat m(x)=\sum_i w_iY_i$. Then

\[
\boxed{
\mathbb E[(\widehat m(x)-m_x)^2\mid X]
=(w^Td)^2+w^T\Sigma w.}
\]

The first term is aggregate neighbor mismatch (transfer bias); the second is
propagated candidate noise. Notice that mismatch is the square of the weighted
*signed* discrepancy, not generally $\sum_iw_i(m_i-m_x)^2$. Opposite
mismatches can cancel when several neighbors are combined.

### Proof

Substitute $Y_i=m_i+\varepsilon_i$:

\[
\widehat m-m_x=w^Td+w^T\varepsilon.
\]

Conditional expansion gives

\[
(w^Td)^2+2(w^Td)\mathbb E[w^T\varepsilon\mid X]
+\mathbb E[(w^T\varepsilon)^2\mid X].
\]

The middle term is zero. Conditional independence makes the last term
$\sum_iw_i^2\sigma_i^2=w^T\Sigma w$.

### Correlated-noise extension

Independence is not essential. With conditional covariance
$C=\operatorname{Cov}(\varepsilon\mid X)$, replace $\Sigma$ by $C$.
Candidate duplication, shared annotators, grouped measurements, or repeated
patients can therefore make covariance—not just individual uncertainty—part
of retrieval risk.

## Theorem A2 — oracle optimal weights (proved, with singular boundary)

Define $H=dd^T+\Sigma\succeq0$. Without nonnegativity,

\[
\min_w w^THw\quad\text{s.t.}\quad\mathbf1^Tw=1.
\]

If $H\succ0$, the unique solution and value are

\[
\boxed{w^*=\frac{H^{-1}\mathbf1}{\mathbf1^TH^{-1}\mathbf1}},
\qquad R^*=\frac1{\mathbf1^TH^{-1}\mathbf1}.}
\]

### Proof

The Lagrangian $w^THw-\lambda(\mathbf1^Tw-1)$ gives
$2Hw=\lambda\mathbf1$. Solving and imposing the affine constraint gives the
displayed formula. Positive definiteness gives uniqueness.

### Singular $H$

The often-quoted blind replacement $H^{-1}\mapsto H^+$ is incomplete.

- If $\mathbf1\in\operatorname{range}(H)$, then
  $w_0=H^+\mathbf1/(\mathbf1^TH^+\mathbf1)$ is a minimizer. Every minimizer
  is $w_0+z$ with $z\in\ker(H)$ and $\mathbf1^Tz=0$, and the minimum is
  $1/(\mathbf1^TH^+\mathbf1)$.
- If $\mathbf1\notin\operatorname{range}(H)$, its projection onto
  $\ker(H)$ is nonzero. A normalized null vector is feasible and has zero
  risk, so the pseudoinverse expression is not the optimizer in general.

With $w\geq0$, the problem is the convex quadratic program

\[
\boxed{\min_w w^THw\quad\text{s.t.}\quad\mathbf1^Tw=1,\ w\geq0.}
\]

The simplex is compact, so a solution exists even when (H) is singular.

## Corollary A3 — one-neighbor risk (proved)

For $w=e_i$,

\[
\boxed{r_i(x)=(m_i-m_x)^2+\sigma_i^2.}
\]

Thus the best single observed target is not generally attached to the closest
row in feature space. It is the candidate with low conditional-target mismatch
and low candidate noise. This statement concerns reuse of the *observed label*;
if a candidate contributes a denoised prediction rather than $Y_i$, its
relevant variance is different.

## Proposition A3b — one-neighbor diagnostics upper-bound aggregate risk

For normalized nonnegative aggregation weights, define the weighted mean of
the individual one-neighbor risks

\[
\bar R_1=\sum_i w_i(d_i^2+\sigma_i^2).
\]

The exact multi-neighbor risk from A1 is

\[
R_{\rm agg}=(\sum_iw_id_i)^2+\sum_iw_i^2\sigma_i^2.
\]

Their gap is

\[
\boxed{
\bar R_1-R_{\rm agg}
=\operatorname{Var}_w(d)+\sum_iw_i(1-w_i)\sigma_i^2\geq0.}
\]

### Proof

The mismatch difference is the weighted-variance identity
$\sum_iw_id_i^2-(\sum_iw_id_i)^2=\operatorname{Var}_w(d)$.  The noise
difference is
$\sum_iw_i\sigma_i^2-\sum_iw_i^2\sigma_i^2
=\sum_iw_i(1-w_i)\sigma_i^2$.  Both terms are nonnegative on the simplex.

### Consequence

Mean top-$k$ one-neighbor risk is a conservative upper bound, not an exact
mechanism metric for a multi-neighbor predictor.  It discards beneficial
signed-mismatch cancellation and ignores the squared-weight dilution of
candidate noise.  It can therefore decrease substantially while actual
prediction risk stays flat or worsens.  The prospective Day-8 follow-up
exhibited exactly this decoupling for TabR and ModernNCA; see
`PROSPECTIVE_RISK_RESULTS.md`.  Any subsequent diagnostic must use the actual
aggregation weights and signed mismatch rather than averaging A3 over a
selected set.

The permanently post-hoc correction did optimize this aggregate expression
directly. It improved the real panel, but a mismatch-only ablation matched the
full objective whereas reliability-only weighting did not. ModernNCA also
failed the frozen S3 transfer gate. Thus A3b correctly diagnoses the earlier
measurement error, but neither A3b nor the corrective QP establishes a novel
candidate-reliability method; see `POSTHOC_AGGREGATION_RESULTS.md`.

## Corollary A4 — local signal metric (proved as a first-order expansion)

If scalar $m$ is differentiable and $x_i=x+\delta_i$, then

\[
m(x_i)-m(x)=\nabla m(x)^T\delta_i+o(\lVert\delta_i\rVert),
\]

and hence

\[
(m_i-m_x)^2
=\delta_i^T G_{\rm signal}(x)\delta_i+o(\lVert\delta_i\rVert^2),
\qquad
\boxed{G_{\rm signal}(x)=\nabla m(x)\nabla m(x)^T.}
\]

For vector conditional means $m:\mathbb R^p\to\mathbb R^q$, squared
Euclidean mismatch gives

\[
\boxed{G_{\rm signal}(x)=J_m(x)^TJ_m(x).}
\]

For class-probability vectors this is a local Brier geometry. Cross-entropy
instead induces a probability-dependent Fisher weighting and is not covered by
the unweighted formula.

This PSD matrix is rank one for scalar regression, so it is generally a
degenerate local seminorm, not a full Riemannian metric. Candidate uncertainty
remains an additive location cost and cannot, in general, be represented by a
symmetric pairwise distance.

## Theorem A5 — induced metric of a TabR-like key encoder (proved locally)

Let the differentiable row representation be $\Phi(x)$ and the key be
$k(x)=W_K\Phi(x)$. For $x'=x+\delta$,

\[
\lVert k(x')-k(x)\rVert^2
=\delta^T G_\theta(x)\delta+o(\lVert\delta\rVert^2),
\]

where

\[
\boxed{G_\theta(x)=J_\Phi(x)^TW_K^TW_KJ_\Phi(x).}
\]

### Proof

Differentiability gives
$k(x+\delta)-k(x)=W_KJ_\Phi(x)\delta+o(\lVert\delta\rVert)$.
Squaring yields the result.

If $\Phi(x)=Bx+c$, $G_\theta=B^TW_K^TW_KB$ is constant: retrieval is a
global Mahalanobis pseudometric. A nonlinear $\Phi$ permits a spatially
varying pullback metric field. It is genuinely Riemannian only where the
Jacobian-projection product has full column rank.

## Derived insight and falsifiable mechanism

The risk law does **not** imply that every useful retrieval metric should equal
$G_{\rm signal}$. It predicts two separable tests:

1. learned distance should rank candidates similarly to
   $(m_i-m_x)^2+\sigma_i^2$, or a cross-fitted proxy;
2. when a model improves, its retrieved top-k set should have lower mismatch
   and/or candidate-noise cost.

An important impossibility follows: ordinary symmetric distance alone cannot
fully encode candidate-only heteroscedastic noise, because
$\sigma_i^2$ is directed (candidate reliability) rather than a symmetric
query–candidate discrepancy. A theory-derived architecture may therefore need
separate *compatibility* and *reliability* terms rather than one geometric
distance. This is more specific than “uncertainty matters.”

## Numerical verification

`run_day8.py theory` checks:

- A1 over 48 random independent-noise systems (30,000 Monte Carlo draws each);
- the correlated covariance extension (60,000 draws);
- A2 against an equality-constrained numerical optimizer;
- the nonnegative simplex QP;
- the singular nullspace zero-risk boundary;
- the symbolic two-candidate expansion;
- A3 against 200,000 one-neighbor noise draws;
- A4 against a finite-difference local signal mismatch on S1; and
- A5 against a finite key-distance expansion for a nonlinear map.

The machine-readable outcomes are in `raw/theory/checks.csv` and the required
summary table. Every frozen algebra check passed in the completed run.

## Failure boundaries and relationship to prior work

- A1 is conditional bias–variance algebra; A2 is a standard quadratic program;
  A4 is a Taylor pullback; A5 is a standard Jacobian pullback. None is claimed
  as a new result in metric learning or differential geometry.
- Classical NCA/LMNN already learns target-useful global metrics; PLML and many
  successors already learn smooth local metrics.
- Kernel and nearest-neighbor regression theory already studies local
  bias/variance and bandwidth.
- PLE, PLR, B-splines, learned knots, GGPL, TabR, ModernNCA, target-aligned
  retrieval, and structural tabular attention are occupied.
- The possible contribution is the joint tabular-retrieval interpretation,
  candidate-noise diagnostic, representation/retrieval branch separation, and
  evidence that the law predicts modern models. If those empirical links fail,
  the theory is explanatory notation rather than an ICLR direction.
