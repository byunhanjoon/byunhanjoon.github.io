# Geometry Transfer Law

## 1. Setup and assumptions

Let `S` be one semantic tabular field, `Z` all remaining covariates, and `Y`
the response. A base rule `b(Z)` is trained without a metric-aware encoding of
`S`. Define

\[
R=Y-b(Z),\qquad \mu_s=\mathbb E[R\mid S=s].
\]

Thus the relevant target is the *conditional residual state signal*, not raw
target smoothness. Let `T` and `U` be disjoint observed and cold states. A
target-independent geometry procedure supplies an operator
`A=A_{UT}` and predicts `A\widehat\mu_T` on `U`. Unless stated otherwise:

1. squared loss is used;
2. `b` and `A` are held fixed while the displayed expectation is taken;
3. `\widehat\mu_T=\mu_T+\epsilon`, with
   `E[epsilon]=0` and `Cov(epsilon)=Sigma`;
4. training estimation error is independent of fresh test outcome noise;
5. `Q` is diagonal, nonnegative, and sums to one;
6. `R=\mu_S+\eta`, with `E[eta|S]=0` and finite second moments.

Cross-fitting is used in experiments to make the state residual means less
biased by fitting `b`. Conditional analyses may condition on the fitted base
rule. If `A` or `b` is selected using outcomes, their selection must occur
inside the relevant training fold.

## 2. Theorem 1 — exact geometry-transfer identity

**Statement.** Let the fallback predict residual zero and the geometry rule
predict `A\widehat\mu_T`. Then

\[
\Delta := \mathcal R_{0}-\mathbb E\mathcal R_A
=\|\mu_U\|_Q^2-\|\mu_U-A\mu_T\|_Q^2
-\operatorname{tr}(QA\Sigma A^\top).
\]

Equivalently,

\[
\Delta=2\mu_U^\top QA\mu_T
-\mu_T^\top A^\top QA\mu_T
-\operatorname{tr}(QA\Sigma A^\top).
\]

Define

\[
G_{\rm transfer}=\|\mu_U\|_Q^2-\|\mu_U-A\mu_T\|_Q^2,
\quad C_{\rm noise}=\operatorname{tr}(QA\Sigma A^\top).
\]

Then `Delta = G_transfer - C_noise`; geometry improves expected squared risk
if and only if `G_transfer > C_noise`.

**Proof.** For a cold state `u`, write `g_u=a_u^T(\mu_T+\epsilon)`.
The fallback loss is `(\mu_u+\eta)^2` and geometry loss is
`(\mu_u+\eta-g_u)^2`. Their difference is

\[
2(\mu_u+\eta)g_u-g_u^2.
\]

The cross terms involving `eta` vanish because `E[eta|S]=0` and fresh test
noise is independent of the training estimate. The remaining expectation is

\[
2\mu_u a_u^T\mu_T-\mathbb E[(a_u^T\mu_T+a_u^T\epsilon)^2]
=2\mu_u a_u^T\mu_T-(a_u^T\mu_T)^2-a_u^T\Sigma a_u.
\]

Weighting and summing over `U` gives the second display because
`sum_u q_u a_u^T Sigma a_u = tr(Q A Sigma A^T)`. Expanding
`||mu_U-A mu_T||_Q^2` gives the first display. No test outcome-variance term
remains: `E[eta^2|S=u]` appears in both risks and cancels. ∎

### Geometry Transfer Ratio

When `C_noise>0`, define `GTR=G_transfer/C_noise`. Negative GTR means systematic
mis-transfer; `0<=GTR<1` means real transferable signal is insufficient to pay
for estimation; `GTR=1` is break-even; and `GTR>1` is beneficial in expectation.
The signed `Delta` remains primary because GTR is unstable near zero cost and
its oracle version depends on unknown cold-state signal.

### Realized versus expected gain

This distinction is essential. Conditional on a realized `mu_hat_T`, the
expected cold-test gain is

\[
2\mu_U^TQA\widehat\mu_T-\|A\widehat\mu_T\|_Q^2.
\]

The explicit noise cost appears after averaging over repeated training-state
estimates. If retrospective test state means are substituted for `mu_U`, the
realized MSE difference is algebraically the same quadratic expression (up to
state weighting). Such an oracle scatter is an arithmetic audit, not evidence
that benefit can be predicted before a new state's outcomes are observed.

## 3. Theorem 2 — no metric-only decision rule

**Statement.** Fix a metric, `T`, `U`, every support/coverage/topology statistic,
sample sizes, `Sigma`, and a nonzero operator `A`. There exist two conditional
state signals on this same experimental design for which geometry helps one and
hurts the other.

**Proof.** Choose `v` such that `g=A v != 0`. For the good target set
`mu_T=v` and `mu_U=g`. Then

\[
\Delta_{good}=\|g\|_Q^2-C_{noise},
\]

which is positive after scaling `v` by a sufficiently large constant. For the
bad target keep the identical `mu_T=v` but set `mu_U=-g`. Then

\[
G_{bad}=\|g\|_Q^2-\|-2g\|_Q^2=-3\|g\|_Q^2,
\]

so `Delta_bad<0` for every positive-semidefinite `Sigma`. Scaling changes no
metric-only quantity. If `A=0`, geometry can never improve and the result is
trivial rather than two-sided. Therefore no rule based only on state positions,
support distance, graph topology, cover radius, metric dimension, cardinality,
or sample size can universally determine the sign. ∎

This theorem explains the old support-distance failure: support controls where
information *could* travel, not whether the conditional target effect aligns
with what is transmitted.

## 4. Theorem 3 — symmetric/transductive spectral special case

**Statement.** Let the same `m` states be smoothed by a symmetric operator

\[
H=V\operatorname{diag}(h_1,\ldots,h_m)V^T,
\quad 0\le h_k\le1,
\]

where `V` is orthonormal. Let `mu=Va`, `mu_hat=mu+epsilon`, and
`Cov(epsilon)=sigma^2 I`. With unnormalized Euclidean aggregate risk,

\[
\Delta=\sum_k[(2h_k-h_k^2)a_k^2-\sigma^2h_k^2].
\]

For `h_k>0`, mode `k` is beneficial precisely when

\[
\frac{a_k^2}{\sigma^2}>\frac{h_k}{2-h_k}.
\]

**Proof.** Orthogonal rotation preserves norm and white covariance. Apply
Theorem 1 with `A=H`, then use `V^T H V=diag(h_k)`:
`2 mu^T H mu-mu^T H^2 mu-sigma^2 tr(H^2)`. Expanding coordinatewise gives the
claim; division by positive `sigma^2 h_k(2-h_k)` gives the threshold. ∎

For mean-per-state risk divide the entire sum by `m`. This exact modal result
does not apply to an arbitrary rectangular cold-state operator.

## 5. Theorem 4 — state-held-out risk estimation

Let semantic states `S_1,...,S_n` be iid/exchangeable draws from `P_S`, with
rows conditionally sampled within state. Let `Alg(D)` denote the *entire*
procedure: base fitting, cross-fitted residual construction, state-effect
estimation, data-dependent operator construction, and hyperparameter selection.
For leave-one-state-out define

\[
\widehat R_{LOSO}=n^{-1}\sum_i \widehat R(S_i;Alg(D_{-i})).
\]

**Statement.** Conditional on within-state sample sizes, LOSO is unbiased for
the risk of training the same symmetric algorithm on `n-1` exchangeable states
and applying it to one fresh state. It is not generally unbiased for an
algorithm trained on `n` states. Group K-fold estimates the analogous risk at
training fraction `(K-1)/K`; under uniform stability/risk convergence as the
number of training states grows, it is consistent for the limiting cold-state
risk.

**Proof.** By exchangeability each ordered pair `(D_{-i},S_i)` has the same
distribution as `n-1` iid training states and an independent fresh state.
Taking expectations of the symmetric average proves LOSO unbiasedness for
that training size. K-fold applies the same argument to a uniformly assigned
held-out block, but its training size differs from the full procedure. If the
risk difference between training on `(1-1/K)n` and `n` states converges to zero
and the state-level empirical average obeys a law of large numbers, the K-fold
estimate is consistent. Without exchangeable states (for example a fixed
spatial extrapolation frontier), it estimates the declared split design, not a
universal new-state distribution. ∎

Every fold must retrain `b(Z)`, residual means, any outcome/data-dependent `A`,
and all hyperparameters. Held-state outcomes may score the fold but may not tune
the model subsequently evaluated on that same fold except inside another nested
layer.

## 6. Finite operator library concentration (optional Theorem 5)

Suppose a fixed library of `K` fully trained procedures has independent
state-level gain variables bounded in an interval of width `B`. For `n` held-out
exchangeable states, Hoeffding plus a union bound gives, with probability at
least `1-delta`, simultaneously for all operators,

\[
|\widehat\Delta_k-\Delta_k|
\le B\sqrt{\frac{\log(2K/\delta)}{2n}}.
\]

Thus the effective confirmation sample size is the number of independent
semantic states, not the raw row count. Conditional rows can reduce uncertainty
in each state mean but cannot manufacture independent states. Dependence among
states, unbounded loss, and tuning on the same held-out gains invalidate this
simple bound; source-level claims require source clustering.

## 7. Proposition — MPE is a factorized similarity map

Let normalized similarity coordinates be `w(x) in R^m`. MPE forms `z=wV` and
an immediately following linear stem forms

\[
u=zW+b=w(VW)+b=wB+b.
\]

Hence it introduces no metric information beyond direct normalized similarity.
Only its rank constraint, optimization path, or induced regularization can
differ. A matched numerical check constructs random `w,V,W` and verifies
`(wV)W=w(VW)` to floating-point precision.

## 8. Scope and limitations

- The exact identity is for squared loss (and, with Hilbert norms, vector Brier
  loss), not log loss, accuracy, or AUROC.
- `A` may be rectangular and need not be a contraction. Kernel interpolation
  can amplify noise dramatically.
- A target-independent metric can still be target-irrelevant. Smooth raw `Y`
  is insufficient when `Z` already explains that structure.
- Oracle `mu_U` is unavailable at deployment. Nested state CV estimates future
  procedure gain under exchangeability; it does not reveal the new state's
  actual residual effect.
- Estimated `Sigma` is itself uncertain. The primary empirical estimator is
  the finite-sample variance of cross-fitted residuals divided by state count.
- Spatial and network states are often dependent and nonexchangeable; results
  then apply to the explicitly declared state split rather than iid new states.

## 9. Vector-valued extension

For `mu_s in R^p`, stack state vectors in a matrix `M`; `A` acts on the state
axis. Replacing every squared `Q` norm by
`||M||_{Q,F}^2=tr(M^T Q M)` and `Sigma` by the block covariance gives the same
proof. Multiclass probability residuals therefore admit the identity under
Brier score, subject to the same unbiasedness and covariance assumptions.
