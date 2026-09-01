# Residual Geometry Trust — theory sketch

## 1. The object

Let `b(Z)` be a fixed tabular base predictor, `R=Y-b(Z)`, and let a declared
metric on semantic field `S` induce a fixed cold-state transfer operator `A`.
From observed states `T`, estimate residual state effects

`mu_hat_T = mu_T + epsilon`, with `E epsilon = 0` and covariance `Sigma`.

For unseen states `U`, use the shrinkable residual expert

`b(Z) + lambda A mu_hat_T`, where `lambda in [0,1]`.

Write `g=A mu_T`, use a diagonal state-weight matrix `Q`, and define

`c = mu_U^T Q g`, `d = ||g||_Q^2`, and
`n = tr(Q A Sigma A^T)`.

## 2. Proposition — exact trust parabola

Relative to the zero-residual fallback, expected squared-risk gain is

`Delta(lambda) = 2 lambda c - lambda^2 (d+n)`.

If `d+n>0`, the best constrained trust is

`lambda_star = clip(c/(d+n), 0, 1)`.

If `d+n=0`, every trust value is equivalent. The oracle gain is nonnegative
because `lambda=0` is feasible.

### Proof

Substitute `lambda A(mu_T+epsilon)` into the Geometry Transfer identity. The
linear cross term is `2 lambda c`; squared transferred signal and propagated
estimation variance scale as `lambda^2 d` and `lambda^2 n`. The resulting
quadratic is concave. Its unconstrained stationary point is `c/(d+n)`, and
projection onto `[0,1]` gives the constrained optimum.

This proposition is elementary shrinkage algebra. It is not itself a novelty
claim. Its role is to expose the correct learning target: not “is the metric
valid?” but “does its conditional residual signal justify its propagated
error, and by how much?”

## 3. Multi-operator extension

For operator predictions collected as columns of `G`, a weight vector `w`
has gain

`Delta(w) = 2 c^T w - w^T (D+N) w`,

where `c=G^T Q mu_U`, `D=G^T Q G`, and `N` is the covariance of operator
estimation errors. Trust selection becomes a constrained convex quadratic
program over a simplex augmented with the zero expert. This unifies fallback,
operator choice, shrinkage, and mixtures without asserting that every supplied
geometry is useful.

## 4. Paper-level theorem program

The possible ICLR contribution needs more than Proposition 1:

1. a state-level cross-fitting theorem for the selected trust/operator under
   exchangeable semantic states;
2. an excess-risk bound depending on the number of states, operator-library
   complexity, covariance-estimation error, and selection margin;
3. a pessimistic lower-confidence trust rule with a high-probability no-harm
   guarantee relative to the neural fallback;
4. a Brier/vector extension and a locally quadratic proper-loss extension;
5. a robustness theorem for nonexchangeable state shifts, using an uncertainty
   set over target alignment rather than pretending ordinary state CV is valid.

## 5. Neural realization

The geometry module should be a zero-initialized residual adapter attached to a
strong neural tabular backbone. The scalar or simplex trust must be learned
only from training-state holdouts. This preserves an exact fallback at zero and
separates the scientific object (conditional value of geometry) from a
particular encoding such as MPE, similarity coordinates, Nyström features, or
graph embeddings.

## 6. Latent-bias Bayes rule for prior-fitted networks

Let `H` index whether a supplied inductive bias generated the task, let `C` be
the labeled context, and let

`m_h(C) = E[Y_Q | C, H=h]` and `p(C) = P(H=1 | C)`.

Under squared loss, the Bayes prediction is

`m(C) = p(C)m_1(C) + (1-p(C))m_0(C)`.

When the irrelevant-geometry regime has zero conditional transfer,
`m_0(C)=0`, this becomes the Day-7 target

`posterior trust × conditional geometry transfer`.

### Proposition — routing regret is disagreement-weighted calibration error

For any soft routing probability `q(C)` and predictor

`m_q(C) = q(C)m_1(C) + (1-q(C))m_0(C)`,

the conditional excess squared risk over the Bayes predictor is exactly

`(q(C)-p(C))^2 ||m_1(C)-m_0(C)||^2`.

### Proof

Conditional squared-risk regret of any prediction `f(C)` is
`||f(C)-E[Y_Q|C]||^2`. Substituting `m_q` and `m` leaves
`(q-p)(m_1-m_0)`; squaring gives the result.

The proposition explains why regime AUROC is not the final objective. A router
may be uncertain or even misclassify regimes where the experts make nearly the
same prediction, with negligible predictive cost. Conversely, small posterior
errors matter greatly where expert disagreement is large. The natural
calibration diagnostic weights `(q-p)^2` by squared expert disagreement.

### Prior-shift identity

If only the regime mixture changes from pretraining prior `pi` to deployment
prior `pi'`, while `P(C|H)` is fixed, posterior odds obey

`logit p_{pi'}(C) = logit p_pi(C) + logit(pi') - logit(pi)`.

Thus task-mixture shift has an exact one-scalar correction when `pi'` is known.
When it is unknown, the residual regret is an identifiable prior-estimation
problem rather than generic representation failure. The learned-PFN experiment
tests the harder uncorrected case and exhibits the predicted under-trust when
smooth tasks become more common.

## 7. Specification-robust optional bias

For a predeclared finite base set `B`, define

`V_robust(a) = min_{b in B} Delta_b(a)`.

Suppose simultaneous lower bounds `L_b(a)` satisfy
`P(for all b,a: Delta_b(a) >= L_b(a)) >= 1-alpha`. If an operator is deployed
only when `max_a min_b L_b(a) > 0`, then on the simultaneous event its gain is
positive for every declared base. This is an immediate union-bound theorem,
but the Day-7 nested replay reveals the statistical price: the rule retained
only ACS and failed its actionability gate. It is therefore a useful safety
ablation, not the promoted lead.
