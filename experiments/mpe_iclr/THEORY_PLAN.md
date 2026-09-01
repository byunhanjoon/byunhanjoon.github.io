# MPE ICLR Theory Plan

Frozen prospectively on 2026-08-29. The final `THEORY.md` must contain complete
statements, assumptions, proofs, numerical checks, and failures. No theorem may
claim optimization or statistical generalization of the full neural network.

## Definitions to fix

For a metric or pseudometric space `(X,d)`, distinct training-state landmarks
`L={l_1,...,l_m}`, bandwidth `h>0`, nonnegative kernel `kappa`, and positive
normalizer `Z_d(x)=sum_j kappa(d(x,l_j)/h)`, define

```text
a_j^d(x) = kappa(d(x,l_j)/h)
w_j^d(x) = a_j^d(x) / Z_d(x)
MPE_d(x) = sum_j w_j^d(x) v_j.
```

The primary kernel is Gaussian. Compact triangular and sparse-k variants are
secondary. All theorems explicitly state when positivity of `Z` is needed.

## Theorem 1 — exact chart/relabeling invariance

Prove by direct substitution that a bijection `pi:X->X'`, transported metric
`d'(pi(x),pi(y))=d(x,y)`, and transported landmarks preserve every affinity,
normalizer, and weight. After identifying the semantic token paired with each
transported landmark, `MPE'(pi(x))=MPE(x)` exactly.

Validation: 32 independently generated relabelings on each synthetic discrete
space and 32 on one frozen real field from each available metric family. Check
weights and aligned representations in float64 with required maximum difference
below `1e-7`; also report float32. Predictive codebook intervention uses the
separate eight-real-codebook rule.

## Theorem 2 — partition-of-unity interpolation bound

For `L_f`-Lipschitz `f:X->R^q` and landmark approximations `a_j`, add and
subtract `sum_j w_j f(l_j)`, apply the triangle inequality, nonnegative weights,
and `sum_j w_j=1` to prove

```text
||f_hat(x)-f(x)||
 <= L_f sum_j w_j(x)d(x,l_j)
    + sum_j w_j(x)||a_j-f(l_j)||.
```

If every landmark error is at most epsilon, obtain `L_f R_w(x)+epsilon`.
For compact support with every active landmark within `h`, obtain
`L_f h+epsilon`. Explicitly note that Gaussian kernels have nonzero tails and
do not satisfy the compact statement without a tail term.

Validation: on every synthetic space/target, calculate the exact empirical
Lipschitz constant when finite, landmark error, `R_w`, observed error, and bound
violation. Smooth Lipschitz targets must have no numerical violation beyond
`1e-10`; discontinuous/random targets are marked assumption failures.

## Theorem 3 — linear-head realizability

Let `A in R^(m x q)` contain desired landmark outputs. If token dimension
`D >= rank(A)`, use a rank factorization `A=VH` with
`V in R^(m x D)` and `H in R^(D x q)`. Then the linear head gives
`MPE(x)H=w(x)VH=w(x)A`, the interpolant of Theorem 2. State the sufficient
special case `D>=q`, the role of an optional bias, and the failure when the
required rank exceeds D.

Validation: random matrices at ranks below/equal/above D, checking constructive
factorization error and the predicted failure above rank D.

## Theorem 4 — equality-metric impossibility

For equality distance and any two states absent from the landmark set, every
landmark distance is one; hence their affinity and weight vectors are identical
whenever `Z>0`, and metric-only MPE maps them to the same token. Gaussian always
has positive Z. A triangular kernel with insufficient support can be undefined,
which must be stated rather than hidden.

Validation: all synthetic nominal unseen states plus the three frozen real
nominal controls. Require exactly identical float64 weights up to `1e-12`.

## Theorem 5 — metric perturbation stability

Assume `|d(x,l_j)-d_tilde(x,l_j)|<=delta`, `kappa` is
`L_kappa`-Lipschitz on the relevant scaled distances, both representations use
the same landmarks/tokens/bandwidth, `Z_d(x)>=z0>0`, and `||v_j||<=V`. First
show `sum_j|a_j-a_tilde_j| <= m L_kappa delta/h`. Then use

```text
||w-w_tilde||_1
 <= sum_j |a_j-a_tilde_j|/Z
    + |Z-Z_tilde|/Z
 <= 2m L_kappa delta/(h z0),
```

and conclude

```text
||MPE_d(x)-MPE_dtilde(x)||
 <= 2 V m L_kappa delta/(h z0).
```

The final proof will verify whether one-sided `Z>=z0` suffices under the chosen
algebra (it does for the displayed decomposition) and will also state the
symmetric assumption used by implementation checks. For Gaussian,
`L_kappa=exp(-1/2)` on nonnegative arguments. This is a worst-case stability
bound, not a claim that predictive loss changes monotonically.

Validation: adversarial and random metric perturbations on interval, cycle,
tree, grid, and real fields; report representation change, RHS, bound ratio,
and partial-corruption predictive curves.

## Theorem 6 — coverage and metric complexity

If the landmarks form an `r`-cover, the compact-kernel conclusion follows from
Theorem 2 only when the bandwidth/support rule ensures at least one active
landmark and all active landmarks are within `h`; then error is bounded by
`L_f h+epsilon`, with `h` chosen no smaller than the cover radius. A minimal
`r`-cover has size at most the covering number `N(X,r)` by definition, so the
required landmark budget is governed by metric coverage rather than storage
cardinality.

For Gaussian tails, derive a non-overclaimed diameter/tail bound. If
`diam(X)<=Delta`, a nearest landmark is within r, and `rho>=r`, then

```text
R_w(x) <= rho
  + Delta (m-1) kappa(rho/h) / kappa(r/h),
```

using monotonicity of the Gaussian kernel. Insert this in Theorem 2. Do not
claim an asymptotic rate without doubling/dimension or covering assumptions.

Validation: measure cover radius, raw state count, empirical covering growth,
and error across landmark budgets. Compare their out-of-sample explanatory
correlations without treating tasks from one source as independent.

## Proposition 7 — interval special case

For ordered equally spaced landmarks with gap `Delta`, triangular
`kappa(t)=max(0,1-t)`, and `h=Delta`, prove that only adjacent landmarks have
positive weight for an interior point and that their normalized weights are
the usual barycentric coefficients. Thus MPE is ordinary piecewise-linear
interpolation on an interval, closely related to PLE rather than a new invention.

Validation: compare weights and predictions against an independently coded
piecewise-linear interpolant at 10,001 points; maximum error must be below
`1e-12` in float64.

## Additional architecture proposition

State and prove the exact algebraic relationship between MPE and normalized
Similarity Encoding. If the similarity vector is `w(x)` and a downstream model
begins with a learned linear map, MPE's first two linear maps compose as
`w(x)VB`. With `m=D` and unrestricted rank, a direct similarity stem can realize
the same map. MPE adds a named token bottleneck/parameterization, not new metric
information. This proposition is required for an honest novelty assessment.

## Optional metric-corruption adversary

Include only if a clean construction is completed before final writing. A
candidate uses two true-metric-separated neighborhoods that a corrupted metric
collapses and a signed distance-to-set Lipschitz target. If all assumptions and
the proportional lower bound are not rigorous, mark it omitted; do not force it.

## Synthetic validation matrix

Spaces: interval, cycle, path graph, balanced tree, unbalanced tree, 2-D grid,
random geometric graph, Watts-Strogatz small-world graph, and equality space.
Targets per space: Lipschitz smooth, piecewise smooth, high frequency, localized
bump, discontinuous, random labels, and metric-misaligned. Each result records
whether theorem assumptions hold.

Support gaps are created by deterministic state removal at increasing native
distance. Required plots are absolute error versus `R_w` and MPE advantage over
lookup/PLE versus nearest training support. Metric/state association corruption
is frozen at 0, 5, 10, 25, 50, and 100 percent while state count and distance
matrix shape remain fixed; graph degree/distance distributions are preserved by
state-label permutations.
