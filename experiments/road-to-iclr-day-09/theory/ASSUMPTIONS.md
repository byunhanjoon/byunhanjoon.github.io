# Formal assumptions and claim types

## Common setup

`T` is a latent task, `D_c` a labeled context, `X_q` query covariates, and
`Y*` a query label. All variables take values in standard Borel spaces, so regular
conditional probabilities exist. A finite group `G` acts measurably on numerical
coordinates of context and query and leaves labels unchanged. Matched action means the
same group element acts on every context and query row of an episode.

`S=q(D_c,X_q)` is a measurable maximal-or-chosen invariant containing empirical
order, ties, missingness and declared invariant metadata. `M` denotes the remaining
observed coordinate-dependent information. The pair `(S,M)` is assumed sufficient for
the full observation used in each stated comparison; no claim is made that the chosen
finite implementation is a mathematically maximal invariant.

Log loss uses natural logarithms. Conditional entropies and mutual information are
assumed finite. Predictors may be arbitrary measurable conditional distributions; model
class and optimization error are separate empirical quantities.

## T1 assumptions

The base task law `P` has a likelihood kernel equivariant under `G`. The symmetrized law
is `P_sym = |G|^{-1} sum_g g#P`. The finite-group theorem does not claim existence of a
Haar probability for the full non-compact group of increasing real bijections. Sampled
finite transform banks are experimental approximations, not applications of a compact-
group theorem.

## T2/T3 assumptions

The Bayes action under log loss is the true regular conditional distribution. `S` is a
measurable function of `(S,M)`. The exact risk gap applies to Bayes risk; learned-model
gaps additionally contain approximation, estimation and optimization error.

## T4 assumptions

Mechanism and warp families have the same finite cardinality `K`, `pi` is a bijection,
the mechanism marginal is uniform, and the uncoupled warp draw is independent and
uniform. T4 describes the ideal population dial. Finite balanced schedules approximate
that law and retain exact uniform marginals, but their joint-cell counts can differ by
one in the uncoupled portion.

## Claim taxonomy

- **Theorem T1:** finite-group symmetrization admits an invariant posterior predictive.
- **Theorem T2:** the Bayes log-risk cost of discarding `M` is `I(Y*;M|S)`.
- **Proposition T3:** any strictly `S`-only predictor pays that positive Bayes regret
  when the conditional mutual information is positive.
- **Proposition T4:** PriorDial's ideal coupling has an exact monotone, nonlinear mutual-
  information calibration while preserving both marginals.
- **Empirical hypothesis:** PriorDial makes the useful part of `M` increase with rho.
- **Empirical analogy:** a learned gate approximates posterior adaptation between
  information regimes. It is not itself a Bayes theorem.
