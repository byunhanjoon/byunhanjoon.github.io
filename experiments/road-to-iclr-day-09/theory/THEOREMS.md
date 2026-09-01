# Coordinate–marginal factorization: core results

## Theorem T1 — finite symmetrization

Let a finite group `G` act measurably on an episode observation `O=(D_c,X_q)` while
leaving `Y*` unchanged. Let `P_sym = |G|^{-1} sum_g g#P`, and assume the likelihood
kernel is equivariant. Then there exists a version of the posterior predictive
`p_sym(dy|O)` satisfying

`p_sym(A | gO) = p_sym(A | O)`

for every measurable label set `A`, every `g in G`, and `P_sym`-almost every `O`.

### Proof

The symmetrized joint law is invariant: for any `h in G`,
`h#P_sym = |G|^{-1} sum_g (hg)#P = P_sym`, because left multiplication permutes the
finite group. Hence both the joint measure of `(O,Y*)` and the marginal measure of `O`
are invariant. Start from any regular conditional kernel `K(A|O)` and define
`K_bar(A|O)=|G|^{-1} sum_g K(A|gO)`. This kernel is invariant because right
multiplication also permutes `G`. Using invariance of the joint and marginal measures,
integration of `K_bar` over any measurable observation set reproduces the same joint
probability as `K`; therefore `K_bar` is a version of the regular conditional law and is
the required invariant posterior predictive. QED.

This theorem is intentionally finite. It does not assert a normalized Haar measure on
the non-compact family of all increasing bijections.

## Theorem T2 — exact log-risk cost

Suppose conditional entropies are finite. Let `R_log(V)` be optimal expected log loss
when the predictor observes `V`. Then

`R_log(S) - R_log(S,M) = I(Y*; M | S) >= 0`.

### Proof

Under log loss the Bayes predictor is the true conditional law, and its risk equals
conditional entropy. Thus `R_log(S)=H(Y*|S)` and
`R_log(S,M)=H(Y*|S,M)`. Their difference is the definition of conditional mutual
information. Non-negativity follows from its conditional-KL representation. QED.

## Proposition T3 — no universal invariant optimum

Assume a strictly invariant predictor is restricted to the quotient observation `S`.
If two task priors agree on the law of `S` but their full predictive distributions depend
on `M`, and `I(Y*;M|S)>0` under their evaluated mixture, then no `S`-only predictor can
match the full-information Bayes risk. Its minimum excess log risk is exactly
`I(Y*;M|S)`.

### Proof

Every `S`-only predictor has risk at least `H(Y*|S)`, with equality only at the
conditional law given `S`. The full-information optimum has risk `H(Y*|S,M)`. T2 gives
the strictly positive gap. Agreement on the quotient law prevents identifying which
full predictive distribution applies from `S` alone; observing `M` resolves predictive
variation on a positive-measure set. QED.

### Mixture-prior corollary

For a mixture `P_lambda=(1-lambda)P_0+lambda P_1`, the regret of the best invariant
predictor relative to the full-information predictor is
`I_{P_lambda}(Y*;M|S)`. This is an exact expression, not a uniform positive lower bound:
it may vanish at particular mixture weights or when the component predictions coincide.

## Proposition T4 — exact information calibration of PriorDial

Let `C` be uniform on `K >= 2` mechanism families. Conditional on `C=c`, set
`W=pi(c)` with probability `rho`; otherwise draw `W` independently and uniformly from
the `K` warp families, where `pi` is a bijection. Then both marginals remain uniform and

`I(C;W) = a log(K a) + (K-1)b log(K b)`,

where `a = rho + (1-rho)/K`, `b = (1-rho)/K`, and the final term is interpreted as zero
when `rho=1`. The information ranges from zero at `rho=0` to `log K` at `rho=1`, is
strictly increasing for `rho>0`, and is convex. Thus `rho` is a coupling-probability
coordinate, not a linear information coordinate.

### Proof

The joint probability is

`P(C=c,W=w) = K^-1 [rho 1{w=pi(c)} + (1-rho)/K]`.

Summing over either coordinate gives a uniform marginal. There are `K` matched cells,
each with joint mass `a/K`, and `K(K-1)` unmatched cells, each with mass `b/K`.
Substitution into `sum p(c,w) log[p(c,w)/(p(c)p(w))]` yields the formula. At the
endpoints it reduces to zero and `log K`. For `0 < rho < 1`, differentiation gives

`dI/drho = (K-1)/K log(a/b) > 0`,

and a second differentiation is positive. QED.

The finite experiment scheduler separately balances the coupled and near-product
portions. Its empirical contingency table can differ slightly from this population law
when the independent repetitions do not complete every cyclic offset; empirical MI is
therefore reported alongside the exact population calibration.

## Proposition T5 — metadata identifiability does not imply routing utility

There exist binary task families for which the family variable is observed perfectly
and is genuinely predictive of the label, yet routing to the nominally matched expert
has strictly higher log risk than a family-independent fixed mixture.

### Construction and proof

Let `C` be uniform on `{-1,+1}` and

`P(Y=1 | C=c) = 1/2 + delta c`,

where `0 < delta < 1/2`. Thus `C` is perfectly identifiable and
`I(Y;C)>0`. Define two family-indexed experts by

`q_c(Y=1) = 1/2 + a c`,

with `delta < a < 1/2`. The matched router observes `C` and selects `q_C`. Its expected
log risk is

`R_match = -(1/2+delta) log(1/2+a) -(1/2-delta) log(1/2-a)`.

The equal fixed mixture of `q_-1` and `q_+1` predicts `1/2` and has risk `log 2`.
For example, `delta=0.1` and `a=0.49` give
`R_match = -0.6 log 0.99 - 0.4 log 0.01 > log 2`. Hence even perfectly recovered,
label-informative family metadata can be an unsafe routing target when the associated
expert is overconfident or otherwise misspecified. If `a=delta`, the matched expert is
Bayes-optimal, showing that the failure is calibration/alignment rather than an
information deficit. QED.

This is an existence result, not a claim that every family router fails. It formalizes
the distinction exposed by the classification experiments.

## Proposition T6 — a mixture may beat the best individual expert

Let `ell(y,p)` be convex in prediction `p`; binary log loss on probabilities and squared
error satisfy this condition. For predictions `p_1,...,p_K` and simplex weights `w`,

`L(sum_j w_j p_j) <= sum_j w_j L(p_j)`,

where `L` is any empirical or expected average of `ell`. Consequently,

`min_{w in simplex} L(sum_j w_j p_j) <= min_j L(p_j)`.

### Proof

Apply Jensen's inequality pointwise in every query prediction and average. Each simplex
vertex selects one individual expert, so minimizing over the full simplex cannot be
worse than minimizing over its vertices. QED.

Therefore the reported `best_individual_oracle` is not an oracle over mixtures. A soft
router can exceed 100% of fixed-to-best-individual headroom without an arithmetic or
leakage error.

## Proposition T7 — deterministic harm bound for prediction shrinkage

Let `L` be an empirical or expected loss convex in predictions, let `p_0` be a fixed
predictor, `p_1` an adaptive predictor, and

`p_lambda = (1-lambda) p_0 + lambda p_1`, `0 <= lambda <= 1`.

Then

`L(p_lambda) - L(p_0) <= lambda [L(p_1) - L(p_0)]`,

or equivalently the gain satisfies

`L(p_0) - L(p_lambda) >= lambda [L(p_0) - L(p_1)]`.

### Proof

Convexity gives
`L(p_lambda) <= (1-lambda)L(p_0) + lambda L(p_1)`.
Subtract `L(p_0)` and rearrange. QED.

Thus if full adaptation is harmful by `h`, a lambda-shrunk prediction is harmed by at
most `lambda h` on the same evaluation distribution. If full adaptation is beneficial,
every positive shrinkage step retains at least a lambda fraction of that gain. Curvature
can make an intermediate predictor better than both endpoints, as on the exploratory
real classification curve. This standard convexity consequence supports the safety
interpretation of shrinkage; it does not bound worst-decile loss, high-loss event rates,
or distribution shift, and is not claimed as a new theorem.
