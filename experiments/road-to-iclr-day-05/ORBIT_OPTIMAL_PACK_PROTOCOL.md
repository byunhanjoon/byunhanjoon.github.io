# Frozen protocol: covariance-optimal symmetry-orbit pack law

Status: frozen after observing the graph-versus-resolution component tradeoff
and before solving the optimization.

## Candidate laws

Generate 32,768 valid four-cover templates from the fixed sequential disjoint
graph sampler and add every four-coset subset of the mixed resolvable design.
Deduplicate templates by their 64-cell union. A deployable law first samples a
template, then applies independent uniform level permutations to all four
factors. This orbit symmetrization gives exact uniform cell marginals and an
isotropic covariance multiplier on every product-ANOVA subspace.

For every template, compute the five surviving triple/four-way covariance
multipliers exactly from ANOVA projection traces. Solve a linear program over
template probabilities that minimizes the largest coefficient normalized by
the sequential graph sampler's coefficient. No real prediction tensor enters
the optimizer.

## Evaluation

Report optimizer support, exact cell marginals, all five coefficients, and the
implied covariance risk on every stored full-product candidate. Compare with
the graph coefficients and their already frozen 99% Monte Carlo intervals.

## Frozen gate

The strict gate passes only if:

1. every optimized coefficient is below the graph coefficient's **99% lower
   endpoint**;
2. the optimized covariance risk is strictly lower on all 23 stored
   full-product candidates; and
3. the LP law has exact uniform marginals and uses only mutually disjoint
   strength-2 four-packs.

If it fails, retain the Pareto frontier: that is evidence of a structural
triple/four-way tradeoff rather than evidence for a tuned new sampler.
