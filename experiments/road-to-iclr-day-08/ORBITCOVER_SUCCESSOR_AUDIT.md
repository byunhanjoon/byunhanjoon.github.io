# OrbitCover successor audit

## Question

Can OrbitCover's same-target coupling result be reframed as a general covariance-optimal coupling theorem that is both useful for tabular prior fitting and substantively different from generic antithetic Monte Carlo?

## Evidence inherited from Days 5--7

No new OrbitCover training was run. This audit uses the already frozen evidence:

- Paired OC2 reduces error strongly when the two evaluations share the same latent target/function.
- Fresh-RNG schema balancing does not reproduce the effect; its paired evaluations do not estimate the same target.
- At convergence, OC2/SRS mean error is approximately `1.002`, so the practical effect is finite-budget rather than an asymptotic advantage.
- Under matched-function comparisons, MLP/ResNet gaps close and FT-Transformer/TabM gaps become small.
- The proposed interaction-order mechanism was weak and did not supply a broader theorem.

These observations support the existing paper's precise, narrow statement. They do not by themselves establish a new general coupling principle.

## Mathematical subtraction

For two unbiased estimators with average \(\bar Z=(Z_1+Z_2)/2\),

\[
\operatorname{Var}(\bar Z)
=\frac14\{\operatorname{Var}(Z_1)+\operatorname{Var}(Z_2)
+2\operatorname{Cov}(Z_1,Z_2)\}.
\]

With fixed marginals, choosing a coupling to minimize estimator variance is therefore exactly choosing a coupling that minimizes covariance. Antithetic variates, countermonotone couplings in one-dimensional special cases, optimal-transport formulations of antithetic sampling, and randomized quasi-Monte Carlo already occupy that general formulation. The 2026 literature identified in `LITERATURE_BOUNDARY.md` further treats optimal antithetic coupling and randomized integration directly.

OrbitCover's valuable content is domain-specific: it constructs a valid same-target pair from orbit structure and demonstrates a finite-compute effect under a carefully matched estimand. Recasting the covariance identity as a new general theorem would overstate novelty.

## Decision

**STOP THIS BRANCH. KEEP CURRENT PAPER SEPARATE.**

No unoccupied general covariance-optimal coupling theorem was found. A successor would need an OrbitCover-specific structural theorem—such as an efficiently constructible optimal coupling class with a nontrivial bound that exploits tabular-prior symmetries—not merely the classical negative-covariance objective.

## Reopen condition

Reopen only if a new result proves all three:

1. a coupling class induced by tabular prior symmetries that is more specific than generic antithetic sampling;
2. an efficiently computable optimizer or approximation guarantee within that class; and
3. a prospective finite-budget improvement under the same-target estimand on multiple architectures.
