# Negative Results

## Uninitialized differentiable discovery collapse — smoke v1, 2026-08-31

The first four-task/two-seed pipeline run used straight-through Gumbel selectors
from random initialization. Compiled TabALU extreme-OOD NRMSE was 8.45 times the
MLP NRMSE; soft TabALU IID NRMSE was 3.43 times the MLP NRMSE; no exact program
was recovered. A fixed-program diagnostic showed that random differentiable
selection failed even on addition and usually selected a single-feature affine
shortcut. Gumbel recovered one multiplication case but failed a two-node
multiply/divide case. This variant is rejected as the Phase-A optimizer.

The smoke generator also admitted a repeated-operand divide expression that was
nearly constant on OOD shells, making normalized error unstable. The generator
now forbids using the current chain value as its own right operand and requires
nontrivial variance at every evaluation multiplier.

The replacement is an explicitly disclosed exhaustive chain-program warm start
followed by the differentiable selector. It isolates the execution hypothesis
but narrows any induction claim: a later general-DAG search ablation remains
mandatory. The original failed artifacts are retained under
`results/phase_a_smoke_v1/` after the next smoke run archives them.

## Conservative operand inference — Phase C, 2026-08-31

Bounded correction and confidence-gated reconstruction improved IID NRMSE by
only 7% at 0.20-SD measurement noise, below the frozen 15% threshold. Both
damaged exact clean prediction to roughly 0.113 NRMSE. The unrestricted encoder
performed similarly and also harmed 8× extrapolation. H2 is rejected for the
current independent-operand setting, and the component is excluded from the
main architecture.

## Context-conditioned coefficients — Phase E, 2026-08-31

An MLP mapping time to executable-program scale and bias achieved future NRMSE
0.1385 versus 0.00639 for discrete regime coefficients, a 21.7× gap. It is
rejected as too flexible and unstable under temporal extrapolation. Shared
discrete structure also failed to beat independent regime programs, so no
sample-efficiency advantage is claimed from this panel.

## Unguarded neural residual under magnitude shift — Phase G, 2026-08-31

The penalized and adaptive residuals produced the desired IID continuum, but
failed the secondary 4× evaluation. At α=1, the pure executable program scored
0.287 NRMSE by safely omitting the bounded non-symbolic term; adaptive, scalar,
and unpenalized residual variants scored 1.587, 2.356, and 3.851. A contribution
penalty does not stop a learned local function from extrapolating aggressively.
The unguarded residual is excluded from the extrapolating core pending a
shift-aware shutdown rule.

## Season routing on real temporal data — UCI Bike Sharing, 2026-08-31

The season-routed typed program scored 0.590 IID NRMSE but 9.779 on all of 2012,
versus 0.602–0.626 for tree ensembles. It selected unconstrained elapsed and
elapsed-squared terms separately within narrow 2011 seasonal windows; exact
execution then amplified those invalid empirical trends. The router fails all
three real-data gates and is excluded. This also demonstrates that exact
execution is only as trustworthy as the discovered computation.

## Deep program discovery — depth scaling, 2026-08-31

Width-128 beam discovery increased from 0.00044 OOD NRMSE at depth 2 to 1.027
at depth 8; functional recovery fell from 93.3% to 6.7%. Oracle execution stayed
exact at every depth. The deep general-program induction claim is rejected;
exact execution does not solve combinatorial search.
