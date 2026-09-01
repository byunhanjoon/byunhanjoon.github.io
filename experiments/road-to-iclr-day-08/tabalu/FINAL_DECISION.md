# Final Component Decision

The proposed full TabALU architecture is a **NO-GO**. The experiments support a
narrow exact-execution result, not the original general tabular learner.

## Retain

- Protected deterministic execution and compilation. It is exact through depth
  8 with oracle graphs and causally outperforms learned neural arithmetic under
  magnitude shift.
- Short-chain constrained discovery, explicitly limited to depth 1–2/3 matched
  families.
- Bounded categorical, ordinal, and periodic datetime operators as a candidate
  library. Their support is synthetic and provisional.

## Exclude from the claimed core

- Random differentiable program discovery: collapsed to shortcuts.
- Learned operand correction: failed its noise/clean-data gate.
- Deep program induction: depth-8 beam discovery failed.
- Context-conditioned coefficients: failed temporal extrapolation.
- Real season routing: catastrophically extrapolated elapsed-time trends.
- Unguarded neural residual: helped IID mixtures but destroyed 4× safety.

## Evidence boundary

The strongest supported statement is:

> When a short arithmetic graph is known or recovered by constrained search,
> exact execution preserves that computation under magnitude extrapolation,
> whereas neural approximations of the same primitives do not.

The evidence does **not** support a claim of general differentiable program
induction, real temporal robustness, broad tabular superiority, or a safe neural
fallback. The first real temporal gate failed, and linear/logistic regression
won all three small general-pilot datasets.

The appropriate research artifact is therefore a transparent execution-versus-
discovery study with negative results, or a redesign centered on search and
shift-aware safety—not a submission claiming the full architecture described in
the original roadmap.
