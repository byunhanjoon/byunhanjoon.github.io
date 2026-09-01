# Method

The implementation separates discovery from execution. A fixed-capacity chain
or DAG selects protected numerical operators and earlier values, then compiles
to a deterministic executable graph with pruning, common-subexpression reuse,
and an affine output. Random straight-through Gumbel discovery is retained as a
failed control; the successful short-task results use a disclosed complete
chain search, while depth scaling uses bounded beam search.

Typed libraries add exact category predicates, ordinal thresholds/ranks, and
bounded or periodic datetime transforms. Regime experiments route rows to
separate compiled experts. A neural residual predicts an additive correction
with scalar or per-row gates and penalties on contribution and gate use. Each
module is evaluated separately, and failed modules are excluded rather than
silently combined into a favorable full model.
