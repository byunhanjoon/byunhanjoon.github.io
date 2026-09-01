# Related Work

## Arithmetic extrapolation

[Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) explicitly
target systematic numerical generalization with specialized differentiable
units. [Extrapolation and learning equations](https://arxiv.org/abs/1610.02995)
introduces equation-learning networks with analytic unary and product units.
TabALU's surviving result is narrower: once an operator graph is known, a
non-learned executor preserves it more reliably than approximating each node.
The experiments do not show superior general graph discovery.

## Symbolic regression and program induction

[Deep Symbolic Regression](https://arxiv.org/abs/1912.04871) searches expression
spaces with risk-seeking policy gradients. [SRBench](https://arxiv.org/abs/2107.14351)
compares modern symbolic-regression systems at scale, and
[PySR](https://arxiv.org/abs/2305.01582) provides an evolutionary symbolic
regression system with explicit complexity control. The complete search and
beam search here are experimental controls, not competitive replacements for
these systems. The missing SRBench comparison remains a publication blocker
for any symbolic-discovery claim.

## Modern tabular learning

[FT-Transformer](https://arxiv.org/abs/2106.11959),
[TabPFN](https://arxiv.org/abs/2207.01848),
[TabR](https://arxiv.org/abs/2307.14338), and
[TabM](https://arxiv.org/abs/2410.24210) are relevant neural baselines for a
broad benchmark. The small general pilot is not a substitute for those
comparisons. It uses tree ensembles, MLPs, and linear models only and therefore
supports no state-of-the-art claim.

## Temporal tabular evaluation

[TabReD](https://arxiv.org/abs/2406.19380) emphasizes realistic temporal splits
and rich real-world tables. Its official downloader requires authenticated
Kaggle access, which was not assumed. The source-pinned UCI Bike Sharing pilot
is a smaller comparable temporal falsification; it is not equivalent to a full
TabReD evaluation.
