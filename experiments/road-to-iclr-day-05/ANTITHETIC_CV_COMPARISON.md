# OrbitCover versus 2026 antithetic Gaussian cross-validation

This is a claim-separation memo, not a performance table. The methods do not
estimate the same object, so numerical ranking without a new common decision
problem would be misleading.

| Axis | OrbitCover | Antithetic Gaussian CV |
|---|---|---|
| Randomized object | Complete trained predictions indexed by a finite product of exact pipeline nuisances and model seeds | Gaussian perturbations of a response vector or asymptotically normal sufficient statistic |
| Marginal law | Declared uniform categorical nuisance product | Normal with declared variance |
| Dependence | Randomized OA cancellation, disjoint cover graphs, or resolvable finite-population packs | Exchangeable equicorrelation at `-1/(K-1)` and a zero-sum perturbation tuple |
| Primary estimand | Held-out Brier/MSE of the uniform nuisance-quotient predictor | Prediction error of an estimator under an independent response copy |
| Bias statement | Exact finite-budget unbiasedness for quadratic independent-block cross-scores | Common marginal determines bias; bias vanishes as perturbation level tends to zero under the stated regime |
| Variance statement | Exact finite covariance operator, resolved by product-fANOVA interaction subspaces | Reducible-variance rates and jointly Gaussian minimax result in the paper's normal-means function class |
| Smoothness | None for the primary quadratic identities | Smoothness/moment assumptions for bounded reducible variance; separate piecewise-smooth rates |
| Closure | Exact when a finite resolution exhausts every nuisance cell | Perturbations zero-sum at every `K`, but the nonlinear CV average need not be constant |
| Defensible novelty | The finite exact-pipeline nuisance estimand and OA/fANOVA/independent-cross-score composition | Antithetic prediction-error randomization and its optimality theory |

The closest sources are [Liu, Panigrahi, and Soloff (JRSS-B,
2026)](https://doi.org/10.1093/jrsssb/qkag073) and [Chattopadhyay, Liu, and
Panigrahi (arXiv, August 2026)](https://arxiv.org/abs/2608.08089). They rule out
selling negative dependence, antithetic risk estimation, or general
optimality as OrbitCover contributions.

Proposition 35 supplies the precise operator boundary. For a resolution of
`H` finite covers, partial without-replacement packs have off-diagonal
coefficient `-1/(H-1)`; a zero-sum `K`-tuple has `-1/(K-1)`. They coincide
only at full resolution `K=H`. The accompanying calibration verifies both
laws. OrbitCover's empirical advantage before closure therefore comes from
interaction-aware structure in the sampled prediction field, not from
attaining the Gaussian theorem's maximal `K`-antithetic law.
