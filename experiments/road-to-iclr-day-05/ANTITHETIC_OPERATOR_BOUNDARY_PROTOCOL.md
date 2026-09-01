# Finite-cover versus Gaussian-antithetic operator boundary

Status: post-literature controlled calibration. This is not an empirical
performance comparison because the two methods estimate different objects.

For a centered population of `H=16` vector-valued cover means, enumerate every
unordered without-replacement pack at `K in {2,4,8,16}`. Verify the exact
finite-population pair-covariance coefficient `-1/(H-1)` and mean-risk ratio
`(H-K)/(H-1)` versus `K` independent cover draws.

Separately simulate the jointly Gaussian antithetic construction for the same
`K`, verifying pairwise coefficient `-1/(K-1)` and the per-realization zero-sum
constraint. Use 100,000 replicates and a deterministic seed.

The purpose is to prevent a category error in related work. Partial OrbitCover
packs are negatively dependent finite-population blocks but are not maximally
antithetic `K`-tuples; they close only when `K=H`. Gaussian antithetic CV
zero-sums its perturbations for every chosen `K`, but applies a nonlinear CV
functional to them and targets a different prediction-error estimand.
