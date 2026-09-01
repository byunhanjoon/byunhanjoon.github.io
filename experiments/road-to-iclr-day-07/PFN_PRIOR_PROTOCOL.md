# Frozen analytic protocol — latent structured-field prior

Status: **FROZEN BEFORE SIMULATION OUTCOMES**

Freeze date: 2026-08-30 (Asia/Seoul).

## Question

If a tabular foundation model is pretrained on a mixture of tasks where a
declared field geometry is sometimes predictive and sometimes irrelevant, what
Bayes rule should it amortize? How fragile is that rule to a changed mixture at
deployment?

## Generative family

- 32 semantic states on a cycle;
- 20 context states and 12 query states, sampled per task;
- latent regime `H` is `smooth` or `unstructured`;
- smooth state effect: zero-mean Gaussian with a cycle heat-kernel covariance;
- unstructured state effect: zero-mean independent Gaussian;
- context observation: state effect plus iid Gaussian estimation noise;
- assumed smooth prior probability: 0.5;
- true smooth probabilities: 0.1, 0.5, and 0.9;
- heat scales: 0.3, 1.0, and 3.0;
- noise standard deviations: 0.1, 0.3, and 1.0;
- 5,000 independent tasks per cell, seed 20260830.

## Fixed predictors

1. `zero`: ignore the supplied geometry;
2. `always_smooth`: Gaussian conditional mean under the smooth prior;
3. `hard_route`: smooth conditional mean iff posterior smooth probability is
   above 0.5;
4. `bayes_mixture`: posterior smooth probability times the smooth conditional
   mean;
5. `regime_oracle`: knows the latent regime, but not query outcomes.

The posterior uses the assumed 0.5 mixture even when the true deployment
mixture is shifted. Query effects never enter routing.

## Checks and reading rule

- At true prior 0.5, `bayes_mixture` must have no larger Monte Carlo mean MSE
  than zero, always-smooth, or hard routing, up to two Monte Carlo standard
  errors. This validates the implementation of the Bayes rule; it is not a new
  theorem.
- Report posterior regime AUROC and calibration.
- At true priors 0.1 and 0.9, report regret from the mismatched assumed prior.
  Do not hide cells where amortized prior mismatch makes a simpler rule win.
- The experiment can motivate a PFN target and prior-shift stress test. It
  cannot establish transformer learnability or real-table utility.

