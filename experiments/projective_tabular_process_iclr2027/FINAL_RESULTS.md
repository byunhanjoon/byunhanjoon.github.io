# Final experimental findings

## Bottom line

The mathematical construction succeeds, but the predeclared performance claim does not. `ProjTabICL` is a valid context-conditional stochastic process whose marginals are exactly those of TabICLv2 and whose predictions are closed under arbitrary linear aggregates. Across the complete 35-dataset OpenML-CTR23 benchmark, however, its learned off-diagonal covariance does not improve proper scores robustly enough to pass the frozen success gates.

This is therefore not evidence for a new state-of-the-art tabular predictor. The strongest paper contribution is instead the distinction between a positive-semidefinite covariance produced for one query batch and a genuinely projective process, together with an exact construction, proofs, and a broad falsification study.

## Frozen evaluation

- 35 OpenML-CTR23 regression tasks, all included.
- 630 primary evaluation episodes: three official folds, two context draws, and context sizes 16, 64, and 256.
- 48 query rows per episode, organized into six disjoint groups.
- Six primary aggregate families: subset means, normalized totals, pair differences, group contrasts, dense signed aggregates, and dense positive aggregates.
- 151,200 scored primary cells after combining episodes and aggregates.
- Six separate development datasets for all tuning; no evaluation dataset was used for model selection.
- Three held-out application datasets and 81 additional episodes: FreMTPL claim count, KDD17 stock return, and bike demand.

The protocol and configuration were frozen before the full evaluation:

- Protocol SHA-256: `a19db1969070fdce89e16c8e60d976a9c49973fc1510373f30a651c48bd4ec89`
- Configuration SHA-256: `4e74dd70c4479418944e43a89626dc90f8a27452d1677e069eec9830452fa1ea`

## Primary hypothesis test

All effects below are `TabICLv2 diagonal score - ProjTabICL score`, so positive values favor the projective covariance.

| Criterion | Result | Frozen requirement | Decision |
|---|---:|---:|---|
| Mean Gaussian-NLL advantage | +0.5267 nat | > 0 | Pass in isolation |
| Dataset NLL wins | 20/35 | at least 21/35 | Fail |
| Paired randomization test | p = 0.0970 | p < 0.05 | Fail |
| Mean CRPS advantage | -0.000624 | > 0 | Fail |
| CRPS dataset wins | 7/35 | descriptive | 28 losses |
| Fixed marginal identity | exactly 0 error | <= 1e-10 | Pass |
| Restriction/permutation audit | exactly 0 error | <= 1e-5 | Pass |
| PSD/symmetry audit | minimum eigenvalue 9.57e-7 | nonnegative within tolerance | Pass |

The mean NLL result is not robust. One Solar Flare result contributes an effect of about +17.98 nat; removing the largest absolute dataset effect reduces the mean advantage to +0.0133 nat. The median dataset effect is +0.00144 and the 10% trimmed mean is +0.00170. The dataset-level bootstrap interval for the untrimmed mean is `[0.00024, 1.56739]`, but the predeclared win-count and randomization criteria still fail.

CRPS gives the clearer negative result: the mean advantage is -0.000624 with a dataset bootstrap interval of `[-0.000900, -0.000374]`, and the projective method loses on 28 of 35 datasets.

## Benchmark comparisons

Distributional results are dataset-macro averages over the six aggregate families. Mean NLL can be dominated by rare variance-collapse failures, so both proper scores and dataset ranks matter.

| Method | NLL | CRPS | MSE | 90% coverage | NLL rank | CRPS rank |
|---|---:|---:|---:|---:|---:|---:|
| ProjTabICL | 7.162 | 0.3495 | 0.5697 | 0.916 | **3.11** | 4.14 |
| TabICLv2 diagonal | 7.689 | **0.3488** | 0.5697 | 0.912 | 3.23 | **3.54** |
| TabPFN-3 diagonal | 2.33e5 | 0.3517 | **0.5093** | 0.886 | 4.40 | 4.03 |
| TabPFN-2.5 diagonal | 2.33e5 | 0.3759 | 0.5244 | 0.858 | 5.03 | 5.34 |
| exact Matérn-3/2 GP | 3.509 | 0.3552 | 0.5786 | 0.866 | 4.14 | 3.83 |
| exact RBF GP | 3.544 | 0.3606 | 0.5902 | 0.860 | 5.00 | 4.91 |
| bootstrap CatBoost process | **3.296** | 0.3701 | 0.5950 | 0.904 | 4.94 | 4.91 |
| Bayesian linear process | 1672.410 | 0.3766 | 0.6060 | 0.851 | 6.14 | 5.29 |

For ordinary point prediction, where the proposed covariance cannot change the shared marginal mean, the nRMSE ordering is:

| Method | nRMSE |
|---|---:|
| TabPFN-3 | **0.6883** |
| TabPFN-2.5 | 0.7045 |
| TabDPT-Turbo 1.2 | 0.7133 |
| TabICLv2 / ProjTabICL marginal | 0.7200 |
| exact Matérn-3/2 GP | 0.7472 |
| bootstrap CatBoost | 0.7569 |
| exact RBF GP | 0.7579 |
| Bayesian linear | 0.7723 |

TabDPT is reported only for point prediction because its public interface does not expose a predictive covariance. TabPFN-3's diagonal distributional comparison is explicitly secondary: the frozen protocol allowed it as a point baseline, and variance support became available only after the official checkpoint was acquired.

## Mechanism controls

All controls share exactly the same TabICLv2 means and marginal variances, isolating off-diagonal covariance quality.

| Covariance mechanism | NLL | CRPS |
|---|---:|---:|
| raw-feature RBF | **6.7490** | 0.3500 |
| shuffled learned features | 7.1254 | 0.3496 |
| learned ProjTabICL head | 7.1618 | **0.3495** |
| hidden-state cosine | 7.1642 | 0.3499 |
| diagonal | 7.6885 | 0.3488 |

The learned head does not separate convincingly from a shuffled-feature control, and a simple raw-feature RBF has lower mean NLL. Thus the experiment does not establish that the frozen TabICLv2 representation contains a particularly useful dependency geometry.

## Projectivity discovery

The first implementation evaluated all query rows together, as is standard for tabular foundation models. It produced a PSD covariance for each batch but failed the defining restriction test: removing other query rows changed a retained row's mean by up to 0.00591, variance by 0.00924, hidden representation by 0.0711, and covariance by 0.00924.

The corrected implementation fits the context once and evaluates each query row separately. This makes the mean, scale, and feature map functions only of `(context, x)`, restoring restriction and permutation identities exactly. The price is substantial: singleton TabICLv2 inference is 30.64 times slower than batched inference in the timing audit, with a median primary episode time of 1.638 seconds.

## Application panel

The three application datasets do not rescue the claim. The projective covariance improves NLL on only one of three datasets; the mean effect (+13.109) is driven by FreMTPL, while the median effect is -0.00617. It loses CRPS on all three datasets, with a mean advantage of -0.000912.

## Submission assessment

The current result has potential as an honest conceptual or benchmark paper about probabilistic consistency, because it contains:

1. a simple exact projective construction with closure under all linear aggregates;
2. formal PSD, consistency, polarization, approximation, and proper-score arguments;
3. a previously underappreciated query-set-dependence failure mode in TFM interfaces; and
4. a complete, frozen, multi-dataset negative study.

It should not be pitched as a performant SOTA method or as already oral-ready. A performance-centered ICLR submission would need a new dependency-learning mechanism—likely process-aware pretraining or a context-adaptive kernel—that beats the fixed-marginal diagonal and strong process baselines without relying on an outlier, while preserving singleton projectivity or finding an efficient query-independent interface.

## Artifacts

- Paper: `paper/iclr2027/projective_tfm.pdf`
- LaTeX source: `paper/iclr2027/projective_tfm.tex`
- Full protocol: `PROTOCOL.md`
- Recorded deviations: `PROTOCOL_DEVIATIONS.md`
- Generated tables: `paper/iclr2027/generated/`
- Generated figures: `paper/iclr2027/figures/`
- Machine-readable results: `/data/byunhanjoon/projective_tabular_process_iclr2027/results/`
- Final audit: `/data/byunhanjoon/projective_tabular_process_iclr2027/results/final_validation.json`

