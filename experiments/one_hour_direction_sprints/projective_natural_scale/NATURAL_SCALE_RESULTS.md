# Natural-scale static projectivity: result and ICLR 2027 decision

Date: 2026-08-31 (Asia/Seoul)

## Verdict

**Go, with a narrower implementation thesis.** Static projectivity now has a
real natural-tabular performance signal, not only a semisynthetic consistency
result:

- the unchanged small projective network beat its capacity-matched direct
  network on aggregate NLL on all 12 natural datasets;
- its point RMSE stayed within 25% of TabPFN on 11/12 datasets;
- its learned off-diagonal covariance improved NLL on 11/12 datasets and CRPS
  on 12/12 when rowwise means and marginal variances were held exactly fixed;
- it remained competitive with exact Bayesian-linear and RBF-GP joints; and
- all 5,040 primary rows and 2,160 covariance-ablation rows passed the frozen
  integrity checks.

The result does **not** yet justify submitting the current 53k-parameter model.
The covariance-only effect is real but modest, and the attempted post-hoc
TabPFN ensemble-view covariance failed its breadth gate. The paper-scale method
should therefore be a jointly pretrained projective process head on a strong
tabular-foundation-model backbone, not an ensemble heuristic.

## What was tested

The primary protocol was frozen before outcomes in `PROTOCOL.md`:

- 12 regression datasets, including insurance counts, stock returns, physical
  simulators, materials, bikes, and wine quality;
- three fresh splits and four independent 16-shot contexts per split;
- 32 disjoint 12-row query groups in validation and test;
- point, subset-mean, difference, dense, and scaled-dense queries;
- validation-only covariance temperature calibration; and
- seven models: the original projective/direct neural checkpoints, Bayesian
  linear regression, RBF GP, TabPFN independent marginals, a TabPFN
  ensemble-correlation joint, and TabICLv2 independent marginals.

This follows the repository blog's strongest experimental rules: a smoke test
does not become evidence, the decisive comparison is matched rather than weak,
and repeated splits address a different uncertainty source than repeated model
seeds. No dataset, split, rho value, or gate was changed after outcomes.

Primary protocol SHA-256:
`144c873f8d8922306a5b415b05825bf84d414d15cbbe4289204dc92cea3239f3`.

## Primary outcome

### Frozen transfer gates

| Gate | Frozen threshold | Result | Pass |
|---|---:|---:|:---:|
| Dense/scaled-dense NLL advantage over direct | positive | `+0.5276` | yes |
| Dense/scaled-dense matched-cell win rate | at least 60% | `89.93%` | yes |
| Datasets within 25% of TabPFN point RMSE | at least 6/12 | `11/12` | yes |
| Integrity | all checks | 5,040/5,040 rows | yes |

Across all four aggregate families, not only the gated dense pair, the
projective model beat the direct model by `0.3461` NLL on average, won `87.3%`
of matched cells, and won the dataset mean on **12/12** datasets. A
dataset-block bootstrap interval for the mean advantage is `[0.2788, 0.4128]`;
the two-sided 12/12 sign-test p-value is `0.00049`.

### Point prediction did not collapse

Dataset-balanced point RMSE was:

| Model | Point RMSE |
|---|---:|
| TabPFN-2.5 | `0.8410` |
| RBF GP | `0.8978` |
| Bayesian linear | `0.9040` |
| **Neural projective** | **`0.9067`** |
| TabICLv2 | `0.9266` |
| Neural direct | `0.9679` |

The projective model is 7.8% behind TabPFN on the pooled point metric, but only
CPU Activity exceeds the frozen 25% tolerance. It is 6.3% better than its
direct control and essentially tied with the exact GP/Bayesian point anchors.
This matters because the aggregate gain was not purchased by ignoring ordinary
prediction.

### Aggregate comparison

| Model | Aggregate NLL | Aggregate CRPS | 90% coverage |
|---|---:|---:|---:|
| **Neural projective** | **`1.6058`** | **`0.7190`** | `0.9070` |
| RBF GP | `1.7431` | `1.0743` | `0.9079` |
| Neural direct | `1.9519` | `0.8068` | `0.9255` |
| TabICLv2 independent | `3.4919` | `1.3643` | `0.9481` |
| Bayesian linear | `37.1541` | `1.1234` | `0.9053` |
| TabPFN-2.5 projective heuristic | `1021.0682` | `1.7121` | `0.9125` |
| TabPFN-2.5 independent | `1021.0825` | `1.9042` | `0.9131` |

The last three pooled NLLs are distorted by all-zero 16-shot claim-count
contexts. TabPFN's documented constant-target branch returns a point mass, and
the exact linear model can also become severely overconfident. This is a useful
robustness failure, not a fair SOTA headline. Excluding only FreMTPL gives:

| Model | Aggregate NLL without FreMTPL |
|---|---:|
| RBF GP | `1.5611` |
| **Neural projective** | **`1.5842`** |
| Bayesian linear | `1.6115` |
| TabPFN-2.5 projective heuristic | `1.6842` |
| TabPFN-2.5 independent | `1.6874` |
| TabICLv2 independent | `1.9007` |
| Neural direct | `1.9488` |

Thus the honest competitive reading is: the projective model is robustly best
against its direct control, competitive but not significantly better than the
RBF GP (7/12 dataset wins), and promising against current row-marginal TFMs in
this extreme 16-shot aggregate setting. The FreMTPL failure must not be used to
claim a thousand-nat advantage over TabPFN.

## Does the learned covariance itself help?

The secondary mechanism protocol in `COVARIANCE_ABLATION_PROTOCOL.md` was
frozen after the primary result but before its own outcomes. It compares:

- the full covariance;
- the same mean and diagonal with independence; and
- the same mean and diagonal with correlations assigned to the wrong query
  pairs by a PSD-preserving shuffle.

Covariance-ablation protocol SHA-256:
`d0f7355fab61df726fa55facc43110b6eeb50a56c66986d53656f27a691864ed`.

| Comparison | Mean NLL advantage | Mean CRPS advantage | Dataset NLL wins |
|---|---:|---:|---:|
| Full vs independent | `+0.00400` | `+0.00176` | **11/12** |
| Full vs shuffled | `+0.00288` | — | — |

All 12 dataset CRPS effects favor the full covariance. Eleven of 12 NLL effects
favor it; the exception is zero-inflated FreMTPL. The two-sided sign-test
p-values are `0.00635` for NLL and `0.00049` for CRPS. A dataset bootstrap
interval is `[-0.00039, 0.00752]` for NLL because the single FreMTPL loss is
large, and `[0.00113, 0.00244]` for CRPS. The strongest repeatable benefit is
on signed differences, where covariance cannot be mimicked by simply
inflating all marginals.

Integrity was exact: the full variant reproduced the primary scores within
`4.44e-16`, marginal diagonals matched exactly, symmetry error was zero, and
the smallest covariance eigenvalue was positive (`0.0474`). Every frozen
mechanism gate passed.

The effect size is modest: full-covariance aggregate NLL is `1.60576` versus
`1.60976` under independence. The much larger `0.3461` advantage over the
direct network is therefore mostly the statistical regularization and query
generalization induced by learning one joint object; learned natural
off-diagonals add a smaller, separately verified gain.

## The TabPFN post-hoc mechanism failed

The marginal-preserving TabPFN covariance layer selected a nonzero correlation
on only 16/36 dataset/split cells and selected independence on 20/36. It
improved pooled NLL (`+0.0142`), CRPS (`+0.1921`), and mean coverage error, but
won dataset-level NLL on only **6/12**, below the frozen 7/12 requirement. Two
datasets tied because validation selected rho zero.

This is a useful negative result. Inference ensemble members are transformed
views designed to stabilize marginal prediction, not posterior function draws;
their across-query correlation is not automatically an epistemic covariance.
Do not develop this heuristic further. Train the joint head for joint scores.

## The theoretical object

For context table `C`, let a network emit a mean, a context-dependent feature
map, and positive residual scale for each row:

`m_C(x)`, `phi_C(x) in R^r`, and `d_C(x) > 0`.

For any finite query set `X_Q = (x_1, ..., x_q)`, define

`mu_i = m_C(x_i)`

and

`Sigma_ij = phi_C(x_i)^T phi_C(x_j) + 1[i=j] d_C(x_i)^2`.

Three facts give the paper its theory spine:

1. **Validity.** `Sigma` is PSD because it is low-rank plus positive diagonal.
2. **Static projectivity.** Permuting query rows permutes the law, and deleting
   rows takes a principal submatrix. Because each row representation depends
   only on `C` and that row, the family of finite Gaussian laws is compatible
   under every restriction. It is an amortized, context-conditional Gaussian
   process.
3. **Functional closure.** Every linear table calculation has
   `a^T Y ~ Normal(a^T mu, a^T Sigma a)`. Scaling, addition, subtraction, and
   overlap identities hold exactly without retraining or a new model call.

Sparse functional training can identify the entire object. Point queries give
means and diagonal variances; pair sums/differences recover off-diagonals by
polarization. More generally, uniform error over bounded linear queries follows
from

`|a^T(mu_hat-mu)| <= ||a|| ||mu_hat-mu||`

and

`|a^T(Sigma_hat-Sigma)a| <= ||a||^2 ||Sigma_hat-Sigma||_op`.

This is the theoretical justification for query generalization: estimate one
mean vector and covariance operator, and control infinitely many downstream
linear queries at once. A direct scalar-query network has to relearn the
linear/quadratic dependence on `a` and can violate all corresponding
identities.

The novelty is **not** that Gaussian processes are projective. It is the
combination of (i) amortized tabular in-context inference, (ii) a learned
context-conditional projective kernel, (iii) training/evaluation on arbitrary
row aggregates, and (iv) evidence that the restriction improves both
query-generalization and natural proper scores.

## Position against recent work

- [TabICL (ICML 2025)](https://arxiv.org/abs/2502.05564),
  [TabDPT (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html),
  [TabPFN-2.5](https://arxiv.org/abs/2511.08667), and
  [TabICLv2](https://arxiv.org/abs/2602.11139) make increasingly strong
  in-context point or row-marginal predictions. They motivate using a strong
  pretrained backbone, but do not by default specify a compatible joint law
  across multiple test rows.
- [Distributional Regression with Tabular Foundation
  Models](https://arxiv.org/abs/2603.08206) argues that current benchmarks
  over-reward means and should use proper scores. This project extends the
  evaluation target from one-row marginals to proper scores for arbitrary
  multi-row functionals.
- [On the Uncertainty Quantification Ability of Tabular Foundation
  Models](https://arxiv.org/abs/2606.01427) finds that GPs can dominate TFMs in
  data-scarce UQ. The 16-shot GP result here agrees; the opportunity is to learn
  GP-like joint structure with a richer amortized tabular prior.
- [JoLT](https://arxiv.org/abs/2502.11877) is the closest title-level work. It
  models several heterogeneous target columns within a row by autoregressive
  LLM factorization. Its paper explicitly notes target-order dependence and
  failure to guarantee a valid exchangeable stochastic process. The proposed
  paper instead models one target across an exchangeable query-row set, has
  exact restriction consistency, answers arbitrary linear aggregates, and is
  fast/numeric rather than text-LLM based. JoLT must be discussed prominently,
  not discovered during review.

## ICLR 2027 build recommendation

Working title: **Projective Tabular Foundation Processes: One Joint Law for
Every Spreadsheet Query**.

Build the next model around these requirements:

1. Use a TabICLv2/TabDPT-style row backbone or a compatible open TFM, with two
   heads: a strong distributional marginal head and a learned low-rank
   projective kernel head. Do not use ensemble-view covariance.
2. Pretrain on variable context lengths, variable query-set sizes, and a mixture
   of point, subset, difference, and random linear-query proper scores. Include
   constant, count, heavy-tail, heteroscedastic, latent-group, and shared-shock
   priors.
3. Scale evaluation to at least 30 untouched regression datasets and context
   sizes 16/64/256/full. Compare TabPFN-2.5 or newer, TabICLv2, TabDPT, exact
   and sparse GPs, Bayesian linear, deep/tree ensembles, neural processes, and
   JoLT where its low-shot setting is computationally feasible.
4. Require point RMSE within 2% of the chosen strong backbone, aggregate
   NLL/CRPS wins on at least 70% of datasets, and a material full-vs-diagonal
   covariance effect. The current 0.004 NLL covariance gain is evidence, not a
   final headline.
5. Add semantically grounded applications: insurance portfolio totals,
   financial baskets/spreads, cohort treatment contrasts, and fleet/energy
   aggregates. These make the oral insight visible: accurate marginals do not
   determine the risk of a sum.
6. Extend beyond a single Gaussian with a projective latent mixture or
   conditional stochastic-process decoder, while retaining exact restriction
   consistency. Count/zero-inflated likelihoods are required by the FreMTPL
   boundary.

## Bottom line

Static projectivity has survived the requested escalation. The strongest
supported claim is now:

> A compatible joint predictive law acts as a query-generalization regularizer
> on natural tabular regression, remains close to strong row predictors, and
> its learned cross-row covariance adds a smaller but measurable proper-score
> gain.

That is enough to justify the next serious model build. It is not enough to
claim a finished oral paper, and the failed TabPFN covariance shortcut tells us
exactly where engineering should stop and joint pretraining should begin.

## Artifacts

- Primary machine-readable result: `results/summary.json`
- Primary cells: `results/cells.csv`
- Dataset tables: `results/point_rmse_by_dataset.csv` and
  `results/tabpfn_projectivity_by_dataset.csv`
- Corrected auditable shards: `results/shards_float64/`
- Preserved pre-precision-fix pass: `results/shards/`
- Covariance mechanism result: `results/covariance_ablation/summary.json`
- Covariance cells and dataset effects:
  `results/covariance_ablation/cells.csv` and
  `results/covariance_ablation/by_dataset.csv`
- Exact package versions and checkpoint hashes: `environment.json`
