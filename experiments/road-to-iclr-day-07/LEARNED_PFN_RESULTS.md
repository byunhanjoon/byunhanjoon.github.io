# Learned optional-geometry PFN — results

Status: **COMPLETE — ALL FROZEN LEARNABILITY GATES PASS**

## Main result

A small geometry-aware set transformer learns nearly the same soft routing law
as the analytic Bayes predictor. At the matched 50/50 pretraining mixture it
beats the parameter-matched geometry-free transformer in all `9/9`
scale-by-noise cells by `0.17639` MSE on average. The advantage is positive in
all three seeds (`0.17538`, `0.17771`, and `0.17609`).

The learned model is only `0.00759` MSE behind the analytic Bayes mixture on
average. It beats both always ignoring and always using geometry in all `9/9`
cells. Its implicit trust coefficient correlates `0.950` with the exact
posterior probability and identifies the latent smooth regime with mean AUROC
`0.863`.

## Phase structure

| Heat scale | Noise | Gain over no geometry | Regret to Bayes | Trust AUROC |
|---:|---:|---:|---:|---:|
| 0.3 | 0.1 | 0.02775 | 0.00273 | 0.746 |
| 0.3 | 0.3 | 0.02527 | 0.00172 | 0.730 |
| 0.3 | 1.0 | 0.01166 | 0.00114 | 0.620 |
| 1.0 | 0.1 | 0.25091 | 0.01604 | 0.986 |
| 1.0 | 0.3 | 0.21977 | 0.00997 | 0.975 |
| 1.0 | 1.0 | 0.08988 | 0.00446 | 0.809 |
| 3.0 | 0.1 | 0.40451 | 0.01443 | 1.000 |
| 3.0 | 0.3 | 0.37157 | 0.01058 | 0.999 |
| 3.0 | 1.0 | 0.18621 | 0.00728 | 0.904 |

The weak-structure/high-noise corner is correctly hard to classify: regime
AUROC falls to `0.620`, yet predictive regret to Bayes is only `0.00114`
because the two experts disagree little there. This supports the regret law in
`THEORY.md`: routing errors matter in proportion to squared expert
disagreement, not regime-classification accuracy alone.

## Deployment prior shift

The model is trained at smooth-task probability 0.5 and receives no deployment
mixture label. At a true rate of 0.1, its mean regret to the fixed-prior Bayes
rule is only `0.00078` and mean implicit trust falls to `0.274`. At a true rate
of 0.9, regret grows to `0.01447` and trust reaches only `0.678`. Context
evidence therefore adapts the router substantially but does not erase its
pretraining prior. This asymmetry is a real limitation and a clean next target.

## Frozen gates

| Gate | Outcome |
|---|---:|
| beat set transformer in at least 7/9 cells | PASS — 9/9 |
| mean advantage at least 0.02 | PASS — 0.17639 |
| mean regret to Bayes at most 0.05 | PASS — 0.00759 |
| trust/posterior correlation at least 0.70 | PASS — 0.950 |
| trust regime AUROC at least 0.75 | PASS — 0.863 |
| beat zero and always-smooth in at least 7/9 | PASS — 9/9 and 9/9 |
| positive advantage in at least two seeds | PASS — 3/3 |
| finite-output integrity | PASS |

## Decision

**KEEP as a successful mechanism test, but do not promote it as the standalone
lead.** The result proves a neural learnability claim that the analytic
simulation could not: a transformer can infer whether supplied state structure
is relevant and approximate posterior soft routing without regime labels.

A literature audit performed after the frozen outcome found a direct boundary:
ACE (AISTATS 2025) explicitly predicts latent GP kernel identity, transformer
neural processes amortize stochastic-process inference across kernel families,
and earlier work amortizes kernel/hyperparameter selection. The remaining
possible novelty is specifically tabular and residual—whether external field
structure adds value after ordinary columns—not generic learned kernel routing.

It is not standalone paper evidence. The cycle, state count, context size, Fourier
coordinates, scale, and noise descriptors are synthetic and known. The
geometry-free control is intentionally unable to distinguish query states;
strong next baselines must include Gaussian-process model averaging, graph
neural processes, explicit plug-in Bayes routing, and generic metric-aware
attention. No real table or classification task has yet tested the learned
prior.

## Reproducibility

- protocol: `LEARNED_PFN_PROTOCOL.md`;
- runner: `learned_structured_pfn.py`;
- cells: `results/learned_pfn/cells.csv`;
- summary: `results/learned_pfn/summary.json`;
- checkpoints and training traces: `results/learned_pfn/{structured,set}/`;
- figures: `results/learned_pfn/figures/`.
