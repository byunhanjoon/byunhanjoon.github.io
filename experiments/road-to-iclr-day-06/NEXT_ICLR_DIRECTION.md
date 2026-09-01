# Next ICLR direction — trust the metric only when it transfers

## Executive decision

**Pivot from “invent a better metric tokenizer” to “detect and prevent harmful
metric transfer under unseen-state tabular shift.”**

The strongest surviving empirical fact is not that one encoding wins
universally. Standardized landmark distances are substantially better when the
declared geometry is task-relevant, but catastrophically worse when a valid
input metric is not predictive of the target. A five-fold, training-state-only
router recovered most of the gain while falling back on the normalized-weight
representation for the harmful medical field. This is the only current path
that is both performant and plausibly defensible, but its evidence is
post-outcome development evidence and it is not yet an ICLR result.

Suggested paper question:

> When should a tabular model trust externally supplied value geometry, and can
> that decision be made using only training states before genuinely new states
> arrive?

Working title: **Trust, Then Encode: Risk-Controlled External Geometry for
Unseen Tabular States**.

## What the experiments established

### Frozen Metric-Field Transport ladder

| Finding | Result | Interpretation |
|---|---:|---|
| Random learned 32x32 factor vs direct weights | -2.86%, 8/18 wins | The current MPE factorization adds an optimization penalty without information. |
| Exact-start ReZero factor vs direct weights | +0.24%, 10/18 wins | Initialization repairs little; it is not a paper contribution. |
| Nine-task Ridge, raw distance m32 vs weights | -2.07% source-balanced | Raw coordinates are not a universal replacement. |
| Four-source neural, raw distance m32 | 17/24 wins | The mechanism is broad at the cell level. |
| ACS / Citi / TLC neural change | +1.18% / +27.70% / +13.63% | Preserving landmark distances matters on hierarchy/spatial fields. |
| String benchmark neural change | -38.68% | An externally supplied metric can cause severe negative transfer. |
| Overall always-raw neural change | -4.82% source-balanced | The frozen E1 gate correctly rejected promotion to transport. |

E2 task transport was **not tested**. It was locked by the prospective gate, so
the evidence rejects the always-raw prerequisite, not transport in isolation.

### Exploratory Metric Trust Router

The separately frozen exploratory router compared `weights_m32` with
`distance_m32` using five-fold cross-validation over training states only. Its
decisions were written before joining any outer validation score.

| Panel | Routed result vs weights | Wins / ties / losses | Worst source |
|---|---:|---:|---:|
| Four-source neural | **+10.28%** source-balanced | **15 / 9 / 0** | 0.00% |
| Nine-task Ridge | +0.19% source-balanced | 24 / 16 / 5 | -0.89% |

The neural router selected raw distances for Citi and TLC, selected them for
one ACS partition, and rejected them for both medical partitions. All five
predeclared feasibility checks passed. This is promising precisely because it
addresses the observed failure rather than hiding that failure. It remains
post-hoc direction finding: the outer outcomes had already been observed when
the router study was motivated.

## Novelty subtraction

Neither the representation nor generic cold-start gating is new:

- [Similarity Encoding](https://arxiv.org/abs/1806.00979) already represents
  dirty categories through similarities to prototype categories.
- [P-GNN](https://arxiv.org/abs/1906.04817) and
  [NodePiece](https://openreview.net/pdf?id=xMJWUKJnFSw) already use distances
  to anchors for inductive representations.
- [Knowledge-Enriched Machine Learning for Tabular Data](https://proceedings.mlr.press/v288/kim25a.html)
  already studies metadata/concept kernels and kernel-enriched tabular
  benchmarks.
- [DropoutNet](https://papers.neurips.cc/paper_files/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html),
  [PT-GNN](https://arxiv.org/abs/2012.07064), and
  [CoMeta](https://arxiv.org/abs/2303.07607) already simulate cold starts or
  transport warm embeddings.
- The recent [GateSID](https://arxiv.org/abs/2603.22916) explicitly uses
  adaptive gating to balance semantic and collaborative signals. A generic
  “learn a gate” claim is therefore crowded too.

The residual novelty hypothesis is narrower:

1. define *value-level metric trust* for one typed tabular field under strict
   state-disjoint risk;
2. show that target-independent geometry can be strongly helpful yet strongly
   harmful across fields, even when all leakage and capacity controls pass;
3. estimate transferability with state-group cross-fitting rather than ordinary
   row-wise validation;
4. provide a fallback/oracle-risk analysis and a benchmark where metric
   corruption, weak metrics, and task-irrelevant metrics are required controls.

This should be presented as a problem + benchmark + risk-control paper, not as
a novel distance embedding.

## What a publishable method still needs

The current router is nested group model selection, which is useful but too
standard by itself. An ICLR-level version needs at least one substantive result:

- an oracle inequality or finite-state risk bound for selecting a metric
  representation from group-cross-fitted losses;
- a confidence-calibrated fallback rule with a stated probability of harmful
  selection under exchangeable state sampling;
- or a cross-field trust estimator that beats per-task group CV while remaining
  calibrated on new source families.

Any learned router must be compared directly with the zero-threshold five-fold
rule. Complexity is justified only if it improves new-source safety.

## Required prospective confirmation

Do not inspect the original test targets and do not reuse the now-observed
outer validation panel as confirmation. Acquire new source families or new
temporal cohorts, freeze them, and test exactly:

1. normalized landmark weights;
2. standardized raw landmark distances with `m=32`;
3. the fixed five-fold state router;
4. similarity encoding, Nyström/kernel features, categorical unknown, and
   qPLE/code baselines;
5. the strongest domain representation available, such as raw coordinates,
   hierarchy ancestors, or graph/spectral features;
6. corrupted-metric and shuffled-router controls.

The next confirmation should require at least 5% source-balanced neural gain,
no source degradation above 2%, improvement on at least three unrelated
semantic geometries, correct fallback on at least two weak or task-irrelevant
metrics, and superiority to the best same-information baseline. The router and
all thresholds must be frozen before outcomes.

## Stop list

- Do not revive MPE by adding depth, landmarks, bandwidths, or another linear
  embedding.
- Do not claim raw landmark distances are novel.
- Do not run E2 transport on this panel after its gate failed.
- Do not sell a generic learned gate as the contribution.
- Do not average away the medical failure; it is the motivation for the new
  question.

## Candid ICLR outlook

- Current evidence as a standalone paper: **not submission-ready**; roughly a
  10–20% ICLR bet because the positive router join is post-hoc and the method is
  standard model selection.
- If a separately frozen new-data panel reproduces the neural gain and safety
  against strong same-information baselines: **plausible 35–50% ICLR bet**.
- If the work also adds a meaningful risk guarantee and releases the strict
  unseen-state metric benchmark: this becomes the strongest version of the
  project. Without prospective confirmation, stop rather than iterate on the
  current validation panel.

## Evidence locations

- Frozen successor results: `metric-field-transport/RESULTS.md`
- Frozen successor audit: `metric-field-transport/FINAL_AUDIT.md`
- Router protocol: `metric-trust-router/EXPLORATORY_PROTOCOL.md`
- Router results: `metric-trust-router/RESULTS.md`
- Router audit: `metric-trust-router/AUDIT.md`
