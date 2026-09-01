# Novelty ledger

Last searched: 2026-08-31 (Asia/Seoul), before any Phase I model result was viewed.

Scope of the search: matched context/query monotone reparameterization, rank/coordinate invariance, marginal-distribution metadata, prior symmetrization, and input adaptation in current tabular foundation models. The search used arXiv, official proceedings, official repositories/model cards, and the current TabArena materials. Absence from this ledger is not proof of novelty; it is a boundary document that must be searched again immediately before a final benchmark and paper draft.

## Closest work

| Work | What it already establishes | Boundary relative to this project |
|---|---|---|
| [TabPFN-3](https://arxiv.org/abs/2605.13986) | Current large-context, fast TabPFN generation and test-time compute scaling. | Mandatory relevance target. The technical report does not make matched featurewise bijective reparameterization its object of study. Local open-weight package access is presently TabPFN-2.5; TabPFN-3 is exposed through the official client/API and requires credentials. |
| [TabICLv2](https://arxiv.org/abs/2602.11139) and [official code](https://github.com/soda-inria/tabicl) | Strong open classification/regression TFM, distribution-aware preprocessing, scalable attention, inference and checkpoints. | Mandatory open model and most practical route for internal analysis or reduced pretraining. Its normalization ensemble is a confound/control, not this project's novelty. |
| [Mitra](https://arxiv.org/abs/2510.21204) and [official classifier model card](https://huggingface.co/autogluon/mitra-classifier) | Mixed synthetic priors improve TFM generalization; prior design matters. | Closest prior-design motivation. RSPF would need to show a specific transformation-orbit intervention and held-out-family robustness, not merely another useful prior mixture. |
| [A Mechanistic Study of Tabular Foundation Models](https://arxiv.org/abs/2605.21288) | Distinct similarity-based TFM readouts, permutation mechanisms, causal interventions, and mechanism-specific attacks. | Closest mechanistic work. A safe extension requires matched whole-task coordinate isomorphisms, marginal-shape analysis, and causal restoration rather than another generic perturbation/readout study. |
| [EquiTabPFN](https://arxiv.org/abs/2502.06684) | Target-dimension permutation equivariance and an equivariance-gap remedy. | Establishes that symmetry gaps can be architectural and consequential, but concerns output/target permutation, not invertible numerical coordinate changes applied to an entire task. |
| [Tabular Numeric Stretch Transformation](https://arxiv.org/abs/2608.09162) | Supervised/unsupervised monotone feature transforms optimized for target smoothness; links to PLE and empirical CDF. | Makes “monotone transforms help” and “learn a useful numerical stretching” unsafe novelty claims. It does not by itself establish TFM posterior stability under matched task isomorphisms or a marginal-metadata tradeoff. |
| [TFM-Retouche](https://arxiv.org/abs/2605.06047) | Architecture-agnostic trainable input-space residual adaptation with validation guard and strong TabArena-Lite results. | Makes “a trainable input adapter improves a frozen TFM” unsafe. It is a mandatory adaptive baseline if reproducible; RSPF must differ in objective, mechanism, and generalization evidence. |
| [TabM](https://arxiv.org/abs/2410.24210) | Parameter-efficient ensembling for strong tabular MLPs. | Conventional trainable neural control; not a TFM symmetry method. |
| [TabR](https://arxiv.org/abs/2307.14338) | Retrieval-augmented tabular model whose predictions depend on learned neighborhood geometry. | Metric-sensitive conventional control and useful mechanism comparator. |
| [TabDPT](https://arxiv.org/abs/2410.18164) | Scaled tabular ICL using real-data self-supervised pretraining and retrieval. | Current retrieval/long-context TFM comparator where installation is practical. |
| [TabArena](https://github.com/autogluon/tabarena) | Living benchmark with curated datasets, repeated splits, strong tuning/ensembling, cached predictions, and current model integrations. | Required clean-IID benchmark after method freeze. The current public version observed during this search is v0.1.8.2, so “v0.1” must be pinned more precisely at final freeze. |
| [BeyondArena](https://arxiv.org/abs/2606.30410) | Unified 142-dataset IID, temporal, grouped, large, high-dimensional, and mixed-feature benchmark. | Required OOD/final stress test after method freeze; not a substitute for the matched-transform audit. |

## Claim ledger

| Proposed claim | Existing closest work | Difference | Required experiment proving difference | Safe to claim? |
|---|---|---|---|---|
| Current TFMs can assign materially different posteriors to matched, information-equivalent coordinate representations of one supervised task. | Mechanistic Study; Numeric Stretch; ordinary preprocessing studies. | Context and query are transformed by the same verified bijection; labels, split, feature identities, and missingness are fixed. | Four-way protocol on exact affine/power/PWL/spline transforms, current TFMs, raw probabilities, inverse audits, tree controls, paired dataset CIs. | No — Phase I pending. |
| Marginal feature shape acts as latent task metadata for TFMs. | TabICLv2 distribution-aware embeddings; Mitra prior-design study. | Separates useful meta-information from accidental cell coordinate through controlled S2/S3/S4 priors. | Controlled synthetic tasks with marginal/function coupling, randomization, and conflict; context-size curves and representation probes. | No — synthetic causal evidence pending. |
| Prior symmetrization reduces coordinate dependence without erasing useful marginal information. | Mitra; EquiTabPFN; augmentation literature. | Symmetrizes featurewise coordinate orbits, uses paired whole-task consistency, holds out transform families, and measures the clean/meta-prior tradeoff. | Matched-compute Base/RSPF-A/RSPF-B, at least two training seeds (three if cheap), held-out spline family, clean and transformed real/synthetic evaluation. | No — method is only a hypothesis. |
| Separating coordinate and marginal streams improves the robustness/clean Pareto frontier. | Raw+quantile features, TabICLv2 embeddings, Numeric Stretch. | Explicit source-feature grouping and ablated information channels rather than duplicated preprocessing views. | Rank/atom/spacing/raw and two-stream ablations after S2 demonstrates a reason to retain marginal metadata. | No — conditional future route only. |

## Explicitly excluded novelty claims

This project will not claim novelty for any of the following:

- TFMs use preprocessing ensembles.
- Quantile transforms can improve tabular learning.
- Monotone distortions can change TFM predictions.
- A trainable input-space adapter can improve a frozen TFM.
- Feature, class, row, or target permutation symmetry matters.
- Different transformed views can be ensembled.

## Search result and next mandatory search

The 2026-08-31 phrase search did not identify a paper whose primary experiment combines all of: a label-independent invertible feature map; the same map on context and query; the four matched/mismatch cells; posterior disagreement; marginal shape as deliberately useful versus nuisance meta-information; and a prior-symmetrization remedy. That is only a provisional novelty boundary. Repeat the same search for work released after 2026-08-31 before any final benchmark and again before paper drafting.

