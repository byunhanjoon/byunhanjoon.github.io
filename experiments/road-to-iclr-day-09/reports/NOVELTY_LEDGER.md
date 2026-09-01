# Novelty ledger

Last checked: 2026-09-01, after the untouched loss-aligned routing test, real transfer,
tail/shrinkage controls, context-rescaled confirmation, and the final 2025–2026 dynamic
ensemble search.

| Nearby work | Verified overlap | Not claimed here | Remaining candidate novelty |
|---|---|---|---|
| [TabPFN-3](https://arxiv.org/abs/2605.13986) | synthetic-only pretraining, preprocessing/OOD checkpoint, test-time ensembling | frontier accuracy or generic preprocessing | controlled quotient-vs-marginal information dial and adaptive frontier |
| [TabICLv2](https://arxiv.org/abs/2602.11139) | diverse synthetic generator and scalable open TFM | a new scalable TFM | audit of its implicit coordinate-information tradeoff |
| [Mechanistic Study](https://arxiv.org/abs/2605.21288) | TFM invariances, perturbations, causal intervention | transforms alter predictions or models have different mechanisms | Bayes cost of quotienting plus a controllable prior and adaptive method |
| [O'Prior](https://arxiv.org/abs/2605.18971) | synthetic marginal realism and robustness depend on prior design | realistic/mixed priors improve TFMs | fixed warp marginals with continuously controlled mechanism–warp dependence |
| [Mitra](https://arxiv.org/abs/2510.21204) | curated mixed synthetic priors improve generalization | mixing priors is new | explicit stable/coordinate channel separation and non-oracle tradeoff adaptation |
| [TabArena](https://github.com/autogluon/tabarena) | 51-dataset living benchmark and cached predictions | a new IID benchmark | matched reparameterization protocol on the same compatible tasks |
| [BeyondArena](https://arxiv.org/abs/2606.30410) | 142 IID/temporal/grouped datasets | broad non-IID benchmarking | external falsification of the factorized method |
| [TabPFN-v2 / Nature](https://doi.org/10.1038/s41586-024-08328-6) | Quantile+Id preprocessing already combines rank-like and original coordinates; preprocessing ensembles include nonlinear transforms | raw+rank views or their fixed ensemble are new | formal cost of discarding coordinate marginals, not the representation recipe |
| [Mechanistic Study](https://arxiv.org/abs/2605.21288) | rank, cube, and soft-exponential monotone attacks; Mitra's quantile front-end; causal model-family comparisons | monotone perturbation audits or rank robustness are new | controlled usefulness of the discarded marginal at fixed marginal prevalence |
| [TFM ensembling study](https://arxiv.org/abs/2605.18696) | broad TFM ensembles have limited diversity; stacking can improve accuracy while damaging log-loss calibration | learned ensembling or a generic calibration warning is new | the specific generator-family-identification versus loss-routing misalignment in PriorDial |
| [META-DES](https://arxiv.org/abs/1810.01270) | dynamic ensemble selection from estimated local classifier competence is established | competence estimation or per-instance dynamic selection is new | controlled separation between identifiable synthetic metadata and loss-aligned usability under a fixed-marginal information dial |
| [DES-AS](https://doi.org/10.1016/j.patcog.2024.110899) | 2025 dynamic selection/weighting uses Shapley-style classifier synergy competence | dynamic weighting or group-competence criteria are new | the fixed-generator target-separation benchmark, not another competence criterion |
| [DES-bADE](https://doi.org/10.1016/j.asoc.2026.115425) | 2026 work optimizes regions of competence for dynamic ensemble selection | optimizing competence neighborhoods is new | loss-target and tail-risk diagnostics under controlled metadata information |
| [MixturePFN](https://openreview.net/forum?id=2fojNANZSv) | mixture-of-experts routing for tabular prior-data fitted networks is established | tabular expert routing or mixture-of-experts architecture is new | a benchmark-level target-alignment result for frozen heterogeneous experts, not a new PFN architecture |
| [Super Learner](https://doi.org/10.2202/1544-6115.1309) | cross-validation-selected convex combinations and oracle-style guarantees are established | CV stacking, convex mixtures, or loss-based weighting is new | controlled four-target separation (metadata, hard selection, correct assignment, calibrated mixture) plus a fixed-marginal information dial |

The successful method conjunction is no longer available because G3 failed. The remaining
candidate contribution is a narrower theory/benchmark package: (i) the exact Bayes
log-risk cost of quotienting, (ii) a fixed-marginal mechanism–warp dependence dial with
an exact nonlinear information calibration, (iii) a scoped TabICL/Mitra stability
contrast, and (iv) a controlled four-target separation: generator metadata can be nearly
perfectly identified yet yield harmful hard routing; predictive-loss estimates require
correct expert assignment; and a calibrated soft mixture can succeed when hard selection
fails. The same frozen soft rule transferred on 12 independently tested real numeric
regression identities and was sensitivity-positive over all 16 completed regression
datasets, while binary classification transfer failed. Immutable pointwise-loss
diagnostics further localize the asymmetry: adaptive weights suppress the regression
squared-error tail but amplify rare high-NLL classification errors without a detectable
aggregate AUC change. This opposite-sign tail pattern sharpens the benchmark-level
target-alignment phenomenon; it is not evidence that competence routing, stacking, soft
expert mixtures, shrinkage, or calibration-risk observations are new algorithms. A
lightly adapted classification mixture later produced a small independent-panel gain,
but because its shrinkage was chosen on earlier real outcomes it is a scoped performance
result rather than part of the synthetic-only novelty claim.

End-of-project recheck: complete for the sources named by the protocol plus the additional
TFM ensembling, classic and 2025–2026 dynamic-selection, Super Learner, and tabular
mixture-of-experts neighbors above. This is a bounded adjacency audit, not proof that no
uncatalogued paper overlaps the remaining conjunction.
