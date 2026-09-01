# Two-hour follow-up: do the first two directions survive?

Date: 2026-08-31. The protocol was frozen before the extended runs in `PROTOCOL.md`.

## Executive decision

Both directions contain real empirical effects, but neither broad paper pitch survives unchanged.

1. **Identification-aware causal models: empirical PURSUE, novelty PIVOT.** An actual pretrained TabPFN is accurate and confident on observational prediction while a plug-in causal estimate is badly wrong under hidden confounding. Exact observationally equivalent worlds force nearly identical model answers despite large differences in true effects. Explicit assumptions repair a raw-row set learner. All five preregistered empirical gates pass. However, the August 2026 paper *Foundation Models for Partial Causal Identification* already addresses the broad solution. The remaining opportunity is a narrow relational/continuous stress-test and benchmark contribution, not another generic identification-aware CFM.
2. **Schema semantics: DEPRIORITIZE as a broad method.** Semantic descriptions strongly help on synthetic held-schema tasks, including opaque field names, but TF-IDF slightly beats BGE/E5/GTE on the standard conditions. The three-domain real panel is effectively tied even against an oracle. A post-hoc negation result is interesting enough for a preregistered robustness benchmark, but not enough to rescue the broad method claim.

## Runtime and audit

The recorded workloads total 92.7 model-runtime minutes: binary TabPFN plus DeepSets 53.2 min, continuous TabPFN 25.9 min, distractor stress 12.6 min, and semantic/real-panel runs 1.0 min. The GPU jobs overlapped, so the longest extended wall-clock path was 53.2 min; protocol design, literature checking, diagnostics, and reporting used the rest of the approximately two-hour investigation. Runs used two NVIDIA H100 NVLs, Python 3.10.16, PyTorch 2.7.0+cu126, TabPFN 6.3.0 with cached v2.5 weights, transformers 4.49.0, and scikit-learn 1.4.2. All result files report `errors=[]`.

## Direction 1 — predictive success is not causal identification

### Exact binary equivalence

The paired SCMs have the same complete population distribution of `(T,Y)`:

- randomized causal: `T ~ Bernoulli(0.5)`, `Y = T xor E_r`, ATE `1-2r`;
- hidden confounding: `T = U xor E_q`, `Y = U xor E_p`, ATE `0`;
- choosing `r=q+p-2qp` makes their observational laws identical.

The full grid used six noise cells, `n={128,512,2048}`, 45 independent data seeds, four nuisance columns, and 16-estimator TabPFN ensembles (810 paired dataset evaluations per model family).

| metric | result |
|---|---:|
| TabPFN AUROC, all cells | 0.804 |
| TabPFN AUROC, informative cells | 0.840 |
| mean predictive confidence | 0.814 |
| hidden-confounding plug-in ATE MAE | 0.629 |
| randomized-causal plug-in ATE MAE | 0.047 |
| predicted difference between paired worlds | 0.046 |
| true difference between paired worlds | 0.608 |
| shuffled-label AUROC | 0.507 |

This is not evidence that TabPFN itself claims causal validity. It demonstrates that the common workflow “fit a powerful predictor, set treatment to 0/1, interpret the contrast causally” cannot solve non-identification.

### Raw rows plus explicit assumptions

A DeepSets estimator received raw datasets rather than summary vectors and was evaluated over five network seeds on held-out noise cells.

| input | ATE MAE |
|---|---:|
| observations only | 0.168 |
| observations + randomized/confounded assumption bit | 0.040 |

The assumption token reduces MAE by 76.4%. This supports the constructive half of the claim: the missing information is structural, not recoverable from more flexible processing of the same observations.

### Continuous equivalence and irrelevant columns

In 540 linear-Gaussian paired evaluations (six cells, three sample sizes, 30 seeds), TabPFN achieved mean predictive R2 0.132. The true paired effect gap averaged 0.867, while its estimated gap averaged only 0.082; causal MAE averaged 0.432. Shuffled outcomes gave R2 -0.006. Predictive signal is modest on average, so this is supporting breadth rather than the headline evidence.

The separate 900-evaluation stress test held the result flat from 0 to 64 irrelevant features:

| nuisance columns | AUROC | confidence | hidden-world causal MAE |
|---:|---:|---:|---:|
| 0 | 0.800 | 0.810 | 0.622 |
| 4 | 0.797 | 0.812 | 0.624 |
| 16 | 0.797 | 0.812 | 0.624 |
| 32 | 0.797 | 0.811 | 0.622 |
| 64 | 0.797 | 0.812 | 0.624 |

### Direction 1 gate verdict

All frozen gates pass: informative AUROC >=0.75, hidden-world causal error >=0.30, confidence >=0.70, shuffled prediction near chance, and at least 50% assumption-aware MAE reduction.

The novelty gate does not pass. [*Foundation Models for Partial Causal Identification*](https://arxiv.org/abs/2608.20841) (posted 2026-08-21) already trains a foundation model to emit identified bounds under explicit structural assumptions. [Do-PFN](https://arxiv.org/abs/2506.06039), [CausalPFN](https://arxiv.org/abs/2506.07918), graph-conditioned causal FMs, and DAG-aware TabPFN generation further crowd the neighborhood. The defensible next experiment is an evaluation/pivot: test existing causal PFNs on relational schemas, continuous identified sets, approximate observational equivalence, and assumption misspecification.

## Direction 2 — semantic transfer is real, but pretrained embeddings do not win

The main study used 18 train and 18 lexically disjoint held-out relation-role pairs, BGE-base, E5-base, and GTE-base frozen encoders, TF-IDF, structure-only, zero-shot prototypes, and oracle roles. Eight additional random 18/18 lexical splits and five numeric seeds per split tested split sensitivity. The protocol requested at least 20 held-out pairs; only 18 suitable disjoint pairs were used, a disclosed deviation partly offset by the eight resplits.

### Cross-split synthetic results

| condition | BGE | E5 | GTE | TF-IDF | structure | oracle |
|---|---:|---:|---:|---:|---:|---:|
| clean orientation accuracy | 0.833 | 0.868 | 0.868 | **0.889** | — | — |
| clean downstream AUROC | 0.773 | 0.795 | 0.795 | **0.808** | 0.615 | 0.882 |
| opaque-name downstream AUROC | 0.759 | 0.763 | 0.767 | **0.785** | 0.612 | 0.881 |
| shuffled-description AUROC | 0.349 | 0.345 | 0.336 | 0.315 | 0.609 | 0.881 |

Descriptions carry real signal: all text methods beat structure-only by more than 0.14 AUROC in clean and opaque-name conditions, and shuffling destroys or reverses that gain. But the key frozen gate required an encoder to beat TF-IDF orientation by 15 points; the best encoder instead trails TF-IDF by 2.1 points. Thus this is a data-design effect, not evidence that pretrained semantic representations are necessary.

### Exploratory negation result

The negated-description condition was added after inspecting the initial fixed split and is therefore post-hoc. Across the resplits, BGE reaches 0.847 orientation accuracy and 0.781 downstream AUROC versus TF-IDF at 0.257 and 0.390. E5 is intermediate; GTE largely fails. This large, encoder-specific interaction is worth a preregistered confirmation on a larger, independently authored corpus, but it cannot be a confirmatory result here.

### Real endpoint panel

Actual local aviation origin/destination, taxi pickup/dropoff, and Citi Bike start/end rows were evaluated with 40,000 calibration, 40,000 train, and 40,000 test rows per domain. Leave-one-domain-out RMSEs are effectively identical across BGE, E5, GTE, TF-IDF, structure-only, and even oracle roles: approximately 0.9925 for aviation, 0.8506–0.8540 for taxi, and 0.9818 for bike share. Within-domain R2 is only 0.001 for aviation, 0.273 for taxi, and 0.040 for bike. This panel supplies no real-world support for the proposed semantic advantage; it is small and should not be treated as a universal negative.

### Direction 2 gate verdict

The downstream, opaque-name, and shuffled-description gates pass. The decisive encoder-over-TF-IDF gate fails. [Relational Transformer](https://openreview.net/forum?id=9yOTJfdzbs) and adjacent relational foundation-model work already use schema names/metadata for zero-shot transfer, further weakening novelty. Deprioritize the broad direction; retain only the negation/compositional-language robustness hypothesis for a clean preregistered test.

## Final ranking after follow-up

1. **Relational partial-identification stress testing / benchmark** — pursue as a narrowed pivot; strong phenomenon, crowded solution space.
2. **Compositional schema-language robustness (especially negation)** — one confirmatory experiment only; kill if it does not replicate.
3. **Generic semantics-aware relational pretraining** — deprioritize; TF-IDF and real-panel results do not justify it.

## Artifacts

- `summary.json`: machine-readable gate decisions and headline aggregates.
- `binary_summary.csv`, `continuous_summary.csv`, `distractor_summary.csv`: Direction 1 aggregates.
- `semantic_orientation_summary.csv`, `semantic_downstream_summary.csv`, `real_endpoint_summary.csv`: Direction 2 aggregates.
- `direction_1/` and `direction_2/`: raw per-run CSV/JSON outputs.
- `figures/followup_direction1_tabpfn_cells.png` and `figures/followup_direction2_crosssplit.png`: main plots.
- `logs/`: complete run logs.
