# Day 5 novelty boundary against recent literature

Audit date: 2026-08-28. This is a claim map, not a complete systematic review.
Primary/official sources were preferred.

## Claims already occupied

| Area | Closest work | Consequence for this project |
|---|---|---|
| Task-irrelevant tabular sensitivity | [Robustness is important: Limitations of LLMs for predictions on tabular data (PNAS Nexus, 2026)](https://doi.org/10.1093/pnasnexus/pgag197) tests row/variable order, names, precision, and formats in general LLMs and tabular foundation models. | Do not claim discovery that superficial tabular representations change predictions. |
| Preprocessing robustness cards | [PREF: Preprocessing Robustness in Heterogeneous Tabular Learning (TMLR submission, 2026)](https://openreview.net/pdf?id=1JhhSxdBS1) defines sensitivity/volatility indices across trees, MLPs, and TFMs. | Do not sell one-knob preprocessing sweeps or generic sensitivity indices as novelty. |
| Target permutation equivariance | [EquiTabPFN (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html) identifies and architecturally closes a target-equivariance gap. | Do not claim class-label permutation as a newly discovered architectural gap or propose a TabPFN-specific class fix as the main contribution. |
| Mechanisms of TFM permutation invariance | [A Mechanistic Study of Tabular Foundation Models (2026)](https://arxiv.org/abs/2605.21288) traces row/column/class behavior to positional parameters and reports exact fixes after removing them. | The TabPFN experiment is supporting evidence only; positional-parameter causality is occupied. |
| Supervised metamorphic relations | [Application of Metamorphic Testing to Supervised Classifiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC3019603/) explicitly includes class-label and attribute permutations. | Exact schema transformations are not themselves novel tests. |
| Functional ANOVA | [Lengerich et al., AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html) develops identifiable functional-ANOVA effects and interactions. | fANOVA is established mathematics; novelty must be its aligned prediction/product-risk role and downstream action. |
| Proper-score variance | [Gruber and Buettner, AISTATS 2023](https://proceedings.mlr.press/v206/gruber23a.html) gives general proper-score bias/variance decompositions via Bregman information. | The Brier identity is a foundation, not a new theorem. The paper can contribute an operational schema-ambiguity estimand built on it. |
| Invariant optimization principle | [Symmetry-Compatible Principle for Optimizer Design (2026)](https://arxiv.org/abs/2605.18106) derives optimizer updates matched to parameter symmetries; [PolarAdamW (2026)](https://arxiv.org/abs/2605.07067) separates spectral control from gauge equivariance. | Do not claim the general idea of symmetry-compatible optimizers. The field-vector experiment is a tabular causal mechanism/control, not the primary novelty. |
| Canonical orbit maps | [A Simple Strategy to Provable Invariance via Orbit Mapping (ACCV 2022)](https://openaccess.thecvf.com/content/ACCV2022/html/Gandikota_A_Simple_Strategy_to_Provable_Invariance_via_Orbit_Mapping_ACCV_2022_paper.html) provides general canonical orbit mapping. | Deterministic schema canonicalization is a required comparator, not a central novelty claim. |
| Latin-hypercube/orthogonal designs | Space-filling designs and Latin hypercubes are classical; e.g. [Williamson 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4440391/) reviews their use for expensive model ensembles. | Do not claim the design itself. The candidate novelty is using a balanced design over exact training-pipeline nuisance factors and random seeds to estimate a supervised prediction quotient at equal fit budget, with aligned proper-risk/fANOVA diagnostics. |
| Orthogonal arrays for ensembles | [On Ensembles, I-Optimality, and Active Learning (Johnson and Broatch, 2021)](https://doi.org/10.1007/s42519-021-00200-4) uses two-level OAs to construct low-correlation half-sample training ensembles. | “Use an OA to build an ensemble” is occupied. The distinction here must be explicit: rows sample a declared product of equivalence-preserving pipeline actions and seeds; no observation is omitted, every member has the full training data, predictions are semantically aligned, and randomized level names target a quotient estimand. |
| Reduced group averaging | [Frame Averaging (Puny et al., ICLR 2022)](https://openreview.net/forum?id=zIUyj55nXR) replaces full group averages with small input-dependent equivariant frames while retaining exact architectural invariance/equivariance. | Do not claim that a subset average is a new general route to invariance. The Day-5 cover is an unbiased randomized finite-population estimator, not an equivariant frame: a realized cover is generally not exactly invariant, but has analyzable fANOVA variance and applies to complete retraining/HPO pipelines as black boxes. |
| Learned probabilistic symmetrization | [Kim et al. (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html) learns an equivariant distribution over group transformations and obtains equivariance in expectation for architecture-agnostic models. | Do not claim probabilistic symmetrization or sampling group actions as new. OrbitCover uses a fixed declared uniform nuisance target and designs *dependent retraining actions* to reduce finite estimator variance; it neither learns a transformation distribution nor guarantees a realized model is equivariant. |
| Orthogonal Monte Carlo | [Choromanski et al., ICML 2019](https://proceedings.mlr.press/v97/choromanski19a.html) couples isotropic continuous Monte Carlo samples through geometric orthogonality. | Variance reduction through dependent, marginally correct Monte Carlo samples is established. The candidate contribution is the mixed-level factorial/group setting, exact component filter, and end-to-end supervised evidence—not generic orthogonal sampling. |
| Randomized-OA integration theory | [Owen (1992)](https://www3.stat.sinica.edu.tw/statistica/j2n2/j2n27/j2n27.htm) established randomized orthogonal arrays for numerical integration and variance reduction; [Ai, Kong, and Li (2016)](https://www3.stat.sinica.edu.tw/statistica/j26n2/j26n217/j26n217.html) gives broader OA-LHS variance theory using functional decompositions. | Removing low-order ANOVA contributions from an integration estimator is occupied. Proposition 7 is best presented as an exact finite, mixed-level, vector-valued specialization that makes the supervised pipeline application auditable—not as invention of the OA/ANOVA filter principle. |
| Sliced/resolvable and antithetic designs | [Hwang, He, and Qian (Technometrics 2016)](https://doi.org/10.1080/00401706.2014.993092) constructs sliced OA-based Latin hypercubes whose coordinated slices improve combined stratification; [L'Ecuyer and Lemieux (MOR 2004)](https://doi.org/10.1287/moor.1040.0101) treats antithetic and other correlation-induction variance reduction generally. | Disjoint/negatively dependent design batches are classical. Proposition 27's narrower contribution is a regular graph over exact mixed-level pipeline covers, its finite prediction-space covariance operator, and an independent outer cross-score for quotient model selection. |
| Antithetic prediction-error estimation | [Liu, Panigrahi, and Soloff, *Cross-validation with antithetic Gaussian randomization* (JRSS-B, 2026)](https://doi.org/10.1093/jrsssb/qkag073) constructs equicorrelated Gaussian train/test randomizations and proves favorable prediction-error bias/variance for smooth prediction rules. | “Use antithetic dependence to improve prediction-risk estimation/model selection” is now clearly occupied and is a close 2026 collision. OrbitCover differs in its sampled object and estimand: it couples fully trained predictions over a finite product of exact pipeline nuisances, uses OA/fANOVA cancellation and independent block cross-products, and does not perturb or split response data. The paper must compare these operators directly at the conceptual level and cannot sell antithetic risk estimation itself as novelty. |
| Optimal antithetic CV laws | [Chattopadhyay, Liu, and Panigrahi, *On the optimality of antithetic randomization for cross-validation* (arXiv, August 2026)](https://arxiv.org/abs/2608.08089) proves that equicorrelation `-1/(K-1)` is necessary and sufficient for bounded reducible variance in its vanishing-bias smooth normal-means regime, and jointly normal randomization is minimax within its class. | General optimal-antithetic language is occupied by a paper released only weeks before this audit. Proposition 35 now makes the boundary explicit: partial sampling from an `H`-cover resolution has coefficient `-1/(H-1)`, not `-1/(K-1)`, until `K=H`. OrbitCover may claim only its exact finite nuisance-product operator identities and empirical interaction-dependent comparison; the failed orbit LP is useful evidence against overclaiming minimax optimality. |
| Modern asymmetric resolvable OAs | [Lin and Pang (Statistics & Probability Letters 2025)](https://doi.org/10.1016/j.spl.2025.110355) generalizes asymmetric resolvable OAs and gives new constructions for mixed-level factors. | The mixed `4x4x2x4` and `4^4` coset resolutions are not new combinatorial-design claims. Their role is an explicit nuisance-quotient compute schedule, the finite-population risk law, and the empirical contrast with a non-resolvable graph pack. |
| Parameter-efficient tabular ensembles | [TabM (Gorishniy et al., ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) shows that efficient ensembles substantially strengthen tabular MLPs. | Heterogeneous representation bags must compare against modern parameter-efficient ensembles before a standalone performance claim. The current MLP/ResNet/FT-Transformer HeteroBag panel is an encouraging secondary result, not yet an ICLR-complete primary method. |
| Training/model multiplicity | [Summers and Dinneen (ICML 2021)](https://proceedings.mlr.press/v139/summers21a.html) dissects supervised neural-network nondeterminism; [Hamman et al. (ICML 2025)](https://openreview.net/forum?id=AXJnqocQpm) studies prediction consistency under tabular-LLM fine-tuning multiplicity. | Do not claim discovery that seeds or minor training variations destabilize predictions. The distinction is a declared multi-factor equivalence product, exact component accounting, and a constructive equal-compute selection action. |
| Multi-seed tabular evaluation | TabM's [official ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf) evaluates tuned models under 15 seeds on most datasets. | Repeating seeds and reporting mean/std is a baseline practice, not novelty. The cover must beat seed blocks and IID joint sampling at the same number of fits. |
| Experimental design for HPO | [Sequential Uniform Designs (JMLR 2021)](https://jmlr.org/papers/v22/20-058.html) treats HPO as a computer experiment; the withdrawn [MOFA ICLR 2021 submission](https://openreview.net/forum?id=OpUJ46CNv43) combines orthogonal Latin hypercubes, OAs, and factorial analysis. | “Use an OA/uniform design in AutoML” is occupied. Here the candidate algorithm/hyperparameters are *not* OA factors: every candidate is evaluated over the same product of equivalence-preserving pipeline nuisances and seeds, and the target is its quotient prediction and selection decision. MOFA must nevertheless be discussed prominently as a close design-language collision. |
| Low-variance selection criteria | [Cawley and Talbot (JMLR 2010)](https://jmlr.org/papers/v11/cawley10a.html) shows that variance of a model-selection criterion can cause selection overfitting and can matter as much as bias. | The general motivation “reduce validation-estimator variance to improve selection” is occupied. The defensible novelty is the exact nuisance-product estimand, fANOVA-filter construction, finite regret link, and equal-compute supervised evidence. |
| Evaluation-sample/CV estimands | [Bates, Hastie, and Tibshirani (JASA 2024)](https://doi.org/10.1080/01621459.2023.2197686) shows that CV and data-splitting estimands can differ from the error of the fitted model at hand and that ordinary uncertainty can be badly calibrated. | Do not describe validation/test instability as newly discovered distribution shift. Proposition 22 is only the exact fixed-pool complement identity for candidate quotient losses; the repeated-partition audits are conditional diagnostics, and repeated training/new-source evaluation remains necessary. |
| Stable set-valued model selection | [Adrian, Soloff, and Willett (TMLR 2025)](https://openreview.net/forum?id=DSDWHsQLgA) stabilizes black-box selection via bagging and an inflated argmax, with leave-one-observation stability guarantees for compact model sets. | Do not claim generic stable model selection. OrbitCover targets randomness from exact pipeline representations/training; its two-replicate winner union is only a finite-randomization diagnostic and has no leave-one-observation or distribution-free coverage guarantee. A paper should compare or explicitly separate these stability notions. |
| Adaptive-stopping bias | [Shin, Ramdas, and Rinaldo (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/65b1e92c585fd4c2159d5f33b5030ff2-Abstract.html) separates bias from adaptive sampling, stopping, and arm choice. | A prediction-dependent stopping rule cannot inherit the fixed-budget cover's unbiased quotient-estimator claim automatically. The Day-5 adaptive schedule is restricted to exploratory validation-only selection and its failed aggressive rule is retained. |
| U-statistic risk objectives | [Clémençon, Colin, and Bellet (JMLR 2016)](https://jmlr.org/papers/v17/15-012.html) studies ERM and model selection with complete and incomplete U-statistic risks. | A pairwise U-statistic as an unbiased risk device is classical. Proposition 18's candidate novelty is narrower: nuisance randomness is the sampled object, strength-matched blocks replace IID members, and the target is the full equivalence×seed quotient loss. |
| Risk estimation for randomized ensembles | [Du et al. (JCGS 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492369/) extrapolates out-of-bag risk across randomized-ensemble sizes; [Bellec et al. (JRSS-B 2025)](https://academic.oup.com/jrsssb/article/87/2/289/7760017) corrects GCV for finite ensembles of penalized estimators. | Do not claim discovery that finite-ensemble loss needs correction. Cross-cover scoring instead estimates the risk of an abstract nuisance-quotient predictor on a held-out validation set, without assuming subsampling ensembles, linear smoothers, or training-sample GCV. These are close conceptual comparators and belong in related work. |
| U-statistic inference for ensembles | [Mentch and Hooker (JMLR 2016)](https://www.jmlr.org/beta/papers/v17/14-168.html) casts subsampled random-forest predictions as incomplete U-statistics for inference. | The Day-5 U-statistic is over independently randomized nuisance predictions and targets a quadratic quotient score, not sampling-distribution inference for a forest prediction. The algebraic ingredient remains classical. |
| Jackknife bias correction | Quenouille's [1949 jackknife precursor](https://doi.org/10.1111/j.2517-6161.1949.tb00023.x) and the classical jackknife/delta-method literature establish smooth-functional bias cancellation. | The two-cover log-loss jackknife is an approximate scope extension, not a new jackknife theorem. Its contribution is only the nuisance-quotient block construction and empirical selection audit; exact unbiasedness remains limited to quadratic cross-scores. |

## Searches that did not reveal a direct collision

Targeted searches for combinations of schema permutations, HPO selection,
orthogonal/Latin prediction ensembles, test-time augmentation, and tabular
supervised pipelines did not reveal work combining all of the following:

1. a declared product of exact schema symmetries applied to the complete fit
   and prediction pipeline;
2. semantic alignment of vector predictions before scoring;
3. a joint `schema x model-seed` Hilbert-risk tensor with exact product fANOVA;
4. a Brier/MSE quotient estimand and selection-path decomposition;
5. an equal-fit strength hierarchy over schema factors and seed, evaluated
   against iid joint draws, lower-strength blocks, seed-only ensembles, and
   deterministic canonicalization on held-out datasets;
6. downstream model selection with validation-only decisions, held-out test
   loss, changed nuisance menus, and a changed data subsample.
7. an independent-cover cross-score that is exactly unbiased for the full
   nuisance-quotient Brier/MSE, with an equal-fit IID member U-statistic rather
   than naive finite-ensemble loss as the comparator;
8. a single empirical chain from prediction-field covariance and fANOVA
   filtering to candidate-ranking inversions, quotient-selection regret, and
   validation-to-test partition instability;
9. disjoint-stream calibration of the exact cross-score covariance-operator
   variance, plus its two component reductions;
10. a small replicated-selector union diagnostic with explicit separation from
   formal stable set-valued selection;
11. a regular disjoint-cover packing graph whose antithetic pair covariance is
   crossed across independent pairs to retain exact quotient-risk unbiasedness.
12. a 16/32/64/128-fit disjoint-packing frontier, including exact mixed-level
   resolutions, an interaction-spectrum comparison among pack laws, and a
   complete-pair IID-U128 control.

Absence from these searches is not proof of novelty. A formal paper review
should additionally search statistical-design, uncertainty-quantification,
AutoML, metamorphic-testing, and test-time-augmentation venues and solicit
expert review.

The August 2026 refresh did find the close antithetic-CV paper above. It does
not instantiate the twelve-item combination, but it materially narrows the
methodological novelty: the submission case now rests on a new finite-nuisance
estimand and operator composition, not on the broad antithetic-risk idea.
An even newer August 2026 optimality preprint further removes any safe claim to
general antithetic optimality. It studies Gaussian response perturbations in a
normal-means asymptotic regime rather than finite trained-pipeline nuisance
orbits, but it must be treated as a central related-work comparator.

## Defensible contribution after current results

The most defensible ICLR-shaped claim is not “models are sensitive to column
order.” It is:

> Exact schema equivalences and ordinary random seeds form a joint nuisance
> product for complete learning pipelines. Aligned proper-risk fANOVA exposes
> how that nuisance propagates through training and model selection. A
> strength-matched cover removes all fANOVA components through its declared
> order and approaches the quotient predictor—and its model-selection
> decision—more efficiently than iid schema/seed sampling, replicated
> lower-strength covers, and spending all fits on seeds.

The frozen six-source-group strength-2 confirmation passed, as did changed
nuisance-menu and changed-subsample repeats, a prospective external exact-risk
panel, and an equal-compute model-selection experiment on the original three
panels. A task-balanced eight-source classification/regression extension also
passed, including its five-source non-enumerating subset. The exact strength-3
result completes the four-factor hierarchy.

The independent-cover cross-score and disjoint packing chain is the most
promising Day-5 extension. At 32 fits the score is calibrated as unbiased,
has lower quotient-loss RMSE than IID-U in 158/171 candidates, and improves
validation agreement/regret in all five panels. At 128 fits, crossing two
independent four-packs beats the complete U-statistic over eight independent
covers on every panel clause and all 23 non-exhaustive candidates. Exact
resolutions and the graph/resolution interaction boundary deepen the mechanism,
while the failed orbit-law optimizer prevents a spurious optimality claim.
At the known 128-cell closure, exhaustive resolution strictly dominates the
randomized cross-score and is the recommended equal-fit method.
This remains a composition of classical designs, antithetics, and U-statistics;
the candidate novelty is their nuisance-pipeline composition and exact-to-
downstream empirical chain, not any ingredient in isolation.

Important limits remain. Ordinary predictive-score gains are small; accuracy
is mixed. The prospective external selection gate failed because validation
and test candidate rankings shifted, and conditional repartitioning attributes
most of that observation to finite evaluation-sample instability rather than
demonstrated population shift. Cross-scoring improves validation fidelity
without a source-stable test-regret gain. Field-wise high-dimensional
strength-2 failed against a marginally balanced control, the HeteroBag
semantic-specificity gate failed on a genuinely untouched panel, and
aggressive adaptive stopping failed. The Cartesian action, quotient-HPO,
HPO-selection, and multiclass selection-ceiling failures must remain prominent
negative evidence.
