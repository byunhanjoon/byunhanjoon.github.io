# OrbitCover claim–evidence matrix

Status: submission synthesis from the frozen final closure. The authoritative
source is `../final_closure/results.md`; older Day-5 summaries are supporting
history, not replacements for this matrix.

## Main-paper claims

| ID | Proposed wording | Status | Direct evidence | Required qualifier / forbidden wording |
| --- | --- | --- | --- | --- |
| C1 | A declared semantic nuisance product defines a quotient predictor for a complete randomized learning pipeline. | Exact definition | `THEORY_FOUNDATIONS.md`, Props. 1–3; final closure Section 14 | Distribution-relative. Do not call the action menu “all equivalent representations.” |
| C2 | Under Brier/MSE, variance around the quotient is exactly expected finite-member loss overhead. | Exact identity | `THEORY_FOUNDATIONS.md`, Props. 1 and 8; tested prediction alignment | Only quadratic scores. Do not extend unchanged to log loss, accuracy, or AUROC. |
| C3 | A randomized strength-t OA cancels product-fANOVA components through order t for its declared finite target. | Exact finite-product result | `THEORY_FOUNDATIONS.md`, Props. 2, 7, and 11; design-balance tests | Classical OA/ANOVA principle. Claim only the pipeline specialization and accounting. |
| C4 | At B=16, OC2-coupled has lower method-relative residual than canonical-independent in 144/144 neural cells and 12/12 datasets. | Confirmed, frozen | `final_closure/summaries/experiment_a_cells.csv`; Fig. 1; equal-source reduction 55.9%, clustered 95% [38.7%, 73.8%] | The methods do not necessarily share an estimand. Say “method-relative residual,” not “unbiased apples-to-apples dominance.” |
| C5 | Coupling schema, initialization, and order is the mechanism of the finite-target gain. | Supported, not fully identified | Same-target Experiment D: all factors 48/48 wins, 47.6% reduction; Fig. 10; fANOVA masses | Say “supports/attributes,” not “proves the only mechanism.” Token/dropout details remain architecture-specific. |
| C6 | Schema strength-2 balance with fresh independent RNG does not beat canonical seed ensembling. | Confirmed negative | OC2-independent: 5/144 cell wins, 0/12 source wins, -7.0%, 95% [-7.8%, -6.3%] | Keep in abstract, results, and limitations. Never collapse OC2-independent and OC2-coupled. |
| C7 | Semantic symmetrization can change the target prediction. | Confirmed descriptive/MC | Mean squared canonical-to-joint distance 2.632e-4; 10/144 over MC threshold; canonical-to-coupled 2.494e-3 | Do not say target shift is always statistically significant or predictively beneficial. |
| C8 | The average strength-2 advantage over SRS does not persist at convergence. | Confirmed negative | Experiment B mean OC2/SRS ratio 1.001683; Figs. 4 and 6 | Do not advertise asymptotic variance reduction. Raw trajectories are primary; slopes are descriptive. |
| C9 | Exact matched initialization nearly removes schema residual for MLP/ResNet and leaves transient architecture-specific residual for FT/TabM. | Confirmed boundary | Experiment B matched-function table; Fig. 9 | Do not claim universal irreducible optimizer-path dependence. |
| C10 | Better quotient estimation improves validation fidelity but not materially held-out selection regret in this panel. | Confirmed boundary | Winner agreement 99.41% vs 96.69%; test regret .004906 vs .005029; validation/test exact winners 19/36 | Do not headline model-selection performance or imply population-shift robustness. |

## Quantitative headline ledger

| Quantity | Value | Unit of aggregation | Source |
| --- | ---: | --- | --- |
| Primary neural cells | 144 | 12 datasets × 3 splits × 4 models | Experiment A |
| OC2-coupled vs canonical cell wins | 144/144 | cell, descriptive | Experiment A/Fig. 1 |
| OC2-coupled vs canonical source wins | 12/12 | dataset mean | Experiment A/Fig. 1 |
| Equal-source mean reduction | 55.9% | dataset | Experiment A |
| Dataset-clustered 95% interval | [38.7%, 73.8%] | dataset bootstrap | Experiment A |
| Architecture reductions (FT/MLP/ResNet/TabM) | 39.1/93.2/58.1/90.2% | architecture-stratified | Experiment A/Fig. 2 |
| OC2-independent reduction | -7.0% | dataset | Experiment A |
| OC2-independent interval | [-7.8%, -6.3%] | dataset bootstrap | Experiment A |
| Canonical-to-joint squared target distance | 2.632e-4 | mean cell | Experiment A/Fig. 3 |
| Cells above target-distance MC threshold | 10/144 | cell | Experiment A |
| Same-target all-factor reduction | 47.6% | 48-cell mean | Experiment D/Fig. 10 |
| Same-target all-factor wins | 48/48 | cell | Experiment D |
| Convergence mean OC2/SRS ratio | 1.001683 | mandatory corners | Experiment B/Fig. 6 |
| Main+pair fraction vs gain | rho=.139, CI [-.017,.320] | dataset-clustered | Experiment C/Fig. 7 |
| Higher-order fraction vs gain | rho=.258, CI [.038,.415] | dataset-clustered | Experiment C/Fig. 7 |
| Exact validation/test winner agreement | 19/36 | partition | prior selection audit |
| Audited unique fits | 140,592 | registry key | final audit |
| Audit tests | 116/116 passed | test | final audit |

## Figure-to-claim map

| Main figure | Message | Claims | Caveat shown with it |
| --- | --- | --- | --- |
| Fig. 1 `figure_1_independent_seed_showdown` | Coupled small-budget residual advantage across the full neural panel | C4, C6 | Label residuals by their estimand; include independent failure |
| Fig. 2 `figure_3_expectation_distance` | The targets are measurably distinct in some cells | C7 | Distance is not performance improvement |
| Fig. 3 `figure_10_coupling_mechanism` | Same-target joint balance outperforms partial balance | C5 | 48-cell mechanism panel; not every possible factor |
| Fig. 4 `figure_6_orbitcover_convergence` | Relative efficiency approaches the SRS boundary | C8 | Nuisance variance itself need not vanish |
| Fig. 5 `figure_9_matched_convergence` | Architecture-specific matched-path boundary | C9 | Initial and converged residuals are different |
| Appendix `figure_7_interaction_predicts_gain` | Interaction order is only a weak predictor | C8 | Transparent predictor is not a reliable cell oracle |
| Appendix `figure_8_failure_cell_spectra` | Failures are retained, not filtered | C8 | Strength 3 has zero recoveries in the designated loss panel |

## Claims excluded from the submission

- “OrbitCover is universally better than independent seed ensembling.”
- “Schema balance alone explains the gain.”
- “All compared methods estimate the same expectation.”
- “Strength 2 or strength 3 always beats SRSWOR.”
- “The efficiency advantage persists at convergence.”
- “Equivalent schemas necessarily induce different converged functions.”
- “OrbitCover substantially improves held-out predictive performance.”
- “Orthogonal arrays, fANOVA filtering, group averaging, antithetic sampling,
  or U-statistic risk estimation are new.”
- “The 232 fit-hours imply a portable wall-clock or energy speedup.”
- “Rows, cached draws, representatives, or split repeats are independent
  inferential units.”

## Reviewer-objection preflight

| Likely objection | Evidence to surface, not bury | Residual weakness |
| --- | --- | --- |
| The baseline reused a favorable seed menu. | 128 canonical independent seeds, unique master seeds, 144-cell B=16 showdown | Reference remains finite Monte Carlo |
| The comparison changes the estimand. | Cross-target distances and 10/144 MC exceedances in main text | Choosing a deployment nuisance law remains application-dependent |
| The benefit is just schema balancing. | OC2-independent failure and same-target eight-arm coupling ablation | Fine-grained token/dropout causal mechanism is incomplete |
| The effect is undertraining. | Nested N×budget grid, exact convergence corners, matched convergence | Scope remains tabular; mean SRS advantage vanishes |
| SRS without replacement is enough. | Show both small-budget win and convergence failure; full failure table | No simple interaction statistic reliably predicts every cell |
| Stable validation should improve test selection. | 19/36 winner agreement and near-null regret change | Evaluation-sample/partition shift remains unsolved |
| The mathematical ingredients are old. | State OA/fANOVA/group averaging as prior art in Introduction and Related Work | Novelty rests on the complete-pipeline estimand, composition, and boundary audit |

## Release gate

The paper can use a claim only if:

1. its wording matches a row above;
2. the reported number regenerates from `../final_closure/summaries/`;
3. its statistical unit is dataset/source unless explicitly labeled
   descriptive;
4. the corresponding negative control appears in the same section or figure;
5. the references do not assign novelty to a classical ingredient.

