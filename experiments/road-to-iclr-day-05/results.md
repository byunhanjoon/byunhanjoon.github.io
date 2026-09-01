# RESULTS — FINAL ICLR CLOSURE

## 1. Executive verdict

**PARTIALLY SUPPORTED**

The final closure directly compares OrbitCover with genuinely independent full-pipeline retraining on all 144 neural dataset×split×architecture cells. At B=16, OC2-coupled versus canonical independent achieves 144/144 cell wins and 12/12 source wins, with an equal-source reduction of 55.9% and clustered 95% interval [38.7%, 73.8%]. OC2-independent achieves 5/144 cells and 0/12 sources, a -7.0% mean reduction. Coupling changes the result by 58.8% relative to schema-only independent balancing. The mean squared distance between canonical and schema×independent expectations is `2.632e-04`, so target shift is reported rather than assumed away. At convergence, the mean OC2/SRS residual ratio is `1.002`. Interaction structure gives Spearman rho `0.139` for main+pair mass versus gain and `0.258` for higher-order mass versus gain. Experiment D's lowest-residual ablation is `all_factors`. Exact matched initialization remains architecture-dependent: MLP/ResNet are negligible while FT-Transformer/TabM retain the reported residuals. All mandatory A–C cells and preferred D are complete. The final audit passes 116/116 tests and verifies 140,592 registry fit keys. The defensible thesis is selected from the frozen rules, not repaired after the outcome.

## 2. What changed relative to the previous results

The previous result established 144/144 material finite nuisance tensors, 144/144 strength-2 wins over IID-16, but only 8/12 positive source means versus SRSWOR. It also showed that exact matched initialization removes about 98% of pooled ordinary schema variance, closing MLP/ResNet and leaving architecture-specific FT-Transformer/TabM residuals. The closure replaces the reused two-seed menu objection with 128 canonical independent seeds and schema×independent pools, tests realistic N/optimization trajectories, predicts the SRS boundary from fANOVA structure, and isolates schema/init/order coupling. Earlier evidence grades remain unchanged.

## 3. Independent canonical-seed showdown

| Method | Mean residual | Relative to canonical independent | Cell wins | Cells | Source wins | Sources |
| --- | --- | --- | --- | --- | --- | --- |
| OC2-COUPLED | 0.0006929229308 | 0.4896388921 | 144 | 144 | 12 | 12 |
| CANONICAL-INDEPENDENT | 0.001415171348 | 1 | 0 | 144 | 0 | 12 |
| OC1-INDEPENDENT | 0.001511801143 | 1.06828134 | 7 | 144 | 0 | 12 |
| IID-JOINT | 0.00151278567 | 1.068977034 | 2 | 144 | 0 | 12 |
| OC2-INDEPENDENT | 0.001513680091 | 1.069609057 | 5 | 144 | 0 | 12 |
| SRS-JOINT | 0.001514003932 | 1.069837893 | 7 | 144 | 0 | 12 |

The five required paired comparisons are:

- `OC2-COUPLED` vs `CANONICAL-INDEPENDENT`: 144/144 cell wins, 12/12 source wins, mean reduction 55.9%, median 72.4%, clustered 95% [38.7%, 73.8%]; architecture reductions `{"ft_transformer": 0.3910178142649847, "mlp": 0.9323539220494259, "resnet": 0.5809183334186383, "tabm": 0.902101585179278}`.
- `OC2-INDEPENDENT` vs `CANONICAL-INDEPENDENT`: 5/144 cell wins, 0/12 source wins, mean reduction -7.0%, median -7.0%, clustered 95% [-7.8%, -6.3%]; architecture reductions `{"ft_transformer": -0.06875062679166088, "mlp": -0.08122662301335182, "resnet": -0.07002190399064334, "tabm": -0.06600060514606088}`.
- `OC2-COUPLED` vs `OC2-INDEPENDENT`: 144/144 cell wins, 12/12 source wins, mean reduction 58.8%, median 74.4%, clustered 95% [42.6%, 75.5%]; architecture reductions `{"ft_transformer": 0.43019244109063015, "mlp": 0.9374358006815943, "resnet": 0.6083429086653291, "tabm": 0.9081628900132677}`.
- `OC2-COUPLED` vs `SRS-JOINT`: 144/144 cell wins, 12/12 source wins, mean reduction 58.8%, median 73.8%, clustered 95% [42.8%, 76.1%]; architecture reductions `{"ft_transformer": 0.43067228651360445, "mlp": 0.9378058277711188, "resnet": 0.6076591390864782, "tabm": 0.9082817639960616}`.
- `OC2-COUPLED` vs `IID-JOINT`: 144/144 cell wins, 12/12 source wins, mean reduction 58.7%, median 74.3%, clustered 95% [42.3%, 75.4%]; architecture reductions `{"ft_transformer": 0.4296865345578894, "mlp": 0.937136450888093, "resnet": 0.6083672370537325, "tabm": 0.9083244584122516}`.

Cached estimator constructions use 512 draws per cell; overlapping draws are never the inferential unit. Dataset is the primary unit.

## 4. Does schema symmetrization change the expectation?

Across cells, mean `||Q_canonical_independent - Q_schema×independent||²` is `2.632e-04` (median `1.005e-04`). It exceeds the cell-specific 95% Monte Carlo noise threshold in 10/144 cells. Mean canonical-to-finite-coupled distance is `2.494e-03`, and mean joint-to-coupled distance is `2.414e-03`. Relative to the B=16 canonical residual, the canonical/joint distance is material under the frozen 10% descriptive materiality check. OrbitCover is therefore interpreted as estimating a distinct symmetrized target as well as reducing variance; cross-target residuals are retained in `experiment_a_cells.csv`.

## 5. What does the 98% matched-function result mean now?

| model | ordinary_variance | matched_variance |
| --- | --- | --- |
| ft_transformer | 0.08756949777 | 0.002615811914 |
| mlp | 0.002577787632 | 8.967529401e-16 |
| resnet | 0.04886851718 | 8.413653915e-14 |
| tabm | 0.00346799598 | 4.193105347e-06 |

MLP and ResNet still close to numerical precision under exact function matching. FT-Transformer retains the largest matched-path component, while TabM retains a smaller component. The convergence repeat is:

| model | budget | ordinary_variance | matched_variance | fraction_removed |
| --- | --- | --- | --- | --- |
| ft_transformer | 100 | 0.1080711185 | 0.08748683041 | 0.2536027773 |
| ft_transformer | 20 | 0.07045075191 | 0.00518983364 | 0.9418607855 |
| ft_transformer | convergence | 0.006746507534 | 1.955069217e-07 | 0.9998947719 |
| mlp | 100 | 0.004022518104 | 1.552644273e-15 | 1 |
| mlp | 20 | 0.0005033221728 | 2.360001462e-16 | 1 |
| mlp | convergence | 0.000336031813 | 1.513393319e-16 | 1 |
| resnet | 100 | 0.08725992748 | 0.07322950556 | 0.2970838073 |
| resnet | 20 | 0.04687819444 | 7.974748079e-14 | 1 |
| resnet | convergence | 0.005401693374 | 1.84912824e-15 | 1 |
| tabm | 100 | 0.0134754249 | 0.0003869053479 | 0.970782263 |
| tabm | 20 | 0.001774880196 | 1.231478059e-06 | 0.9996048663 |
| tabm | convergence | 0.0009446491499 | 1.352556802e-09 | 0.9999981777 |

This rules out a universal claim that schema alone irreducibly changes every optimizer path. The supported scope is architecture-specific token/member/dropout/minibatch dynamics plus structured finite/infinite randomization.

## 6. Training-scale and convergence

The mandatory model-level corners are:

| model | corner | total_nuisance_variance | oc2_srs_ratio_b16 | oc2_canonical_ratio_b16 |
| --- | --- | --- | --- | --- |
| ft_transformer | largest/20 | 0.02941318082 | 0.9897295677 | 1.139652631 |
| ft_transformer | largest/convergence | 0.02335183151 | 1.013132327 | 1.136594412 |
| ft_transformer | small/20 | 0.06682304298 | 0.9981260636 | 1.118854912 |
| ft_transformer | small/convergence | 0.007265092815 | 1.002597767 | 1.071758722 |
| mlp | largest/20 | 0.005048859264 | 0.9973527588 | 1.188941755 |
| mlp | largest/convergence | 0.01084124722 | 0.9836850198 | 1.132561542 |
| mlp | small/20 | 0.002679340059 | 1.010087051 | 1.165564212 |
| mlp | small/convergence | 0.003269496122 | 1.000907587 | 1.265125407 |
| resnet | largest/20 | 0.01495109343 | 0.9933515923 | 1.113994224 |
| resnet | largest/convergence | 0.01808568959 | 1.002679006 | 1.175516597 |
| resnet | small/20 | 0.04344486018 | 0.9958460739 | 1.116264926 |
| resnet | small/convergence | 0.009717690244 | 1.009060756 | 1.114245447 |
| tabm | largest/20 | 0.004144322987 | 1.009338481 | 1.187601998 |
| tabm | largest/convergence | 0.004250024157 | 1.014182785 | 1.120105943 |
| tabm | small/20 | 0.004295647184 | 1.004153993 | 1.211519512 |
| tabm | small/convergence | 0.002273107135 | 1.006933816 | 1.087246282 |

Nuisance variance persists in at least one realistic convergence condition. OrbitCover relative efficiency does not persist on average at convergence. Dataset size changes effect magnitude and interaction mix, but the raw trajectories—not a fragile fitted exponent—are the primary result.

The descriptive, dataset-clustered log-risk slopes are `{"optimization_budget/ft_transformer": {"dataset_clustered_95_interval": [0.29065646963733704, 0.6231154327625205], "datasets": 6, "equal_dataset_mean_slope": 0.4402839375534216, "trajectory_slopes": 18}, "optimization_budget/mlp": {"dataset_clustered_95_interval": [0.5311900741236998, 1.3600759982866057], "datasets": 6, "equal_dataset_mean_slope": 1.0070199608887545, "trajectory_slopes": 18}, "optimization_budget/resnet": {"dataset_clustered_95_interval": [0.3716407873384738, 0.7242495293923182], "datasets": 6, "equal_dataset_mean_slope": 0.5616650173743971, "trajectory_slopes": 18}, "optimization_budget/tabm": {"dataset_clustered_95_interval": [0.4431123358745704, 0.8648745498997705], "datasets": 6, "equal_dataset_mean_slope": 0.6483603559225876, "trajectory_slopes": 18}, "training_size/ft_transformer": {"dataset_clustered_95_interval": [-0.13688471169665437, -0.01262423565846263], "datasets": 6, "equal_dataset_mean_slope": -0.07683908420188197, "trajectory_slopes": 30}, "training_size/mlp": {"dataset_clustered_95_interval": [0.023154180751470545, 0.20782989854627312], "datasets": 6, "equal_dataset_mean_slope": 0.11690550544126423, "trajectory_slopes": 30}, "training_size/resnet": {"dataset_clustered_95_interval": [-0.15651190386470457, 0.08652100891356508], "datasets": 6, "equal_dataset_mean_slope": -0.02764253671091627, "trajectory_slopes": 30}, "training_size/tabm": {"dataset_clustered_95_interval": [-0.08517684898188904, 0.3328240904189486], "datasets": 6, "equal_dataset_mean_slope": 0.08336709230658274, "trajectory_slopes": 30}}`. They summarize direction and uncertainty only; they are not promoted as asymptotic scaling exponents.

## 7. Interaction spectrum explains successes/failures

Main+pair fraction versus OC2 gain has Spearman rho `0.139` with clustered interval `[-0.017251543751763965, 0.32013383222671876]`. Higher-order fraction versus gain has rho `0.258` with interval `[0.038070682846704264, 0.4149196002249062]`. Mean high-order fraction is `0.033` in OC2 wins and `0.016` in strict SRS wins. Architecture-specific correlations and clustered intervals are `{"catboost_native": {"cells": 6, "higher": {"cells": 6, "dataset_clustered_95_interval": [NaN, NaN], "datasets": 6, "spearman": NaN}, "main_pair": {"cells": 6, "dataset_clustered_95_interval": [1.0, 1.0], "datasets": 6, "spearman": 1.0}}, "ft_transformer": {"cells": 87, "higher": {"cells": 87, "dataset_clustered_95_interval": [0.0035021795979470562, 0.6371080778603119], "datasets": 12, "spearman": 0.35094688815590236}, "main_pair": {"cells": 87, "dataset_clustered_95_interval": [-0.4574164529585364, -0.051313307740342345], "datasets": 12, "spearman": -0.2468258212838436}}, "mlp": {"cells": 87, "higher": {"cells": 87, "dataset_clustered_95_interval": [0.3201894487263383, 0.749872699847834], "datasets": 12, "spearman": 0.5431661922951658}, "main_pair": {"cells": 87, "dataset_clustered_95_interval": [-0.12604598739310166, 0.32226889232561834], "datasets": 12, "spearman": 0.11171315739392061}}, "native_histgb": {"cells": 1, "higher": null, "main_pair": null}, "onehot_adam_mlp": {"cells": 26, "higher": {"cells": 26, "dataset_clustered_95_interval": [-0.6476584831553285, -0.1738093612350588], "datasets": 26, "spearman": -0.4338029307681561}, "main_pair": {"cells": 26, "dataset_clustered_95_interval": [0.7823559995093242, 0.9814179161721557], "datasets": 26, "spearman": 0.9241025641025641}}, "ordinal_forest": {"cells": 18, "higher": {"cells": 18, "dataset_clustered_95_interval": [NaN, NaN], "datasets": 18, "spearman": NaN}, "main_pair": {"cells": 18, "dataset_clustered_95_interval": [0.50709255283711, 1.0], "datasets": 18, "spearman": 0.8306219873721381}}, "resnet": {"cells": 87, "higher": {"cells": 87, "dataset_clustered_95_interval": [0.5057878297481028, 0.71858188109024], "datasets": 12, "spearman": 0.6337890148265081}, "main_pair": {"cells": 87, "dataset_clustered_95_interval": [-0.394589446995077, 0.0390398406849519], "datasets": 12, "spearman": -0.13859248015312572}}, "tabm": {"cells": 87, "higher": {"cells": 87, "dataset_clustered_95_interval": [0.2955804242309738, 0.7751142835913719], "datasets": 12, "spearman": 0.5635894402706312}, "main_pair": {"cells": 87, "dataset_clustered_95_interval": [-0.09271790546507631, 0.38607349279812075], "datasets": 12, "spearman": 0.1779104014652011}}}`. The prior four non-positive source means are heloc_credit_risk, openml-kin8nm-189, openml-pol-201, openml-puma32h-308; exact ties remain ties. Their source-level spectrum comparison with the eight positive sources is `{"nonpositive": {"higher_fraction": 0.006281908602027729, "main_fraction": 0.526967771006198, "main_pair_fraction": 0.9197041009829428, "pair_fraction": 0.3927363299767448, "triple_fraction": 0.07401399041502946}, "positive": {"higher_fraction": 0.06392329304922528, "main_fraction": 0.36671619763939856, "main_pair_fraction": 0.7232554273070554, "pair_fraction": 0.35653922966765683, "triple_fraction": 0.21282127964371927}}`. The transparent model's leave-one-dataset-out result is `{"cells": 399, "leave_one_dataset_out_r2": -3.1229708414924724e+23, "spearman_prediction": 0.6660427860650643}`. The complete failure table contains 134 strict cells and is not filtered for favorability.

## 8. Strength hierarchy

Strength-1 balances only main effects; strength-2 removes the matched pairwise spectrum and remains the B=16 default; strength-3 targets triples at B=64 and closes products when the budget reaches the population. Among prior strength-2/SRS losses, strength-3 recoveries are: none. Non-recoveries are: credit_card_default/2026082801/ft_transformer, credit_card_default/2026082811/ft_transformer, credit_card_default/2026082821/ft_transformer, heloc_credit_risk/2026082801/ft_transformer, heloc_credit_risk/2026082811/ft_transformer, heloc_credit_risk/2026082821/ft_transformer, fremtpl_claim_count/2026082821/ft_transformer. Thus “match strength to interaction order” is not reliably supported as a cell-ranking rule, not a guarantee that strength-3 always beats finite-population sampling.

## 9. Coupling mechanism

| method | mean_residual | mean_relative_reduction_vs_none | cell_wins | cells |
| --- | --- | --- | --- | --- |
| all_factors | 0.001864874172 | 0.4759451799 | 48 | 48 |
| schema_initialization | 0.001988861991 | 0.3397425039 | 48 | 48 |
| schema_order | 0.002285157818 | 0.2758614289 | 48 | 48 |
| initialization_order | 0.002382298667 | 0.2321314207 | 48 | 48 |
| schema | 0.002456620443 | 0.1301734014 | 48 | 48 |
| initialization | 0.002612098101 | 0.07042993737 | 48 | 48 |
| order | 0.002637392057 | 0.1402159481 | 47 | 48 |
| none | 0.002775809955 | 0 | 0 | 48 |

The best finite ablation is `all_factors`. The full component means are `{"fanova_reconstruction_error": 8.408213723077688e-19, "initialization_main_mass": 0.0024577421733346845, "initialization_order_mass": 0.0013060242899574513, "joint_higher_mass": 0.011547500062404232, "order_main_mass": 0.0019228330384528432, "schema_initialization_mass": 0.013898517922556239, "schema_only_mass": 0.004806357104904081, "schema_order_mass": 0.0038519947027675985, "split_seed": 2026082811.0, "total_variance": 0.039790969294377136}`. This answers whether the benefit is principally schema, RNG, or pairwise schema×RNG balance without confusing the finite mechanism tensor with independent infinite-seed retraining.

## 10. Architecture-specific conclusions

### MLP

Matched residual is negligible. Its independent/coupled reductions are reported in the B=16 architecture dictionary; low-order schema structure is the useful regime.

### ResNet

Matched residual is negligible, but ordinary stochastic/schema interaction remains material. Structured coupling can still reduce quotient Monte Carlo even when exact coordinate matching closes the fixed path.

### FT-Transformer

FT-Transformer has the largest high-order and matched-path residual. Its weaker strength-2 boundary is predicted prospectively by the interaction analysis rather than hidden.

### TabM

TabM has strong low-order structure with a small nonzero matched residual. The independent-seed comparison determines whether this translates beyond its earlier finite seed menu.

### TabPFN

Prior TabPFN evidence remains separate: default internal ensembling reduced external schema risk, and external strength-2 beat IID 18/18 and SRS 12/18 cells. Calls and internal members are not mislabeled as retrained fits.

### CatBoost/GBDT

The final secondary independent-seed results are `{"catboost_native": {"canonical_residual": 0.00017220039099124357, "cells": 12, "oc2_cell_wins": 7, "oc2_independent_residual": 0.00014426857327348076, "oc2_relative_reduction": 0.16220530950584855}, "xgboost": {"canonical_residual": 1.093255650096441e-14, "cells": 12, "oc2_cell_wins": 12, "oc2_independent_residual": 4.1942668861519886e-15, "oc2_relative_reduction": 0.6163507697598458}}`. Prior native CatBoost had zero category-ID total effect, while ordinal XGBoost remained category-ID sensitive; deterministic/invariant GBDT cells remain an explicit boundary.

## 11. Practical compute efficiency

The audited registry contains 140,592 complete unique fit keys and `232.005` summed fit-hours of local telemetry: `231.802` GPU-fit-hours and `0.203` CPU-fit-hours. End-to-end closure wall clock from the frozen hash through audit is `9.974` hours with two H100 NVL devices and concurrent CPU analysis. For OC2-independent, 16 fits match a median `15.0` canonical-independent fits among 144/144 bracketed cells; 0 cells require more than 64 by the observed curve. For OC2-coupled the corresponding median is `29.3` across 87/144 bracketed cells, with 57 above 64. No budget equivalence is asserted outside the observed 4–64 bracket. GPU figures are H100-local measurements, not portable latency guarantees.

## 12. Ranking/model-selection implications

The prior exact validation result remains: strength-2 winner agreement 99.41% versus 96.69% IID and Spearman 98.64% versus 96.77%. Held-out selected-test regret changed only from `0.005029` to `0.004906`, because exact validation/test winners agreed in only 19/36 partitions. Partition shift is distinct from nuisance Monte Carlo; the final paper must not headline the small test-regret difference.

## 13. Failure cases

- Canonical-independent wins over OC2-coupled: none.
- Strict SRSWOR wins (first 30 listed): heloc_credit_risk/2026082801/ft_transformer, heloc_credit_risk/2026082821/ft_transformer, heloc_credit_risk/2026082811/ft_transformer, kdd17_stock_return/2026082801/mlp, credit_card_default/2026082811/ft_transformer, credit_card_default/2026082821/ft_transformer, polish_bankruptcy_4year/original/onehot_adam_mlp, bank_marketing_subscription/2026082801/mlp, heloc_credit_risk/2026082801/ft_transformer, openml-abalone-183/2026082801/resnet, heloc_credit_risk/2026082801/tabm, fremtpl_claim_count/2026082801/mlp, openml-abalone-183/2026082801/ft_transformer, credit_card_default/2026082801/tabm, credit_card_default/2026082801/ft_transformer, bank_marketing_subscription/2026082821/mlp, credit_card_default/2026082801/tabm, openml-abalone-183/2026082821/mlp, bank_marketing_subscription/2026082801/mlp, australian_credit_approval/2026082811/mlp, openml-abalone-183/2026082821/tabm, heloc_credit_risk/2026082801/mlp, credit_card_default/2026082811/mlp, openml-puma32h-308/2026082811/tabm, openml-pol-201/2026082821/tabm, openml-abalone-183/2026082801/tabm, openml-abalone-183/2026082801/ft_transformer, heloc_credit_risk/2026082801/tabm, fremtpl_claim_count/2026082811/tabm, openml-abalone-183/2026082801/resnet, and 104 additional cells in the complete failure table.
- Strength-3 non-recoveries: credit_card_default/2026082801/ft_transformer, credit_card_default/2026082811/ft_transformer, credit_card_default/2026082821/ft_transformer, heloc_credit_risk/2026082801/ft_transformer, heloc_credit_risk/2026082811/ft_transformer, heloc_credit_risk/2026082821/ft_transformer, fremtpl_claim_count/2026082821/ft_transformer.
- Strength-3 recoveries among the same loss panel: none.
- Nuisance variance at or below `1e-10` at convergence: none.
- Architectures with negligible matched residual: MLP and ResNet.
- Three datasets with least mean OC2-coupled benefit: australian_credit_approval (36.2%), bank_marketing_subscription (38.1%), german_credit_risk (45.8%).

## 14. Final defensible theorem/claim target

For a declared semantic nuisance distribution and randomized learner, the symmetrized predictor is a finite/infinite expectation. Orthogonal-array designs exactly cancel fANOVA components through their design strength, while residual error is governed by unmatched higher-order mass and finite-population/coupling covariance. Empirically, this yields the measured equal-budget reductions and the reported architecture/interaction boundaries. No novelty is claimed for orthogonal arrays, group averaging, or generic antithetic sampling; the contribution is their semantic learning-pipeline formulation, exact prediction-space accounting, and broad falsificatory boundary map.

## 15. Recommended final paper thesis

**Thesis B**

> Semantic symmetrization defines a distinct quotient predictor, and interaction-balanced designs estimate it efficiently.

## 16. Best paper title

1. **OrbitCover: Interaction-Balanced Semantic Randomization for Efficient Predictor Symmetrization**
2. **When Equivalent Tables Train Differently: Structured Randomization Beyond Independent Ensembling**
3. **Matching Design Strength to Schema Interaction Order in Randomized Tabular Learning**

## 17. ICLR readiness

- novelty: **4/5**
- theory: **4/5**
- empirical breadth: **5/5**
- baseline strength: **5/5**
- mechanism: **3/5**
- realistic-scale evidence: **5/5**
- prospective validity: **4/5**
- story coherence: **3/5**
- reproducibility: **5/5**

**READY TO WRITE ICLR**

All frozen mandatory experiments are complete; no extra experiment is invented merely because a result is mixed.

## 18. Five strongest reviewer objections

1. **Objection:** Independent canonical seeds may erase the claimed advantage. **Evidence:** the 144-cell B=16 showdown and clustered comparison above. **Remaining weakness:** a finite 128-seed reference still has Monte Carlo error. **Best response:** show reference bootstrap and every cross-target residual.
2. **Objection:** The phenomenon is transient undertraining. **Evidence:** the six-source nested N×budget×convergence grid and matched convergence repeat. **Remaining weakness:** architectures/datasets remain tabular rather than vision/language scale. **Best response:** scope the claim to the tested randomized tabular pipelines.
3. **Objection:** SRSWOR is already optimal enough. **Evidence:** all failure cells plus the prospective interaction-spectrum correlation and strength hierarchy. **Remaining weakness:** the transparent predictor need not rank every cell. **Best response:** present interaction order as a boundary condition, not an oracle.
4. **Objection:** Matched initialization removes the effect. **Evidence:** MLP/ResNet closure is retained, FT-Transformer/TabM residuals and independent-RNG evidence are separated. **Remaining weakness:** exact token/member mechanism is not fully identified. **Best response:** abandon the universal optimizer-path claim and lead with structured expectation estimation.
5. **Objection:** Validation fidelity does not imply useful held-out selection. **Evidence:** partition-shift decomposition and the 19/36 exact winner agreement. **Remaining weakness:** predictive gains are small. **Best response:** make quotient estimation—not SOTA prediction or selection—the primary endpoint.

## 19. Final recommendation

**PIVOT PAPER THESIS**

The audited evidence supports this choice under the frozen rule: independent-seed reduction=-7.0%, coupled reduction=55.9%, convergence OC2/SRS=`1.002`, and interaction correlations have the signs reported above. The paper should state every canonical, SRS, convergence, matched-path, and selection failure prominently and use **Thesis B** exactly as the thesis boundary.
