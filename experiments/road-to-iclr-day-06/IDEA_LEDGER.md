# Day-6 idea ledger

This is a chronological decision ledger, not the final ranking.  “Keep” means
retain for the next falsification step; it does not imply an ICLR-ready claim.

| Idea | Theory prediction | Evidence state | Decision now | Principal risk |
|---|---|---|---|---|
| H1 Semantic Arithmetic Amplification / IEA64 | exact schema conjugacy can be broken only by the first reduction order; a higher-precision interface can close the whole path | confirmed on 3 datasets × 3 models × 8 seeds, 72 bundles / 576 paths; 9/9 exact IEA64 cells | **keep, narrowed** | generic bit-level amplification is known; no mean accuracy gain |
| H2 Precision-Delay Law | hitting time is monotonically ordered by nominal unit roundoff | 18 frozen bundles; float16 separates before bfloat16 in all FT datasets | **discard as written** | effective kernel/quantization error is not a scalar unit-roundoff law |
| H3 Full-Scale Closure | IEA64 stays exact and FP32 instability persists on all rows through 200 epochs at acceptable overhead | complete 36 bundles / 288 paths; exact-cell 0/9 and stable-control 3/6 fail, while FT-material 3/3, timing 3/3, and canonical-loss +.282% pass | **discard universal claim** | closure is finite-horizon and architecture boundaries are not stable |
| H4 Semantic Shadow Forecast | epoch-2 schema discrepancy ranks epoch-20 discrepancy across optimizer configs | complete 324 bundles / 972 paths; pooled AUROC .916, but 0/3 FT per-dataset rho gates and stable-control fraction .861 fail | **discard as written** | pooled discrimination does not overcome weak datasetwise rank transfer or unstable controls |
| H5 Cross-Perturbation Fragility Transfer | the same response operator makes an early schema shadow rank later independent-seed prediction variance | complete tensor reuse; 0/3 FT rho gates, pooled rho .375, top-quartile AUROC .704 | **discard as written** | schema roundoff and independent-seed perturbations do not share a reliable configuration ranking |
| H6 Semantic Lyapunov Screen | log orbit-growth slope through epoch 20 predicts material epoch-200 divergence better than the epoch-20 level | complete 33 tests; AUROC 1.0, rank rho .701, sensitivity 1.0, specificity .800, but zero AUROC gain over the raw level | **discard as an incremental screen** | the raw epoch-20 level is already perfect by AUROC |
| H7 Rounding-Cell Survival | IEA64 suppresses rather than eliminates the conditional hazard of crossing a float32 rounding boundary | complete 31 bundles / 93 paths; later-hit .952, exact-early .968, final wins 1.0, IEA64 material failures .204 | **keep, narrowly supported** | Credit failure rate is .333 and only three datasets were tested |
| H8 Level-or-Acceleration Screen | log-orbit slope increase detects unstable-mode takeover while an epoch-20 level branch handles already-material paths | complete 29 tests; balanced accuracy .944, delayed recall 1.0, but zero accuracy improvement over H6 | **discard as an incremental successor** | high accuracy is not incremental value |
| H9 Post-Breach Arithmetic Attenuation | smaller interface-injection covariance can reduce final orbit energy even after exact closure breaks | complete 25 bundles / 75 paths; 51/51 eligible final wins, 36/51 rescues, no twofold worsening, all 3 ratio gates, canonical loss +.760% | **keep, narrowly supported** | thresholds were development-informed and the prospective split is runtime-ordered/unbalanced |

## Current scientific update

H1 survives H2 because exact IEA64 closure is a pathwise observation rather
than a nominal-precision scaling law.  H2 removes the tempting but unsupported
claim that “more bits simply delay divergence.”  Partial H3 further changes the
architecture story: Bank ResNet is stable at epoch 20 and unstable at epoch
50+, so short runs cannot establish a stable-model class.

The strongest Day-6 route would require a useful consequence beyond numerical
reproducibility.  H4 tests same-source early forecasting, H5 tests transfer to
ordinary seed fragility, and H6 tests long-horizon delayed-instability
screening.  Each has a frozen failure rule and none is promoted yet.
