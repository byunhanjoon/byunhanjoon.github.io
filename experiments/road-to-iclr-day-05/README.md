# Road to ICLR — Day 5

This directory contains the protocols, implementations, complete prediction
tensors, exact analyses, failures, and paper synthesis for the Day-5
**OrbitCover** direction.

The core question is whether exact tabular representation equivalences and
training seeds can be treated as a finite nuisance product, then integrated
more efficiently than IID repetition using randomized strength-matched covers.
The answer is strongly positive for quotient Brier/MSE estimation and
validation-side model selection, with explicit high-order and
validation-to-test counterexamples.

Start with:

- [`PAPER_BLUEPRINT.md`](PAPER_BLUEPRINT.md) — evidence-weighted paper claim;
- [`THEORY_FOUNDATIONS.md`](THEORY_FOUNDATIONS.md) — Propositions 1--35;
- [`RECENT_LITERATURE_AUDIT.md`](RECENT_LITERATURE_AUDIT.md) — novelty boundary;
- [`ANTITHETIC_CV_COMPARISON.md`](ANTITHETIC_CV_COMPARISON.md) — closest-work operator separation;
- [`REVIEWER_ATTACK_AUDIT.md`](REVIEWER_ATTACK_AUDIT.md) — adversarial submission-readiness audit;
- [`results.md`](results.md) — required evidence-weighted scientific verdict;
- `DAY5_FINAL_REPORT.md` — pre-matched-control sprint report and detailed evidence history;
- `EXPERIMENT_LEDGER.md` — pass/fail/evidence-status ledger (written at sprint end).

## Evidence hierarchy

Protocols labeled **frozen before outcomes** are the strongest evidence.
Changed-menu and changed-subsample runs test conditional replication.
Prospective external panels were specified before their tensors were fit.
Files labeled **post-gate**, **post-failure**, or **diagnostic** are supporting
mechanism or scope analyses and never retroactively enlarge a frozen gate.

The major retained failure modes are external validation-to-test rank instability,
high-dimensional marginal-balance competition, prediction-dependent stopping,
quotient-HPO selection, and the semantic HeteroBag placebo.

## Reproduction levels

Use a Python environment with NumPy, pandas, SciPy, scikit-learn, matplotlib,
pytest, and the model libraries referenced by the fit scripts. The original
run used:

```bash
PYTHON=/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python
```

### 1. Algebra and construction tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" -m pytest -q
```

These tests do not refit models. They verify fANOVA identities, cover margins,
mixed-level constructions, exact covariance coefficients, SRSWOR identities,
phase boundaries, cross-score unbiasedness, and related invariants.

The current suite contains 103 tests, including exact matched-parameter
function preservation, modern five-factor maximum-unique strength-2/3 arrays,
TabPFN rendering, classical rendering, and semantic row-order controls.

### 2. Reanalyze stored complete tensors

The `results/*_cover/` and `results/tier1_*` directories contain the complete
prediction tensors needed by the analysis scripts. Representative entrypoints:

```bash
"$PYTHON" analyze_strength2_cover.py
"$PYTHON" analyze_strength3_cover.py
"$PYTHON" analyze_taskbalanced_model_selection.py
"$PYTHON" analyze_cross_quotient_selection.py
"$PYTHON" analyze_cross_score_efficiency.py
"$PYTHON" analyze_cross_variance_identity.py
"$PYTHON" analyze_cross_variance_decomposition.py
"$PYTHON" analyze_selection_error_decomposition.py
"$PYTHON" analyze_repeated_holdout_shift.py
"$PYTHON" analyze_repartitioned_cross_selection.py
"$PYTHON" analyze_stability_sets.py
"$PYTHON" analyze_stability_set_coupling.py
"$PYTHON" analyze_multiclass_cross_score.py
"$PYTHON" analyze_log_quotient_jackknife.py
"$PYTHON" analyze_log_jackknife_frontier.py
"$PYTHON" analyze_disjoint_log_loss.py
"$PYTHON" analyze_mixed_resolvable_packing.py
"$PYTHON" analyze_orbit_optimal_pack.py
"$PYTHON" analyze_disjoint_pack_cross128.py
"$PYTHON" analyze_pack_cross128_variance.py
"$PYTHON" analyze_pack_cross128_power.py
"$PYTHON" analyze_packed_unbiased_frontier.py
"$PYTHON" analyze_exact_closure_repartition.py
"$PYTHON" analyze_exhaustive128_control.py
"$PYTHON" analyze_packing_metric_scope.py
"$PYTHON" audit_systems_efficiency.py
"$PYTHON" analyze_accuracy_margin_diagnostic.py
"$PYTHON" analyze_log_loss_support.py
"$PYTHON" analyze_smoothed_log_packing.py
"$PYTHON" analyze_smoothed_log_taylor.py
"$PYTHON" analyze_late_source_extension.py
# Second source block uses the same analyzer with --config/--tensors/--output-prefix.
"$PYTHON" analyze_combined_packing_sources.py
"$PYTHON" analyze_source_sensitivity.py
"$PYTHON" analyze_late_source_metric_scope.py
"$PYTHON" analyze_timed_refit.py
"$PYTHON" analyze_late_strength_failure.py
"$PYTHON" analyze_combined_late_metric_sources.py
"$PYTHON" analyze_modern_model_extension.py
"$PYTHON" analyze_expanded_model_sources.py
"$PYTHON" analyze_repeated_split_modern.py
"$PYTHON" analyze_repeated_split_metric_scope.py
"$PYTHON" analyze_partition_nuisance_scale.py
"$PYTHON" analyze_antithetic_operator_boundary.py
"$PYTHON" analyze_late_source_c_audit.py
"$PYTHON" analyze_final_combined_sources.py
"$PYTHON" analyze_source_c_operator_prediction.py
"$PYTHON" make_paper_figures.py
```

Monte Carlo analyzers use deterministic hash-derived seeds. Set BLAS thread
counts to one when running several analyzers concurrently.

### 3. Refit complete tensors

Fit scripts and frozen JSON configs are paired by name, for example:

```bash
"$PYTHON" tier1_orbit.py --config tier1_confirmation_config.json \
  --dataset credit_card_default --model onehot_adam_mlp \
  --output-dir results/tier1_confirmation
"$PYTHON" openml_external_cover.py --config openml_external_cover_config.json \
  --dataset openml-sonar-40 --model ordinal_forest \
  --output-dir results/openml_external_cover
```

The local Day-1--4 datasets are expected under the `data_root` recorded in each
config. OpenML configs record dataset IDs, split seeds, subsampling seeds,
factor levels, model budgets, and threads per fit. Refit runs are much more
expensive than tensor reanalysis.

The modern-family extension is reproduced by fitting every dataset/model pair
in `openml_modern_model_cover_config.json`, then running:

```bash
"$PYTHON" analyze_strength2_cover.py \
  --config openml_modern_model_cover_config.json \
  --input-dir results/openml_modern_model_cover \
  --output-prefix modern_model_strength2
"$PYTHON" analyze_late_source_extension.py \
  --config openml_modern_model_cover_config.json \
  --tensors results/openml_modern_model_cover \
  --strength-prefix modern_model_strength2 --output-prefix modern_model \
  --minimum-cell-wins 13 --required-source-wins 7
"$PYTHON" analyze_late_source_metric_scope.py \
  --config openml_modern_model_cover_config.json \
  --tensors results/openml_modern_model_cover --output-prefix modern_model
```

## Principal machine-readable outputs

Final completion entrypoints and outputs:

```bash
"$PYTHON" analyze_completion_panel.py
"$PYTHON" analyze_completion_tabpfn.py
"$PYTHON" completion_menu_size.py --analyze
"$PYTHON" completion_row_order.py --analyze
"$PYTHON" make_completion_outputs.py
"$PYTHON" audit_result_integrity.py
```

- `results/completion_panel_summary.json` — 144-cell modern-neural and
  60-cell classical consolidated analysis;
- `results/completion_tabpfn_summary.json` — internal/external TabPFN study;
- `results/completion_menu_size_summary.json` — 256-state menu approximation;
- `results/completion_row_order_summary.json` — semantic row-order control;
- `results/completion_tables/` — six required tables in CSV and Markdown;
- `results/completion_figures/` — ten required figures in PNG and PDF;
- `results/integrity_audit_summary.json` — final 587-tensor audit;

- `results/exact_panel_meta_summary.json` — 57-cell descriptive exact roll-up;
- `results/source_generalization_uncertainty_summary.json` — source bootstrap;
- `results/taskbalanced_model_selection_summary.json` — external
  classification/regression selection;
- `results/cross_quotient_selection_summary.json` — frozen unbiased score gate;
- `results/cross_score_efficiency_summary.json` — candidate-level score RMSE;
- `results/cross_variance_identity_summary.json` — Proposition 19 calibration;
- `results/cross_variance_decomposition_summary.json` — operator components;
- `results/selection_error_decomposition_summary.json` — exact target shift;
- `results/repeated_holdout_shift_summary.json` — conditional partition audit;
- `results/repartitioned_cross_selection_summary.json` — paired selector audit;
- `results/stability_set_summary.json` — replicated-selector union diagnostic;
- `results/stability_set_independent_summary.json` — coupling repeat;
- `results/multiclass_cross_score_summary.json` — vector-Brier scope addendum;
- `results/log_quotient_jackknife_summary.json` — approximate nonlinear scope;
- `results/log_jackknife_frontier_summary.json` — 32/64-fit nonlinear frontier;
- `results/synthetic_selection_phase_summary.json` — controlled ranking boundary;
- `results/screen_then_cross_summary.json` — failed first allocation rule;
- `results/cheap_screen_precise_deploy_summary.json` — successful paired allocator;
- `results/disjoint_pair32_summary.json` — packed 32-fit prediction/selection;
- `results/disjoint_pair_coupling_summary.json` — candidate-independent repeat;
- `results/disjoint_pair_cross_summary.json` — unbiased antithetic U64 score;
- `results/disjoint_pair_uncertainty_summary.json` — source/non-partition audit;
- `results/disjoint_pack64_summary.json` — mutually disjoint four-cover frontier;
- `results/pack64_operator_summary.json` — pure-component pack covariance audit;
- `results/multiclass_disjoint_pack_summary.json` — vector-valued packing scope;
- `results/resolvable_coset_summary.json` — scalable `GF(4)` packing frontier;
- `results/disjoint_log_loss_summary.json` — nonlinear disjoint-packing scope;
- `results/mixed_resolvable_summary.json` — mixed-product resolution and graph comparison;
- `results/orbit_optimal_pack_summary.json` — failed strict orbit-law optimizer;
- `results/disjoint_pack_cross128_summary.json` — unbiased packed 128-fit score;
- `results/pack_cross128_uncertainty_summary.json` — non-exhaustive source scope;
- `results/pack_cross128_variance_summary.json` — four-pack score operator audit;
- `results/pack_cross128_power_summary.json` — gap-calibrated ranking power;
- `results/packed_unbiased_frontier_summary.json` — unbiased 32/64/128 frontier;
- `results/exact_closure_repartition_summary.json` — exact selector transfer boundary;
- `results/exhaustive128_control_summary.json` — exact equal-fit closure control;
- `results/packing_metric_scope_summary.json` — Brier/log/AUC/accuracy packing scope;
- `results/systems_efficiency_audit_summary.json` — timing-telemetry limitation;
- `results/accuracy_margin_diagnostic_summary.json` — unresolved margin mechanism;
- `results/log_loss_support_summary.json` — clipped-log support boundary;
- `results/smoothed_log_packing_summary.json` — interior-supported log robustness;
- `results/smoothed_log_taylor_summary.json` — local Taylor and global-bound calibration;
- `results/late_source_extension_summary.json` — four-source prospective extension;
- `results/late_source_b_extension_summary.json` — second-block failure/packing boundary;
- `results/combined_packing_source_summary.json` — eleven-source packing inference;
- `results/combined_packing_source_sensitivity_summary.json` — post-hoc source-concentration audit;
- `results/late_source_metric_scope_summary.json` — untouched metric boundary;
- `results/late_source_b_metric_scope_summary.json` — stronger nonlinear/ranking boundary;
- `results/timed_refit_summary.json` — deterministic local refit telemetry;
- `results/late_strength_failure_summary.json` — high-order failure mechanism;
- `results/combined_late_metric_source_summary.json` — eight-source metric scope;
- `results/modern_model_extension_audit_summary.json` — HistGB/CatBoost transport audit;
- `results/expanded_model_source_summary.json` — eleven-source, expanded-model sensitivity;
- `results/repeated_split_modern_summary.json` — frozen three-partition transport audit;
- `results/repeated_split_metric_scope_summary.json` — post-primary nonlinear split scope;
- `results/partition_nuisance_scale_summary.json` — partition/nuisance error-scale audit;
- `results/antithetic_operator_boundary_summary.json` — finite/Gaussian antithesis boundary;
- `results/late_source_c_audit_summary.json` — final prospective four-source/five-model gate;
- `results/late_source_c_metric_scope_summary.json` — source-C nonlinear/ranking scope;
- `results/final_combined_source_summary.json` — fifteen-source combined sensitivity;
- `results/source_c_operator_prediction_summary.json` — source-C graph/fANOVA calibration;
- `results/late_source_c_split_audit_summary.json` — conditional alternate-partition repeat;
- `results/source_c_two_split_summary.json` — descriptive two-partition source-C roll-up;
- `results/matched_function_summary.json` — late prospective exact
  matched-initial-function control;
- `results/interaction_phase_diagram_summary.json` — favorable/adverse regions;
- `results/highdim_strength3_bootstrap_summary.json` — high-dimensional scope;
- `results/predictive_metric_uncertainty_summary.json` — nonlinear metrics;
- `results/figures/` — paper-ready PDF and PNG figures.

The stored tensors and repeated transformed representatives are dependent
measurements. Source-level inference is kept separate from exact
within-tensor expectations and action Monte Carlo uncertainty.
