# `day3_agent.md` completion audit

This audit maps the literal experiment brief to generated evidence. A
negative result is complete when the required test was run and retained; it
does not become a positive scientific result.

| Requirement | Status | Evidence |
| --- | --- | --- |
| `1_day2_anchors` | **PASS** | `{"adult_expected_scores": {"basis_blend": 0.8597137767950371, "cumulative_ple": 0.8590381426202321, "local_ple": 0.8586081935999017}, "adult_rows": 4, "black_friday_expected_scores": {"basis_blend": 0.6931962505786373, "cumulative_ple": 0.6966277030051913, ...` |
| `2_controlled_condition_sweep` | **PASS** | `{"diagnostics_present": true, "kappas": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0], "mlp_datasets": ["adult", "california", "diamond"], "resnet_endpoints": [["adult", 1.0], ["adult", 1000.0], ["california", 1.0], ["california", 1000.0], ["diamond"...` |
| `3_ple_identity_whitening` | **PASS** | `{"gap_reduction": 0.9122807017543859, "max_recorded_reconstruction_error_or_angle": 1.1606496459088164e-09, "rows": 50}` |
| `4_invariant_regularization` | **PASS** | `{"aggregate_spread": {"invariant": 0.023873269396613585, "no_first_wd": 0.021174027681903747, "standard": 0.023244641577610282}, "regularizers": ["invariant", "no_first_wd", "standard"], "rows": 123, "verdict": "negative result retained"}` |
| `5_ordinal_geometry` | **PASS** | `{"controlled_dataset_models": [["adult", "mlp"], ["diamond", "mlp"], ["diamond", "resnet"]], "controlled_rows": 88, "datasets": ["adult", "black-friday", "diamond"], "natural_basis_rows": 90}` |
| `6_nominal_categorical_geometry` | **PASS** | `{"dataset_models": [["adult", "mlp"], ["adult", "resnet"], ["diamond", "mlp"], ["diamond", "resnet"]], "kappas": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0], "rows": 92}` |
| `7_block_residualization` | **PASS** | `{"representations": ["block_residualized", "block_residualized_whitened", "blockwise_whitened", "joint_whitened", "raw_joint", "standardized_categorical"], "rows": 60, "verdict": "geometry succeeded; predictive intervention failed"}` |
| `8_secondary_branches` | **PASS** | `{"cross_atoms": "predeclared P2 cut; no result claimed", "cyclic_rows": 70, "frequency_preconditioning_rows": 30, "residual_te_rows": 30}` |
| `9_dataset_and_model_integrity` | **PASS** | `{"frozen_prospective_datasets_retained": ["Food_Delivery_Time", "credit_card_clients_default", "heloc", "miami_housing", "seismic-bumps", "wine_quality"], "later_broad_models": ["dense_stem_ft_transformer", "dense_stem_tabm", "mlp", "resnet"], "later_broad_...` |
| `10_required_controls` | **PASS** | `{"global_scale": "geometric mean singular value 1", "kappa_1": "random orthogonal control", "normalization": ["diagonal standardization", "whitening"], "optimizers": ["AdamW", "SGD follow-up"], "regularization": ["standard WD", "no first-layer WD", "invaria...` |
| `11_deliverables` | **PASS** | `{"figures": {"block_residualization_diamonds": true, "categorical_basis_conditioning": true, "kappa_vs_convergence": true, "kappa_vs_performance": true, "ordinal_local_vs_cumulative": true, "ple_identity_before_after_whitening": true, "standard_vs_invariant...` |
| `12_verification` | **PASS** | `{"broad_freeze_changed": ["/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-03/experiments/day3/broad_data.py", "/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-03/tests/test_broad_benchmark.py"], "broad_freeze_match...` |

**Overall: COMPLETE.**

The original success criteria were only partially supported: H1/H2 and
the ordinal/nominal extensions succeeded, while the proposed invariant
regularizer failed. The correct conclusion is therefore a completed,
decisive mechanism study—not full confirmation of every hypothesis.

Provenance warning: the later broad-benchmark freeze audit remains red
because commit `768e0c0` added a trajectory helper to `broad_data.py` and
its test after the broad manifest was frozen. All coverage and scientific
invariants pass, but this warning is intentionally not suppressed.

Machine-readable evidence: `results/day3/day3_agent_completion_audit.json`.
