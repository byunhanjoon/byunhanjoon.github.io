"""Algebra and construction checks for regular disjoint cover packing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_disjoint_pair32 import paired_ids
from analyze_disjoint_pair_cross import cover_graph, graph_theory
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_resolvable_coset_packing import coset_resolution
from analyze_mixed_resolvable_packing import mixed_coset_resolution
from analyze_packing_metric_scope import metric_values
from analyze_strength2_cover import assert_strength


HERE = Path(__file__).resolve().parent


def test_observed_cover_graphs_are_regular() -> None:
    expected = {
        (4, 1, 1, 4): (1, 0),
        (4, 1, 2, 4): (18, 1),
        (4, 2, 2, 4): (144, 21),
        (4, 4, 1, 4): (144, 21),
        (4, 4, 2, 4): (1_728, 485),
    }
    for shape, (vertices, degree) in expected.items():
        ids, _, adjacency, _ = cover_graph(shape)
        assert len(ids) == vertices
        assert np.all(adjacency.sum(axis=1) == degree)
        assert np.array_equal(adjacency, adjacency.T)


def test_disjoint_pair_is_exact_on_32_cell_product() -> None:
    shape = (4, 1, 2, 4)
    packed, _ = paired_ids(shape, "test", "exact")
    assert all(len(np.intersect1d(left, right)) == 0
               for left, right in zip(*packed))
    assert all(len(np.union1d(left, right)) == 32
               for left, right in zip(*packed))
    field = np.random.default_rng(7).normal(size=np.prod(shape))
    estimate = (field[packed[0]].mean(axis=1) + field[packed[1]].mean(axis=1)) / 2
    assert np.allclose(estimate, field.mean(), atol=1e-14)


def test_full_graph_reduces_all_surviving_covariance_components() -> None:
    theory = graph_theory((4, 4, 2, 4))
    ratios = [value for key, value in theory["pair_to_single_ratios"].items()
              if abs(theory["single_cover_coefficients"][key]) > 1e-10]
    assert len(ratios) == 5
    assert all(0 < value < .5 for value in ratios)


def test_frozen_disjoint_pair_gates_pass() -> None:
    result32 = json.loads((HERE / "results/disjoint_pair32_summary.json").read_text())
    result64 = json.loads((HERE / "results/disjoint_pair_cross_summary.json").read_text())
    uncertainty = json.loads((HERE / "results/disjoint_pair_uncertainty_summary.json").read_text())
    assert result32["frozen_gate_passed"]
    assert result64["frozen_gate_passed"]
    assert uncertainty["frozen_addendum_passed"]


def test_four_pack_is_mutually_disjoint_and_exhausts_64_cells() -> None:
    shape = (4, 2, 2, 4)
    pack, _, attempts = sample_pack_and_pairs(shape, "test", "pack")
    assert attempts == 1
    for current in pack[:32]:
        assert len(np.unique(current)) == 64
        assert np.array_equal(np.sort(np.unique(current)), np.arange(64))


def test_four_pack_gates_and_operator_calibration() -> None:
    result = json.loads((HERE / "results/disjoint_pack64_summary.json").read_text())
    scope = json.loads((HERE / "results/disjoint_pack64_uncertainty_summary.json").read_text())
    operator = json.loads((HERE / "results/pack64_operator_summary.json").read_text())
    assert result["frozen_gate_passed"]
    assert result["exact_partition_candidates"] == 148
    assert not scope["frozen_addendum_passed"]
    assert operator["frozen_gate_passed"]
    assert operator["components_passing_99_percent"] == 5


def test_resolvable_cosets_partition_full_4_power_4_product() -> None:
    resolution = coset_resolution()
    assert resolution.shape == (16, 16, 4)
    for design in resolution:
        assert_strength(design, (4, 4, 4, 4), 2)
    ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), (4, 4, 4, 4))
    assert np.array_equal(np.sort(ids.reshape(-1)), np.arange(256))


def test_resolvable_and_multiclass_scope_outcomes() -> None:
    coset = json.loads((HERE / "results/resolvable_coset_summary.json").read_text())
    multiclass = json.loads((HERE / "results/multiclass_disjoint_pack_summary.json").read_text())
    coupling = json.loads((HERE / "results/disjoint_pair_coupling_summary.json").read_text())
    assert coset["frozen_gate_passed"]
    assert not multiclass["frozen_gate_passed"]
    assert multiclass["pack64_max_absolute_quotient_score_error"] == 0
    assert coupling["frozen_gate_passed"]


def test_mixed_resolution_partitions_observed_product() -> None:
    resolution = mixed_coset_resolution()
    assert resolution.shape == (8, 16, 4)
    for design in resolution:
        assert_strength(design, (4, 4, 2, 4), 2)
    ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), (4, 4, 2, 4))
    assert np.array_equal(np.sort(ids.reshape(-1)), np.arange(128))
    result = json.loads((HERE / "results/mixed_resolvable_summary.json").read_text())
    assert result["construction_gate_passed"]
    assert not result["stronger_empirical_gate_passed"]
    assert result["candidate_wins_vs_graph64"] == 2


def test_disjoint_log_loss_scope_closes_and_passes() -> None:
    result = json.loads((HERE / "results/disjoint_log_loss_summary.json").read_text())
    pair = result["comparisons"]["pair32"]
    pack = result["comparisons"]["pack64"]
    assert pair["frozen_gate_passed"]
    assert pack["frozen_gate_passed"]
    assert pair["exact_partition_candidates"] == 117
    assert pack["exact_partition_candidates"] == 127
    assert pair["exact_partition_max_absolute_error"] == 0
    assert pack["exact_partition_max_absolute_error"] == 0
    assert pair["panels_passing_by_clause"]["rmse_lower"] == 6
    assert pack["panels_passing_by_clause"]["rmse_lower"] == 6


def test_orbit_pack_optimizer_strict_gate_is_retained_failure() -> None:
    result = json.loads((HERE / "results/orbit_optimal_pack_summary.json").read_text())
    assert result["lp_success"]
    assert result["support_size"] <= 6
    assert result["all_templates_are_64_distinct_cells"]
    assert result["orbit_cell_marginals_exact_by_transitivity"]
    assert not result["frozen_gate_passed"]
    assert result["components_below_graph_lower_99"] == 0
    assert result["real_candidates_below_graph_point"] == 0


def test_pack_cross128_unbiased_frontier_passes() -> None:
    result = json.loads((HERE / "results/disjoint_pack_cross128_summary.json").read_text())
    assert result["frozen_gate_passed"]
    assert result["panels_passing_by_clause"] == {"rmse": 5, "agreement": 5, "regret": 5}
    assert result["full_product_candidates"] == 23
    assert result["full_product_rmse_wins"] == 23
    assert result["exact_partition_candidates"] == 148
    assert result["exact_partition_max_absolute_error"] < 1e-12
    uncertainty = json.loads((HERE / "results/pack_cross128_uncertainty_summary.json").read_text())
    assert uncertainty["frozen_addendum_passed"]
    assert uncertainty["represented_panels"] == 4
    assert uncertainty["panels_passing"] == 4
    variance = json.loads((HERE / "results/pack_cross128_variance_summary.json").read_text())
    assert variance["frozen_gate_passed"]
    assert variance["candidates"] == 23
    assert variance["candidates_within_2_58_combined_se"] == 22
    power = json.loads((HERE / "results/pack_cross128_power_summary.json").read_text())
    assert power["frozen_gate_passed"]
    assert power["clauses_passing"] == power["panel_gap_coupling_clauses"] == 32
    assert power["strict_pair_cells"] == power["pair_gap_coupling_cells"] == 344
    frontier = json.loads((HERE / "results/packed_unbiased_frontier_summary.json").read_text())
    assert frontier["frozen_gate_passed"]
    assert frontier["candidates_improving_32_to_64"] == 23
    assert frontier["candidates_improving_64_to_128"] == 23
    assert frontier["candidates_with_128_best"] == 23
    repartition = json.loads((HERE / "results/exact_closure_repartition_summary.json").read_text())
    assert repartition["method_gate_passed"]
    assert not repartition["transfer_gate_passed"]
    assert repartition["frozen_interpretation"] == "validation_only_pass"
    assert repartition["eligible_sources"] == 15
    exhaustive = json.loads((HERE / "results/exhaustive128_control_summary.json").read_text())
    assert exhaustive["stronger_control_gate_passed"]
    assert exhaustive["candidates_with_strictly_lower_exhaustive_rmse"] == 23
    assert not exhaustive["pack_cross128_compute_optimal_at_closure"]
    metrics = json.loads((HERE / "results/packing_metric_scope_summary.json").read_text())
    assert metrics["frozen_gate_passed"]
    assert metrics["frozen_scope_interpretation"] == "probabilistic_ranking_pass_accuracy_boundary"
    assert not metrics["comparisons"]["pair32"]["accuracy_scope_passed"]


def test_vectorized_binary_auc_handles_ties() -> None:
    y = np.asarray([0, 0, 1, 1])
    scores = np.asarray([[.1, .5, .5, .9], [.1, .2, .8, .9]])
    predictions = np.stack((1 - scores, scores), axis=-1)
    auc = metric_values(y, predictions)["roc_auc"]
    assert np.allclose(auc, [.875, 1.0])


def test_log_loss_support_boundary_is_recorded() -> None:
    result = json.loads((HERE / "results/log_loss_support_summary.json").read_text())
    assert result["classification_candidates"] == 150
    assert result["interpretation"] == "clip_active_boundary"
    assert not result["unclipped_smooth_log_assumption_holds_on_audited_population"]
    assert result["methods"]["exact_quotient"]["candidates_touching_clip"] == 10
    assert result["methods"]["exact_quotient"]["candidates_with_exact_zero"] == 5
    assert result["methods"]["mutually_disjoint_pack64"]["pooled_clip_fraction"] < 1e-4


def test_smoothed_log_packing_is_robust() -> None:
    result = json.loads((HERE / "results/smoothed_log_packing_summary.json").read_text())
    assert result["all_smoothing_levels_passed"]
    assert result["interpretation"] == "interior_supported_robustness_pass"
    for comparison in ("pair32", "pack64"):
        for alpha in ("1e-06", "0.0001", "0.01"):
            current = result["comparisons"][comparison][alpha]
            assert current["gate_passed"]
            assert current["exact_partition_max_absolute_error"] < 1e-12


def test_smoothed_log_taylor_mechanism_passes() -> None:
    result = json.loads((HERE / "results/smoothed_log_taylor_summary.json").read_text())
    assert result["classification_candidates"] == 23
    assert result["local_taylor_gate_passed"]
    assert result["global_bound_audit_passed"]
    assert result["by_alpha"]["0.01"]["cells_passing_both_local_clauses"] == 88
    assert result["by_alpha"]["0.01"]["global_bound_violations"] == 0


def test_late_source_extension_and_clustered_scope_pass() -> None:
    late = json.loads((HERE / "results/late_source_extension_summary.json").read_text())
    assert late["complete_tensors"] == 12
    assert late["represented_complete_product_fits"] == 1_536
    assert late["primary_strength2_gate"]["passed"]
    assert late["primary_strength2_gate"]["source_mean_wins_vs_iid_and_strength1"] == 4
    combined = json.loads((HERE / "results/combined_packing_source_summary.json").read_text())
    assert combined["unique_sources"] == 11
    assert combined["all_scope_gates_passed"]
    assert all(item["positive_sources"] == 11 for item in combined["comparisons"].values())
    sensitivity = json.loads(
        (HERE / "results/combined_packing_source_sensitivity_summary.json").read_text()
    )
    assert sensitivity["evidence_status"] == "post_hoc_diagnostic"
    assert sensitivity["all_sensitivity_checks_positive"]
    assert all(item["leave_one_source_out_mean_range"][0] > 0
               for item in sensitivity["comparisons"].values())
    late_b = json.loads((HERE / "results/late_source_b_extension_summary.json").read_text())
    assert not late_b["primary_strength2_gate"]["passed"]
    assert all(item["source_mean_rmse_wins"] == 4
               for item in late_b["packing_and_selection"].values())
    mechanism = json.loads((HERE / "results/late_strength_failure_summary.json").read_text())
    assert mechanism["strength1_failures"] == 1
    assert mechanism["spearman_pair_to_higher_vs_log_advantage"] > .8
    assert mechanism["permutation_two_sided_pvalue"] < .001
    metric = json.loads((HERE / "results/late_source_metric_scope_summary.json").read_text())
    assert not metric["all_probabilistic_ranking_gates_passed"]
    assert metric["comparisons"]["pack64"]["metrics"]["log_loss"]["candidate_strict_wins"] == 11
    assert metric["comparisons"]["pack64"]["metrics"]["roc_auc"]["candidate_strict_wins"] == 8
    metric_b = json.loads((HERE / "results/late_source_b_metric_scope_summary.json").read_text())
    assert not metric_b["all_probabilistic_ranking_gates_passed"]
    assert metric_b["comparisons"]["pack64"]["metrics"]["log_loss"]["candidate_strict_wins"] == 8
    assert metric_b["comparisons"]["pack64"]["metrics"]["accuracy"]["candidate_strict_wins"] == 3
    metric_sources = json.loads((HERE / "results/combined_late_metric_source_summary.json").read_text())
    assert metric_sources["sources"] == 8
    assert metric_sources["brier_log_source_scope_passed"]
    assert all(metric_sources["comparisons"][name]["log_loss"]["strictly_positive_sources"] == 8
               for name in ("pair32", "pack64"))
    timing = json.loads((HERE / "results/timed_refit_summary.json").read_text())
    assert timing["exact_artifact_matches"] == timing["timed_cells"] == 12
    assert timing["total_timed_fits"] == 1_536
    assert not timing["portable_runtime_claimed"]


def test_modern_model_extension_retains_declared_boundaries() -> None:
    result = json.loads(
        (HERE / "results/modern_model_extension_audit_summary.json").read_text()
    )
    assert result["complete_tensors"] == 16
    assert result["represented_complete_product_fits"] == 2_048
    assert result["strength2_frozen_gate_passed"]
    assert result["strength2"]["material_cell_wins_vs_all_controls"] == 14
    assert not result["all_packing_frozen_strict_gates_passed"]
    assert result["all_packing_candidates_nonadverse"]
    assert all(item["nondegenerate_candidate_wins"] == 8
               for item in result["packing"].values())
    expanded = json.loads((HERE / "results/expanded_model_source_summary.json").read_text())
    assert expanded["evidence_status"] == "post_outcome_sensitivity"
    assert expanded["all_sensitivity_checks_positive"]
    assert all(item["positive_sources"] == 11
               for item in expanded["comparisons"].values())


def test_repeated_split_modern_transport_and_boundary() -> None:
    result = json.loads((HERE / "results/repeated_split_modern_summary.json").read_text())
    assert result["complete_tensors"] == 48
    assert result["represented_complete_product_fits"] == 6_144
    assert result["manifest_issues"] == []
    assert result["all_frozen_nuisance_transport_gates_passed"]
    assert result["strength2"]["material_wins"] == 35
    assert result["strength2"]["material_cells"] == 43
    assert all(item["nondegenerate_candidate_losses"] == 0
               for item in result["packing"].values())
    transfer = result["exact_validation_to_test_transfer"]
    assert transfer["winner_agreements"] == 21
    assert transfer["dataset_split_pairs"] == 24
    assert transfer["four_split_sensitivity"]["winner_agreements"] == 28
    metric = json.loads(
        (HERE / "results/repeated_split_metric_scope_summary.json").read_text()
    )
    assert metric["comparisons"]["pair32"]["brier"]["nondegenerate_losses"] == 0
    assert metric["comparisons"]["pack64"]["log_loss"]["nondegenerate_losses"] == 0
    scale = json.loads((HERE / "results/partition_nuisance_scale_summary.json").read_text())
    assert scale["winner_flips"] == 4
    assert scale["median_partition_to_nuisance_scale_ratio"] > 50
    assert not scale["formal_inference_claimed"]
