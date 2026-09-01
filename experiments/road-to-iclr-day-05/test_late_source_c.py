"""Regression checks for the final prospective source-C panel."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def test_source_c_is_complete_factorial_panel():
    config = json.loads((HERE / "openml_late_source_c_cover_config.json").read_text())
    directory = RESULTS / "openml_late_source_c_cover"
    assert len(config["datasets"]) == 4
    assert len(config["models"]) == 5
    for dataset in config["datasets"]:
        for model in config["models"]:
            stem = f"{dataset}__{model}"
            manifest = json.loads((directory / f"{stem}.json").read_text())
            assert manifest["status"] == "complete"
            assert manifest["full_product_verified"] is True
            assert manifest["fits"] == 128
            assert manifest["effective_split_seed"] == 2026083041
            with np.load(directory / f"{stem}.npz") as archive:
                assert archive["validation_predictions"].shape[:4] == (4, 4, 2, 4)
                assert archive["test_predictions"].shape[:4] == (4, 4, 2, 4)


def test_source_c_audit_encodes_frozen_thresholds():
    audit = json.loads((RESULTS / "late_source_c_audit_summary.json").read_text())
    assert audit["complete_tensors"] == 20
    assert audit["represented_complete_product_fits"] == 2560
    assert audit["strength2"]["required_all_cell_wins"] == 16
    assert audit["strength2"]["required_source_mean_wins"] == 4
    for row in audit["packing"].values():
        expected = (
            row["nondegenerate_candidates"] > 0
            and row["nondegenerate_win_fraction"] >= .8
            and row["nondegenerate_losses"] == 0
            and row["source_mean_wins"] == 4
        )
        assert row["frozen_gate_passed"] is expected


def test_final_combination_has_fifteen_unique_sources():
    combined = json.loads((RESULTS / "final_combined_source_summary.json").read_text())
    assert combined["unique_sources"] == 15
    assert all(row["sources"] == 15 for row in combined["comparisons"].values())


def test_source_c_operator_prediction_respects_exact_coefficient_range():
    audit = json.loads((RESULTS / "source_c_operator_prediction_summary.json").read_text())
    triple = audit["exact_graph_coefficients"]["triple_packed_to_independent_pair_range"]
    four = audit["exact_graph_coefficients"]["four_way_packed_to_independent_pair"]
    predicted = audit["predicted_ratio_range"]
    assert 0 < triple[0] <= triple[1] < four < 1
    assert triple[0] - 1e-14 <= predicted[0] <= predicted[1] <= four + 1e-14
    assert audit["evidence_status"] == "post_outcome_mechanism_diagnostic"


def test_source_c_alternate_split_is_complete():
    directory = RESULTS / "openml_late_source_c_split_cover"
    manifests = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    assert len(manifests) == 20
    assert sum(row["fits"] for row in manifests) == 2560
    assert all(row["effective_split_seed"] == 2026083051 for row in manifests)
    assert all(row["full_product_verified"] is True for row in manifests)


def test_source_c_split_preserves_nuisance_but_not_partition_transfer():
    audit = json.loads((RESULTS / "late_source_c_split_audit_summary.json").read_text())
    assert audit["all_frozen_gates_passed"] is True
    assert audit["strength2"]["all_cell_wins_vs_iid_and_strength1"] == 20
    assert all(row["nondegenerate_losses"] == 0 for row in audit["packing"].values())
    assert audit["validation_test_winner_agreements"] == 2
    assert audit["mean_validation_selected_test_regret"] > 0


def test_source_c_two_split_rollup_keeps_cluster_boundary():
    summary = json.loads((RESULTS / "source_c_two_split_summary.json").read_text())
    assert summary["unique_sources"] == 4
    assert summary["dataset_split_pairs"] == 8
    assert summary["strength2"]["material_wins"] == summary["strength2"]["material_cells"] == 25
    assert summary["exact_validation_test_transfer"]["winner_agreements"] == 6
    assert all(row["nondegenerate_losses"] == 0 for row in summary["packing"].values())


def test_source_c_two_split_artifacts_are_not_reused():
    summary = json.loads((RESULTS / "source_c_two_split_summary.json").read_text())
    check = summary["artifact_independence_check"]
    assert check["passed"] is True
    assert check["paired_tensor_artifacts"] == check["byte_distinct_tensor_pairs"] == 20
    assert check["effective_split_seeds"] == [2026083041, 2026083051]
