"""Numerical checks for the exact identities used by the concept memo."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from orbit_anova import (
    budgeted_marginalization_frontier,
    decompose,
    log_orbit_summary,
    risk_summary,
)
from schema_radius_analysis import radius
from selection_rule_orbit_pilot import selection_margin_diagnostic, switch_decomposition
from selection_split_confirmation import exact_binomial_sign_p, exact_sign_flip_p


HERE = Path(__file__).resolve().parent


def _simplex_predictions(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    logits = rng.normal(size=shape)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def test_brier_gap_equals_hilbert_variance() -> None:
    rng = np.random.default_rng(4)
    predictions = _simplex_predictions((2, 3, 17, 4), rng)
    y = rng.integers(0, 4, size=17)
    result = risk_summary(predictions, y, ("left", "right"))
    assert result["risk_identity_absolute_error"] < 1e-14


def test_balanced_fanova_is_orthogonal_and_reconstructs() -> None:
    rng = np.random.default_rng(7)
    predictions = rng.normal(size=(2, 3, 4, 11, 2))
    result = decompose(predictions, ("a", "b", "c"))
    assert result["component_sum_error"] < 1e-14
    assert result["prediction_reconstruction_max_error"] < 1e-14


def test_log_gap_is_label_free_reverse_kl() -> None:
    rng = np.random.default_rng(9)
    predictions = _simplex_predictions((2, 3, 13, 5), rng)
    first = log_orbit_summary(predictions, 2, rng.integers(0, 5, size=13))
    second = log_orbit_summary(predictions, 2, rng.integers(0, 5, size=13))
    assert first["risk_identity_absolute_error"] < 1e-14
    assert second["risk_identity_absolute_error"] < 1e-14
    assert first["label_free_gap"] == second["label_free_gap"]


def test_schema_radius_matches_bruteforce_for_two_points() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(2, 19, 3))
    result = radius(points)
    squared_distance = np.mean(np.sum((points[0] - points[1]) ** 2, axis=-1))
    assert abs(result["worst_distribution_schema_radius_squared"] - squared_distance / 4) < 1e-10
    assert result["duality_absolute_error"] < 1e-10


def test_pairwise_transport_lower_bound_scaling() -> None:
    """For deterministic members, the pairwise formula equals variance."""
    rng = np.random.default_rng(13)
    points = rng.normal(size=(5, 23, 2))
    centroid = points.mean(axis=0)
    variance = np.mean(np.sum((points - centroid) ** 2, axis=-1))
    pair_sum = sum(
        np.mean(np.sum((points[i] - points[j]) ** 2, axis=-1))
        for i, j in itertools.combinations(range(len(points)), 2)
    )
    assert abs(variance - pair_sum / len(points) ** 2) < 1e-14


def test_budgeted_marginalization_matches_exact_draw_enumeration() -> None:
    """The hybrid residual formula is exact for iid complement draws."""
    predictions = np.asarray(
        [
            [[[0.0]], [[2.0]], [[5.0]]],
            [[[2.0]], [[4.0]], [[7.0]]],
        ]
    )
    frontier = budgeted_marginalization_frontier(
        predictions, ("a", "b"), budget=4
    )
    action = next(item for item in frontier if item["factors"] == ["a"])
    q_a = predictions.mean(axis=0)
    grand = predictions.mean(axis=(0, 1))
    realized = []
    for left, right in itertools.product(range(3), repeat=2):
        estimate = (q_a[left] + q_a[right]) / 2
        realized.append(float(np.mean((estimate - grand) ** 2)))
    assert action["conditional_draws"] == 2
    assert action["realized_cost"] == 4
    assert abs(action["expected_residual_risk"] - np.mean(realized)) < 1e-14


def test_budgeted_marginalization_contains_iid_baseline() -> None:
    rng = np.random.default_rng(17)
    predictions = rng.normal(size=(2, 3, 7, 2))
    frontier = budgeted_marginalization_frontier(
        predictions, ("a", "b"), budget=5
    )
    iid = frontier[0]
    assert iid["factors"] == []
    assert iid["conditional_draws"] == 5
    assert iid["unused_budget"] == 0
    assert abs(iid["expected_residual_risk"] - iid["residual_risk"] / 5) < 1e-14


def test_configuration_switch_variance_decomposition() -> None:
    rng = np.random.default_rng(19)
    frozen = rng.normal(size=(2, 3, 2, 17, 4))
    selected = frozen + rng.normal(scale=0.3, size=frozen.shape)
    result = switch_decomposition(frozen, selected)
    assert result["change_reconstruction_error"] < 1e-14
    assert abs(
        result["schema_risk_change"]
        - result["switch_dispersion"]
        - result["twice_frozen_switch_cross_covariance"]
    ) < 1e-14


def test_exact_sign_flip_resolution() -> None:
    # With seven same-signed nonzero differences, only the two all-equal sign
    # assignments are as extreme as the observed mean.
    assert exact_sign_flip_p(np.ones(7)) == 2 / 2**7
    assert exact_binomial_sign_p(np.ones(7)) == 2 / 2**7
    assert exact_binomial_sign_p(np.asarray([1, 1, 1, 1, 1, 1, -1])) == 0.125


def test_selection_margin_certificate() -> None:
    stable = np.asarray(
        [
            [[[0.10, 0.20, 0.30]], [[0.11, 0.19, 0.31]]],
            [[[0.09, 0.21, 0.29]], [[0.12, 0.22, 0.28]]],
        ]
    )
    result = selection_margin_diagnostic(stable)
    assert result["sufficient_certificate_holds"]
    unstable = stable.copy()
    unstable[1, 1, 0, 1] = 0.05
    result = selection_margin_diagnostic(unstable)
    assert not result["sufficient_certificate_holds"]
    assert result["minimum_identity_winner_gap_over_orbit"] < 0


def test_selection_confirmation_headline_artifact() -> None:
    data = json.loads((HERE / "selection_split_confirmation.json").read_text())
    expected = {
        "adult/catboost_native": (7, 0.015625, 6, 0.125),
        "churn/ordinal_forest": (6, 0.125, 6, 0.03125),
        "churn/catboost_native": (7, 0.015625, 5, 0.21875),
    }
    for case, (risk_count, risk_sign_p, repair_count, repair_sign_p) in expected.items():
        result = data["cases"][case]
        summary = result["confirmation_summary"]
        assert summary["higher_schema_risk_splits"] == risk_count
        assert summary["schema_risk_difference_exact_binomial_sign_p"] == risk_sign_p
        assert summary["heldout_repair_lower_schema_risk_splits"] == repair_count
        assert summary["heldout_repair_schema_risk_exact_binomial_sign_p"] == repair_sign_p
        selected = result["heldout_selected_joint_partition"][
            "same_split_schema_variance"
        ]
        repaired = result["heldout_development_pooled_joint_partition"][
            "same_split_schema_variance"
        ]
        assert 0.27 < 1 - repaired / selected < 0.36


def test_field_topology_headline_artifact() -> None:
    data = json.loads((HERE / "field_topology_bayes_suite.json").read_text())
    comparisons = {
        "ordinal": ("path", {"isotropic": 6, "ring": 7, "permuted_path": 9}),
        "cyclic": ("ring", {"isotropic": 6, "path": 6, "permuted_path": 9}),
        "nominal": ("isotropic", {"path": 9, "ring": 9, "permuted_path": 9}),
    }
    for task, (matched, expected) in comparisons.items():
        for rival, count in expected.items():
            key = f"matched_minus_{rival}"
            actual = sum(
                scenario[task]["paired_matched_contrasts"][key][
                    "mean_difference_ci95"
                ][1]
                < 0
                for scenario in data["scenarios"].values()
            )
            assert actual == count, (task, matched, rival, actual)


def test_adaptive_topology_search_fails_nominal_admissibility() -> None:
    data = json.loads(
        (HERE / "field_topology_strength_selection.json").read_text()
    )
    expected = {
        "path_tuned": 8,
        "ring_tuned": 8,
        "permuted_path_tuned": 9,
    }
    for family, count in expected.items():
        key = f"matched_minus_{family}"
        significant_false_topology_wins = sum(
            scenario["nominal"]["contrasts"][key]["mean_difference_ci95"][0]
            > 0
            for scenario in data["scenarios"].values()
        )
        assert significant_false_topology_wins == count

        zero_rates = [
            scenario["nominal"]["families"][family][
                "zero_stiffness_selection_rate"
            ]
            for scenario in data["scenarios"].values()
        ]
        assert 0.55 <= min(zero_rates) <= max(zero_rates) <= 0.715
        assert 0.62 < float(np.mean(zero_rates)) < 0.64


def test_selection_partition_sensitivity_headline_artifact() -> None:
    data = json.loads(
        (HERE / "selection_partition_sensitivity.json").read_text()
    )
    expected = {
        "adult/catboost_native": (222 / 252, 192 / 252, 4, 0.27672955974842767),
        "churn/ordinal_forest": (214 / 252, 190 / 252, 2, 0.4393724318266716),
        "churn/catboost_native": (186 / 252, 144 / 252, 3, 0.5142118863049097),
    }
    for case, (agreement, optimal, constant_splits, interaction) in expected.items():
        summary = data["cases"][case]["summary"]
        assert summary["partition_count"] == 252
        assert summary["fraction_matching_full_menu_choice"] == agreement
        assert summary["fraction_heldout_optimal"] == optimal
        assert summary["splits_with_one_choice_across_all_partitions"] == constant_splits
        assert summary["decision_menu_by_split_fraction"] == interaction


def test_balanced_fold_average_equals_full_menu_average() -> None:
    rng = np.random.default_rng(31)
    losses = rng.normal(size=(4, 4, 2, 6))
    pairs = tuple(itertools.combinations(range(4), 2))
    fold_means = [
        losses[np.ix_(feature, category, range(2), range(6))].mean(
            axis=(0, 1, 2)
        )
        for feature in pairs
        for category in pairs
    ]
    error = np.max(
        np.abs(np.mean(fold_means, axis=0) - losses.mean(axis=(0, 1, 2)))
    )
    assert error < 1e-14


def test_menu_averaged_output_repair_headline_artifact() -> None:
    data = json.loads((HERE / "selection_menu_output_risk.json").read_text())
    expected = {
        "adult/catboost_native": (228, 0.29196450363360826, 0.19988431555258548, 7, 6),
        "churn/ordinal_forest": (180, 0.3753073148389635, 0.2712455644634547, 4, 2),
        "churn/catboost_native": (198, 0.34658065597464616, 0.40023569380383794, 5, 4),
    }
    for case, (menu_wins, reduction, menu_fraction, robust_2, robust_4) in expected.items():
        result = data["cases"][case]
        summary = result["split_summary"]
        assert summary["individual_menus_with_lower_schema_risk"] == menu_wins
        assert summary["individual_menu_count"] == 252
        assert summary["splits_with_lower_mean_schema_risk"] == 7
        assert summary["schema_risk_exact_sign_flip_p"] == 0.015625
        assert summary["schema_risk_exact_binomial_sign_p"] == 0.015625
        assert summary["relative_mean_schema_risk_reduction"] == reduction
        assert summary["splits_robustly_lower_at_density_ratio_cap_2"] == robust_2
        assert summary["splits_robustly_lower_at_density_ratio_cap_4"] == robust_4
        assert result["joint_output_menu_total_fraction"] == menu_fraction


def test_otto_prospective_decision_and_promoted_output_artifacts() -> None:
    decisions = json.loads(
        (HERE / "selection_otto_prospective_decisions.json").read_text()
    )
    expected = {
        "otto/ordinal_forest": (1, 5),
        "otto/native_histgb": (0, 4),
        "otto/catboost_native": (0, 4),
    }
    for case, (unstable, certificates) in expected.items():
        result = decisions["cases"][case]
        assert result["selection_unstable_splits"] == unstable
        assert result["certificate_holds_splits"] == certificates

    forest = decisions["cases"]["otto/ordinal_forest"]["records"]
    switched = [r for r in forest if r["fraction_different_from_identity"] > 0]
    assert len(switched) == 1
    assert switched[0]["split_seed"] == 20_260_830
    assert switched[0]["decision_fanova"]["feature"] == 0.375
    assert switched[0]["decision_fanova"]["total"] == 0.375

    promoted = json.loads(
        (
            HERE
            / "selection_split_otto_screen"
            / "otto_ordinal_forest_s20260830.json"
        ).read_text()
    )
    assert promoted["selection_to_frozen_schema_risk_ratio"] == 2.3505759526604613
    assert promoted["selection_minus_frozen_orbit_mean_brier"] == -0.005104429808984179
    assert promoted["configuration_switch_decomposition"][
        "change_reconstruction_error"
    ] == 0.0

    overlap = json.loads(
        (
            HERE
            / "selection_split_otto_screen"
            / "otto_ordinal_forest_s20260827.json"
        ).read_text()
    )
    assert overlap["selection_counts"] == forest[0]["selection_counts"]
    assert np.max(
        np.abs(
            np.asarray(overlap["mean_validation_brier_by_config"])
            - np.asarray(forest[0]["mean_validation_brier_by_config"])
        )
    ) < 1e-15
