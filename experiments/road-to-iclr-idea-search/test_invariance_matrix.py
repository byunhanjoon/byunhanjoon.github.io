"""Integrity checks for the Day 3 one-factor invariance extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invariance_matrix import (
    CONFIG_PATH,
    align_probabilities,
    build_views,
    render_x,
    transformed_categorical_indices,
)
from analyze_invariance_matrix import aggregate, flatten


HERE = Path(__file__).resolve().parent


def fixture() -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[int, int]]:
    x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 2.0],
            [4.0, 1.0, 0.0],
        ]
    )
    y = np.asarray([0, 1, 2, 1])
    return x, y, (2,), {2: 3}


def test_config_declares_controls_before_results() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["status"] == "prospective_day3_invariance_matrix_extension"
    assert len(config["families"]) == 6
    assert config["factors"] == [
        "feature_order",
        "category_ids",
        "class_ids",
        "numeric_units",
    ]
    assert "not_invariant" in config["expected_behavior"]["ordinal_forest_sqrt"]["category_ids"]
    assert "invariant" in config["expected_behavior"]["native_histgb"]["category_ids"]


def test_each_view_menu_changes_only_its_declared_factor() -> None:
    x, y, categorical, cardinalities = fixture()
    for factor in ("feature_order", "category_ids", "class_ids", "numeric_units"):
        views = build_views(factor, x, y, categorical, cardinalities, 1, 3)
        assert views[0].name == "identity"
        for view in views[1:]:
            rendered = render_x(x, view, 1)
            if factor == "feature_order":
                inverse = np.argsort(view.feature_permutation)
                np.testing.assert_allclose(rendered[:, inverse], x)
                assert not view.category_maps
                np.testing.assert_array_equal(view.class_permutation, np.arange(3))
            elif factor == "category_ids":
                np.testing.assert_allclose(rendered[:, :2], x[:, :2])
                np.testing.assert_array_equal(np.sort(np.unique(rendered[:, 2])), [0, 1, 2])
                np.testing.assert_array_equal(view.feature_permutation, np.arange(3))
            elif factor == "class_ids":
                np.testing.assert_allclose(rendered, x)
                np.testing.assert_array_equal(np.sort(view.class_permutation), np.arange(3))
            else:
                np.testing.assert_allclose(rendered[:, 1:], x[:, 1:])
                assert np.all(view.unit_scale > 0)


def test_categorical_indices_follow_feature_permutation() -> None:
    x, y, categorical, cardinalities = fixture()
    views = build_views("feature_order", x, y, categorical, cardinalities, 1, 3)
    for view in views:
        transformed = transformed_categorical_indices(categorical, view)
        assert len(transformed) == 1
        assert view.feature_permutation[transformed[0]] == 2


def test_output_alignment_inverts_semantic_class_rendering() -> None:
    raw = np.asarray([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    mapping = np.asarray([2, 0, 1])
    aligned = align_probabilities(raw, mapping)
    np.testing.assert_allclose(aligned, raw[:, [2, 0, 1]])


def test_completed_artifact_if_present() -> None:
    path = HERE / "invariance_matrix_results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["status"] == "completed_day3_invariance_matrix_extension"
    assert set(data["cells"]) == {"adult", "churn", "otto"}
    for dataset in data["cells"].values():
        assert set(dataset["families"]) == set(data["config"]["families"])
        for family in dataset["families"].values():
            assert set(family) == set(data["config"]["factors"])


def test_completed_artifact_controls_and_analysis_if_present() -> None:
    path = HERE / "invariance_matrix_results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = flatten(data)
    tolerance = data["config"]["design"]["invariance_tolerance"]
    summary = aggregate(rows, tolerance)
    assert summary["applicable_cells"] == 66

    def cell(dataset: str, family: str, factor: str) -> dict[str, object]:
        return data["cells"][dataset]["families"][family][factor]

    for dataset in ("adult", "churn"):
        assert cell(dataset, "native_histgb", "category_ids")["frozen"][
            "max_probability_deviation_from_identity"
        ] == 0.0
        assert cell(dataset, "catboost_native", "category_ids")["frozen"][
            "max_probability_deviation_from_identity"
        ] == 0.0
        assert cell(dataset, "ordinal_forest_sqrt", "category_ids")["frozen"][
            "max_probability_deviation_from_identity"
        ] > 0.2

    assert cell("adult", "onehot_logistic", "feature_order")[
        "selection_switch_fraction"
    ] == 0.0
    assert cell("churn", "onehot_logistic", "numeric_units")["frozen"][
        "max_probability_deviation_from_identity"
    ] < 1e-12
