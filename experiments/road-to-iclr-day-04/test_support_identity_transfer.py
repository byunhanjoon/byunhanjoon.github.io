from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import support_identity_transfer_pilot as pilot
import signal_gated_support_pilot as signal_pilot
import universal_mass_identity_pilot as universal_pilot
from trichart_shared_pilot import TriChartModel
from trichart_tsafe_pilot import TAnchoredTriChart
from trichart_frozen_anchor_pilot import FrozenAnchorResidual


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def test_exact_support_uses_zero_for_unseen_values() -> None:
    parts = {
        "train": np.array([[1.0], [2.0], [1.0], [3.0]]),
        "val": np.array([[2.0], [4.0]]),
        "test": np.array([[3.0], [5.0]]),
    }
    selected, cardinalities, codes = pilot.exact_support_codes(parts, 128)
    assert selected == [0]
    assert cardinalities == [3]
    assert codes["val"].tolist() == [[2], [0]]
    assert codes["test"].tolist() == [[3], [0]]


def test_quantile_control_fits_exact_table_sizes() -> None:
    parts = {
        "train": np.array([[0.0], [0.0], [1.0], [1.0]]),
        "val": np.array([[0.25], [0.75]]),
        "test": np.array([[0.0], [1.0]]),
    }
    edges = np.linspace(0.0, 1.0, 17)[None]
    codes = pilot.quantile_bin_codes(parts, edges, [0], [2])
    for values in codes.values():
        assert values.min() >= 1
        assert values.max() <= 2


def test_target_aware_edges_are_strict_and_fixed_width() -> None:
    x = np.arange(40, dtype=np.float64)[:, None]
    y = (x[:, 0] > 20).astype(np.float64)
    edges = pilot.target_aware_edges(x, y, 8, 2, 0.0)
    assert edges.shape == (1, 9)
    assert np.all(np.diff(edges, axis=1) > 0)


def test_signal_selector_finds_repeatable_exact_advantage() -> None:
    rng = np.random.default_rng(17)
    level = rng.integers(0, 4, 1000).astype(np.float32)
    noise = rng.normal(size=1000).astype(np.float32)
    x = np.column_stack((level, noise))
    y = np.array([0.0, 2.0, -1.0, 1.0], dtype=np.float32)[level.astype(int)]
    config = {
        "support_cardinality_max": 128,
        "selector_folds": 5,
        "qple_bins": 2,
        "selector_smoothing": 20.0,
        "selector_min_positive_folds": 4,
        "selector_min_relative_gain_pct": 0.1,
    }
    rows = signal_pilot.signal_scores(x, y, config)
    selected = [row["column"] for row in rows if row["selected"]]
    assert selected == [0]


def test_zero_linear_gate_is_function_preserving_and_trainable() -> None:
    tokenizer = pilot.SupportTokenizer(
        edges=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        n_bin_fields=0,
        category_cardinalities=[],
        support_columns=[0],
        support_cardinalities=[2],
        d_token=4,
        use_support=True,
        support_gate_mode="zero_linear",
    )
    x = np.array([[0.0], [1.0]], dtype=np.float32)
    output = tokenizer(
        torch.from_numpy(x),
        torch.empty(2, 0),
        torch.empty(2, 0, dtype=torch.long),
        torch.tensor([[1], [2]]),
    )
    baseline = torch.einsum(
        "nfb,fbd->nfd",
        pilot.ple_basis(torch.from_numpy(x), tokenizer.edges),
        tokenizer.num_weight,
    ) + tokenizer.num_bias
    assert torch.allclose(output, baseline)
    output.square().mean().backward()
    assert tokenizer.support_gate_logits.grad is not None


def test_empirical_atom_interval_has_its_probability_mass() -> None:
    parts = {
        "train": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "val": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "test": np.array([0.0], dtype=np.float32),
    }
    rank, lower, upper, *_ = universal_pilot.encode_field(parts, 8, 4, 4)
    assert rank["val"].tolist() == [0.375, 0.75, 0.875]
    assert np.allclose(upper["val"] - lower["val"], [0.75, 0.0, 0.25])


def test_interval_ple_matches_midpoint_for_zero_width_and_numerical_average() -> None:
    edges = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    lower = torch.tensor([[0.1], [0.2]])
    upper = torch.tensor([[0.1], [0.8]])
    actual = universal_pilot.ple_interval_basis(lower, upper, edges)
    expected_point = pilot.ple_basis(lower[:1], edges)
    assert torch.allclose(actual[:1], expected_point, atol=1e-6)
    grid = torch.linspace(0.2, 0.8, 100_001)[:, None]
    numerical = pilot.ple_basis(grid, edges).mean(dim=0)
    assert torch.allclose(actual[1], numerical, atol=2e-5)


def test_trichart_uses_one_trainable_backbone_for_three_views() -> None:
    data = SimpleNamespace(
        x_num={"train": np.zeros((8, 1), dtype=np.float32)},
        x_bin=None,
        category_cardinalities=[],
    )
    universal = SimpleNamespace(n_fields=1)
    encoding = SimpleNamespace(
        qple_edges=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        tple_edges=np.array([[0.0, 0.25, 1.0]], dtype=np.float32),
        selected_columns=[],
        cardinalities=[],
    )
    config = {
        "d_token": 4,
        "rank_ple_bins": 2,
        "top_exact_levels": 2,
        "rare_hash_buckets": 2,
        "frequency_modes": 1,
        "seed": 17,
        "width": 8,
        "depth": 1,
        "ft_feedforward_width": 8,
        "dropout": 0.0,
    }
    model = TriChartModel(data, universal, encoding, config, "mlp")
    rows = 5
    rank = torch.linspace(0.1, 0.9, rows)[:, None]
    views = model(
        rank,
        torch.empty(rows, 0),
        torch.empty(rows, 0, dtype=torch.long),
        rank,
        rank,
        rank,
        torch.ones(rows, 1, dtype=torch.long),
        torch.zeros(rows, 1),
    )
    assert views.shape == (rows, 3)
    loss = views.square().mean() + 0.1 * views.var(dim=1, unbiased=False).mean()
    loss.backward()
    assert model.backbone.head.weight.grad is not None


def test_t_anchored_trichart_is_exact_tple_at_zero_gates() -> None:
    data = SimpleNamespace(
        x_num={"train": np.zeros((8, 1), dtype=np.float32)},
        x_bin=None,
        category_cardinalities=[],
    )
    universal = SimpleNamespace(n_fields=1)
    encoding = SimpleNamespace(
        qple_edges=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        tple_edges=np.array([[0.0, 0.25, 1.0]], dtype=np.float32),
        selected_columns=[],
        cardinalities=[],
    )
    config = {
        "d_token": 4,
        "rank_ple_bins": 2,
        "top_exact_levels": 2,
        "rare_hash_buckets": 2,
        "frequency_modes": 1,
        "seed": 17,
        "width": 8,
        "depth": 1,
        "ft_feedforward_width": 8,
        "dropout": 0.0,
    }
    torch.manual_seed(31)
    baseline = pilot.SupportModel(
        data=data,
        encoding=encoding,
        method="tple",
        architecture="mlp",
        d_token=4,
        width=8,
        depth=1,
        ft_feedforward_width=8,
        dropout=0.0,
    )
    torch.manual_seed(31)
    anchored = TAnchoredTriChart(
        data, universal, encoding, config, "mlp", "field_gate"
    )
    rows = 5
    raw = torch.linspace(0.1, 0.9, rows)[:, None]
    empty_float = torch.empty(rows, 0)
    empty_cat = torch.empty(rows, 0, dtype=torch.long)
    code = torch.ones(rows, 1, dtype=torch.long)
    information = torch.zeros(rows, 1)
    baseline_prediction = baseline(raw, empty_float, empty_cat, code[:, :0])
    anchored_prediction = anchored(
        raw,
        empty_float,
        empty_cat,
        raw,
        raw,
        raw,
        code,
        information,
    )
    assert torch.equal(anchored.q_gate, torch.zeros_like(anchored.q_gate))
    assert torch.equal(anchored.rank_gate, torch.zeros_like(anchored.rank_gate))
    assert torch.equal(anchored_prediction, baseline_prediction)
    anchored_prediction.square().mean().backward()
    assert anchored.q_gate.grad is not None
    assert anchored.rank_gate.grad is not None


def test_frozen_anchor_residual_keeps_exact_epoch_zero_fallback() -> None:
    data = SimpleNamespace(
        x_num={"train": np.zeros((8, 1), dtype=np.float32)},
        x_bin=None,
        category_cardinalities=[],
    )
    universal = SimpleNamespace(n_fields=1)
    encoding = SimpleNamespace(
        qple_edges=np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        tple_edges=np.array([[0.0, 0.25, 1.0]], dtype=np.float32),
        selected_columns=[],
        cardinalities=[],
    )
    config = {
        "d_token": 4,
        "rank_ple_bins": 2,
        "top_exact_levels": 2,
        "rare_hash_buckets": 2,
        "frequency_modes": 1,
        "seed": 17,
        "width": 8,
        "depth": 1,
        "ft_feedforward_width": 8,
        "dropout": 0.0,
    }
    anchor = pilot.SupportModel(
        data=data,
        encoding=encoding,
        method="tple",
        architecture="mlp",
        d_token=4,
        width=8,
        depth=1,
        ft_feedforward_width=8,
        dropout=0.0,
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    model = FrozenAnchorResidual(
        anchor, data, universal, encoding, config, "mlp"
    )
    rows = 5
    raw = torch.linspace(0.1, 0.9, rows)[:, None]
    features = (
        raw,
        torch.empty(rows, 0),
        torch.empty(rows, 0, dtype=torch.long),
        raw,
        raw,
        raw,
        torch.ones(rows, 1, dtype=torch.long),
        torch.zeros(rows, 1),
    )
    prediction, anchor_prediction = model(*features)
    assert torch.equal(model.residual_gate, torch.zeros_like(model.residual_gate))
    assert torch.equal(prediction, anchor_prediction)
    model.train()
    assert model.anchor.training is False
    prediction.square().mean().backward()
    assert model.residual_gate.grad is not None


def test_fixed_heterogeneous_policy_preserves_failed_gate() -> None:
    decision = json.loads(
        (RESULTS / "multiview_equal_compute_decision.json").read_text()
    )["fixed_policy_predeclared_gate"]
    assert decision["test_wins"] == 10
    assert decision["test_cells"] == 12
    assert decision["positive_dataset_means"] == 3
    assert decision["clauses"]["at_least_8_of_12_test_wins"] is True
    assert decision["clauses"]["positive_overall_mean_relative_test_gain"] is False
    assert decision["clauses"]["positive_mean_within_both_task_families"] is False
    assert decision["passed"] is False


def test_adult_exact_multiview_additivity_gate_and_parameter_control() -> None:
    import csv

    rows = list(csv.DictReader((RESULTS / "adult_exact_multiview.csv").open()))
    assert len(rows) == 3
    assert sum(float(row["relative_val_logloss_gain_pct"]) > 0 for row in rows) == 2
    assert all(
        row["anchor_parameters"] == row["q_member_parameters"]
        == row["t_member_parameters"]
        for row in rows
    )
    decision = json.loads(
        (RESULTS / "adult_exact_multiview_decision.json").read_text()
    )
    assert decision["gate_evaluable"] is True
    assert decision["gate_passed"] is True
    assert decision["validation_wins"] == 2
    assert decision["mean_relative_validation_logloss_gain_pct"] == pytest.approx(
        0.42440705988236876
    )

    hybrid = list(
        csv.DictReader((RESULTS / "adult_exact_multiview_hybrid.csv").open())
    )[0]
    assert float(hybrid["relative_val_logloss_gain_pct"]) > 0
    assert hybrid["anchor_parameters"] == hybrid["q_member_parameters"]
    assert hybrid["anchor_parameters"] == hybrid["t_member_parameters"]


def test_heterobag_three_member_prospective_gate() -> None:
    decision = json.loads(
        (RESULTS / "heterobag_three_member_decision.json").read_text()
    )
    assert decision["prospective_gate_passed"] is True
    assert decision["test_wins"] == 10
    assert decision["test_cells"] == 12
    assert decision["positive_dataset_means"] == 4
    assert all(decision["clauses"].values())
    assert decision["mean_relative_test_gain_pct"] == pytest.approx(
        0.8430600035979848
    )
    assert decision["mean_relative_test_gain_pct_by_task"] == pytest.approx(
        {"classification": 1.084319211408605, "regression": 0.6018007957873647}
    )
