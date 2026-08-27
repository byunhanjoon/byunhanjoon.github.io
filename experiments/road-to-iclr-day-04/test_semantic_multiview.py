"""Focused checks for the neural-only semantic multi-view pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import semantic_multiview_pilot as pilot  # noqa: E402
import field_local_distillation_pilot as local  # noqa: E402


def test_float32_quantile_edges_remain_strict_with_discrete_fields() -> None:
    train = np.column_stack(
        (
            np.zeros(128, dtype=np.float32),
            np.repeat(np.arange(4, dtype=np.float32), 32),
        )
    )
    edges = pilot.quantile_edges(train, n_bins=16)
    assert edges.dtype == np.float32
    assert np.all(np.diff(edges, axis=1) > 0)
    basis = pilot.ple_basis(torch.from_numpy(train), torch.from_numpy(edges))
    assert torch.isfinite(basis).all()


def test_float32_quantile_edges_remain_strict_at_large_offsets() -> None:
    # At this scale, a float64 epsilon based on the field range can disappear
    # when the padded knots are cast back to float32 (as observed on POL).
    train = np.column_stack(
        (
            np.full(128, 1.0e8, dtype=np.float32),
            np.repeat(np.array([1.0e8, 1.0e8 + 8], dtype=np.float32), 64),
        )
    )
    edges = pilot.quantile_edges(train, n_bins=16)
    assert edges.dtype == np.float32
    assert np.all(np.diff(edges, axis=1) > 0)
    assert np.all(np.isfinite(edges))
    basis = pilot.ple_basis(torch.from_numpy(train), torch.from_numpy(edges))
    assert torch.isfinite(basis).all()


def test_fourier_chart_respects_the_ring_seam() -> None:
    phases = torch.tensor([[0.001], [0.999], [0.5]], dtype=torch.float32)
    basis = pilot.cyclic_fourier_basis(phases, n_bins=16)[:, 0]
    seam_distance = torch.linalg.vector_norm(basis[0] - basis[1])
    opposite_distance = torch.linalg.vector_norm(basis[0] - basis[2])
    assert seam_distance < opposite_distance / 4


def test_wrong_geometry_changes_only_declared_cyclic_fields() -> None:
    train = np.column_stack(
        (
            np.linspace(-2, 2, 64, dtype=np.float32),
            np.arange(64, dtype=np.float32) % 24,
        )
    )
    edges = pilot.quantile_edges(train, n_bins=8)
    common = dict(
        edges=edges,
        n_bin_fields=0,
        category_cardinalities=[],
        d_token=8,
        cyclic_columns=[1],
        cyclic_periods=[24.0],
        cyclic_origins=[0.0],
    )
    correct = pilot.FieldTokenizer(view="topology", **common)
    wrong = pilot.FieldTokenizer(view="wrong", **common)
    x = torch.from_numpy(train)
    correct_basis = correct.numerical_basis(x)
    wrong_basis = wrong.numerical_basis(x)
    torch.testing.assert_close(correct_basis[:, 0], wrong_basis[:, 0])
    assert not torch.allclose(correct_basis[:, 1], wrong_basis[:, 1])


def test_all_backbones_accept_single_and_paired_views() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(64, 4)).astype(np.float32)
    edges = pilot.quantile_edges(train, n_bins=8)
    x_num = torch.from_numpy(train[:7])
    x_bin = torch.from_numpy(rng.normal(size=(7, 2)).astype(np.float32))
    x_cat = torch.from_numpy(rng.integers(0, 4, size=(7, 1), dtype=np.int64))
    for backbone in pilot.BACKBONES:
        for method in ("ple", "topology", "multiview_noalign", "multiview_vicreg"):
            torch.manual_seed(0)
            model = pilot.SemanticMultiViewModel(
                method=method,
                backbone=backbone,
                edges=edges,
                n_bin_fields=2,
                category_cardinalities=[4],
                cyclic_columns=[3],
                cyclic_periods=[24.0],
                cyclic_origins=[0.0],
                d_token=8,
                width=16,
                depth=1,
            )
            first, first_latent, second, second_latent = model(x_num, x_bin, x_cat)
            assert first.shape == (7,)
            assert first_latent.shape[0] == 7
            assert torch.isfinite(first).all()
            if method in pilot.DUAL_METHODS:
                assert second is not None and second.shape == (7,)
                assert second_latent is not None and second_latent.shape == first_latent.shape
                alignment, pieces = pilot.vicreg_loss(first_latent, second_latent)
                assert torch.isfinite(alignment)
                assert set(pieces) == {"invariance", "variance", "covariance"}
            else:
                assert second is None and second_latent is None


def test_schema_declarations_match_official_suffixes() -> None:
    assert [name for name, _, _ in pilot.CYCLIC_SUFFIXES["weather"]] == [
        "day_of_week",
        "day_of_month",
        "minute_of_day",
        "hour_of_day",
        "month",
    ]
    for dataset in ("cooking-time", "delivery-eta", "maps-routing"):
        assert [name for name, _, _ in pilot.CYCLIC_SUFFIXES[dataset]] == [
            "day_of_week",
            "minute_of_day",
            "hour_of_day",
        ]
    summary = json.loads(
        (HERE / "results/semantic_multiview_summary.json").read_text()
    )
    assert summary["status"] == "falsified_as_broad_default"
    vicreg = next(
        row for row in summary["method_summary"]
        if row["method"] == "multiview_vicreg"
    )
    assert vicreg["pairs"] == 6
    assert vicreg["wins_vs_ple"] == 1
    assert abs(vicreg["mean_gain_vs_ple_pct"] + 0.6898932233748399) < 1e-12
    mechanism = summary["correct_vs_wrong_geometry"]
    assert mechanism["correct_wins"] == 4
    assert abs(mechanism["mean_correct_gain_pct"] - 0.03340063284209742) < 1e-12


def _local_fixture() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    rng = np.random.default_rng(23)
    train = rng.normal(size=(64, 4)).astype(np.float32)
    train[:, 3] = np.arange(64, dtype=np.float32) % 24
    edges = pilot.quantile_edges(train, n_bins=8)
    x_num = torch.from_numpy(train[:9])
    x_bin = torch.from_numpy(rng.normal(size=(9, 1)).astype(np.float32))
    x_cat = torch.from_numpy(rng.integers(0, 3, size=(9, 1), dtype=np.int64))
    common: dict[str, object] = dict(
        edges=edges,
        n_bin_fields=1,
        category_cardinalities=[3],
        cyclic_columns=[3],
        cyclic_periods=[24.0],
        cyclic_origins=[0.0],
        d_token=8,
    )
    return edges, x_num, x_bin, x_cat, common


def test_field_local_adapter_is_exact_ple_at_initialization() -> None:
    edges, x_num, x_bin, x_cat, common = _local_fixture()
    model_common = dict(
        backbone="mlp",
        edges=edges,
        n_bin_fields=common["n_bin_fields"],
        category_cardinalities=common["category_cardinalities"],
        cyclic_columns=common["cyclic_columns"],
        cyclic_periods=common["cyclic_periods"],
        cyclic_origins=common["cyclic_origins"],
        d_token=8,
        width=16,
        depth=1,
    )
    torch.manual_seed(9)
    baseline = local.FieldLocalModel(method="ple", **model_common)
    torch.manual_seed(9)
    adapted = local.FieldLocalModel(method="semantic_local", **model_common)
    baseline.eval()
    adapted.eval()
    baseline_prediction, _, _ = baseline(x_num, x_bin, x_cat)
    adapted_prediction, _, _ = adapted(x_num, x_bin, x_cat)
    torch.testing.assert_close(adapted_prediction, baseline_prediction, rtol=0, atol=0)
    assert isinstance(adapted.tokenizer, local.FieldLocalTokenizer)
    assert torch.count_nonzero(adapted.tokenizer.gate_logits) == 0


def test_field_local_residual_updates_only_declared_tokens() -> None:
    _, x_num, x_bin, x_cat, common = _local_fixture()
    tokenizer = local.FieldLocalTokenizer(method="semantic_noalign", **common)
    tokenizer.gate_logits.data.fill_(0.5)
    base_tokens = tokenizer.base(x_num, x_bin, x_cat)
    tokens, _, _ = tokenizer(x_num, x_bin, x_cat)
    delta = tokens - base_tokens
    noncyclic = [0, 1, 2, 4, 5]
    torch.testing.assert_close(delta[:, noncyclic], torch.zeros_like(delta[:, noncyclic]))
    assert torch.count_nonzero(delta[:, 3]) > 0


def test_local_distillation_stops_gradient_into_ple_teacher() -> None:
    _, x_num, x_bin, x_cat, common = _local_fixture()
    tokenizer = local.FieldLocalTokenizer(method="semantic_local", **common)
    _, teacher, student = tokenizer(x_num, x_bin, x_cat)
    loss = local.local_distillation_loss(teacher, student)
    loss.backward()
    assert tokenizer.base.num_weight.grad is None
    assert tokenizer.residual_weight.grad is not None
    assert torch.linalg.vector_norm(tokenizer.residual_weight.grad) > 0


def test_field_local_wrong_geometry_preserves_shape_but_changes_chart() -> None:
    _, x_num, _, _, common = _local_fixture()
    correct = local.FieldLocalTokenizer(method="semantic_local", **common)
    wrong = local.FieldLocalTokenizer(method="semantic_wrong_local", **common)
    correct_basis = correct.residual_basis(x_num)
    wrong_basis = wrong.residual_basis(x_num)
    assert correct_basis.shape == wrong_basis.shape == (9, 1, 8)
    assert not torch.allclose(correct_basis, wrong_basis)
    summary = json.loads(
        (HERE / "results/field_local_distillation_summary.json").read_text()
    )
    assert summary["status"] == "stop_branch"
    assert summary["test_metrics_computed"] is False
    assert summary["predeclared_gate"] == {
        "cleared_cells": 3,
        "passed": False,
        "required_cells": 5,
        "rule": (
            "semantic_local must have lower validation RMSE than PLE, the "
            "parameter-matched PLE adapter, and semantic_wrong_local in at "
            "least 5/6 Weather/Cooking architecture cells"
        ),
    }
    assert abs(summary["semantic_local"]["mean_gain_vs_ple_pct"] - 0.004129637721143849) < 1e-12
