from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tournament.common import PANEL_PATH, ROOT, bd
from tournament.models import initial_predictions
from tournament.optimizers import BlockAdaptiveOptimizer
from tournament.representations import audit_orbit_coordinates, build_interface


def synthetic_pair(seed: int = 11):
    rng = np.random.default_rng(seed)
    splits = [rng.normal(size=(n, 8)) for n in (96, 32, 40)]
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    transformed = [values @ q for values in splits]
    common = dict(
        family="test",
        variant="reference",
        scope="all",
        columns=[f"x::{index}" for index in range(8)],
        feature_blocks={"x": list(range(8))},
        categorical_blocks={},
        metadata={},
    )
    reference = bd.Representation(
        representation_id="reference",
        member=-1,
        X_train=splits[0],
        X_validation=splits[1],
        X_test=splits[2],
        transforms={"x": np.eye(8)},
        is_reference=True,
        **common,
    )
    rotated = bd.Representation(
        representation_id="rotated",
        member=0,
        X_train=transformed[0],
        X_validation=transformed[1],
        X_test=transformed[2],
        transforms={"x": q},
        is_reference=False,
        **common,
    )
    return reference, rotated, q


@pytest.mark.parametrize(
    "method,parameters,tolerance",
    [
        ("gram_anchor", {"anchors": 16, "selection": "gram_pivot", "normalize": True}, 1e-10),
        ("gram_distance", {"anchors": 16, "selection": "gram_pivot", "kernel": "rbf"}, 1e-10),
        ("nystrom_gram", {"anchors": 16, "selection": "gram_pivot"}, 1e-8),
        ("pca", {}, 1e-8),
        ("hybrid_spectral", {"tau": 0.05, "anchors": 16}, 1e-8),
    ],
)
def test_interface_is_orthogonally_invariant(method, parameters, tolerance):
    reference, rotated, _ = synthetic_pair()
    mapped_reference = build_interface(reference, method, "synthetic", **parameters)
    mapped_rotated = build_interface(rotated, method, "synthetic", **parameters)
    audit = audit_orbit_coordinates(mapped_reference, [mapped_rotated])[0]
    assert audit["shape_match"]
    assert audit["train_relative_error"] < tolerance
    assert audit["validation_relative_error"] < tolerance
    assert audit["test_relative_error"] < tolerance


@pytest.mark.parametrize("method", ["block_scalar_adam", "block_adam", "matrix_adam"])
def test_block_optimizer_one_step_is_equivariant(method):
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    weight = torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float64, requires_grad=True)
    rotated = torch.tensor(weight.detach().numpy() @ q, dtype=torch.float64, requires_grad=True)
    gradient = torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float64)
    rotated_gradient = torch.tensor(gradient.numpy() @ q, dtype=torch.float64)
    first = BlockAdaptiveOptimizer(weight, [range(8)], method=method, lr=1e-3)
    second = BlockAdaptiveOptimizer(rotated, [range(8)], method=method, lr=1e-3)
    weight.grad = gradient
    rotated.grad = rotated_gradient
    first.step()
    second.step()
    assert np.allclose(rotated.detach().numpy(), weight.detach().numpy() @ q, atol=1e-9, rtol=1e-9)


def test_data_equivariant_initialization_matches_initial_predictions():
    reference, rotated, _ = synthetic_pair()
    config = {
        "width": 32,
        "hidden_layers": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 1,
    }
    first = initial_predictions(
        "controlled_mlp", "regression", reference, 3, "cpu", config, "data_equivariant"
    )
    second = initial_predictions(
        "controlled_mlp", "regression", rotated, 3, "cpu", config, "data_equivariant"
    )
    assert np.max(np.abs(first - second)) < 1e-5


def test_tabm_data_equivariant_initialization_matches_initial_predictions():
    reference, rotated, _ = synthetic_pair()
    config = {
        "arch_type": "tabm",
        "k": 4,
        "d_block": 32,
        "n_blocks": 2,
        "dropout": 0.0,
        "share_training_batches": True,
    }
    first = initial_predictions(
        "tabm_d", "regression", reference, 3, "cpu", config, "data_equivariant"
    )
    second = initial_predictions(
        "tabm_d", "regression", rotated, 3, "cpu", config, "data_equivariant"
    )
    assert np.max(np.abs(first - second)) < 1e-5


def test_prospective_panel_is_locked_and_hash_recorded():
    panel = json.loads(PANEL_PATH.read_text())
    assert panel["status"] == "LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS"
    assert panel["selection_evidence"]["outcomes_accessed_before_lock"] is False
    assert 6 <= len(panel["datasets"]) <= 8
    assert len({record["key"] for record in panel["datasets"]}) == len(panel["datasets"])
    assert all(1000 <= record["rows"] <= 30000 for record in panel["datasets"])
    assert all(record["raw_columns"] <= 100 for record in panel["datasets"])
    expected = hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest()
    recorded = (ROOT / "configs" / "NEW_PROSPECTIVE_PANEL.sha256").read_text().split()[0]
    assert recorded == expected
