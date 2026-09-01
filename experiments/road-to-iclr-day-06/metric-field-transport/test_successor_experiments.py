from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

import successor_experiments as successor


HERE = Path(__file__).resolve().parent


def test_frozen_protocol_hash_matches() -> None:
    expected = (HERE / "PROTOCOL_SHA256.txt").read_text().split()[0]
    observed = hashlib.sha256((HERE / "DEVELOPMENT_PROTOCOL.md").read_bytes()).hexdigest()
    assert observed == expected


def test_inner_split_and_target_seal_exclude_original_test() -> None:
    task = successor.load_task("employee_salaries")
    state_parts = successor.split_state_indices(task, 0)
    row_parts = successor.split_row_indices(task, 0)
    inner_train, inner_validation = successor.inner_state_split(task, 0)

    assert set(inner_train).isdisjoint(set(inner_validation))
    assert set(inner_train) | set(inner_validation) == set(state_parts["train"])
    assert (set(inner_train) | set(inner_validation)).isdisjoint(set(state_parts["test"]))

    target = successor.sealed_raw_target(task, row_parts["test"])
    assert np.isfinite(target[row_parts["train"]]).all()
    assert np.isfinite(target[row_parts["validation"]]).all()
    assert np.isnan(target[row_parts["test"]]).all()


def test_representations_use_only_training_landmarks_and_are_conditioned() -> None:
    task = successor.load_task("employee_salaries")
    training_states = successor.split_state_indices(task, 0)["train"]
    tables, metadata = successor.representation_tables(task, training_states)

    assert set(metadata["landmark_indices"]["m32"]).issubset(set(training_states.tolist()))
    assert np.allclose(tables["weights_m32"].sum(axis=1), 1.0, atol=1e-6)
    for name, table in tables.items():
        assert table.shape[0] == len(task.states), name
        assert np.isfinite(table).all(), name

    training_coordinates = tables["distance_m32"][training_states]
    assert np.allclose(training_coordinates.mean(axis=0), 0.0, atol=2e-5)
    standard_deviations = training_coordinates.std(axis=0)
    assert np.all(np.isclose(standard_deviations, 0.0, atol=2e-5) | np.isclose(standard_deviations, 1.0, atol=2e-5))


def test_identity_and_rezero_are_exact_paired_initial_controls() -> None:
    x = torch.randn(11, 39)

    torch.manual_seed(91)
    direct = successor.ControlledFactorMLP(39, 32, "weights_direct").eval()
    torch.manual_seed(91)
    identity = successor.ControlledFactorMLP(39, 32, "factor_identity_learned").eval()
    torch.manual_seed(91)
    rezero = successor.ControlledFactorMLP(39, 32, "factor_rezero").eval()

    with torch.inference_mode():
        expected = direct(x)
        assert torch.equal(expected, identity(x))
        assert torch.equal(expected, rezero(x))


def test_sparse_batch_accepts_torch_indices() -> None:
    design = sparse.csr_matrix(np.arange(30, dtype=np.float32).reshape(10, 3))
    result = successor.dense_batch(design, torch.tensor([7, 2]), torch.device("cpu"))
    assert torch.equal(result, torch.tensor([[21.0, 22.0, 23.0], [6.0, 7.0, 8.0]]))
