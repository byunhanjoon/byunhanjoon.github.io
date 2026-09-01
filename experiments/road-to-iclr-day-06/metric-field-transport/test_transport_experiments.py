from __future__ import annotations

import numpy as np
import torch

import successor_experiments as successor
import transport_experiments as transport


def test_transport_graph_uses_training_anchors_and_excludes_self() -> None:
    task = successor.load_task("employee_salaries")
    parts = successor.split_state_indices(task, 0)
    tables, _ = successor.representation_tables(task, parts["train"])
    graph = transport.build_transport_graph(
        task,
        parts["train"],
        parts["validation"],
        tables["distance_m32"],
        shuffled=False,
        split_index=0,
    )
    assert np.allclose(graph.neighbor_weights[graph.valid_query].sum(axis=1), 1.0)
    for local, state in enumerate(parts["train"]):
        assert local not in graph.neighbor_train_indices[state]
    assert graph.valid_query[parts["train"]].all()
    assert graph.valid_query[parts["validation"]].all()
    assert not graph.valid_query[parts["test"]].any()


def test_transport_branch_is_exact_raw_base_at_initialization() -> None:
    state_count = 14
    training_states = np.arange(10, dtype=np.int64)
    neighbors = np.tile(np.arange(8, dtype=np.int64), (state_count, 1))
    weights = np.full((state_count, 8), 1 / 8, dtype=np.float32)
    differences = np.zeros((state_count, 8, 7), dtype=np.float32)
    graph = transport.TransportGraph(
        neighbors,
        weights,
        differences,
        np.ones(state_count, dtype=bool),
        np.arange(state_count),
    )
    x = torch.randn(19, 23)
    states = torch.arange(19) % state_count

    successor.seed_everything(441)
    raw = transport.TransportMLP(23, state_count, training_states, "raw_base", None, 7).eval()
    successor.seed_everything(441)
    zero = transport.TransportMLP(
        23, state_count, training_states, "transport_zero", graph, 7
    ).eval()
    successor.seed_everything(441)
    first = transport.TransportMLP(
        23, state_count, training_states, "transport_first_order", graph, 7
    ).eval()
    with torch.inference_mode():
        expected = raw(x, states, evaluation=True)
        assert torch.equal(expected, zero(x, states, evaluation=True))
        assert torch.equal(expected, first(x, states, evaluation=True))


def test_first_order_auxiliary_is_finite_and_differentiable() -> None:
    state_count = 12
    training_states = np.arange(9, dtype=np.int64)
    neighbors = np.tile(np.arange(8, dtype=np.int64), (state_count, 1))
    weights = np.full((state_count, 8), 1 / 8, dtype=np.float32)
    differences = np.ones((state_count, 8, 5), dtype=np.float32)
    graph = transport.TransportGraph(
        neighbors,
        weights,
        differences,
        np.ones(state_count, dtype=bool),
        np.arange(state_count),
    )
    model = transport.TransportMLP(
        6, state_count, training_states, "transport_first_order", graph, 5
    )
    mask = torch.tensor([True, False, True, False, True, False, True, False, True])
    loss = model.auxiliary_loss(mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.correction_left is not None
    assert model.correction_left.grad is not None
    assert torch.isfinite(model.correction_left.grad).all()
