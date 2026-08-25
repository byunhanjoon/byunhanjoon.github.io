import numpy as np
import torch

import experiments.day3.trajectory_decomposition as trajectory
from experiments.day3.broad_models import make_model
from experiments.day3.core import Dataset, condition_transform, make_prepared


def _prepared_pair(kappa: float = 100.0):
    rng = np.random.default_rng(12)
    sizes = {"train": 96, "val": 32, "test": 40}
    reference = {part: rng.normal(size=(size, 5)) for part, size in sizes.items()}
    transform = condition_transform(5, kappa, seed=51)
    changed = {part: values @ transform for part, values in reference.items()}
    y = {
        part: (values[:, 0] - 0.4 * values[:, 2] + rng.normal(scale=0.1, size=len(values))).astype(
            np.float32
        )
        for part, values in reference.items()
    }
    dataset = Dataset("synthetic", "regression", reference, None, None, y, 1, "synthetic")
    return make_prepared(dataset, reference, {}), make_prepared(dataset, changed, {}), transform


def test_function_matching_preserves_first_layer_outputs():
    reference_data, changed_data, transform = _prepared_pair(1000.0)
    torch.manual_seed(7)
    reference = make_model("mlp", 5, 1)
    changed = make_model("mlp", 5, 1)
    changed.load_state_dict(reference.state_dict())
    trajectory.function_match_first_layer(changed.first, transform)
    reference.eval()
    changed.eval()
    with torch.inference_mode():
        reference_prediction = reference(torch.from_numpy(reference_data.x["val"]))
        changed_prediction = changed(torch.from_numpy(changed_data.x["val"]))
    assert torch.allclose(reference_prediction, changed_prediction, atol=2e-5, rtol=2e-5)
    assert trajectory.mapped_weight_drift(reference.first, changed.first, transform) < 1e-5


def test_symmetric_prediction_drift_is_scale_symmetric():
    left = np.array([[-2.0], [1.0], [3.0]])
    right = np.array([[-1.0], [2.0], [2.5]])
    assert np.isclose(
        trajectory.symmetric_prediction_drift(left, right),
        trajectory.symmetric_prediction_drift(right, left),
    )
    assert trajectory.symmetric_prediction_drift(left, left) == 0.0


def test_matched_adamw_starts_together_then_can_drift(monkeypatch):
    reference, changed, transform = _prepared_pair(30.0)
    cfg = {
        "training": {
            "updates": 1,
            "trajectory_steps": [0, 1],
            "batch_size": 48,
            "probe_rows_per_split": 32,
            "adamw_learning_rate": 0.003,
            "adamw_weight_decay": 0.0001,
            "natural_first_learning_rate": 0.03,
            "natural_later_adamw_learning_rate": 0.001,
        },
        "analysis_gates": {"matched_step0_max_drift": 0.0001},
    }
    monkeypatch.setattr(trajectory, "config", lambda: cfg)
    _, observations = trajectory.train_pair(
        reference,
        changed,
        model_name="mlp",
        arm="matched_adamw",
        transform=transform,
        seed=0,
        device="cpu",
    )
    validation = {int(row["step"]): row for row in observations if row["probe_split"] == "val"}
    assert validation[0]["prediction_drift"] < 1e-5
    assert validation[1]["prediction_drift"] > validation[0]["prediction_drift"]


def test_matched_input_natural_is_a_one_step_closure_control(monkeypatch):
    reference, changed, transform = _prepared_pair(30.0)
    cfg = {
        "training": {
            "updates": 1,
            "trajectory_steps": [0, 1],
            "batch_size": 96,
            "probe_rows_per_split": 32,
            "adamw_learning_rate": 0.003,
            "adamw_weight_decay": 0.0001,
            "natural_first_learning_rate": 0.03,
            "natural_later_adamw_learning_rate": 0.001,
        },
        "analysis_gates": {"matched_step0_max_drift": 0.0001},
    }
    monkeypatch.setattr(trajectory, "config", lambda: cfg)
    _, observations = trajectory.train_pair(
        reference,
        changed,
        model_name="mlp",
        arm="matched_input_natural",
        transform=transform,
        seed=0,
        device="cpu",
    )
    validation = {int(row["step"]): row for row in observations if row["probe_split"] == "val"}
    assert validation[0]["prediction_drift"] < 1e-5
    assert validation[1]["prediction_drift"] < 2e-4
