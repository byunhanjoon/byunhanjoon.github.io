from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_task, transform_frame
from src.analysis.runner import _combined_gpu_peak, _transform, load_config, selected_jobs
from src.metrics import classification_metrics, disagreement_metrics, regression_metrics
from src.transforms import SignedPowerTransform


def test_builtin_split_has_no_overlap(tmp_path: Path):
    task = load_task(
        {"dataset": "breast_cancer", "source": "sklearn", "problem_type": "binary"},
        seed=42,
        max_context=300,
        max_query=100,
        cache_dir=tmp_path,
    )
    audit = task.audit()
    assert audit["train_validation_disjoint"]
    assert audit["train_test_disjoint"]
    assert audit["validation_test_disjoint"]
    assert audit["columns_identical"]


def test_transform_frame_preserves_non_numeric_columns_and_masks():
    frame = pd.DataFrame(
        {
            "number": [1.0, np.nan, 3.0, 5.0],
            "category": pd.Series(["a", "b", "a", "c"], dtype="category"),
        }
    )
    transform = SignedPowerTransform(2.0).fit(frame[["number"]].to_numpy())
    result = transform_frame(frame, ["number"], transform)
    assert result["category"].equals(frame["category"])
    assert np.array_equal(result["number"].isna(), frame["number"].isna())


def test_classification_metrics_and_disagreement():
    y = np.asarray([0, 1, 1, 0])
    clean = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]])
    same = classification_metrics(y, clean)
    assert 0 < same["loss"] < 1
    assert same["accuracy"] == 1.0
    disagreement = disagreement_metrics(clean, clean, "binary")
    assert disagreement == {"js_divergence": 0.0, "total_variation": 0.0, "argmax_flip_rate": 0.0}


def test_regression_metrics_and_disagreement():
    y = np.asarray([1.0, 2.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["loss"] == 0.0
    assert metrics["r2"] == 1.0
    disagreement = disagreement_metrics(y, y, "regression", normalization_scale=1.0)
    assert disagreement["normalized_absolute_disagreement"] == 0.0
    assert disagreement["prediction_spearman"] == 1.0


def test_job_selection_filters_transform_and_seed_before_sharding():
    config = {
        "datasets": [{"dataset": "a"}, {"dataset": "b"}],
        "models": ["m1", "m2"],
        "transforms": [
            {"name": "identity", "values": [0.0]},
            {"name": "warp", "values": [0.5, 1.0]},
        ],
        "seeds": [11, 12],
        "split_seed": 20260831,
    }
    jobs = selected_jobs(
        config,
        datasets={"b"},
        models={"m2"},
        transforms={"warp"},
        seeds={12},
        split_seeds=None,
        shard_index=0,
        num_shards=1,
    )
    assert jobs == [
        ({"dataset": "b"}, "m2", "warp", 0.5, 12, 20260831),
        ({"dataset": "b"}, "m2", "warp", 1.0, 12, 20260831),
    ]


def test_job_selection_expands_and_filters_split_seeds():
    config = {
        "datasets": [{"dataset": "a"}],
        "models": ["m"],
        "transforms": [{"name": "identity", "values": [0.0]}],
        "seeds": [11],
        "split_seeds": [101, 102],
    }
    all_jobs = selected_jobs(
        config, datasets=None, models=None, split_seeds=None, shard_index=0, num_shards=1
    )
    assert [job[-1] for job in all_jobs] == [101, 102]
    filtered = selected_jobs(
        config, datasets=None, models=None, split_seeds={102}, shard_index=0, num_shards=1
    )
    assert len(filtered) == 1 and filtered[0][-1] == 102


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("negative_affine", 1.0),
        ("empirical_cdf", 1.0),
        ("quantile_gaussian", 1e-5),
        ("atomic_spacing", 1.0),
        ("composition", 1.0),
    ],
)
def test_phase2_transform_factory(name, value):
    train = np.arange(24, dtype=np.float64).reshape(8, 3)
    transform = _transform(name, value, 17).fit(train)
    warped = transform.transform(train)
    assert warped.shape == train.shape
    assert np.isfinite(warped).all()


def test_frozen_phase2_config_has_expected_grid_size():
    path = Path(__file__).parents[1] / "configs" / "audit" / "main.yaml"
    config = load_config(path)
    jobs = selected_jobs(config, datasets=None, models=None, shard_index=0, num_shards=1)
    assert config["phase"] == "phase2_development"
    assert len(config["datasets"]) == 20
    assert len(jobs) == 13_440


def test_combined_gpu_peak_includes_isolated_worker(monkeypatch):
    monkeypatch.setattr("src.analysis.runner._gpu_peak", lambda device: 17)
    telemetry = {
        "clean": {"peak_gpu_memory_bytes": 101},
        "matched": {"peak_gpu_memory_bytes": 303},
        "context_only": {"peak_gpu_memory_bytes": None},
    }
    assert _combined_gpu_peak(telemetry, "cuda:0") == 303
