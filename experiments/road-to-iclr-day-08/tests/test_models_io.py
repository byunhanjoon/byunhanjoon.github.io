from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.analysis.io import atomic_save_npz, code_digest, load_completed_run, load_run_predictions, sha256_file
from src.analysis.runner import build_run_snapshot
from src.data import load_task
from src.models import fit_predict_many


def test_xgboost_prediction_shape(tmp_path: Path):
    task = load_task(
        {"dataset": "breast_cancer", "source": "sklearn", "problem_type": "binary"},
        seed=43,
        max_context=200,
        max_query=60,
        cache_dir=tmp_path,
    )
    outcomes = fit_predict_many(
        "xgboost",
        task.problem_type,
        task.X_train,
        task.y_train,
        {"first": task.X_test, "second": task.X_test.copy()},
        categorical_columns=task.categorical_columns,
        categorical_indices=task.categorical_indices,
        seed=43,
        device="cpu",
    )
    outcome = outcomes["first"]
    assert outcome.prediction.shape == (len(task.X_test), task.n_classes)
    np.testing.assert_allclose(outcome.prediction.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(outcomes["first"].prediction, outcomes["second"].prediction)
    assert outcomes["first"].telemetry["shared_fit_id"] == outcomes["second"].telemetry["shared_fit_id"]


def test_sklearn_controls_share_fit_and_predict_probabilities(tmp_path: Path):
    task = load_task(
        {"dataset": "breast_cancer", "source": "sklearn", "problem_type": "binary"},
        seed=47,
        max_context=120,
        max_query=30,
        cache_dir=tmp_path,
    )
    for model in ("random_forest", "linear"):
        outcomes = fit_predict_many(
            model,
            task.problem_type,
            task.X_train,
            task.y_train,
            {"clean": task.X_test, "copy": task.X_test.copy()},
            categorical_columns=task.categorical_columns,
            categorical_indices=task.categorical_indices,
            seed=47,
            device="cpu",
        )
        assert outcomes["clean"].prediction.shape == (len(task.X_test), task.n_classes)
        np.testing.assert_allclose(outcomes["clean"].prediction.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(outcomes["clean"].prediction, outcomes["copy"].prediction, atol=1e-15)
        assert outcomes["clean"].telemetry["shared_fit_id"] == outcomes["copy"].telemetry["shared_fit_id"]


def test_cached_result_reload_checks_checksum(tmp_path: Path):
    result = tmp_path / "result.npz"
    metadata = tmp_path / "result.json"
    manifest = tmp_path / "MANIFEST.jsonl"
    atomic_save_npz(result, {"prediction__clean": np.arange(5)})
    metadata.write_text("{}\n")
    record = {
        "job_key": "job",
        "code_sha256": "code",
        "status": "complete",
        "run_id": "run",
        "result_path": str(result),
        "metadata_path": str(metadata),
        "result_sha256": sha256_file(result),
    }
    manifest.write_text(json.dumps(record) + "\n")
    loaded = load_completed_run(manifest, "job", "code")
    assert loaded is not None
    arrays = load_run_predictions(loaded)
    np.testing.assert_array_equal(arrays["prediction__clean"], np.arange(5))


def test_code_digest_ignores_generated_python_bytecode(tmp_path: Path):
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    first = code_digest(tmp_path)
    cache = source.parent / "__pycache__" / "module.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"generated")
    assert code_digest(tmp_path) == first
    source.write_text("VALUE = 2\n")
    assert code_digest(tmp_path) != first


def test_command_provenance_snapshot_is_reusable(monkeypatch, tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("phase: test\n")
    monkeypatch.setattr("src.analysis.runner.git_provenance", lambda repository: {"git_commit": "a", "tracked_diff_sha256": "b", "worktree_status_sha256": "c"})
    monkeypatch.setattr("src.analysis.runner._packages", lambda: {"numpy": "test"})
    snapshot = build_run_snapshot(tmp_path, tmp_path, config)
    assert snapshot["code_sha256"] == code_digest(tmp_path)
    assert snapshot["package_versions"] == {"numpy": "test"}
    assert snapshot["config_sha256"] == sha256_file(config)
