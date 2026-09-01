"""Shared protocol, data, metric, hashing, and artifact helpers.

The confirmation round remains the authoritative implementation of the frozen
data split, RBF feature blocks, orthogonal orbits, natural basis pairs, and
headline metrics.  This module loads it under a private module name so the
tournament does not silently fork those definitions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DAY9_ROOT = ROOT.parent
CONFIRMATION_ROOT = DAY9_ROOT / "basis_dependence_confirmation"
PROTOCOL_PATH = ROOT / "configs" / "TOURNAMENT_PROTOCOL.json"
PANEL_PATH = ROOT / "configs" / "NEW_PROSPECTIVE_PANEL.json"
PRIOR_PANEL_PATH = CONFIRMATION_ROOT / "configs" / "dataset_panel.json"
PRIOR_PROTOCOL_PATH = CONFIRMATION_ROOT / "configs" / "development_protocol.yaml"
PYTHON = Path("/home/byunhanjoon/miniconda3/bin/python")


def _load_confirmation_module() -> Any:
    name = "_basis_tournament_confirmation_core"
    if name in sys.modules:
        return sys.modules[name]
    path = CONFIRMATION_ROOT / "src" / "basis_dependence.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen confirmation source at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bd = _load_confirmation_module()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_protocol() -> dict[str, Any]:
    return load_json(PROTOCOL_PATH)


def load_prior_protocol() -> dict[str, Any]:
    return yaml.safe_load(PRIOR_PROTOCOL_PATH.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_hashes() -> dict[str, str]:
    return {
        "tournament_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "new_prospective_panel_sha256": sha256_file(PANEL_PATH),
        "prior_dataset_panel_sha256": sha256_file(PRIOR_PANEL_PATH),
    }


def data_config(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        key: protocol[key]
        for key in (
            "split_seed",
            "max_train_rows",
            "max_validation_rows",
            "max_test_rows",
            "minimum_continuous_unique_values",
            "basis_dimension",
        )
    }


def development_specs(protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol = load_protocol() if protocol is None else protocol
    wanted = protocol["development_datasets"]
    records = {record["key"]: record for record in load_json(PRIOR_PANEL_PATH)["datasets"]}
    missing = set(wanted) - set(records)
    if missing:
        raise RuntimeError(f"development datasets missing from frozen prior panel: {sorted(missing)}")
    result = []
    for key in wanted:
        spec = dict(records[key])
        spec["panel"] = "development"
        result.append(spec)
    return result


def prospective_specs() -> list[dict[str, Any]]:
    panel = load_json(PANEL_PATH)
    if panel["status"] != "LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS":
        raise RuntimeError("prospective panel is not locked")
    return [
        {
            key: value
            for key, value in record.items()
            if key in {"key", "openml_id", "openml_version", "problem_type", "panel", "cyclic_periods"}
        }
        for record in panel["datasets"]
    ]


def load_blocks(spec: dict[str, Any], protocol: dict[str, Any] | None = None) -> Any:
    protocol = load_protocol() if protocol is None else protocol
    dataset = bd.load_dataset(spec, data_config(protocol))
    return bd.build_rbf_feature_matrix(dataset, data_config(protocol))


def orthogonal_all_orbit(blocks: Any, protocol: dict[str, Any] | None = None) -> list[Any]:
    protocol = load_protocol() if protocol is None else protocol
    reps = bd.build_primary_representations(blocks, int(protocol["orbit_members"]))
    selected = [rep for rep in reps if rep.is_reference or rep.variant == "orthogonal_all"]
    if len(selected) != int(protocol["orbit_members"]) + 1:
        raise RuntimeError(f"expected reference plus orthogonal orbit, got {len(selected)}")
    selected.sort(key=lambda rep: (-int(rep.is_reference), rep.member))
    return selected


def task_error(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> float:
    metrics = bd.prediction_metrics(problem_type, y, prediction)
    return float(metrics["log_loss"] if problem_type == "classification" else metrics["rmse"])


def disagreement(problem_type: str, y: np.ndarray, reference: np.ndarray, prediction: np.ndarray) -> float:
    metrics = bd.disagreement_metrics(problem_type, y, reference, prediction)
    key = "probability_rmse" if problem_type == "classification" else "prediction_rmse_normalized"
    return float(metrics[key])


def relative_task_change(method_error: float, raw_error: float) -> float:
    return float((method_error - raw_error) / max(abs(raw_error), 1e-12))


def disagreement_reduction(method_disagreement: float, raw_disagreement: float) -> float:
    if raw_disagreement <= 1e-12:
        return 0.0 if method_disagreement <= 1e-12 else float("-inf")
    return float(1.0 - method_disagreement / raw_disagreement)


def max_gpu_memory_mb(device: str) -> float:
    try:
        import torch

        if str(device).startswith("cuda"):
            return float(torch.cuda.max_memory_allocated(device) / 2**20)
    except Exception:
        pass
    return 0.0


def reset_gpu_memory(device: str) -> None:
    try:
        import torch

        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def environment_metadata() -> dict[str, Any]:
    result = bd.environment_metadata()
    result.update(
        {
            "platform": platform.platform(),
            "pid": os.getpid(),
            "time_utc_epoch": time.time(),
            "protocol_hashes": protocol_hashes(),
        }
    )
    return result


def ensure_finite(arrays: Iterable[np.ndarray], label: str) -> None:
    for array in arrays:
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite values in {label}")


def jsonable(value: Any) -> Any:
    return bd.jsonable(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_prediction_bundle(
    path: Path,
    validation: np.ndarray,
    test: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, validation=np.asarray(validation), test=np.asarray(test))
    write_json(path.with_suffix(".json"), metadata)


def read_prediction_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = np.load(path)
    metadata = load_json(path.with_suffix(".json"))
    return np.asarray(arrays["validation"]), np.asarray(arrays["test"]), metadata
