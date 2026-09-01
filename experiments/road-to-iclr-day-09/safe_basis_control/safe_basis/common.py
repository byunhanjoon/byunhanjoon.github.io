"""Shared data, provenance, prediction, and metric helpers.

The frozen Day-9 confirmation implementation remains authoritative for OpenML
splits, target encoding, RBF blocks, basis orbits, and headline metrics.  This
round adds only the safety/rank/embedding logic requested by its own protocol.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAY9_ROOT = ROOT.parent
TOURNAMENT_ROOT = DAY9_ROOT / "basis_controlled_tournament"
CONFIRMATION_ROOT = DAY9_ROOT / "basis_dependence_confirmation"
PROTOCOL_PATH = ROOT / "configs" / "SAFE_BASIS_PROTOCOL.json"
PANEL_PATH = ROOT / "configs" / "NEW_TAIL_PROSPECTIVE_PANEL.json"
TAIL_FINALISTS_PATH = ROOT / "configs" / "TAIL_FINALISTS.json"
TAIL_FINALISTS_SHA_PATH = ROOT / "configs" / "TAIL_FINALISTS.sha256"
PRIOR_PANEL_PATH = CONFIRMATION_ROOT / "configs" / "dataset_panel.json"
PRIOR_TAIL_PANEL_PATH = TOURNAMENT_ROOT / "configs" / "NEW_PROSPECTIVE_PANEL.json"
PRIOR_PROTOCOL_PATH = CONFIRMATION_ROOT / "configs" / "development_protocol.yaml"

if str(TOURNAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOURNAMENT_ROOT))

from tournament.common import bd  # noqa: E402


EPS = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(bd.jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    return load_json(PROTOCOL_PATH)


def data_config(protocol: dict[str, Any] | None = None, *, basis_dimension: int | None = None) -> dict[str, Any]:
    protocol = load_protocol() if protocol is None else protocol
    result = {
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
    if basis_dimension is not None:
        result["basis_dimension"] = int(basis_dimension)
    return result


def _spec_index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["key"]): dict(record) for record in load_json(path)["datasets"]}


def development_specs(protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol = load_protocol() if protocol is None else protocol
    records = {**_spec_index(PRIOR_PANEL_PATH), **_spec_index(PRIOR_TAIL_PANEL_PATH)}
    missing = set(protocol["development_datasets"]) - set(records)
    if missing:
        raise RuntimeError(f"development datasets missing from frozen prior panels: {sorted(missing)}")
    result = []
    for key in protocol["development_datasets"]:
        spec = records[key]
        spec["panel"] = "tail_development"
        result.append(spec)
    return result


def finalist_lock() -> tuple[dict[str, Any], str]:
    if not TAIL_FINALISTS_PATH.exists() or not TAIL_FINALISTS_SHA_PATH.exists():
        raise RuntimeError("prospective access refused: TAIL_FINALISTS.json and SHA256 are required")
    expected = TAIL_FINALISTS_SHA_PATH.read_text().strip().split()[0]
    actual = sha256_file(TAIL_FINALISTS_PATH)
    if expected != actual:
        raise RuntimeError("prospective access refused: TAIL_FINALISTS SHA256 mismatch")
    config = load_json(TAIL_FINALISTS_PATH)
    finalists = config.get("finalists", [])
    if config.get("status") != "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS":
        raise RuntimeError("prospective access refused: finalist status is not frozen")
    if not 1 <= len(finalists) <= 4:
        raise RuntimeError(f"prospective access refused: expected 1--4 finalists, got {len(finalists)}")
    if config.get("prospective_panel_sha256") != sha256_file(PANEL_PATH):
        raise RuntimeError("prospective access refused: prospective panel hash drift")
    return config, actual


def prospective_specs() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    finalists, finalist_hash = finalist_lock()
    panel = load_json(PANEL_PATH)
    if panel.get("status") != "LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS":
        raise RuntimeError("prospective panel status is not locked")
    specs = []
    for record in panel["datasets"]:
        specs.append(
            {
                key: value
                for key, value in record.items()
                if key in {"key", "openml_id", "openml_version", "problem_type", "panel", "cyclic_periods"}
            }
        )
    return specs, finalists, finalist_hash


def load_blocks(spec: dict[str, Any], protocol: dict[str, Any] | None = None, *, basis_dimension: int | None = None) -> Any:
    protocol = load_protocol() if protocol is None else protocol
    dataset = bd.load_dataset(spec, data_config(protocol, basis_dimension=basis_dimension))
    return bd.build_rbf_feature_matrix(dataset, data_config(protocol, basis_dimension=basis_dimension))


def orthogonal_orbit(blocks: Any, protocol: dict[str, Any] | None = None) -> list[Any]:
    protocol = load_protocol() if protocol is None else protocol
    representations = bd.build_primary_representations(blocks, int(protocol["orbit_members"]))
    selected = [rep for rep in representations if rep.is_reference or rep.variant == "orthogonal_all"]
    selected.sort(key=lambda rep: (-int(rep.is_reference), rep.member))
    expected = int(protocol["orbit_members"]) + 1
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} orbit members, got {len(selected)}")
    return selected


def prediction_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column == "prediction" or column.startswith("prediction_")]
    if columns == ["prediction"]:
        return columns
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))


def _prior_development_raw(model: str, dataset: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    root = CONFIRMATION_ROOT / "results" / "raw" / "development" / "replication" / model / dataset / f"seed_{seed}"
    if not (root / "predictions.csv.gz").exists():
        raise FileNotFoundError(root)
    frame = pd.read_csv(root / "predictions.csv.gz")
    frame = frame[(frame["is_reference"]) | (frame["variant"] == "orthogonal_all")]
    columns = prediction_columns(frame)
    output: dict[str, dict[str, np.ndarray]] = {}
    for (representation_id, split), part in frame.groupby(["representation_id", "split"], sort=False):
        part = part.sort_values("row_id")
        values = part[columns].to_numpy(dtype=float)
        if columns == ["prediction"]:
            values = values[:, 0]
        output.setdefault(str(representation_id), {})[str(split)] = values
    if len(output) != 9:
        raise RuntimeError(f"incomplete prior raw orbit at {root}: {len(output)}")
    return output


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {"validation": np.asarray(arrays["validation"]), "test": np.asarray(arrays["test"])}


def _prior_prospective_raw(model: str, dataset: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    root = TOURNAMENT_ROOT / "results" / "raw" / "prospective" / model / dataset / f"seed_{seed}" / "Raw"
    paths = sorted(root.glob("*.npz"))
    if len(paths) != 9:
        raise RuntimeError(f"incomplete prior prospective raw orbit at {root}: {len(paths)}")
    return {path.stem: _read_npz(path) for path in paths}


def load_frozen_development_predictions(
    model: str, dataset: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    """Load previously frozen Raw and m=16 GramAnchor predictions.

    The five old prospective datasets are reused as development only in this
    new round; the three original development datasets come from the earlier
    confirmation/tournament artifacts.
    """

    prior_tail = set(_spec_index(PRIOR_TAIL_PANEL_PATH))
    if dataset in prior_tail:
        raw = _prior_prospective_raw(model, dataset, seed)
        gram_root = TOURNAMENT_ROOT / "results" / "raw" / "prospective" / model / dataset / f"seed_{seed}" / "GramAnchor"
        gram_paths = sorted(gram_root.glob("*.npz"))
        if len(gram_paths) != 1:
            raise RuntimeError(f"expected one invariant Gram bundle at {gram_root}, got {len(gram_paths)}")
        gram = _read_npz(gram_paths[0])
        source = {"raw": str(gram_root.parent / "Raw"), "gram": str(gram_paths[0]), "prior_panel": "tournament_prospective"}
    else:
        raw = _prior_development_raw(model, dataset, seed)
        gram_path = TOURNAMENT_ROOT / "results" / "raw" / "stage2_representation" / model / dataset / f"seed_{seed}" / "GramAnchor.npz"
        if not gram_path.exists():
            raise FileNotFoundError(gram_path)
        gram = _read_npz(gram_path)
        source = {"raw": str(CONFIRMATION_ROOT / "results/raw/development/replication" / model / dataset / f"seed_{seed}"), "gram": str(gram_path), "prior_panel": "confirmation_development"}
    return raw, gram, source


def save_prediction_bundle(path: Path, train: np.ndarray | None, validation: np.ndarray, test: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"validation": np.asarray(validation), "test": np.asarray(test)}
    if train is not None:
        arrays["train"] = np.asarray(train)
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), metadata)


def load_prediction_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as stored:
        arrays = {key: np.asarray(stored[key]) for key in stored.files}
    return arrays, load_json(path.with_suffix(".json"))


def per_row_loss(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if problem_type == "classification":
        probability = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        return -np.log(probability[np.arange(len(y)), y.astype(int)])
    return (np.asarray(prediction, dtype=float).reshape(-1) - y.astype(float)) ** 2


def task_error(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> float:
    if problem_type == "classification":
        return float(np.mean(per_row_loss(problem_type, y, prediction)))
    return float(np.sqrt(np.mean(per_row_loss(problem_type, y, prediction))))


def trivial_prediction(problem_type: str, y_train: np.ndarray, count: int) -> np.ndarray:
    if problem_type == "classification":
        classes = int(np.max(y_train)) + 1
        counts = np.bincount(y_train.astype(int), minlength=classes).astype(float)
        prior = counts / counts.sum()
        return np.repeat(prior[None, :], int(count), axis=0)
    return np.full(int(count), float(np.mean(y_train)))


def normalized_excess_risk(
    problem_type: str,
    y: np.ndarray,
    raw_prediction: np.ndarray,
    method_prediction: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    raw = task_error(problem_type, y, raw_prediction)
    method = task_error(problem_type, y, method_prediction)
    trivial = task_error(problem_type, y, trivial_prediction(problem_type, y_train, len(y)))
    denominator = max(trivial - raw, float(epsilon))
    return {
        "raw_loss": raw,
        "method_loss": method,
        "trivial_loss": trivial,
        "absolute_task_difference": method - raw,
        "relative_task_difference": (method - raw) / max(abs(raw), EPS),
        "safety_denominator": denominator,
        "normalized_excess_risk": (method - raw) / denominator,
        "denominator_sensitive": bool(trivial - raw < 1e-6),
    }


def disagreement(problem_type: str, y: np.ndarray, reference: np.ndarray, prediction: np.ndarray) -> float:
    metrics = bd.disagreement_metrics(problem_type, y, reference, prediction)
    key = "probability_rmse" if problem_type == "classification" else "prediction_rmse_normalized"
    return float(metrics[key])


def mix_predictions(raw: np.ndarray, invariant: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * np.asarray(raw) + float(alpha) * np.asarray(invariant)


def calibration_metrics(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if problem_type != "classification":
        residual = np.asarray(prediction).reshape(-1) - np.asarray(y)
        return {"calibration_kind": "regression_residual", "mean_residual": float(np.mean(residual)), "residual_std": float(np.std(residual))}
    probability = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    confidence = probability.max(axis=1)
    correct = (probability.argmax(axis=1) == np.asarray(y)).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    one_hot = np.eye(probability.shape[1])[np.asarray(y).astype(int)]
    return {"calibration_kind": "classification", "ece_10bin": ece, "brier_multiclass": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))}


def ensure_finite(values: Iterable[np.ndarray], label: str) -> None:
    for value in values:
        if not np.isfinite(np.asarray(value)).all():
            raise RuntimeError(f"non-finite values in {label}")


def environment_metadata() -> dict[str, Any]:
    result = bd.environment_metadata()
    result.update(
        {
            "platform": platform.platform(),
            "time_utc_epoch": time.time(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prospective_panel_sha256": sha256_file(PANEL_PATH),
        }
    )
    return result
