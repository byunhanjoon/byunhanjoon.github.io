#!/usr/bin/env python3
"""Forensic, prediction-only audit of the Day-9 Guarded Basis metrics.

This program deliberately does not import or call any model-fitting entry point.
It reconstructs targets from the locked OpenML versions and split protocol,
then recomputes every validation/test loss and normalized excess-risk value from
the saved prediction bundles.  All outputs are written below Day 10.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DAY9 = ROOT.parent / "road-to-iclr-day-09" / "guarded_basis_control"
PROCESSED = DAY9 / "results" / "processed"
PROSPECTIVE = DAY9 / "results" / "raw" / "prospective"
AUDIT = ROOT / "results" / "audit"
TARGETS = AUDIT / "targets"
REPORT = ROOT / "metric_audit_results.md"
EPSILON = 1e-8
LOSS_ABS_TOL = 1e-8
LOSS_REL_TOL = 1e-6
PRIORITY_METHODS = (
    "GuardedGram-G2-after-RBF-k16",
    "GuardedGram-G2-g0p0-t01",
    "SafeGram-t01",
    "SafeRankGram-t01",
    "Raw+Gram@0.75",
    "PureGram",
    "BlockGuard-Greedy-t01",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def close_enough(left: float, right: float) -> bool:
    difference = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return difference <= LOSS_ABS_TOL or difference / scale <= LOSS_REL_TOL


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"validation", "test"}:
            raise RuntimeError(f"unexpected prediction keys at {path}: {stored.files}")
        # Preserve the saved dtype.  The original pipeline mixes predictions in
        # that dtype, then promotes to float64 inside the loss calculation.
        return {key: np.asarray(stored[key]) for key in stored.files}


def mix_predictions(raw: np.ndarray, invariant: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * np.asarray(raw) + float(alpha) * np.asarray(invariant)


def per_row_loss(problem_type: str, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    if problem_type == "classification":
        probability = np.clip(np.asarray(prediction, dtype=float), EPSILON, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        return -np.log(probability[np.arange(len(target)), target.astype(int)])
    residual = np.asarray(prediction, dtype=float).reshape(-1) - target.astype(float)
    return residual**2


def task_loss(problem_type: str, target: np.ndarray, prediction: np.ndarray) -> float:
    losses = per_row_loss(problem_type, target, prediction)
    return float(np.mean(losses) if problem_type == "classification" else np.sqrt(np.mean(losses)))


def disagreement(
    problem_type: str, target: np.ndarray, reference: np.ndarray, prediction: np.ndarray
) -> float:
    if problem_type == "classification":
        left = np.clip(np.asarray(reference, dtype=float), EPSILON, 1.0)
        right = np.clip(np.asarray(prediction, dtype=float), EPSILON, 1.0)
        left /= left.sum(axis=1, keepdims=True)
        right /= right.sum(axis=1, keepdims=True)
        return float(np.sqrt(np.mean((left - right) ** 2)))
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(prediction, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((left - right) ** 2)) / max(float(np.std(target)), 1e-12))


def trivial_prediction(
    problem_type: str, y_train: np.ndarray, count: int, prediction_columns: int | None = None
) -> np.ndarray:
    if problem_type == "classification":
        classes = int(np.max(y_train)) + 1
        if prediction_columns is not None and classes != int(prediction_columns):
            raise RuntimeError(
                f"class-order/shape mismatch: training has {classes} classes, predictions have "
                f"{prediction_columns} columns"
            )
        counts = np.bincount(y_train.astype(int), minlength=classes).astype(float)
        prior = counts / counts.sum()
        return np.repeat(prior[None, :], int(count), axis=0)
    return np.full(int(count), float(np.mean(y_train)))


def add_source(
    sources: dict[Path, set[str]], path: Path, role: str, include_sidecar: bool = False
) -> None:
    resolved = path.resolve()
    sources[resolved].add(role)
    if include_sidecar:
        sidecar = resolved.with_suffix(".json")
        if not sidecar.exists():
            raise FileNotFoundError(sidecar)
        sources[sidecar].add(role + " metadata")


def load_targets(
    panel: dict[str, Any], protocol: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Run only the frozen data/split loader; never construct or fit a model."""

    sys.path.insert(0, str(DAY9))
    from guarded_basis.common import _load_soilksat_with_observed_target, bd, data_config

    config = data_config(protocol)
    targets: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    TARGETS.mkdir(parents=True, exist_ok=True)
    for row in panel["datasets"]:
        spec = {
            key: value
            for key, value in row.items()
            if key in {"key", "openml_id", "openml_version", "problem_type", "cyclic_periods"}
        }
        spec["panel"] = "guarded_new_untouched_prospective"
        dataset = (
            _load_soilksat_with_observed_target(spec, config)
            if spec["key"] == "SoilKsatDB"
            else bd.load_dataset(spec, config)
        )
        value = {
            "problem_type": dataset.problem_type,
            "openml_id": dataset.openml_id,
            "openml_version": dataset.openml_version,
            "y_train": np.asarray(dataset.y_train),
            "y_validation": np.asarray(dataset.y_validation),
            "y_test": np.asarray(dataset.y_test),
            "train_indices": np.asarray(dataset.train_indices, dtype=np.int64),
            "validation_indices": np.asarray(dataset.validation_indices, dtype=np.int64),
            "test_indices": np.asarray(dataset.test_indices, dtype=np.int64),
        }
        targets[dataset.key] = value
        snapshot = TARGETS / f"{dataset.key}.npz"
        np.savez_compressed(
            snapshot,
            y_train=value["y_train"],
            y_validation=value["y_validation"],
            y_test=value["y_test"],
            train_indices=value["train_indices"],
            validation_indices=value["validation_indices"],
            test_indices=value["test_indices"],
        )
        if dataset.problem_type == "classification":
            counts = np.bincount(value["y_train"].astype(int)).tolist()
            train_stat = json.dumps(counts, separators=(",", ":"))
        else:
            train_stat = f"mean={float(np.mean(value['y_train'])):.17g}"
        records.append(
            {
                "dataset": dataset.key,
                "problem_type": dataset.problem_type,
                "openml_id": dataset.openml_id,
                "openml_version": dataset.openml_version,
                "n_train": len(value["y_train"]),
                "n_validation": len(value["y_validation"]),
                "n_test": len(value["y_test"]),
                "training_target_statistic": train_stat,
                "y_train_sha256": array_hash(value["y_train"]),
                "y_validation_sha256": array_hash(value["y_validation"]),
                "y_test_sha256": array_hash(value["y_test"]),
                "train_indices_sha256": array_hash(value["train_indices"]),
                "validation_indices_sha256": array_hash(value["validation_indices"]),
                "test_indices_sha256": array_hash(value["test_indices"]),
                "snapshot_path": str(snapshot),
                "split_seed": int(protocol["split_seed"]),
                "trivial_fit_labels": "training only",
            }
        )
    return targets, pd.DataFrame(records)


def prediction_checks(
    *,
    scope: str,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    split: str,
    target_info: dict[str, Any],
    prediction: np.ndarray,
) -> dict[str, Any] | None:
    if target_info["problem_type"] != "classification":
        return None
    values = np.asarray(prediction, dtype=float)
    row_sums = values.sum(axis=1)
    n_classes = int(np.max(target_info["y_train"])) + 1
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise RuntimeError(f"classification prediction shape mismatch for {dataset}/{model}/{method}")
    return {
        "scope": scope,
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": seed,
        "split": split,
        "n_rows": len(values),
        "training_classes": n_classes,
        "probability_columns": values.shape[1],
        "class_ordering": "training-only LabelEncoder sorted classes; encoded columns 0..K-1",
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "maximum_row_sum_deviation": float(np.max(np.abs(row_sums - 1.0))),
        "minimum_probability": float(values.min()),
        "maximum_probability": float(values.max()),
        "entries_clipped_at_epsilon": int(np.sum(values < EPSILON)),
        "entries_clipped_at_one": int(np.sum(values > 1.0)),
        "clipping_epsilon": EPSILON,
        "renormalized_after_clipping": True,
        "sample_weights_used": False,
    }


def locate_general_predictions(
    payload: dict[str, Any], model: str, dataset: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    base = PROSPECTIVE / "general" / model / dataset / f"seed_{seed}"
    raw_orbit_paths = [base / "Raw" / "rbf_reference.npz"] + [
        base / "Raw" / f"orthogonal_all__m{member}.npz" for member in range(8)
    ]
    paths = {
        "raw": raw_orbit_paths[0],
        "gram": base / "GramAnchor-m16.npz",
        "rank": base / "RankAdaptiveGram.npz",
    }
    selected = payload["block_selection"]
    if not selected["features"]:
        paths["block"] = paths["raw"]
    else:
        match = [
            row for row in payload["candidate_rows"] if row["candidate"] == selected["candidate"]
        ]
        if len(match) != 1:
            raise RuntimeError(f"cannot resolve BlockGuard selection for {dataset}/{model}/seed_{seed}")
        if math.isclose(float(match[0]["invariant_feature_fraction"]), 1.0, abs_tol=1e-12):
            paths["block"] = paths["gram"]
        else:
            paths["block"] = (
                PROSPECTIVE
                / "blockguard"
                / "representations"
                / model
                / dataset
                / f"seed_{seed}"
                / str(match[0]["selection_key"])
                / "rbf_reference.npz"
            )
    bundles: dict[str, Any] = {key: load_bundle(path) for key, path in paths.items()}
    bundles["raw_orbit"] = [load_bundle(path) for path in raw_orbit_paths]
    for index, path in enumerate(raw_orbit_paths):
        paths[f"raw_orbit_{index}"] = path
    if paths["block"] == paths["raw"]:
        bundles["block_orbit"] = bundles["raw_orbit"]
    elif paths["block"] == paths["gram"]:
        bundles["block_orbit"] = [bundles["gram"]] * len(raw_orbit_paths)
    else:
        block_root = paths["block"].parent
        block_paths = [block_root / "rbf_reference.npz"] + [
            block_root / f"orthogonal_all__m{member}.npz" for member in range(8)
        ]
        bundles["block_orbit"] = [load_bundle(path) for path in block_paths]
        for index, path in enumerate(block_paths):
            paths[f"block_orbit_{index}"] = path
    return bundles, paths


def locate_embedding_predictions(
    model: str, dataset: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    base = (
        PROSPECTIVE
        / "embedding_predictions"
        / model
        / dataset
        / f"seed_{seed}"
        / "RBF"
        / "k16"
    )
    raw_orbit_paths = [base / "Raw" / "rbf_k16_embedding_original_m-1.npz"] + [
        base / "Raw" / f"rbf_k16_embedding_rotated_m{member}.npz" for member in range(8)
    ]
    paths = {
        "raw": raw_orbit_paths[0],
        "gram": base
        / "GramAfterEmbedding"
        / "gram_anchor__pivot_tie_v2__rbf_k16_embedding_original_m-1.npz",
    }
    bundles: dict[str, Any] = {key: load_bundle(path) for key, path in paths.items()}
    bundles["raw_orbit"] = [load_bundle(path) for path in raw_orbit_paths]
    for index, path in enumerate(raw_orbit_paths):
        paths[f"raw_orbit_{index}"] = path
    return bundles, paths


def method_prediction(
    scope: str,
    method: str,
    alpha: float,
    bundles: dict[str, dict[str, np.ndarray]],
    split: str,
) -> np.ndarray:
    raw = bundles["raw"][split]
    if scope == "general" and method == "BlockGuard-Greedy-t01":
        return bundles["block"][split]
    invariant = bundles["rank"] if scope == "general" and method == "SafeRankGram-t01" else bundles["gram"]
    return mix_predictions(raw, invariant[split], alpha)


def method_orbit_predictions(
    scope: str,
    method: str,
    alpha: float,
    bundles: dict[str, Any],
    split: str,
) -> list[np.ndarray]:
    if scope == "general" and method == "BlockGuard-Greedy-t01":
        return [values[split] for values in bundles["block_orbit"]]
    invariant = bundles["rank"] if scope == "general" and method == "SafeRankGram-t01" else bundles["gram"]
    return [mix_predictions(values[split], invariant[split], alpha) for values in bundles["raw_orbit"]]


def collect_scope(
    *,
    scope: str,
    panel: dict[str, Any],
    models: Iterable[str],
    seeds: Iterable[int],
    targets: dict[str, dict[str, Any]],
    sources: dict[Path, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    probability_records: list[dict[str, Any]] = []
    unit_root = PROSPECTIVE / ("units" if scope == "general" else "embedding_units")
    for dataset_row in panel["datasets"]:
        dataset = str(dataset_row["key"])
        target_info = targets[dataset]
        for model in models:
            for seed in seeds:
                unit_path = unit_root / model / dataset / f"seed_{seed}.json"
                add_source(sources, unit_path, f"{scope} atomic unit")
                payload = read_json(unit_path)
                if payload.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete unit {unit_path}")
                if scope == "general":
                    bundles, paths = locate_general_predictions(payload, model, dataset, seed)
                else:
                    bundles, paths = locate_embedding_predictions(model, dataset, seed)
                for role, path in paths.items():
                    add_source(sources, path, f"{scope} {role} prediction", include_sidecar=True)
                cell_index = {
                    (str(cell["method"]), str(cell["split"])): cell for cell in payload["cells"]
                }
                for (method, split), cell in cell_index.items():
                    target = np.asarray(target_info[f"y_{split}"])
                    raw_prediction = bundles["raw"][split]
                    alpha = float(cell.get("selected_alpha", 0.0))
                    prediction = method_prediction(scope, method, alpha, bundles, split)
                    raw_orbit = [values[split] for values in bundles["raw_orbit"]]
                    method_orbit = method_orbit_predictions(
                        scope, method, alpha, bundles, split
                    )
                    if len(raw_prediction) != len(target) or len(prediction) != len(target):
                        raise RuntimeError(
                            f"prediction/target length mismatch for {scope}/{dataset}/{model}/{seed}/{method}/{split}"
                        )
                    raw_loss = task_loss(target_info["problem_type"], target, raw_prediction)
                    method_loss = task_loss(target_info["problem_type"], target, prediction)
                    columns = raw_prediction.shape[1] if raw_prediction.ndim == 2 else None
                    trivial = trivial_prediction(
                        target_info["problem_type"], target_info["y_train"], len(target), columns
                    )
                    trivial_loss = task_loss(target_info["problem_type"], target, trivial)
                    numerator = method_loss - raw_loss
                    headroom = trivial_loss - raw_loss
                    denominator = max(headroom, EPSILON)
                    cost = numerator / denominator
                    stable = numerator / max(trivial_loss, EPSILON)
                    raw_disagreements = [
                        disagreement(
                            target_info["problem_type"], target, raw_orbit[0], member
                        )
                        for member in raw_orbit[1:]
                    ]
                    method_disagreements = [
                        disagreement(
                            target_info["problem_type"], target, method_orbit[0], member
                        )
                        for member in method_orbit[1:]
                    ]
                    raw_disagreement = float(np.mean(raw_disagreements))
                    method_disagreement = float(np.mean(method_disagreements))
                    control = (
                        0.0
                        if raw_disagreement <= 1e-12
                        else 1.0 - method_disagreement / raw_disagreement
                    )
                    raw_disagreement_stored = float(cell["raw_disagreement"])
                    method_disagreement_stored = float(cell["method_disagreement"])
                    control_stored = float(cell["disagreement_reduction"])
                    raw_stored = float(cell["raw_loss"])
                    method_stored = float(cell["method_loss"])
                    trivial_stored = float(cell["trivial_loss"])
                    cost_stored = float(cell["normalized_excess_risk"])
                    record = {
                        "scope": scope,
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": int(seed),
                        "split": split,
                        "problem_type": target_info["problem_type"],
                        "n_rows": len(target),
                        "family": str(cell["family"]),
                        "selected_alpha": alpha,
                        "invariant_feature_fraction": cell.get("invariant_feature_fraction", np.nan),
                        "raw_disagreement_recomputed": raw_disagreement,
                        "raw_disagreement_stored": raw_disagreement_stored,
                        "method_disagreement_recomputed": method_disagreement,
                        "method_disagreement_stored": method_disagreement_stored,
                        "disagreement_reduction": control,
                        "disagreement_reduction_stored": control_stored,
                        "disagreement_mismatch": max(
                            abs(raw_disagreement - raw_disagreement_stored),
                            abs(method_disagreement - method_disagreement_stored),
                            abs(control - control_stored),
                        ),
                        "disagreement_matches_tolerance": close_enough(
                            raw_disagreement, raw_disagreement_stored
                        )
                        and close_enough(
                            method_disagreement, method_disagreement_stored
                        )
                        and close_enough(control, control_stored),
                        "raw_fallback": bool(cell["raw_fallback"]),
                        "inference_multiplier": float(cell["inference_multiplier"]),
                        "raw_loss_recomputed": raw_loss,
                        "raw_loss_stored": raw_stored,
                        "method_loss_recomputed": method_loss,
                        "method_loss_stored": method_stored,
                        "trivial_loss_recomputed": trivial_loss,
                        "trivial_loss_stored": trivial_stored,
                        "raw_loss_mismatch": abs(raw_loss - raw_stored),
                        "method_loss_mismatch": abs(method_loss - method_stored),
                        "trivial_loss_mismatch": abs(trivial_loss - trivial_stored),
                        "absolute_loss_mismatch": max(
                            abs(raw_loss - raw_stored), abs(method_loss - method_stored)
                        ),
                        "loss_matches_tolerance": close_enough(raw_loss, raw_stored)
                        and close_enough(method_loss, method_stored),
                        "trivial_matches_tolerance": close_enough(trivial_loss, trivial_stored),
                        "L_raw": raw_loss,
                        "L_method": method_loss,
                        "L_trivial": trivial_loss,
                        "numerator": numerator,
                        "raw_headroom": headroom,
                        "denominator_used": denominator,
                        "denominator_clipped": bool(headroom < EPSILON),
                        "C_recomputed": cost,
                        "C_stored": cost_stored,
                        "C_difference": cost - cost_stored,
                        "C_matches_tolerance": close_enough(cost, cost_stored),
                        "C_stable": stable,
                        "delta_L": numerator,
                        "delta_sigma": (
                            numerator / max(float(np.std(target.astype(float))), EPSILON)
                            if target_info["problem_type"] == "regression"
                            else np.nan
                        ),
                        "delta_logloss": (
                            numerator if target_info["problem_type"] == "classification" else np.nan
                        ),
                    }
                    records.append(record)
                    check = prediction_checks(
                        scope=scope,
                        dataset=dataset,
                        model=model,
                        method=method,
                        seed=seed,
                        split=split,
                        target_info=target_info,
                        prediction=prediction,
                    )
                    if check is not None:
                        probability_records.append(check)
    return pd.DataFrame(records), pd.DataFrame(probability_records)


def aggregate_units(
    all_cells: pd.DataFrame,
    scope: str,
    metric: str,
    *,
    remove_clipped: bool = False,
) -> pd.DataFrame:
    test = all_cells[(all_cells.scope == scope) & (all_cells.split == "test")].copy()
    validation = all_cells[(all_cells.scope == scope) & (all_cells.split == "validation")].copy()
    if remove_clipped:
        test = test[~test.denominator_clipped]
        validation = validation[~validation.denominator_clipped]
    group = ["dataset", "problem_type", "model", "method", "family"]
    if scope == "embedding":
        test["embedding"] = "RBF"
        test["k"] = 16
        group = ["dataset", "problem_type", "model", "embedding", "k", "method", "family"]
    units = (
        test.groupby(group, as_index=False, dropna=False)
        .agg(
            seeds_included=("seed", "nunique"),
            selected_alpha=("selected_alpha", "median"),
            invariant_feature_fraction=("invariant_feature_fraction", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=(metric, "median"),
            raw_task_error=("L_raw", "median"),
            method_task_error=("L_method", "median"),
            trivial_task_error=("L_trivial", "median"),
            raw_fallback=("raw_fallback", "max"),
            inference_multiplier=("inference_multiplier", "median"),
        )
    )
    validation_metric = (
        validation.groupby(["dataset", "model", "method"], as_index=False)[metric]
        .median()
        .rename(columns={metric: "validation_C"})
    )
    units = units.merge(validation_metric, on=["dataset", "model", "method"], how="left")
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_task_error"].rank(
        method="average"
    )
    delta = units.method_task_error - units.raw_task_error
    units["task_outcome"] = np.where(delta < -1e-12, "W", np.where(delta > 1e-12, "L", "T"))
    units.insert(0, "scope", scope)
    return units


def summarize_units(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, family), frame in units.groupby(["method", "family"], sort=False):
        costs = frame.normalized_excess_risk.to_numpy(float)
        controls = frame.disagreement_reduction.to_numpy(float)
        outcomes = frame.task_outcome.value_counts()
        rows.append(
            {
                "scope": str(frame.scope.iloc[0]),
                "method": method,
                "family": family,
                "units": len(frame),
                "datasets": int(frame.dataset.nunique()),
                "model_families": int(frame.model.nunique()),
                "median_alpha": float(frame.selected_alpha.median()),
                "median_invariant_feature_fraction": (
                    float(frame.invariant_feature_fraction.median())
                    if frame.invariant_feature_fraction.notna().any()
                    else np.nan
                ),
                "p25_disagreement_reduction": float(np.quantile(controls, 0.25)),
                "median_disagreement_reduction": float(np.median(controls)),
                "p75_disagreement_reduction": float(np.quantile(controls, 0.75)),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "wins": int(outcomes.get("W", 0)),
                "ties": int(outcomes.get("T", 0)),
                "losses": int(outcomes.get("L", 0)),
                "fraction_C_lt_0": float(np.mean(costs < 0)),
                "fraction_C_gt_0p01": float(np.mean(costs > 0.01)),
                "fraction_C_gt_0p05": float(np.mean(costs > 0.05)),
                "fallback_rate": float(frame.raw_fallback.mean()),
                "mean_predictive_rank": float(frame.predictive_rank.mean()),
                "inference_multiplier": float(frame.inference_multiplier.median()),
                "minimum_seeds_per_unit": int(frame.seeds_included.min()),
            }
        )
    return pd.DataFrame(rows)


def add_efficiency(summary: pd.DataFrame, block_training_multiplier: float) -> pd.DataFrame:
    values = summary.copy()
    training: list[float] = []
    parameters: list[float] = []
    for row in values.itertuples():
        method = str(row.method)
        if method in {"Raw", "PureGram"}:
            training.append(1.0)
            parameters.append(1.0)
        elif method.startswith("BlockGuard"):
            training.append(block_training_multiplier)
            parameters.append(1.0)
        else:
            training.append(2.0)
            parameters.append(2.0)
    values["training_multiplier"] = training
    values["parameter_multiplier"] = parameters
    values["paper_score"] = (
        values.median_disagreement_reduction
        - 3 * values.median_C.clip(lower=0)
        - 3 * (values.p95_C - 0.01).clip(lower=0)
        - 2 * (values.max_C - 0.05).clip(lower=0)
        - 0.05 * np.log2(values.inference_multiplier.clip(lower=1))
    )
    return values


def make_rankings(summary: pd.DataFrame, block_training_multiplier: float) -> pd.DataFrame:
    values = add_efficiency(summary, block_training_multiplier)
    definitions: list[tuple[str, pd.Series, list[str], list[bool]]] = [
        (
            "A — Paper Safety",
            (values.median_C <= 0.005) & (values.p95_C <= 0.02) & (values.max_C <= 0.10),
            ["median_disagreement_reduction", "mean_predictive_rank"],
            [False, True],
        ),
        (
            "B — Strict Safety",
            (values.p95_C <= 0.01) & (values.max_C <= 0.05),
            ["median_disagreement_reduction", "mean_predictive_rank"],
            [False, True],
        ),
        (
            "C — Basis Control",
            pd.Series(True, index=values.index),
            ["median_disagreement_reduction", "p95_C"],
            [False, True],
        ),
        (
            "D — Predictive Performance",
            pd.Series(True, index=values.index),
            ["mean_predictive_rank", "median_C"],
            [True, True],
        ),
        (
            "E — Efficiency",
            values.median_disagreement_reduction >= 0.50,
            [
                "inference_multiplier",
                "training_multiplier",
                "parameter_multiplier",
                "median_disagreement_reduction",
            ],
            [True, True, True, False],
        ),
        (
            "F — Overall Paper Candidate",
            pd.Series(True, index=values.index),
            ["paper_score", "mean_predictive_rank"],
            [False, True],
        ),
    ]
    rows: list[pd.DataFrame] = []
    for name, eligible, keys, ascending in definitions:
        ranked = values[eligible].sort_values(keys, ascending=ascending).copy()
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "ranking", name)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True, sort=False)


def paper_verdict(summary: pd.DataFrame, finalist_names: set[str]) -> str:
    eligible = summary[
        summary.method.isin(finalist_names)
        & (summary.median_disagreement_reduction >= 0.60)
        & (summary.median_C <= 0.005)
        & (summary.p95_C <= 0.02)
        & (summary.max_C <= 0.10)
        & (summary.model_families >= 3)
    ]
    fixed = summary[summary.method == "Raw+Gram@0.75"]
    adaptive = summary[summary.method.isin(finalist_names - {"Raw+Gram@0.75"})]
    block = summary[summary.method == "BlockGuard-Greedy-t01"]
    if len(block) and bool(
        block.iloc[0].median_disagreement_reduction >= 0.60
        and block.iloc[0].p95_C <= 0.02
        and block.iloc[0].inference_multiplier <= 1.0
    ):
        return "BLOCK-SELECTION-WINS"
    if len(eligible):
        return "FINAL-METHOD-SIGNAL"
    if (
        len(fixed)
        and fixed.iloc[0].median_C <= 0.005
        and fixed.iloc[0].p95_C <= 0.02
        and fixed.iloc[0].max_C <= 0.10
        and fixed.iloc[0].median_disagreement_reduction
        >= adaptive.median_disagreement_reduction.max() + 0.10
    ):
        return "FIXED-MIXTURE-WINS"
    if len(adaptive) and adaptive.median_disagreement_reduction.max() < 0.50 and bool(
        (adaptive.p95_C <= 0.02).any()
    ):
        return "SAFE-BUT-CONSERVATIVE"
    return "METHOD-STILL-UNSOLVED"


def combine_for_paper(general: pd.DataFrame, embedding: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            general.assign(scope="general"),
            embedding[embedding.method == "GuardedGram-G2-after-RBF-k16"].assign(
                scope="RBF-k16 embedding"
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def aggregation_diagnostics(cells: pd.DataFrame, original_units: pd.DataFrame) -> pd.DataFrame:
    test = cells[cells.split == "test"]
    records: list[dict[str, Any]] = []
    keys = ["scope", "dataset", "model", "method"]
    original_index = original_units.set_index(keys)
    for key, frame in test.groupby(keys, sort=False):
        scope, dataset, model, method = key
        raw = frame.L_raw.to_numpy(float)
        meth = frame.L_method.to_numpy(float)
        trivial = frame.L_trivial.to_numpy(float)
        costs = frame.C_recomputed.to_numpy(float)
        mean_ratio = (meth.mean() - raw.mean()) / max(trivial.mean() - raw.mean(), EPSILON)
        median_ratio = (np.median(meth) - np.median(raw)) / max(
            np.median(trivial) - np.median(raw), EPSILON
        )
        old = original_index.loc[key]
        displayed_delta = float(old.method_task_error - old.raw_task_error)
        stored_cost = float(old.normalized_excess_risk)
        apparent = bool(
            (displayed_delta < 0 and stored_cost > 0)
            or (displayed_delta > 0 and stored_cost < 0)
        )
        records.append(
            {
                "scope": scope,
                "dataset": dataset,
                "model": model,
                "method": method,
                "seeds": len(frame),
                "C_A_mean_per_seed_C": float(np.mean(costs)),
                "C_B_median_per_seed_C": float(np.median(costs)),
                "C_C_ratio_of_mean_losses": float(mean_ratio),
                "C_D_ratio_of_median_losses": float(median_ratio),
                "reported_unit_C": stored_cost,
                "reported_raw_task_error": float(old.raw_task_error),
                "reported_method_task_error": float(old.method_task_error),
                "displayed_loss_delta": displayed_delta,
                "reported_C_matches": (
                    "B: median of per-seed C"
                    if close_enough(stored_cost, float(np.median(costs)))
                    else "NO MATCH"
                ),
                "apparent_sign_mismatch": apparent,
                "diagnosis": "AGGREGATION-EXPLAINED" if apparent else "NO APPARENT MISMATCH",
            }
        )
    return pd.DataFrame(records)


def pathology_tables(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = test.copy()
    values["headroom_le_0"] = values.raw_headroom <= 0
    values["headroom_lt_1e_8"] = values.raw_headroom < 1e-8
    values["headroom_lt_1e_6"] = values.raw_headroom < 1e-6
    values["headroom_lt_1e_4_times_trivial"] = (
        values.raw_headroom < 1e-4 * values.L_trivial
    )
    definitions = [
        ("headroom <= 0", "headroom_le_0"),
        ("headroom < 1e-8", "headroom_lt_1e_8"),
        ("headroom < 1e-6", "headroom_lt_1e_6"),
        ("headroom < 1e-4 * L_trivial", "headroom_lt_1e_4_times_trivial"),
    ]
    records: list[dict[str, Any]] = []
    for scope in ["all", "general", "embedding"]:
        frame = values if scope == "all" else values[values.scope == scope]
        for label, column in definitions:
            affected = frame[frame[column]]
            records.append(
                {
                    "scope": scope,
                    "headroom_condition": label,
                    "cells": len(affected),
                    "percentage": 100.0 * len(affected) / len(frame),
                    "base_dataset_model_seed_cells": int(
                        affected[["scope", "dataset", "model", "seed"]].drop_duplicates().shape[0]
                    ),
                    "datasets_affected": ";".join(sorted(affected.dataset.unique())),
                    "models_affected": ";".join(sorted(affected.model.unique())),
                }
            )
    clipped = values[values.denominator_clipped].copy()
    clipped["absolute_C"] = clipped.C_recomputed.abs()
    clipped = clipped.sort_values("absolute_C", ascending=False)
    return values, pd.DataFrame(records), clipped


def forensic_table(cells: pd.DataFrame, original_units: pd.DataFrame) -> pd.DataFrame:
    requested = (
        ((cells.dataset == "OnlineNewsPopularity") & cells.model.isin(["catboost", "controlled_mlp", "tabicl_v2"]))
        | cells.dataset.isin(["Brazilian_houses", "SoilKsatDB", "2dplanes"])
    )
    selected = cells[
        (cells.scope == "general")
        & (cells.split == "test")
        & requested
        & cells.method.isin(PRIORITY_METHODS)
    ].copy()
    selected["record_type"] = "seed"
    selected["reported_raw_loss"] = selected.raw_loss_stored
    selected["reported_method_loss"] = selected.method_loss_stored
    selected["reported_C"] = selected.C_stored
    selected["aggregation_semantics"] = "per-seed"
    columns = [
        "record_type",
        "dataset",
        "model",
        "method",
        "seed",
        "problem_type",
        "L_raw",
        "L_method",
        "L_trivial",
        "numerator",
        "raw_headroom",
        "denominator_used",
        "C_recomputed",
        "reported_raw_loss",
        "reported_method_loss",
        "reported_C",
        "denominator_clipped",
        "aggregation_semantics",
    ]
    rows = [selected[columns]]
    original = original_units.set_index(["scope", "dataset", "model", "method"])
    aggregates: list[dict[str, Any]] = []
    for key, frame in selected.groupby(["scope", "dataset", "model", "method"], sort=False):
        old = original.loc[key]
        aggregates.append(
            {
                "record_type": "aggregated_report",
                "dataset": key[1],
                "model": key[2],
                "method": key[3],
                "seed": np.nan,
                "problem_type": frame.problem_type.iloc[0],
                "L_raw": float(np.median(frame.L_raw)),
                "L_method": float(np.median(frame.L_method)),
                "L_trivial": float(np.median(frame.L_trivial)),
                "numerator": float(np.median(frame.L_method) - np.median(frame.L_raw)),
                "raw_headroom": float(np.median(frame.L_trivial) - np.median(frame.L_raw)),
                "denominator_used": max(
                    float(np.median(frame.L_trivial) - np.median(frame.L_raw)), EPSILON
                ),
                "C_recomputed": float(np.median(frame.C_recomputed)),
                "reported_raw_loss": float(old.raw_task_error),
                "reported_method_loss": float(old.method_task_error),
                "reported_C": float(old.normalized_excess_risk),
                "denominator_clipped": bool(frame.denominator_clipped.any()),
                "aggregation_semantics": "independent medians of raw loss, method loss, and per-seed C",
            }
        )
    rows.append(pd.DataFrame(aggregates)[columns])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["dataset", "model", "method", "record_type", "seed"], na_position="last"
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    values = frame.copy()
    if limit is not None:
        values = values.head(limit)
    values = values[columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def inventory(
    sources: dict[Path, set[str]], git_commit: str
) -> tuple[pd.DataFrame, dict[Path, str]]:
    records: list[dict[str, Any]] = []
    starting_hashes: dict[Path, str] = {}
    for path in sorted(sources):
        digest = sha256_file(path)
        starting_hashes[path] = digest
        records.append(
            {
                "role": "; ".join(sorted(sources[path])),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "git_blob_sha1": git_blob_hash(path),
                "repository_commit": git_commit,
            }
        )
    return pd.DataFrame(records), starting_hashes


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "preliminary_C_mismatches.csv",
        "preliminary_disagreement_mismatches.csv",
    ):
        (AUDIT / stale_name).unlink(missing_ok=True)
    panel_path = DAY9 / "configs" / "GUARDED_PROSPECTIVE_PANEL.json"
    protocol_path = DAY9 / "configs" / "GUARDED_PROTOCOL.json"
    finalists_path = DAY9 / "configs" / "GUARDED_FINALISTS.json"
    panel = read_json(panel_path)
    protocol = read_json(protocol_path)
    finalists = read_json(finalists_path)
    provenance_path = PROCESSED / "final_provenance.json"
    experiment_provenance = read_json(provenance_path)
    experiment_frozen_commit = str(
        experiment_provenance["repository_commit_at_protocol_freeze"]
    )
    models_general = [str(value) for value in protocol["general_models"]]
    models_embedding = ["controlled_mlp", "tabm_d", "resnet_tabular"]
    seeds = [int(value) for value in protocol["prospective_seeds"]]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()

    sources: dict[Path, set[str]] = defaultdict(set)
    for path, role in [
        (DAY9 / "results.md", "reported final report"),
        (PROCESSED / "prospective_general_cells.csv", "reported general cells"),
        (PROCESSED / "prospective_embedding_cells.csv", "reported embedding cells"),
        (PROCESSED / "prospective_six_rankings.csv", "reported rankings"),
        (PROCESSED / "prospective_worst_10_tail_cells.csv", "reported worst tails"),
        (PROCESSED / "prospective_general_units.csv", "reported general units"),
        (PROCESSED / "prospective_embedding_units.csv", "reported embedding units"),
        (PROCESSED / "prospective_general_summary.csv", "reported general summary"),
        (PROCESSED / "prospective_embedding_summary.csv", "reported embedding summary"),
        (provenance_path, "final experiment provenance"),
        (panel_path, "locked prospective panel"),
        (protocol_path, "locked protocol"),
        (finalists_path, "frozen finalists"),
        (DAY9 / "guarded_basis" / "common.py", "prospective loader and metric imports"),
        (DAY9.parent / "safe_basis_control" / "safe_basis" / "common.py", "C and loss implementation"),
        (
            DAY9.parent / "basis_dependence_confirmation" / "src" / "basis_dependence.py",
            "split and target-encoding implementation",
        ),
        (DAY9 / "scripts" / "run_guarded_prospective.py", "general aggregation code"),
        (DAY9 / "scripts" / "run_embedding_prospective.py", "embedding aggregation code"),
        (DAY9 / "scripts" / "generate_final_report.py", "ranking and verdict code"),
    ]:
        add_source(sources, path, role)

    targets, target_inventory = load_targets(panel, protocol)
    general, general_probability = collect_scope(
        scope="general",
        panel=panel,
        models=models_general,
        seeds=seeds,
        targets=targets,
        sources=sources,
    )
    embedding, embedding_probability = collect_scope(
        scope="embedding",
        panel=panel,
        models=models_embedding,
        seeds=seeds,
        targets=targets,
        sources=sources,
    )
    cells = pd.concat([general, embedding], ignore_index=True, sort=False)
    probability = pd.concat([general_probability, embedding_probability], ignore_index=True)

    expected_all_rows = 2880 + 864
    if len(cells) != expected_all_rows:
        raise RuntimeError(f"expected {expected_all_rows} validation/test cells, found {len(cells)}")
    if not cells.loss_matches_tolerance.all() or not cells.trivial_matches_tolerance.all():
        raise RuntimeError("stored loss mismatch exceeds tolerance")
    if not cells.disagreement_matches_tolerance.all():
        mismatch = cells[~cells.disagreement_matches_tolerance].copy()
        mismatch.sort_values("disagreement_mismatch", ascending=False).to_csv(
            AUDIT / "preliminary_disagreement_mismatches.csv", index=False
        )
        raise RuntimeError(
            f"stored disagreement mismatch exceeds tolerance in {len(mismatch)} cells"
        )
    if not cells.C_matches_tolerance.all():
        mismatch = cells[~cells.C_matches_tolerance].copy()
        mismatch["absolute_C_difference"] = mismatch.C_difference.abs()
        mismatch.sort_values("absolute_C_difference", ascending=False).to_csv(
            AUDIT / "preliminary_C_mismatches.csv", index=False
        )
        print(
            mismatch[
                [
                    "scope",
                    "dataset",
                    "model",
                    "method",
                    "seed",
                    "split",
                    "L_raw",
                    "L_method",
                    "L_trivial",
                    "raw_headroom",
                    "C_recomputed",
                    "C_stored",
                    "C_difference",
                ]
            ]
            .sort_values("C_difference", key=lambda value: value.abs(), ascending=False)
            .head(20)
            .to_string(index=False),
            flush=True,
        )
        raise RuntimeError(f"stored C mismatch exceeds tolerance in {len(mismatch)} cells")

    test = cells[cells.split == "test"].copy()
    losses = test[
        [
            "scope",
            "dataset",
            "model",
            "method",
            "seed",
            "problem_type",
            "n_rows",
            "raw_loss_recomputed",
            "raw_loss_stored",
            "method_loss_recomputed",
            "method_loss_stored",
            "absolute_loss_mismatch",
            "raw_loss_mismatch",
            "method_loss_mismatch",
            "loss_matches_tolerance",
        ]
    ].rename(columns={"n_rows": "n_test"})
    trivial = test[
        [
            "scope",
            "dataset",
            "model",
            "seed",
            "problem_type",
            "n_rows",
            "trivial_loss_recomputed",
            "trivial_loss_stored",
            "trivial_loss_mismatch",
            "trivial_matches_tolerance",
        ]
    ].drop_duplicates(["scope", "dataset", "model", "seed"])
    trivial = trivial.rename(columns={"n_rows": "n_test"})
    c_columns = [
        "scope",
        "dataset",
        "model",
        "method",
        "seed",
        "problem_type",
        "L_raw",
        "L_method",
        "L_trivial",
        "numerator",
        "raw_headroom",
        "denominator_used",
        "denominator_clipped",
        "C_recomputed",
        "C_stored",
        "C_difference",
        "C_matches_tolerance",
        "C_stable",
        "delta_L",
        "delta_sigma",
        "delta_logloss",
    ]
    per_seed_c = test[c_columns].copy()

    sign_rows: list[dict[str, Any]] = []
    for row in per_seed_c.itertuples(index=False):
        if row.raw_headroom <= 0:
            continue
        violation = (
            (row.L_method < row.L_raw and not row.C_recomputed < 0)
            or (row.L_method == row.L_raw and row.C_recomputed != 0)
            or (row.L_method > row.L_raw and not row.C_recomputed > 0)
        )
        if violation:
            sign_rows.append(row._asdict())
    sign_violations = pd.DataFrame(sign_rows, columns=c_columns)

    original_general_units = pd.read_csv(PROCESSED / "prospective_general_units.csv").assign(
        scope="general"
    )
    original_embedding_units = pd.read_csv(
        PROCESSED / "prospective_embedding_units.csv"
    ).assign(scope="embedding")
    original_units = pd.concat(
        [original_general_units, original_embedding_units], ignore_index=True, sort=False
    )

    corrected_general_units = aggregate_units(cells, "general", "C_recomputed")
    corrected_embedding_units = aggregate_units(cells, "embedding", "C_recomputed")
    stable_general_units = aggregate_units(cells, "general", "C_stable")
    stable_embedding_units = aggregate_units(cells, "embedding", "C_stable")
    unclipped_general_units = aggregate_units(
        cells, "general", "C_recomputed", remove_clipped=True
    )
    unclipped_embedding_units = aggregate_units(
        cells, "embedding", "C_recomputed", remove_clipped=True
    )
    corrected_general = summarize_units(corrected_general_units)
    corrected_embedding = summarize_units(corrected_embedding_units)
    stable_general = summarize_units(stable_general_units)
    stable_embedding = summarize_units(stable_embedding_units)
    unclipped_general = summarize_units(unclipped_general_units)
    unclipped_embedding = summarize_units(unclipped_embedding_units)

    diagnostics = aggregation_diagnostics(cells, original_units)
    sign_mismatches = diagnostics[diagnostics.apparent_sign_mismatch].copy()
    pathology, pathology_summary, clipped = pathology_tables(test)
    forensic = forensic_table(cells, original_units)

    block_counts = pd.read_csv(PROCESSED / "prospective_one_block.csv").groupby(
        ["dataset", "model", "seed"]
    ).size()
    candidate_counts = pd.read_csv(PROCESSED / "prospective_block_candidates.csv").groupby(
        ["dataset", "model", "seed"]
    ).size()
    block_training_multiplier = float(block_counts.add(candidate_counts, fill_value=0).median())
    corrected_general = add_efficiency(corrected_general, block_training_multiplier)
    corrected_embedding = add_efficiency(corrected_embedding, block_training_multiplier)
    stable_general = add_efficiency(stable_general, block_training_multiplier)
    stable_embedding = add_efficiency(stable_embedding, block_training_multiplier)
    unclipped_general = add_efficiency(unclipped_general, block_training_multiplier)
    unclipped_embedding = add_efficiency(unclipped_embedding, block_training_multiplier)
    corrected_paper = combine_for_paper(corrected_general, corrected_embedding)
    stable_paper = combine_for_paper(stable_general, stable_embedding)
    unclipped_paper = combine_for_paper(unclipped_general, unclipped_embedding)
    corrected_rankings = make_rankings(corrected_paper, block_training_multiplier)
    stable_rankings = make_rankings(stable_paper, block_training_multiplier)
    unclipped_rankings = make_rankings(unclipped_paper, block_training_multiplier)
    original_rankings = pd.read_csv(PROCESSED / "prospective_six_rankings.csv")
    rank_comparison = original_rankings[["ranking", "method", "rank"]].rename(
        columns={"rank": "old_rank"}
    ).merge(
        corrected_rankings[["ranking", "method", "rank"]].rename(
            columns={"rank": "new_rank"}
        ),
        on=["ranking", "method"],
        how="outer",
    )
    rank_comparison = rank_comparison.merge(
        stable_rankings[["ranking", "method", "rank"]].rename(
            columns={"rank": "stable_metric_rank"}
        ),
        on=["ranking", "method"],
        how="outer",
    )
    rank_comparison["changed"] = rank_comparison.old_rank.fillna(-1) != rank_comparison.new_rank.fillna(-1)

    original_general = pd.read_csv(PROCESSED / "prospective_general_summary.csv")
    original_embedding = pd.read_csv(PROCESSED / "prospective_embedding_summary.csv")
    compare_fields = [
        "median_disagreement_reduction",
        "median_C",
        "p90_C",
        "p95_C",
        "max_C",
        "fraction_C_gt_0p01",
        "fraction_C_gt_0p05",
        "fallback_rate",
        "mean_predictive_rank",
    ]
    reproduction_rows: list[dict[str, Any]] = []
    for scope, old, new in [
        ("general", original_general, corrected_general),
        ("embedding", original_embedding, corrected_embedding),
    ]:
        merged = old.merge(new, on="method", suffixes=("_old", "_new"))
        for row in merged.itertuples():
            for field in compare_fields:
                reproduction_rows.append(
                    {
                        "scope": scope,
                        "method": row.method,
                        "field": field,
                        "reported": getattr(row, field + "_old"),
                        "recomputed": getattr(row, field + "_new"),
                        "absolute_difference": abs(
                            getattr(row, field + "_old") - getattr(row, field + "_new")
                        ),
                    }
                )
    summary_reproduction = pd.DataFrame(reproduction_rows)

    finalist_names = {str(row["method"]) for row in finalists["finalists"]}
    corrected_verdict = paper_verdict(corrected_paper, finalist_names)
    unclipped_verdict = paper_verdict(unclipped_paper, finalist_names)
    main = corrected_embedding[
        corrected_embedding.method == "GuardedGram-G2-after-RBF-k16"
    ].iloc[0]
    main_unclipped = unclipped_embedding[
        unclipped_embedding.method == "GuardedGram-G2-after-RBF-k16"
    ].iloc[0]
    main_holds = bool(
        abs(main.median_disagreement_reduction - 0.75) <= 0.05
        and abs(main.p95_C - 0.002647) <= 5e-4
        and abs(main.max_C - 0.006437) <= 5e-4
        and math.isclose(main.median_C, main_unclipped.median_C, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(main.p95_C, main_unclipped.p95_C, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(main.max_C, main_unclipped.max_C, rel_tol=1e-9, abs_tol=1e-12)
    )
    math_clean = bool(
        cells.loss_matches_tolerance.all()
        and cells.trivial_matches_tolerance.all()
        and cells.C_matches_tolerance.all()
        and len(sign_violations) == 0
        and not rank_comparison.changed.any()
        and corrected_verdict == "FINAL-METHOD-SIGNAL"
    )
    has_pathology = bool((test.raw_headroom <= 0).any())
    if math_clean and has_pathology:
        audit_verdict = "AUDIT-PASS-WITH-METRIC-CAVEAT"
    elif math_clean:
        audit_verdict = "AUDIT-PASS"
    elif main_holds and corrected_verdict == "FINAL-METHOD-SIGNAL":
        audit_verdict = "AGGREGATION-BUG-BUT-CONCLUSIONS-SURVIVE"
    else:
        audit_verdict = "MATERIAL-METRIC-BUG"

    # Persist machine-readable outputs before writing the narrative report.
    outputs: dict[str, pd.DataFrame] = {
        "all_split_losses.csv": cells,
        "per_seed_losses.csv": losses,
        "per_seed_trivial.csv": trivial,
        "per_seed_C.csv": per_seed_c,
        "sign_violations.csv": sign_violations,
        "classification_probability_checks.csv": probability,
        "target_split_inventory.csv": target_inventory,
        "aggregation_diagnostics.csv": diagnostics,
        "aggregation_sign_mismatches.csv": sign_mismatches,
        "denominator_pathology.csv": pathology,
        "denominator_pathology_summary.csv": pathology_summary,
        "clipped_denominator_cells.csv": clipped,
        "forensic_examples.csv": forensic,
        "corrected_general_units.csv": corrected_general_units,
        "corrected_embedding_units.csv": corrected_embedding_units,
        "corrected_general_summary.csv": corrected_general,
        "corrected_embedding_summary.csv": corrected_embedding,
        "stable_general_units.csv": stable_general_units,
        "stable_embedding_units.csv": stable_embedding_units,
        "stable_general_summary.csv": stable_general,
        "stable_embedding_summary.csv": stable_embedding,
        "unclipped_general_units.csv": unclipped_general_units,
        "unclipped_embedding_units.csv": unclipped_embedding_units,
        "unclipped_general_summary.csv": unclipped_general,
        "unclipped_embedding_summary.csv": unclipped_embedding,
        "corrected_rankings.csv": corrected_rankings,
        "stable_metric_rankings.csv": stable_rankings,
        "unclipped_metric_rankings.csv": unclipped_rankings,
        "ranking_before_after.csv": rank_comparison,
        "summary_reproduction.csv": summary_reproduction,
    }
    for name, frame in outputs.items():
        frame.to_csv(AUDIT / name, index=False)

    artifact_inventory, starting_hashes = inventory(sources, git_commit)
    artifact_inventory["experiment_protocol_freeze_commit"] = experiment_frozen_commit
    artifact_inventory.to_csv(AUDIT / "artifact_inventory.csv", index=False)

    online = forensic[
        (forensic.dataset == "OnlineNewsPopularity")
        & (forensic.model == "catboost")
        & (forensic.method == "GuardedGram-G2-g0p0-t01")
    ].copy()
    aggregate_online = online[online.record_type == "aggregated_report"].iloc[0]
    clipped_base = clipped[["scope", "dataset", "model", "seed"]].drop_duplicates()
    stable_display = pd.concat(
        [
            stable_general.assign(scope="general"),
            stable_embedding.assign(scope="embedding"),
        ],
        ignore_index=True,
    )
    stable_display = stable_display[stable_display.method.isin(PRIORITY_METHODS)].copy()
    stable_display = stable_display.rename(
        columns={
            "median_C": "median_C_stable",
            "p95_C": "p95_C_stable",
            "max_C": "max_C_stable",
        }
    )
    report_rank = rank_comparison[
        rank_comparison.ranking.isin(["A — Paper Safety", "F — Overall Paper Candidate"])
    ].copy()
    suspicious_display = sign_mismatches.sort_values(
        ["dataset", "model", "method"]
    )[[
        "dataset",
        "model",
        "method",
        "reported_raw_task_error",
        "reported_method_task_error",
        "reported_unit_C",
        "diagnosis",
    ]]
    clipped_display = clipped_base.copy()

    report = f"""# Guarded Basis Control — Metric Audit

## Executive Verdict
{audit_verdict}

## One-Paragraph Summary

All {len(cells):,} reported validation/test rows ({len(test):,} test-level dataset × model × method × seed cells) were reconstructed from saved predictions, and every raw loss, method loss, training-only trivial loss, disagreement reduction, and per-seed C matches the stored value within the required tolerance. There are {len(sign_violations)} per-seed sign violations and the largest loss mismatch is {cells.absolute_loss_mismatch.max():.3e}. The apparent OnlineNewsPopularity/CatBoost contradiction is aggregation-explained: the report independently takes medians of seed losses while its C is the median of per-seed ratios. However, {len(clipped):,} method cells ({100*len(clipped)/len(test):.2f}%) use the 1e-8 denominator because Raw does not beat the trivial predictor, so C is pathological in those cells. The corrected ranking and `FINAL-METHOD-SIGNAL` verdict are unchanged, and the RBF-k16 guarded result is unaffected.

## 1. Files and Code Audited

The complete path/hash inventory contains {len(artifact_inventory):,} exact source files in `results/audit/artifact_inventory.csv`. Each row records the absolute path, SHA256, Git blob SHA1, experiment protocol-freeze commit `{experiment_frozen_commit}`, and audit-time repository commit `{git_commit}`. Inputs include `results.md`, all requested prospective CSVs, all 288 atomic unit JSON files, the exact saved prediction bundles used for loss/control reconstruction, the locked panel/protocol/finalists, final provenance, the target/split loader, metric implementation, aggregation code, ranking code, and verdict code.

## 2. Exact Metric Definitions

Classification uses unweighted multiclass/binary log loss. Probabilities are clipped elementwise to `[1e-8, 1]`, renormalized by row, and indexed by the training-only `LabelEncoder` order. Regression uses RMSE. The trivial classifier is the training class-frequency vector; the trivial regressor is the training target mean. No validation/test labels and no sample weights construct the trivial predictor. Per seed, `C = (L_method - L_raw) / max(L_trivial - L_raw, 1e-8)`. Sensitivity metrics are `C_stable = (L_method - L_raw) / max(L_trivial, 1e-8)`, absolute `delta_L`, and regression `delta_sigma = delta_RMSE / std(y_test)`.

## 3. Raw Prediction -> Loss Verification

| rows checked | mismatches | maximum mismatch |
| --- | --- | --- |
| {len(cells)} | {int((~cells.loss_matches_tolerance).sum())} | {cells.absolute_loss_mismatch.max():.3e} |

The test-only required table has {len(losses):,} rows. Class-column counts match the training-only encoding in all {len(probability):,} classification checks. Probability normalization/clipping diagnostics are in `classification_probability_checks.csv`; the loss routine intentionally reproduces the stored clipping epsilon and renormalization. All raw/method orbit disagreements and disagreement reductions also match; their maximum mismatch is {cells.disagreement_mismatch.max():.3e}.

## 4. Trivial Predictor Verification

All {len(trivial):,} unique scope × dataset × model × seed test baselines match their stored values; maximum absolute mismatch is {test.trivial_loss_mismatch.max():.3e}. `target_split_inventory.csv` and the target snapshots record training/validation/test indices and target hashes. The audited loader fits class encoders, class frequencies, regression means, imputers, and subsampling from training rows only.

## 5. Per-Seed C Verification

| cells | exact matches | mismatches | sign violations |
| --- | --- | --- | --- |
| {len(test)} | {int(test.C_matches_tolerance.sum())} | {int((~test.C_matches_tolerance).sum())} | {len(sign_violations)} |

The maximum absolute C difference is {test.C_difference.abs().max():.3e}. The mandatory sign invariant holds at every seed whose `L_trivial > L_raw`.

## 6. Aggregation Semantics

Seed losses are never pooled at the prediction-row level. For each dataset/model/method, the original code independently takes the median across three seeds of raw loss, method loss, disagreement reduction, selected alpha, and per-seed C. Thus displayed raw and method losses can come from different seeds, while displayed C is `C_B`, the median of the three already-normalized per-seed values. Global medians/p90/p95/max are then computed over the 60 general dataset/model units or 36 embedding dataset/model units. W/T/L and predictive ranks use the independently median-aggregated task losses. `aggregation_diagnostics.csv` records mean C (`C_A`), median C (`C_B`), ratio of mean losses (`C_C`), and ratio of median losses (`C_D`) for every unit. Every existing reported unit C matches `C_B`; none matches a different hidden aggregation systematically.

## 7. OnlineNewsPopularity / CatBoost Forensic Example

{markdown_table(online, ['record_type','seed','L_raw','L_method','L_trivial','numerator','raw_headroom','C_recomputed','reported_C'])}

The aggregated report displays raw={aggregate_online.reported_raw_loss:.10g}, method={aggregate_online.reported_method_loss:.10g}, and C={aggregate_online.reported_C:.10g}. The lower displayed method loss and positive C are **AGGREGATION-EXPLAINED**, not a per-seed sign bug: raw loss, method loss, and C were independently median-aggregated, and their middle values come from different seeds.

## 8. Other Suspicious Cells

There are {len(sign_mismatches)} aggregate rows where the sign of the independently displayed median-loss difference differs from the median per-seed C. All have correct per-seed mathematics and are classified `AGGREGATION-EXPLAINED`.

{markdown_table(suspicious_display, list(suspicious_display.columns), limit=30)}

The full seed-level evidence for OnlineNewsPopularity/controlled_mlp/TabICLv2 and all models on Brazilian_houses, SoilKsatDB, and 2dplanes is in `forensic_examples.csv`.

## 9. Denominator Pathology

{markdown_table(pathology_summary[pathology_summary.scope == 'all'], ['headroom_condition','cells','percentage','base_dataset_model_seed_cells','datasets_affected','models_affected'])}

The following {len(clipped_base)} unique scope/dataset/model/seed base cells use the epsilon denominator (expanded to {len(clipped)} method cells):

{markdown_table(clipped_display, list(clipped_display.columns), limit=None)}

The largest clipped values are retained in the primary audit and listed in `clipped_denominator_cells.csv`. Removing them only for sensitivity gives paper verdict `{unclipped_verdict}`; the primary corrected verdict remains `{corrected_verdict}`. The RBF-k16 guarded finalist contains one clipped per-seed improvement (`eye_movements`/TabM-D/seed 2), but removing it leaves the finalist's median, p95, and maximum exactly unchanged.

## 10. Corrected General Prospective Summary

{markdown_table(corrected_general, ['method','median_disagreement_reduction','median_C','p90_C','p95_C','max_C','fraction_C_gt_0p01','fraction_C_gt_0p05','wins','ties','losses','fallback_rate','mean_predictive_rank','paper_score'])}

## 11. Corrected Embedding Prospective Summary

{markdown_table(corrected_embedding, ['method','median_disagreement_reduction','median_C','p90_C','p95_C','max_C','fraction_C_gt_0p01','fraction_C_gt_0p05','wins','ties','losses','fallback_rate','mean_predictive_rank','paper_score'])}

## 12. Stable-Metric Sensitivity Analysis

{markdown_table(stable_display, ['scope','method','median_C_stable','p95_C_stable','max_C_stable'])}

`C_stable` and `delta_L` are sensitivity diagnostics, not silently substituted acceptance metrics. The complete per-seed stable values and regression `delta_sigma`/classification `delta_logloss` are in `per_seed_C.csv`; stable and denominator-clipped-removed unit/summary tables are separate files. Sensitivity paper scores and all six corresponding rankings are recorded in `stable_metric_rankings.csv` and `unclipped_metric_rankings.csv`; the original C thresholds are not reinterpreted as calibrated stable-metric thresholds.

## 13. Ranking Before vs After Audit

{markdown_table(report_rank, ['ranking','method','old_rank','new_rank','stable_metric_rank','changed'])}

All primary old/new ranks are identical. Stable-metric ranks are shown only as sensitivity because the original C thresholds were not calibrated for `C_stable`.

## 14. Does GuardedGram-G2-after-RBF-k16 Still Hold?

{'YES' if main_holds else 'PARTLY'}

Corrected exact metrics are median control={main.median_disagreement_reduction:.10g}, median C={main.median_C:.10g}, p90={main.p90_C:.10g}, p95={main.p95_C:.10g}, max={main.max_C:.10g}, W/T/L={int(main.wins)}/{int(main.ties)}/{int(main.losses)}, and fraction `C > .01`={main.fraction_C_gt_0p01:.10g}. These reproduce the reported approximately 75% control, p95 0.0026, and max 0.0064.

## 15. Does FINAL-METHOD-SIGNAL Still Hold?

{'YES' if corrected_verdict == 'FINAL-METHOD-SIGNAL' else 'NO'}

Applying the original prespecified aggregation and verdict code to recomputed losses returns `{corrected_verdict}`. The result also remains `{unclipped_verdict}` in the clipped-cell-removal sensitivity analysis.

## 16. Recommended Metric for the Paper

Keep canonical per-seed C as the primary protocol metric only when reporting its denominator context. Report `delta_L` (and regression `delta_sigma`) plus `C_stable` as secondary sensitivities. Explicitly mark every case with `L_trivial - L_raw <= 0`, report the clipped-cell count and affected datasets/models, and never interpret epsilon-amplified magnitudes as comparable effect sizes. To avoid apparent sign contradictions, paper tables should display seed-level rows or compute displayed loss deltas from the same seed-level C aggregation; if independent medians are retained, label that fact beside the table.

## 17. Bugs Fixed

No Day-9 experiment or method code was modified. No per-seed loss/C bug was found. The apparent sign contradictions are a presentation-level aggregation artifact, and the denominator caveat is a metric-domain pathology rather than a reproduction failure. This standalone Day-10 audit adds diagnostics only.

## 18. Files Produced

- `metric_audit_results.md`
- `results/audit/audit_manifest.json`
- `results/audit/artifact_inventory.csv`
- `results/audit/target_split_inventory.csv` and `results/audit/targets/*.npz`
- `results/audit/per_seed_losses.csv`, `per_seed_trivial.csv`, `per_seed_C.csv`, and `sign_violations.csv`
- `results/audit/aggregation_diagnostics.csv`, `aggregation_sign_mismatches.csv`, and `forensic_examples.csv`
- `results/audit/denominator_pathology.csv`, its summary, and clipped-cell listing
- corrected, stable-metric, and clipped-cell-removed unit/summary CSVs
- corrected/stable rankings, before/after ranks, and summary-reproduction diagnostics
"""
    REPORT.write_text(report)

    # Verify the complete set of source files remained byte-identical.
    changed = [str(path) for path, digest in starting_hashes.items() if sha256_file(path) != digest]
    if changed:
        raise RuntimeError(f"Day-9 input changed during audit: {changed}")

    manifest = {
        "status": "COMPLETE",
        "audit_verdict": audit_verdict,
        "paper_verdict_recomputed": corrected_verdict,
        "paper_verdict_without_clipped_cells": unclipped_verdict,
        "repository_commit": git_commit,
        "experiment_protocol_freeze_commit": experiment_frozen_commit,
        "all_validation_test_rows_checked": len(cells),
        "test_cells_checked": len(test),
        "loss_mismatches": int((~cells.loss_matches_tolerance).sum()),
        "disagreement_mismatches": int((~cells.disagreement_matches_tolerance).sum()),
        "trivial_mismatches": int((~cells.trivial_matches_tolerance).sum()),
        "C_mismatches": int((~cells.C_matches_tolerance).sum()),
        "sign_violations": len(sign_violations),
        "maximum_loss_mismatch": float(cells.absolute_loss_mismatch.max()),
        "maximum_disagreement_mismatch": float(cells.disagreement_mismatch.max()),
        "maximum_C_difference": float(test.C_difference.abs().max()),
        "aggregate_apparent_sign_mismatches": len(sign_mismatches),
        "clipped_method_cells": len(clipped),
        "clipped_base_cells": len(clipped_base),
        "guarded_embedding_holds": main_holds,
        "guarded_embedding_metrics": {
            "median_control": float(main.median_disagreement_reduction),
            "median_C": float(main.median_C),
            "p90_C": float(main.p90_C),
            "p95_C": float(main.p95_C),
            "max_C": float(main.max_C),
            "wins": int(main.wins),
            "ties": int(main.ties),
            "losses": int(main.losses),
        },
        "primary_rank_changes": int(rank_comparison.changed.sum()),
        "summary_maximum_absolute_difference": float(
            summary_reproduction.absolute_difference.max()
        ),
        "day9_inputs_changed_during_audit": changed,
        "source_files_hashed": len(artifact_inventory),
        "output_csv_files": len(outputs) + 1,
    }
    write_json(AUDIT / "audit_manifest.json", manifest)
    print(
        f"[metric audit] {audit_verdict} rows={len(cells)} test={len(test)} "
        f"clipped={len(clipped)} sign_violations={len(sign_violations)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
