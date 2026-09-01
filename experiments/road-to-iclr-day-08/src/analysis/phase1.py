"""Dataset-level integrity checks and summaries for the frozen Phase I audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.io import iter_records, load_run_predictions, sha256_file
from src.analysis.runner import selected_jobs


TREE_MODELS = {"xgboost", "catboost", "lightgbm", "random_forest"}

# The transforms are mathematical bijections, but their float64 implementations
# include interpolation and numerical inversion.  The frozen pilot's worst
# round-trip relative error is 3.51e-7 (with zero strict-order violations), so
# 1e-6 is a conservative implementation tolerance rather than a claim of
# bitwise inversion.
NUMERICAL_INVERSE_RTOL = 1e-6
# A float64 affine map can merge adjacent one-ULP representations even though
# the real-valued map is strictly monotone.  Permit an observed order tie only
# when the round-trip error proves that the merged inputs differ by no more than
# numerical dust.  Material collisions remain fatal.
NUMERICAL_ORDER_TIE_RTOL = 1e-12


def job_tuple(
    dataset: str,
    model: str,
    transform: str,
    transform_value: float,
    seed: int,
    split_seed: int,
) -> tuple[str, str, str, float, int, int]:
    return dataset, model, transform, float(transform_value), int(seed), int(split_seed)


def expected_jobs(config: dict[str, Any]) -> list[tuple[str, str, str, float, int, int]]:
    jobs = selected_jobs(
        config,
        datasets=None,
        models=None,
        transforms=None,
        seeds=None,
        split_seeds=None,
        shard_index=0,
        num_shards=1,
    )
    return [
        job_tuple(spec["dataset"], model, transform, value, seed, split_seed)
        for spec, model, transform, value, seed, split_seed in jobs
    ]


def record_tuple(record: dict[str, Any], fallback_split_seed: int) -> tuple[str, str, str, float, int, int]:
    return job_tuple(
        record["dataset"],
        record["model"],
        record["transformation"]["name"],
        record["transformation"]["parameter"],
        record["seed"],
        record.get("split_seed", fallback_split_seed),
    )


def collect_records(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    code_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exactly one latest checksum-valid record for every expected job."""
    code_sha256 = code_sha256 or code_digest(root)
    config_sha256 = sha256_file(config_path)
    expected = expected_jobs(config)
    expected_set = set(expected)
    datasets = {item[0] for item in expected}
    models = set(config["models"])
    candidates = [
        record
        for record in (iter_records(root / "results" / "MANIFEST.jsonl") or [])
        if record.get("phase") == config["phase"]
        and record.get("config_sha256") == config_sha256
        and record.get("code_sha256") == code_sha256
        and record.get("dataset") in datasets
        and record.get("model") in models
    ]
    fallback_split_seed = int(config.get("split_seed", config.get("split_seeds", [0])[0]))
    latest_complete: dict[tuple[str, str, str, float, int, int], dict[str, Any]] = {}
    latest_status: dict[tuple[str, str, str, float, int, int], str] = {}
    for record in candidates:
        key = record_tuple(record, fallback_split_seed)
        if key not in expected_set:
            continue
        latest_status[key] = str(record["status"])
        if record["status"] == "complete":
            latest_complete[key] = record
    missing = [key for key in expected if key not in latest_complete]
    coverage = {
        "phase": config["phase"],
        "config": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "expected_jobs": len(expected),
        "complete_jobs": len(latest_complete),
        "missing_jobs": len(missing),
        "latest_failed_jobs": sum(latest_status.get(key) == "failed" for key in expected),
        "latest_unavailable_jobs": sum(latest_status.get(key) == "unavailable" for key in expected),
        "missing": [list(key) for key in missing],
    }
    records = [latest_complete[key] for key in expected if key in latest_complete]
    return records, coverage


def validate_transform_audit(audit: dict[str, Any], run_id: str) -> None:
    """Validate structural and float64 round-trip invariants for one transform."""
    if audit["metadata"]["exactness_class"] == "bijection on declared categorical support":
        if not all(
            audit.get(key, False)
            for key in ("equality_classes_preserved", "missing_mask_preserved", "exact_round_trip")
        ):
            raise ValueError(f"categorical transform audit failed: {run_id}")
        return
    if not audit["missing_mask_preserved"] or not audit["all_finite_inputs_have_finite_outputs"]:
        raise ValueError(f"transform audit failed: {run_id}")
    error = float(audit["max_rel_reconstruction_error"])
    exactness = audit["metadata"]["exactness_class"]
    if exactness != "order-preserving but lossy because of ties/finite precision":
        if audit.get("strict_order_reversals", 0):
            raise ValueError(f"order reversal failed transform audit: {run_id}")
        tie_scale = audit.get("max_order_tie_relative_input_gap", error)
        if audit["strict_order_violations"] and tie_scale > NUMERICAL_ORDER_TIE_RTOL:
            raise ValueError(f"material order collision failed transform audit: {run_id}")
    if exactness in {"exact analytic bijection", "bijection on observed support"}:
        if error > NUMERICAL_INVERSE_RTOL:
            raise ValueError(
                f"numerical inverse tolerance failed ({error:.6g} > "
                f"{NUMERICAL_INVERSE_RTOL:.6g}): {run_id}"
            )


def validate_record(record: dict[str, Any]) -> None:
    result = Path(record["result_path"])
    metadata = Path(record["metadata_path"])
    if not result.exists() or not metadata.exists():
        raise FileNotFoundError(f"run artifact is missing: {record['run_id']}")
    if sha256_file(result) != record["result_sha256"]:
        raise ValueError(f"checksum mismatch: {result}")
    arrays = load_run_predictions(record)
    required = {f"prediction__{condition}" for condition in ("clean", "matched", "context_only", "query_only")}
    required |= {"y_test", "test_row_ids", "train_row_ids"}
    if not required.issubset(arrays):
        raise ValueError(f"prediction bundle lacks {sorted(required - set(arrays))}: {result}")
    n_test = len(arrays["y_test"])
    for condition in ("clean", "matched", "context_only", "query_only"):
        prediction = arrays[f"prediction__{condition}"]
        if len(prediction) != n_test or not np.isfinite(prediction).all():
            raise ValueError(f"invalid {condition} predictions: {result}")
    validate_transform_audit(record["transform_audit"], record["run_id"])
    pairing = record.get("fit_pairing")
    if pairing != {
        "original_context": ["clean", "query_only"],
        "transformed_context": ["matched", "context_only"],
    }:
        raise ValueError(f"fitted-context pairing is absent: {record['run_id']}")
    telemetry = record["telemetry"]
    if telemetry["clean"].get("shared_fit_id") != telemetry["query_only"].get("shared_fit_id"):
        raise ValueError(f"original-context fit was not shared: {record['run_id']}")
    if telemetry["matched"].get("shared_fit_id") != telemetry["context_only"].get("shared_fit_id"):
        raise ValueError(f"transformed-context fit was not shared: {record['run_id']}")


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        validate_record(record)
        problem = record["problem_type"]
        disagreement_name = "normalized_absolute_disagreement" if problem == "regression" else "js_divergence"
        row: dict[str, Any] = {
            "run_id": record["run_id"],
            "dataset": record["dataset"],
            "model": record["model"],
            "model_family": "tree" if record["model"] in TREE_MODELS else "tfm",
            "problem_type": problem,
            "transform": record["transformation"]["name"],
            "transform_value": float(record["transformation"]["parameter"]),
            "transform_severity": float(record["transform_audit"]["metadata"]["severity"]),
            "seed": int(record["seed"]),
            "split_seed": int(record.get("split_seed", 0)),
            "wall_clock_seconds": float(record["wall_clock_seconds"]),
            "peak_gpu_memory_bytes": record["peak_gpu_memory_bytes"],
            "result_path": record["result_path"],
            "metadata_path": record["metadata_path"],
            "disagreement_metric": disagreement_name,
        }
        for condition in ("clean", "matched", "context_only", "query_only"):
            metrics = record["metrics"][condition]
            row[f"{condition}_loss"] = float(metrics["loss"])
            row[f"{condition}_loss_gap"] = float(metrics["isomorphism_gap"])
            row[f"{condition}_normalized_loss_gap"] = float(metrics["normalized_isomorphism_gap"])
            row[f"{condition}_disagreement"] = float(metrics[disagreement_name])
        rows.append(row)
    frame = pd.DataFrame(rows)
    identity = frame[frame["transform"] == "identity"].set_index(
        ["dataset", "model", "seed", "split_seed"]
    )
    if identity.index.has_duplicates:
        raise ValueError("identity baseline is not unique by dataset/model/seed")
    baseline = identity["matched_disagreement"].rename("identity_refit_disagreement")
    frame = frame.join(baseline, on=["dataset", "model", "seed", "split_seed"])
    if frame["identity_refit_disagreement"].isna().any():
        raise ValueError("a transformed job lacks its paired identity noise baseline")
    frame["matched_excess_disagreement"] = frame["matched_disagreement"] - frame["identity_refit_disagreement"]
    frame["mean_mismatch_disagreement"] = frame[["context_only_disagreement", "query_only_disagreement"]].mean(axis=1)
    return frame


def bootstrap_mean(values: np.ndarray, seed: int = 20260831, draws: int = 10_000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def dataset_level_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed = frame[frame["transform"] != "identity"].copy()
    dataset = (
        transformed.groupby(["dataset", "model", "model_family", "problem_type"], as_index=False)
        .agg(
            matched_normalized_loss_gap=("matched_normalized_loss_gap", "mean"),
            matched_disagreement=("matched_disagreement", "mean"),
            identity_refit_disagreement=("identity_refit_disagreement", "mean"),
            matched_excess_disagreement=("matched_excess_disagreement", "mean"),
            mean_mismatch_disagreement=("mean_mismatch_disagreement", "mean"),
        )
    )
    rows = []
    tolerance = 1e-4
    for (model, problem), group in dataset.groupby(["model", "problem_type"], sort=True):
        loss = bootstrap_mean(group["matched_normalized_loss_gap"].to_numpy())
        disagreement = bootstrap_mean(group["matched_excess_disagreement"].to_numpy())
        values = group["matched_normalized_loss_gap"].to_numpy()
        rows.append(
            {
                "model": model,
                "problem_type": problem,
                "datasets": len(group),
                "mean_matched_normalized_loss_gap": loss["mean"],
                "loss_gap_ci_low": loss["ci_low"],
                "loss_gap_ci_high": loss["ci_high"],
                "mean_excess_disagreement": disagreement["mean"],
                "excess_disagreement_ci_low": disagreement["ci_low"],
                "excess_disagreement_ci_high": disagreement["ci_high"],
                "loss_wins": int(np.sum(values < -tolerance)),
                "loss_ties": int(np.sum(np.abs(values) <= tolerance)),
                "loss_losses": int(np.sum(values > tolerance)),
            }
        )
    return dataset, pd.DataFrame(rows)


def transform_summary(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame[frame["transform"] != "identity"]
    dataset = (
        transformed.groupby(
            ["dataset", "model", "problem_type", "transform", "transform_value", "transform_severity"],
            as_index=False,
        )
        .agg(
            matched_normalized_loss_gap=("matched_normalized_loss_gap", "mean"),
            matched_excess_disagreement=("matched_excess_disagreement", "mean"),
            matched_disagreement=("matched_disagreement", "mean"),
            mean_mismatch_disagreement=("mean_mismatch_disagreement", "mean"),
        )
    )
    rows = []
    group_columns = ["model", "problem_type", "transform", "transform_value", "transform_severity"]
    for keys, group in dataset.groupby(group_columns, sort=True):
        loss = bootstrap_mean(group["matched_normalized_loss_gap"].to_numpy())
        disagreement = bootstrap_mean(group["matched_excess_disagreement"].to_numpy())
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "datasets": len(group),
                "mean_matched_normalized_loss_gap": loss["mean"],
                "loss_gap_ci_low": loss["ci_low"],
                "loss_gap_ci_high": loss["ci_high"],
                "mean_excess_disagreement": disagreement["mean"],
                "excess_disagreement_ci_low": disagreement["ci_low"],
                "excess_disagreement_ci_high": disagreement["ci_high"],
                "median_matched_disagreement": float(group["matched_disagreement"].median()),
                "median_mismatch_disagreement": float(group["mean_mismatch_disagreement"].median()),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
