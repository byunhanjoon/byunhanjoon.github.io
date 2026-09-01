#!/usr/bin/env python3
"""Natural-data covariance mechanism ablation for the projective checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_benchmark as bench


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "covariance_ablation"
SHARDS = OUT / "shards"
VARIANTS = ("projective_full", "projective_independent", "projective_shuffled")


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "COVARIANCE_ABLATION_PROTOCOL.md").read_bytes()).hexdigest()


def independent_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    return np.stack([np.diag(row) for row in diagonal])


def shuffled_covariance(covariance: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), bench.EPS)
    scale = np.sqrt(diagonal)
    correlation = covariance / (scale[:, :, None] * scale[:, None, :])
    result = np.empty_like(covariance)
    for group in range(bench.GROUPS):
        permutation = rng.permutation(bench.Q)
        shuffled_correlation = correlation[group][np.ix_(permutation, permutation)]
        result[group] = (
            scale[group, :, None] * shuffled_correlation * scale[group, None, :]
        )
        result[group, np.arange(bench.Q), np.arange(bench.Q)] = diagonal[group]
    return 0.5 * (result + result.transpose(0, 2, 1))


def run_cell(
    dataset: str,
    split_seed: int,
    models: list[Any],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    data = bench.prepare_data(bench.dataset_spec(dataset), split_seed)
    validation_store: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    test_store: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    audits = []

    for replicate in range(bench.REPLICATES):
        rng = np.random.default_rng(bench.stable_seed("context", dataset, split_seed, replicate))
        context_indices = rng.choice(len(data.train_y), size=bench.CONTEXT, replace=False)
        raw_y = data.train_y[context_indices].astype(np.float64)
        context_mean = float(raw_y.mean())
        context_scale = float(raw_y.std())
        if not np.isfinite(context_scale) or context_scale < bench.EPS:
            context_scale = data.metric_y_scale
        context_y = ((raw_y - context_mean) / context_scale).astype(np.float32)
        validation = bench.make_query_bundle(
            data.validation_full,
            data.validation_pca,
            data.validation_y,
            data.metric_y_mean,
            data.metric_y_scale,
            bench.stable_seed("validation", dataset, split_seed, replicate),
        )
        test = bench.make_query_bundle(
            data.test_full,
            data.test_pca,
            data.test_y,
            data.metric_y_mean,
            data.metric_y_scale,
            bench.stable_seed("test", dataset, split_seed, replicate),
        )
        audit = {
            "replicate": replicate,
            "mean_max_abs": 0.0,
            "diagonal_max_abs": 0.0,
            "symmetry_max_abs": 0.0,
            "minimum_eigenvalue": float("inf"),
            "neural_forward_max_abs": 0.0,
        }

        for partition_name, bundle, store in (
            ("validation", validation, validation_store),
            ("test", test, test_store),
        ):
            mean, full_covariance, forward_error = bench.predict_neural_projective(
                models,
                data.train_pca[context_indices],
                context_y,
                bundle.pca,
                device,
            )
            mean, full_covariance = bench.convert_joint_scale(
                mean,
                full_covariance,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            variants = {
                "projective_full": full_covariance,
                "projective_independent": independent_covariance(full_covariance),
                "projective_shuffled": shuffled_covariance(
                    full_covariance,
                    bench.stable_seed("shuffle", partition_name, dataset, split_seed, replicate),
                ),
            }
            truth = bench.targets(bundle)
            full_diagonal = np.diagonal(full_covariance, axis1=1, axis2=2)
            full_projection = bench.project_joint(mean, full_covariance, bundle.coefficients)
            for variant, covariance in variants.items():
                prediction = (
                    full_projection
                    if variant == "projective_full"
                    else bench.project_joint(mean, covariance, bundle.coefficients)
                )
                bench.add_prediction(store, variant, prediction, truth)
                audit["mean_max_abs"] = max(
                    audit["mean_max_abs"],
                    float(np.max(np.abs(mean - mean))),
                )
                audit["diagonal_max_abs"] = max(
                    audit["diagonal_max_abs"],
                    float(
                        np.max(
                            np.abs(
                                np.diagonal(covariance, axis1=1, axis2=2)
                                - full_diagonal
                            )
                        )
                    ),
                )
                symmetry, minimum_eigenvalue = bench.covariance_audit(covariance)
                audit["symmetry_max_abs"] = max(audit["symmetry_max_abs"], symmetry)
                audit["minimum_eigenvalue"] = min(
                    audit["minimum_eigenvalue"], minimum_eigenvalue
                )
            audit["neural_forward_max_abs"] = max(
                audit["neural_forward_max_abs"], forward_error
            )
        audits.append(audit)

    chosen = {}
    calibration_rows = []
    for variant in VARIANTS:
        temperature, validation_nll = bench.temperature_for(validation_store[variant])
        chosen[variant] = temperature
        calibration_rows.append(
            {
                "dataset": dataset,
                "split_seed": split_seed,
                "model": variant,
                "variance_temperature": temperature,
                "validation_nll": validation_nll,
            }
        )

    rows = []
    for variant in VARIANTS:
        temperature = chosen[variant]
        for family in bench.QUERY_FAMILIES:
            for replicate, (mean, variance, truth) in enumerate(test_store[variant][family]):
                rows.append(
                    {
                        "dataset": dataset,
                        "split_seed": split_seed,
                        "context_replicate": replicate,
                        "query_family": family,
                        "model": variant,
                        "n_queries": len(truth),
                        "variance_temperature": temperature,
                        **bench.metrics(mean, temperature * variance, truth),
                    }
                )
    audit = {
        "dataset": dataset,
        "split_seed": split_seed,
        "protocol_sha256": protocol_hash(),
        "primary_protocol_sha256": bench.protocol_hash(),
        "wall_seconds": time.perf_counter() - started,
        "episode_audits": audits,
    }
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows), audit


def analyze() -> dict[str, Any]:
    expected = [
        (spec["name"], int(split))
        for spec in bench.CONFIG["datasets"]
        for split in bench.CONFIG["split_seeds"]
    ]
    cells, calibration, audits, missing = [], [], [], []
    for dataset, split in expected:
        stem = f"{dataset}_seed{split}"
        paths = (
            SHARDS / f"{stem}_cells.csv",
            SHARDS / f"{stem}_calibration.csv",
            SHARDS / f"{stem}_audit.json",
        )
        if not all(path.exists() for path in paths):
            missing.append(stem)
            continue
        cells.append(pd.read_csv(paths[0]))
        calibration.append(pd.read_csv(paths[1]))
        audits.append(json.loads(paths[2].read_text()))
    if missing:
        raise FileNotFoundError(missing)
    cells_frame = pd.concat(cells, ignore_index=True)
    calibration_frame = pd.concat(calibration, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cells_frame.to_csv(OUT / "cells.csv", index=False)
    calibration_frame.to_csv(OUT / "calibration.csv", index=False)

    primary = pd.read_csv(bench.OUT / "cells.csv")
    primary = primary[primary.model == "neural_projective"].sort_values(
        ["dataset", "split_seed", "context_replicate", "query_family"]
    )
    full = cells_frame[cells_frame.model == "projective_full"].sort_values(
        ["dataset", "split_seed", "context_replicate", "query_family"]
    )
    reproduction_error = float(
        np.max(
            np.abs(
                primary[["rmse", "nll", "crps", "coverage90"]].to_numpy()
                - full[["rmse", "nll", "crps", "coverage90"]].to_numpy()
            )
        )
    )

    aggregate = cells_frame[cells_frame.query_family.isin(bench.AGGREGATE_FAMILIES)]
    pivot_nll = aggregate.pivot(
        index=["dataset", "split_seed", "context_replicate", "query_family"],
        columns="model",
        values="nll",
    )
    pivot_crps = aggregate.pivot(
        index=["dataset", "split_seed", "context_replicate", "query_family"],
        columns="model",
        values="crps",
    )
    independent_nll_advantage = pivot_nll.projective_independent - pivot_nll.projective_full
    independent_crps_advantage = pivot_crps.projective_independent - pivot_crps.projective_full
    shuffled_nll_advantage = pivot_nll.projective_shuffled - pivot_nll.projective_full
    by_dataset = independent_nll_advantage.groupby("dataset").mean()
    dataset_table = pd.DataFrame(
        {
            "nll_advantage_over_independent": by_dataset,
            "crps_advantage_over_independent": independent_crps_advantage.groupby("dataset").mean(),
            "nll_advantage_over_shuffled": shuffled_nll_advantage.groupby("dataset").mean(),
        }
    )
    dataset_table.to_csv(OUT / "by_dataset.csv")

    maximum_diagonal_error = max(
        episode["diagonal_max_abs"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    maximum_symmetry_error = max(
        episode["symmetry_max_abs"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    minimum_eigenvalue = min(
        episode["minimum_eigenvalue"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    expected_rows = len(expected) * bench.REPLICATES * len(bench.QUERY_FAMILIES) * len(VARIANTS)
    integrity = bool(
        len(cells_frame) == expected_rows
        and np.isfinite(cells_frame[["rmse", "nll", "crps", "coverage90"]]).all().all()
        and all(audit["protocol_sha256"] == protocol_hash() for audit in audits)
        and reproduction_error <= 1e-7
        and maximum_diagonal_error <= 1e-10
        and maximum_symmetry_error <= 1e-8
        and minimum_eigenvalue >= -1e-8
    )
    gates = {
        "mean_nll_improves_over_independent": float(independent_nll_advantage.mean()) > 0,
        "mean_crps_improves_over_independent": float(independent_crps_advantage.mean()) > 0,
        "dataset_nll_wins_over_independent_at_least_7": int((by_dataset > 0).sum()) >= 7,
        "mean_nll_improves_over_shuffled": float(shuffled_nll_advantage.mean()) > 0,
    }
    result = {
        "status": "complete_covariance_ablation",
        "protocol_sha256": protocol_hash(),
        "integrity": {
            "pass": integrity,
            "expected_rows": expected_rows,
            "observed_rows": int(len(cells_frame)),
            "primary_reproduction_max_abs": reproduction_error,
            "maximum_diagonal_error": maximum_diagonal_error,
            "maximum_symmetry_error": maximum_symmetry_error,
            "minimum_eigenvalue": minimum_eigenvalue,
        },
        "metrics": {
            "mean_nll_advantage_over_independent": float(independent_nll_advantage.mean()),
            "mean_crps_advantage_over_independent": float(independent_crps_advantage.mean()),
            "cell_nll_win_rate_over_independent": float((independent_nll_advantage > 0).mean()),
            "dataset_nll_wins_over_independent": int((by_dataset > 0).sum()),
            "mean_nll_advantage_over_shuffled": float(shuffled_nll_advantage.mean()),
        },
        "gates": gates,
        "mechanism_positive": integrity and all(gates.values()),
        "total_wall_seconds": float(sum(audit["wall_seconds"] for audit in audits)),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[spec["name"] for spec in bench.CONFIG["datasets"]])
    parser.add_argument("--split", type=int, choices=bench.CONFIG["split_seeds"])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    if args.dataset is None or args.split is None:
        parser.error("--dataset and --split are required unless --analyze is used")
    SHARDS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models, _ = bench.load_neural_models(device)
    cells, calibration, audit = run_cell(args.dataset, args.split, models, device)
    stem = f"{args.dataset}_seed{args.split}"
    cells.to_csv(SHARDS / f"{stem}_cells.csv", index=False)
    calibration.to_csv(SHARDS / f"{stem}_calibration.csv", index=False)
    (SHARDS / f"{stem}_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": stem, "rows": len(cells), "wall_seconds": audit["wall_seconds"]}))


if __name__ == "__main__":
    main()
