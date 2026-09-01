#!/usr/bin/env python3
"""Run surviving invariant interfaces on all five model families."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT.parent / "basis_dependence_confirmation"
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    bd,
    development_specs,
    disagreement,
    environment_metadata,
    load_blocks,
    load_json,
    load_prior_protocol,
    load_protocol,
    orthogonal_all_orbit,
    protocol_hashes,
    read_prediction_bundle,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


METHODS: dict[str, dict[str, Any]] = {
    "PCA": {"interface": "pca", "parameters": {}},
    "GramAnchor": {
        "interface": "gram_anchor",
        "parameters": {"anchors": 16, "selection": "gram_pivot", "normalize": True},
    },
    # Predeclared anchor-count rescue, promoted only after the full controlled-
    # MLP anchor ablation showed lower task cost than m=16.
    "GramAnchor-m8": {
        "interface": "gram_anchor",
        "parameters": {"anchors": 8, "selection": "gram_pivot", "normalize": True},
    },
    "GramDistance": {
        "interface": "gram_distance",
        "parameters": {"anchors": 16, "selection": "gram_pivot", "kernel": "rbf"},
    },
    "NystromGram": {
        "interface": "nystrom_gram",
        "parameters": {
            "anchors": 16,
            "selection": "gram_pivot",
            "energy": 0.99,
            "max_rank": 8,
            "floor_fraction": 1e-6,
        },
    },
    "HybridSpectral-t0.01": {
        "interface": "hybrid_spectral",
        "parameters": {"tau": 0.01, "anchors": 16},
    },
    "HybridSpectral-t0.05": {
        "interface": "hybrid_spectral",
        "parameters": {"tau": 0.05, "anchors": 16},
    },
    "HybridSpectral-t0.10": {
        "interface": "hybrid_spectral",
        "parameters": {"tau": 0.10, "anchors": 16},
    },
}


def prediction_values(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame.columns if column == "prediction" or column.startswith("prediction_")]
    if columns == ["prediction"]:
        return frame["prediction"].to_numpy(dtype=float)
    columns.sort(key=lambda value: int(value.rsplit("_", 1)[1]))
    return frame[columns].to_numpy(dtype=float)


def load_prior_predictions(model: str, dataset: str, seed: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    bundle = (
        CONFIRMATION
        / "results"
        / "raw"
        / "development"
        / "replication"
        / model
        / dataset
        / f"seed_{seed}"
    )
    metadata = bundle / "metadata.json"
    if not metadata.exists():
        raise RuntimeError(f"missing authoritative prior baseline bundle {bundle}")
    frame = pd.read_csv(bundle / "predictions.csv.gz")
    metrics = pd.read_csv(bundle / "metrics.csv")
    wanted = frame[(frame["is_reference"]) | (frame["variant"] == "orthogonal_all")]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for (representation_id, split), part in wanted.groupby(["representation_id", "split"], sort=False):
        part = part.sort_values("row_id")
        predictions.setdefault(str(representation_id), {})[str(split)] = prediction_values(part)
    fit_seconds = {
        str(row.representation_id): float(row.fit_seconds)
        for row in metrics[(metrics["split"] == "test") & ((metrics["is_reference"]) | (metrics["variant"] == "orthogonal_all"))].itertuples()
    }
    if len(predictions) != 9:
        raise RuntimeError(f"prior raw orbit incomplete at {bundle}: {len(predictions)}")
    return predictions, fit_seconds


def interface_path(model: str, dataset: str, seed: int, method: str) -> Path:
    return (
        ROOT
        / "results"
        / "raw"
        / "stage2_representation"
        / model
        / dataset
        / f"seed_{seed}"
        / f"{method}.npz"
    )


def add_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    problem_type: str,
    model: str,
    seed: int,
    method: str,
    track: str,
    orbit: list[Any],
    predictions: dict[str, dict[str, np.ndarray]],
    y_validation: np.ndarray,
    y_test: np.ndarray,
    fit_seconds: dict[str, float],
    reused: bool,
) -> None:
    reference_id = orbit[0].representation_id
    for rep in orbit:
        for split, target in (("validation", y_validation), ("test", y_test)):
            prediction = predictions[rep.representation_id][split]
            reference_prediction = predictions[reference_id][split]
            rows.append(
                {
                    "dataset": dataset,
                    "problem_type": problem_type,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "track": track,
                    "representation_id": rep.representation_id,
                    "member": rep.member,
                    "is_reference": rep.is_reference,
                    "split": split,
                    "task_error": task_error(problem_type, target, prediction),
                    "disagreement": disagreement(
                        problem_type, target, reference_prediction, prediction
                    ),
                    "fit_seconds": float(fit_seconds.get(rep.representation_id, 0.0)),
                    "prediction_reused_due_to_coordinate_identity": reused and not rep.is_reference,
                }
            )


def run_unit(spec: dict[str, Any], model: str, seed: int, device: str) -> None:
    protocol = load_protocol()
    prior_protocol = load_prior_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    raw, raw_fit_seconds = load_prior_predictions(model, blocks.dataset.key, seed)
    rows: list[dict[str, Any]] = []
    add_rows(
        rows,
        dataset=blocks.dataset.key,
        problem_type=blocks.dataset.problem_type,
        model=model,
        seed=seed,
        method="Raw",
        track="representation",
        orbit=orbit,
        predictions=raw,
        y_validation=blocks.dataset.y_validation,
        y_test=blocks.dataset.y_test,
        fit_seconds=raw_fit_seconds,
        reused=False,
    )
    coordinate_records = []
    invariant_predictions: dict[str, dict[str, np.ndarray]] = {}
    for method, definition in METHODS.items():
        mapped = [
            build_interface(
                rep,
                definition["interface"],
                blocks.dataset.key,
                **definition["parameters"],
            )
            for rep in orbit
        ]
        audits = audit_orbit_coordinates(mapped[0], mapped[1:])
        maximum_coordinate_error = max(
            max(record["train_relative_error"], record["validation_relative_error"], record["test_relative_error"])
            for record in audits
        )
        if maximum_coordinate_error >= 1e-8:
            raise RuntimeError(
                f"{blocks.dataset.key}/{method} is not coordinate-invariant: {maximum_coordinate_error}"
            )
        coordinate_records.extend(
            {
                "dataset": blocks.dataset.key,
                "model": model,
                "seed": seed,
                "method": method,
                **record,
            }
            for record in audits
        )
        path = interface_path(model, blocks.dataset.key, seed, method)
        hashes = {**protocol_hashes(), "prior_protocol_sha256": sha256_file(CONFIRMATION / "configs" / "development_protocol.yaml")}
        if path.exists() and path.with_suffix(".json").exists():
            validation, test, metadata = read_prediction_bundle(path)
            if metadata["protocol_hashes"] != hashes:
                raise RuntimeError(f"protocol drift in {path}")
        else:
            started = time.time()
            validation, test, telemetry = bd.fit_predict(
                model,
                blocks.dataset.problem_type,
                mapped[0],
                blocks.dataset.y_train,
                blocks.dataset.y_validation,
                seed,
                device,
                prior_protocol,
            )
            metadata = {
                "stage": 2,
                "track": "representation",
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": seed,
                "method": method,
                "definition": definition,
                "reference_interface_id": mapped[0].representation_id,
                "maximum_orbit_coordinate_error": maximum_coordinate_error,
                "protocol_hashes": hashes,
                "telemetry": telemetry,
                "wall_seconds": time.time() - started,
            }
            save_prediction_bundle(path, validation, test, metadata)
        # Coordinate identity makes each transformed interface the same model
        # input. Store that fact explicitly instead of spending eight fits on
        # bytewise-equivalent designs.
        predictions = {
            rep.representation_id: {"validation": validation, "test": test} for rep in mapped
        }
        invariant_predictions[method] = {"validation": validation, "test": test}
        fit_map = {mapped[0].representation_id: float(metadata["telemetry"]["fit_seconds"])}
        add_rows(
            rows,
            dataset=blocks.dataset.key,
            problem_type=blocks.dataset.problem_type,
            model=model,
            seed=seed,
            method=method,
            track="representation",
            orbit=mapped,
            predictions=predictions,
            y_validation=blocks.dataset.y_validation,
            y_test=blocks.dataset.y_test,
            fit_seconds=fit_map,
            reused=True,
        )
        print(f"[stage2 interface] {model} {blocks.dataset.key} seed={seed} {method}", flush=True)

    # H1 prediction mixtures require no refit and are evaluated over every raw
    # orbit member. Alpha is selected later from development validation only.
    for method, invariant in invariant_predictions.items():
        for alpha in protocol["hybrids"]["prediction_mixture_alphas"]:
            label = f"Raw+{method}@{alpha:g}"
            mixed: dict[str, dict[str, np.ndarray]] = {}
            for rep in orbit:
                mixed[rep.representation_id] = {
                    split: (1.0 - float(alpha)) * raw[rep.representation_id][split]
                    + float(alpha) * invariant[split]
                    for split in ("validation", "test")
                }
            add_rows(
                rows,
                dataset=blocks.dataset.key,
                problem_type=blocks.dataset.problem_type,
                model=model,
                seed=seed,
                method=label,
                track="hybrid_prediction_mixture",
                orbit=orbit,
                predictions=mixed,
                y_validation=blocks.dataset.y_validation,
                y_test=blocks.dataset.y_test,
                fit_seconds={},
                reused=True,
            )
    destination = ROOT / "results" / "processed" / "stage2_representation_cells"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{model}__{blocks.dataset.key}__seed_{seed}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(coordinate_records).to_csv(
        destination / f"{model}__{blocks.dataset.key}__seed_{seed}__coordinate_audit.csv",
        index=False,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": seed,
            "raw_source": "immutable prior confirmation replication bundle",
            "environment": environment_metadata(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["controlled_mlp", "tabm_d", "tabicl_v2", "tabpfn_2_6", "catboost"],
        required=True,
    )
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    survivors = load_json(ROOT / "configs" / "STAGE1_SURVIVORS.json")
    expected = {"PCA", "GramAnchor", "GramDistance", "NystromGram"}
    if not expected.issubset(set(survivors["survivors"])):
        raise RuntimeError("Stage-2 method set is not supported by frozen Stage-1 survivors")
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
        if not specs:
            raise RuntimeError(f"unknown dataset {args.dataset}")
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.device)


if __name__ == "__main__":
    main()
