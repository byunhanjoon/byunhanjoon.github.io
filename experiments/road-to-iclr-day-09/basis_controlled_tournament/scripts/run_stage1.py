#!/usr/bin/env python3
"""Run the preregistered three-dataset Stage-1 method screen."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    development_specs,
    disagreement,
    environment_metadata,
    load_blocks,
    load_protocol,
    orthogonal_all_orbit,
    protocol_hashes,
    read_prediction_bundle,
    save_prediction_bundle,
    task_error,
    write_json,
)
from tournament.models import fit_model  # noqa: E402
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


METHODS: dict[str, dict[str, Any]] = {
    "AdamW": {"track": "optimizer", "optimizer": "adamw"},
    "BlockScalarAdam": {"track": "optimizer", "optimizer": "block_scalar_adam"},
    "BlockAdam": {"track": "optimizer", "optimizer": "block_adam"},
    "MatrixAdam": {"track": "optimizer", "optimizer": "matrix_adam"},
    "PCA": {"track": "representation", "interface": "pca", "parameters": {}},
    "GramAnchor": {
        "track": "representation",
        "interface": "gram_anchor",
        "parameters": {"anchors": 16, "selection": "gram_pivot", "normalize": True},
    },
    "GramDistance": {
        "track": "representation",
        "interface": "gram_distance",
        "parameters": {"anchors": 16, "selection": "gram_pivot", "kernel": "rbf"},
    },
    "NystromGram": {
        "track": "representation",
        "interface": "nystrom_gram",
        "parameters": {
            "anchors": 16,
            "selection": "gram_pivot",
            "energy": 0.99,
            "max_rank": 8,
            "floor_fraction": 1e-6,
        },
    },
}


def bundle_path(model: str, dataset: str, seed: int, method: str, representation_id: str) -> Path:
    safe_representation = representation_id.replace("/", "_")
    return (
        ROOT
        / "results"
        / "raw"
        / "stage1"
        / model
        / dataset
        / f"seed_{seed}"
        / method
        / f"{safe_representation}.npz"
    )


def load_or_fit(
    path: Path,
    *,
    model: str,
    method: str,
    method_config: dict[str, Any],
    rep: Any,
    blocks: Any,
    seed: int,
    device: str,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hashes = protocol_hashes()
    if path.exists() and path.with_suffix(".json").exists():
        validation, test, metadata = read_prediction_bundle(path)
        if metadata.get("protocol_hashes") != hashes:
            raise RuntimeError(f"cached bundle protocol drift: {path}")
        return validation, test, metadata
    started = time.time()
    result = fit_model(
        model,
        blocks.dataset.problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
        protocol,
        optimizer_method=str(method_config.get("optimizer", "adamw")),
    )
    metadata = {
        "stage": 1,
        "dataset": blocks.dataset.key,
        "problem_type": blocks.dataset.problem_type,
        "model": model,
        "seed": seed,
        "method": method,
        "method_config": method_config,
        "representation_id": rep.representation_id,
        "variant": rep.variant,
        "member": rep.member,
        "is_reference": rep.is_reference,
        "protocol_hashes": hashes,
        "telemetry": result.telemetry,
        "completed_at_epoch": time.time(),
        "wall_seconds": time.time() - started,
    }
    save_prediction_bundle(path, result.validation, result.test, metadata)
    return result.validation, result.test, metadata


def run_dataset(spec: dict[str, Any], model: str, seed: int, device: str) -> None:
    protocol = load_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    max_reconstruction = 0.0
    for rep in orbit:
        for audit in rep.metadata.get("equivalence", {}).values():
            max_reconstruction = max(max_reconstruction, float(audit["reconstruction_error"]))
    if max_reconstruction >= 1e-6:
        raise RuntimeError(f"equivalent-pair reconstruction failed: {max_reconstruction}")

    rows: list[dict[str, Any]] = []
    coordinate_audits: list[dict[str, Any]] = []
    for method, method_config in METHODS.items():
        mapped = orbit
        if method_config["track"] == "representation":
            mapped = [
                build_interface(
                    rep,
                    str(method_config["interface"]),
                    blocks.dataset.key,
                    **method_config["parameters"],
                )
                for rep in orbit
            ]
            for audit in audit_orbit_coordinates(mapped[0], mapped[1:]):
                coordinate_audits.append(
                    {
                        "dataset": blocks.dataset.key,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        **audit,
                    }
                )
        predictions: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for rep in mapped:
            path = bundle_path(model, blocks.dataset.key, seed, method, rep.representation_id)
            predictions[rep.representation_id] = load_or_fit(
                path,
                model=model,
                method=method,
                method_config=method_config,
                rep=rep,
                blocks=blocks,
                seed=seed,
                device=device,
                protocol=protocol,
            )
        reference = mapped[0]
        ref_validation, ref_test, _ = predictions[reference.representation_id]
        for rep in mapped:
            validation, test, metadata = predictions[rep.representation_id]
            for split, target, ref_prediction, prediction in (
                ("validation", blocks.dataset.y_validation, ref_validation, validation),
                ("test", blocks.dataset.y_test, ref_test, test),
            ):
                rows.append(
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "track": method_config["track"],
                        "representation_id": rep.representation_id,
                        "member": rep.member,
                        "is_reference": rep.is_reference,
                        "split": split,
                        "task_error": task_error(blocks.dataset.problem_type, target, prediction),
                        "disagreement": disagreement(
                            blocks.dataset.problem_type, target, ref_prediction, prediction
                        ),
                        "fit_seconds": float(metadata["telemetry"]["fit_seconds"]),
                        "gpu_peak_memory_mb": float(metadata["telemetry"].get("gpu_peak_memory_mb", 0.0)),
                        "max_input_reconstruction_error": max_reconstruction,
                    }
                )
        print(f"[complete] {model} {blocks.dataset.key} seed={seed} {method}", flush=True)

    processed = ROOT / "results" / "processed" / "stage1_cells"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / f"{model}__{blocks.dataset.key}__seed_{seed}.csv", index=False)
    pd.DataFrame(coordinate_audits).to_csv(
        processed / f"{model}__{blocks.dataset.key}__seed_{seed}__coordinate_audit.csv", index=False
    )
    write_json(
        processed / f"{model}__{blocks.dataset.key}__seed_{seed}__metadata.json",
        {
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": seed,
            "max_input_reconstruction_error": max_reconstruction,
            "environment": environment_metadata(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["controlled_mlp", "tabm_d"], required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    allowed = set(protocol["stage1_datasets"])
    if args.seed not in protocol["stage1_seeds"]:
        raise RuntimeError(f"seed {args.seed} is not frozen for Stage 1")
    specs = [spec for spec in development_specs(protocol) if spec["key"] in allowed]
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
        if not specs:
            raise RuntimeError(f"unknown or non-Stage-1 dataset {args.dataset}")
    for spec in specs:
        run_dataset(spec, args.model, args.seed, args.device)


if __name__ == "__main__":
    main()
