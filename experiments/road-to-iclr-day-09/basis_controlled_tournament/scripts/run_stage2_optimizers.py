#!/usr/bin/env python3
"""Run optimizer rescues on the full six-dataset development panel."""

from __future__ import annotations

import argparse
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


METHODS: dict[str, dict[str, Any]] = {
    "AdamW": {"optimizer": "adamw", "initialization": "default", "overrides": {}},
    "BlockAdam": {"optimizer": "block_adam", "initialization": "default", "overrides": {}},
    "MatrixAdam": {"optimizer": "matrix_adam", "initialization": "default", "overrides": {}},
    "BlockAdam+DataInit": {
        "optimizer": "block_adam",
        "initialization": "data_equivariant",
        "overrides": {},
    },
    "MatrixAdam+DataInit": {
        "optimizer": "matrix_adam",
        "initialization": "data_equivariant",
        "overrides": {},
    },
    "SoftBlockAdam-a0.1+DataInit": {
        "optimizer": "soft_block_adam",
        "initialization": "data_equivariant",
        "overrides": {"alpha": 0.1},
    },
    "SoftBlockAdam-a0.25+DataInit": {
        "optimizer": "soft_block_adam",
        "initialization": "data_equivariant",
        "overrides": {"alpha": 0.25},
    },
}


def path_for(model: str, dataset: str, seed: int, method: str, representation_id: str) -> Path:
    return (
        ROOT
        / "results"
        / "raw"
        / "stage2_optimizer"
        / model
        / dataset
        / f"seed_{seed}"
        / method
        / f"{representation_id}.npz"
    )


def run_unit(spec: dict[str, Any], model: str, seed: int, device: str) -> None:
    protocol = load_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    hashes = protocol_hashes()
    rows = []
    for method, definition in METHODS.items():
        predictions = {}
        for rep in orbit:
            path = path_for(model, blocks.dataset.key, seed, method, rep.representation_id)
            if path.exists() and path.with_suffix(".json").exists():
                validation, test, metadata = read_prediction_bundle(path)
                if metadata["protocol_hashes"] != hashes:
                    raise RuntimeError(f"protocol drift in {path}")
            else:
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
                    optimizer_method=definition["optimizer"],
                    optimizer_overrides=definition["overrides"],
                    initialization=definition["initialization"],
                )
                validation, test = result.validation, result.test
                metadata = {
                    "stage": 2,
                    "track": "optimizer",
                    "dataset": blocks.dataset.key,
                    "problem_type": blocks.dataset.problem_type,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "definition": definition,
                    "representation_id": rep.representation_id,
                    "member": rep.member,
                    "is_reference": rep.is_reference,
                    "protocol_hashes": hashes,
                    "telemetry": result.telemetry,
                    "wall_seconds": time.time() - started,
                }
                save_prediction_bundle(path, validation, test, metadata)
            predictions[rep.representation_id] = (validation, test, metadata)
        reference_id = orbit[0].representation_id
        ref_validation, ref_test, _ = predictions[reference_id]
        for rep in orbit:
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
                        "track": "optimizer",
                        "optimizer": definition["optimizer"],
                        "initialization": definition["initialization"],
                        "representation_id": rep.representation_id,
                        "member": rep.member,
                        "is_reference": rep.is_reference,
                        "split": split,
                        "task_error": task_error(blocks.dataset.problem_type, target, prediction),
                        "disagreement": disagreement(
                            blocks.dataset.problem_type, target, ref_prediction, prediction
                        ),
                        "fit_seconds": float(metadata["telemetry"]["fit_seconds"]),
                        "gpu_peak_memory_mb": float(
                            metadata["telemetry"].get("gpu_peak_memory_mb", 0.0)
                        ),
                    }
                )
        print(f"[stage2 optimizer] {model} {blocks.dataset.key} seed={seed} {method}", flush=True)
    destination = ROOT / "results" / "processed" / "stage2_optimizer_cells"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{model}__{blocks.dataset.key}__seed_{seed}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(
        output.with_suffix(".json"),
        {
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": seed,
            "environment": environment_metadata(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["controlled_mlp", "tabm_d"], required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
        if not specs:
            raise RuntimeError(f"unknown dataset {args.dataset}")
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    if not set(seeds).issubset(protocol["model_seeds"]):
        raise RuntimeError("seed outside frozen protocol")
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.device)


if __name__ == "__main__":
    main()
