#!/usr/bin/env python3
"""Equal three-learning-rate HPO for raw AdamW and optimizer rescues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    development_specs,
    disagreement,
    load_blocks,
    load_protocol,
    orthogonal_all_orbit,
    read_prediction_bundle,
    save_prediction_bundle,
    task_error,
)
from tournament.models import fit_model  # noqa: E402


METHODS: dict[str, dict[str, Any]] = {
    "AdamW": {"optimizer": "adamw", "initialization": "default", "overrides": {}},
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
}


def run_unit(spec, model: str, seed: int, device: str) -> None:
    protocol = load_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    base_lr = float(protocol["models"][model]["learning_rate"])
    multipliers = protocol["optimizer_equal_hpo"]["learning_rate_multipliers"]
    trial_rows = []
    selected_rows = []
    for method, definition in METHODS.items():
        per_rep: dict[str, list[dict[str, Any]]] = {}
        for multiplier in multipliers:
            lr = base_lr * float(multiplier)
            for rep in orbit:
                path = (
                    ROOT
                    / "results"
                    / "raw"
                    / "equal_hpo"
                    / model
                    / blocks.dataset.key
                    / f"seed_{seed}"
                    / method
                    / f"lr_multiplier_{multiplier:g}"
                    / f"{rep.representation_id}.npz"
                )
                if path.exists():
                    validation, test, metadata = read_prediction_bundle(path)
                else:
                    overrides = {**definition["overrides"], "learning_rate": lr}
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
                        optimizer_overrides=overrides,
                        initialization=definition["initialization"],
                    )
                    validation, test = result.validation, result.test
                    metadata = {
                        "method": method,
                        "definition": definition,
                        "learning_rate": lr,
                        "learning_rate_multiplier": multiplier,
                        "telemetry": result.telemetry,
                    }
                    save_prediction_bundle(path, validation, test, metadata)
                record = {
                    "representation_id": rep.representation_id,
                    "member": rep.member,
                    "is_reference": rep.is_reference,
                    "multiplier": float(multiplier),
                    "learning_rate": lr,
                    "validation": validation,
                    "test": test,
                    "validation_task_error": task_error(
                        blocks.dataset.problem_type, blocks.dataset.y_validation, validation
                    ),
                    "test_task_error": task_error(
                        blocks.dataset.problem_type, blocks.dataset.y_test, test
                    ),
                    "fit_seconds": float(metadata["telemetry"]["fit_seconds"]),
                }
                per_rep.setdefault(rep.representation_id, []).append(record)
                trial_rows.append(
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        **{key: value for key, value in record.items() if key not in {"validation", "test"}},
                    }
                )
        chosen = {
            representation_id: min(records, key=lambda row: (row["validation_task_error"], row["multiplier"]))
            for representation_id, records in per_rep.items()
        }
        reference_id = orbit[0].representation_id
        for rep in orbit:
            record = chosen[rep.representation_id]
            reference = chosen[reference_id]
            for split, target in (
                ("validation", blocks.dataset.y_validation),
                ("test", blocks.dataset.y_test),
            ):
                selected_rows.append(
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "track": "optimizer_equal_hpo",
                        "representation_id": rep.representation_id,
                        "member": rep.member,
                        "is_reference": rep.is_reference,
                        "split": split,
                        "selected_lr_multiplier": record["multiplier"],
                        "selected_learning_rate": record["learning_rate"],
                        "task_error": record[f"{split}_task_error"],
                        "disagreement": disagreement(
                            blocks.dataset.problem_type,
                            target,
                            reference[split],
                            record[split],
                        ),
                        "fit_seconds": record["fit_seconds"],
                    }
                )
        print(f"[equal HPO] {model} {blocks.dataset.key} seed={seed} {method}", flush=True)
    destination = ROOT / "results" / "processed" / "equal_hpo"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{blocks.dataset.key}__seed_{seed}"
    pd.DataFrame(trial_rows).to_csv(destination / f"{stem}__trials.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(destination / f"{stem}__selected.csv", index=False)


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
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.device)


if __name__ == "__main__":
    main()
