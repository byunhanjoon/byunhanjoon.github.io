#!/usr/bin/env python3
"""Materialize and audit every task in a frozen audit panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.io import atomic_write_json, sha256_file  # noqa: E402
from src.analysis.runner import load_config  # noqa: E402
from src.data import load_task  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "audit" / "pilot.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    output = args.output or ROOT / "results" / "panel" / f"pilot_panel__{config_sha256[:16]}.json"
    output = output.resolve()
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("config_sha256") != config_sha256:
            raise FileExistsError(f"refusing to overwrite panel audit from another config: {output}")
        print(output)
        return

    tasks = []
    for spec in config["datasets"]:
        task = load_task(
            spec,
            seed=int(config["split_seed"]),
            max_context=config.get("max_context"),
            max_query=config.get("max_query"),
            cache_dir=Path(config["openml_cache"]).expanduser(),
        )
        audit = task.audit()
        if not all(
            audit[key]
            for key in ("train_validation_disjoint", "train_test_disjoint", "validation_test_disjoint")
        ):
            raise AssertionError(f"split leakage in {task.dataset}: {audit}")
        if not task.numeric_columns:
            raise ValueError(f"Phase I numerical audit task has no numerical features: {task.dataset}")
        tasks.append(
            {
                "dataset": task.dataset,
                "task_id": task.task_id,
                "problem_type": task.problem_type,
                "n_classes": task.n_classes,
                "split_id": task.split_id,
                "audit": audit,
                "numeric_columns": task.numeric_columns,
                "categorical_columns": task.categorical_columns,
                "descriptors": task.descriptors,
            }
        )
        print(f"validated {task.dataset}: split={task.split_id}", flush=True)
    payload = {
        "protocol_version": config["protocol_version"],
        "phase": config["phase"],
        "config": str(config_path),
        "config_sha256": config_sha256,
        "split_seed": int(config["split_seed"]),
        "model_warp_seeds": [int(seed) for seed in config["seeds"]],
        "max_context": config.get("max_context"),
        "max_query": config.get("max_query"),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    atomic_write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
