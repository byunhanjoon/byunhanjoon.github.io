#!/usr/bin/env python3
"""Analyze fresh context-rescaled classification and regression confirmations."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, prediction_loss, weighted_prediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hierarchical(values: dict[str, np.ndarray], seed: int, draws: int) -> dict[str, object]:
    names = sorted(values)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            rng.choice(values[str(name)], len(values[str(name)]), replace=True).mean()
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    per_dataset = {name: float(values[name].mean()) for name in names}
    return {
        "gain": float(np.mean(list(per_dataset.values()))),
        "ci_low": float(low), "ci_high": float(high),
        "positive_datasets": int(sum(value > 0 for value in per_dataset.values())),
        "per_dataset_gain": per_dataset,
    }


def main() -> None:
    paths = sorted((ROOT / "results" / "raw").glob("context_rescaled_confirmation_*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one raw bundle, found {paths}")
    raw_path = paths[0]
    metadata = json.loads(raw_path.with_suffix(".metadata.json").read_text())
    config = yaml.safe_load((ROOT / metadata["config"]).read_text())
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    rows = []
    for index in range(len(arrays["dataset"])):
        task_type = str(arrays["task_type"][index])
        tuning = metadata["synthetic_tuning"][task_type]
        experts = arrays["expert_prediction"][index].astype(float)
        y = arrays["query_y"][index].astype(float)
        fixed = weighted_prediction(experts, np.asarray(tuning["fixed_weights"], dtype=float))
        competence = weighted_prediction(
            experts,
            competence_weights(
                arrays["cv_expert_loss"][index].astype(float),
                float(tuning["temperature"]), float(tuning["uniform_shrinkage"]),
            ),
        )
        candidate = 0.9 * fixed + 0.1 * competence
        methods = {"fixed": fixed, "competence": competence}
        if task_type == "classification":
            methods["shrink_0.1"] = candidate
        for method, prediction in methods.items():
            rows.append({
                "episode_index": index, "dataset": str(arrays["dataset"][index]),
                "task_type": task_type, "method": method,
                "loss": prediction_loss(y, prediction, task_type),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(ROOT / "results" / "processed" / "context_rescaled_confirmation_detail_v1.csv", index=False)
    parent_path = ROOT / metadata["processed_summary"]
    parent = pd.read_csv(parent_path)
    parity = detail[detail["method"].isin(["fixed", "competence"])].merge(
        parent[["episode_index", "method", "loss"]],
        on=["episode_index", "method"], validate="one_to_one",
        suffixes=("_recomputed", "_parent"),
    )
    max_parent_error = float(np.max(np.abs(parity["loss_recomputed"] - parity["loss_parent"])))
    if max_parent_error > 1e-5:
        raise AssertionError(f"parent loss mismatch: {max_parent_error}")
    tasks = {}
    for offset, task_type in enumerate(("classification", "regression")):
        task = detail[detail["task_type"] == task_type]
        right = "shrink_0.1" if task_type == "classification" else "competence"
        pivot = task.pivot(index=["dataset", "episode_index"], columns="method", values="loss")
        delta = pivot["fixed"] - pivot[right]
        values = {
            str(dataset): group.to_numpy()
            for dataset, group in delta.groupby(level="dataset", sort=True)
        }
        result = hierarchical(values, 235501 + offset, int(config["bootstrap_draws"]))
        result["method"] = right
        result["gate_pass"] = bool(result["ci_low"] > 0 and result["positive_datasets"] >= 3)
        tasks[task_type] = result
    audit = {
        "protocol": "CONTEXT_RESCALED_CONFIRMATION_PROTOCOL.md",
        "parent_run": metadata["run_key"],
        "episodes": int(len(arrays["dataset"])),
        "episode_rescale": bool(config["episode_rescale"]),
        "parent_max_abs_loss_error": max_parent_error,
        "raw_bundle_sha256": sha256(raw_path),
        "parent_cells_sha256": sha256(parent_path),
        "tasks": tasks,
        "joint_robustness_pass": bool(all(result["gate_pass"] for result in tasks.values())),
    }
    path = ROOT / "results" / "processed" / "context_rescaled_confirmation_audit_v1.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
