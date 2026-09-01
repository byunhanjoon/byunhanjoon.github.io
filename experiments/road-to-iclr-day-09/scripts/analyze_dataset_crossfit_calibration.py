#!/usr/bin/env python3
"""Frozen leave-one-dataset-out calibration over immutable real-panel predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights

TEMPERATURES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0)
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def load_bundle(prefix: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    raw_path = unique(f"results/raw/{prefix}_*.npz")
    cells_path = unique(f"results/processed/{prefix}_*_cells.csv")
    metadata_path = unique(f"results/raw/{prefix}_*.metadata.json")
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, pd.read_csv(cells_path), json.loads(metadata_path.read_text())


def loss(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        p = np.clip(prediction, 1e-6, 1 - 1e-6)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return float(np.mean((y - prediction) ** 2))


def hierarchical(values: dict[str, np.ndarray], seed: int) -> dict:
    names = sorted(values)
    observed = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    samples = np.empty(10_000)
    for draw in range(10_000):
        chosen = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            np.mean(rng.choice(values[str(name)], len(values[str(name)]), replace=True))
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def paired(frame: pd.DataFrame, left: str, right: str) -> dict[str, np.ndarray]:
    result = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        pivot = group.pivot(index="episode_uid", columns="method", values="loss")
        result[str(dataset)] = (pivot[left] - pivot[right]).to_numpy()
    return result


def main() -> None:
    sources = []
    for source_index, prefix in enumerate(("real_panel_competence", "openml_breadth_competence")):
        arrays, cells, metadata = load_bundle(prefix)
        parent = cells[cells["method"].isin(["fixed", "competence"])].copy()
        parent["source_index"] = source_index
        sources.append((source_index, arrays, parent, metadata))

    episode_records = []
    candidate_losses: dict[tuple[int, float, float], float] = {}
    baseline_errors = []
    tuning = sources[0][3]["synthetic_tuning"]
    uid = 0
    for source_index, arrays, parent, _ in sources:
        for index in range(len(arrays["dataset"])):
            dataset = str(arrays["dataset"][index])
            task_type = str(arrays["task_type"][index])
            predictions = arrays["expert_prediction"][index].astype(float)
            cv_loss = arrays["cv_expert_loss"][index].astype(float)
            y = arrays["query_y"][index].astype(float)
            fixed_weight = np.asarray(tuning[task_type]["fixed_weights"], dtype=float)
            fixed_loss = loss(y, fixed_weight @ predictions, task_type)
            synthetic_weight = competence_weights(
                cv_loss, float(tuning[task_type]["temperature"]),
                float(tuning[task_type]["uniform_shrinkage"]),
            )
            synthetic_loss = loss(y, synthetic_weight @ predictions, task_type)
            parent_rows = parent[
                (parent["episode_index"] == index) & (parent["source_index"] == source_index)
            ].set_index("method")["loss"]
            baseline_errors.extend([
                abs(fixed_loss - float(parent_rows["fixed"])),
                abs(synthetic_loss - float(parent_rows["competence"])),
            ])
            candidates = {}
            for temperature in TEMPERATURES:
                soft = competence_weights(cv_loss, temperature, 0.0)
                for alpha in ALPHAS:
                    weight = (1 - alpha) * soft + alpha * fixed_weight
                    candidates[(temperature, alpha)] = loss(y, weight @ predictions, task_type)
            episode_records.append({
                "episode_uid": uid, "source_index": source_index, "source_episode": index,
                "dataset": dataset, "task_type": task_type,
                "fixed": fixed_loss, "synthetic_competence": synthetic_loss,
                "candidates": candidates,
            })
            uid += 1
    max_error = float(max(baseline_errors))
    if max_error > 1e-5:
        raise AssertionError(f"parent baseline mismatch: {max_error}")

    output_records = []
    selected = {}
    for task_type in ("classification", "regression"):
        task = [row for row in episode_records if row["task_type"] == task_type]
        datasets = sorted({row["dataset"] for row in task})
        for held_out in datasets:
            training = [row for row in task if row["dataset"] != held_out]
            train_datasets = sorted({row["dataset"] for row in training})
            candidates = []
            for temperature in TEMPERATURES:
                for alpha in ALPHAS:
                    dataset_means = [
                        np.mean([
                            row["candidates"][(temperature, alpha)]
                            for row in training if row["dataset"] == name
                        ])
                        for name in train_datasets
                    ]
                    candidates.append((float(np.mean(dataset_means)), temperature, alpha))
            _, temperature, alpha = min(candidates)
            selected[held_out] = {"task_type": task_type, "temperature": temperature, "alpha": alpha}
            for row in task:
                if row["dataset"] != held_out:
                    continue
                methods = {
                    "fixed": row["fixed"],
                    "synthetic_competence": row["synthetic_competence"],
                    "dataset_crossfit": row["candidates"][(temperature, alpha)],
                }
                for method, value in methods.items():
                    output_records.append({
                        "episode_uid": row["episode_uid"], "dataset": held_out,
                        "task_type": task_type, "method": method, "loss": value,
                    })
    frame = pd.DataFrame(output_records)
    detail_path = ROOT / "results/processed/dataset_crossfit_calibration_detail_v1.csv"
    audit_path = ROOT / "results/processed/dataset_crossfit_calibration_audit_v1.json"
    for output in (detail_path, audit_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    frame.to_csv(detail_path, index=False)
    audit = {
        "protocol": "DATASET_CROSSFIT_CALIBRATION_PROTOCOL.md",
        "parent_max_abs_loss_error": max_error,
        "grid": {"temperatures": TEMPERATURES, "alphas": ALPHAS},
        "selected_by_heldout_dataset": selected,
        "tasks": {},
    }
    passing, no_harm = [], {}
    for task_index, task_type in enumerate(("classification", "regression")):
        task = frame[frame["task_type"] == task_type]
        fixed_contrast = hierarchical(paired(task, "fixed", "dataset_crossfit"), 22_000 + task_index)
        synthetic_contrast = hierarchical(
            paired(task, "synthetic_competence", "dataset_crossfit"), 22_100 + task_index
        )
        if fixed_contrast["ci_low"] > 0:
            passing.append(task_type)
        threshold = 0.005 if task_type == "classification" else 0.02
        no_harm[task_type] = fixed_contrast["ci_high"] >= -threshold
        audit["tasks"][task_type] = {
            "dataset_crossfit_vs_fixed": fixed_contrast,
            "dataset_crossfit_vs_synthetic_competence": synthetic_contrast,
            "mean_losses": task.groupby("method")["loss"].mean().to_dict(),
            "dataset_count": int(task["dataset"].nunique()),
        }
    audit["passing_task_types"] = passing
    audit["no_material_harm"] = no_harm
    audit["strong_gate_pass"] = len(passing) == 2
    audit["scoped_gate_pass"] = bool(passing and all(no_harm.values()))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
