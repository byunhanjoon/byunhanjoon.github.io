#!/usr/bin/env python3
"""Frozen cyclic expert-assignment control on synthetic and real predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.methods import competence_weights


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1: raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def load(pattern: str) -> dict[str, np.ndarray]:
    with np.load(unique(pattern), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metric(y: np.ndarray, prediction: np.ndarray, task: str) -> float:
    if task == "classification":
        p = np.clip(prediction, 1e-6, 1 - 1e-6)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return float(np.mean((y - prediction) ** 2))


def bootstrap_equal_cells(frame: pd.DataFrame, seed: int) -> dict:
    cells = [group["gain"].to_numpy() for _, group in frame.groupby(
        ["context_size", "feature_count", "rho"], sort=True
    )]
    observed = float(np.mean([x.mean() for x in cells])); rng = np.random.default_rng(seed)
    samples = np.empty(10_000)
    for start in range(0, 10_000, 250):
        stop = min(start + 250, 10_000); chunk = np.zeros(stop - start)
        for values in cells:
            index = rng.integers(0, len(values), size=(stop - start, len(values)))
            chunk += values[index].mean(axis=1)
        samples[start:stop] = chunk / len(cells)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def bootstrap_datasets(frame: pd.DataFrame, seed: int) -> dict:
    values = {name: group["gain"].to_numpy() for name, group in frame.groupby("dataset")}
    names = sorted(values); observed = float(np.mean([values[n].mean() for n in names]))
    rng = np.random.default_rng(seed); samples = np.empty(10_000)
    for draw in range(10_000):
        chosen = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            np.mean(rng.choice(values[str(name)], len(values[str(name)]), replace=True))
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def episode_gain(arrays: dict[str, np.ndarray], index: int, tuning: dict, task: str) -> tuple[float, float]:
    predictions = arrays["expert_prediction"][index].astype(float)
    y = arrays["query_y"][index].astype(float)
    cv_loss = arrays["cv_expert_loss"][index].astype(float)
    temperature = float(tuning[task]["temperature"])
    shrinkage = float(tuning[task]["uniform_shrinkage"])
    aligned_weight = competence_weights(cv_loss, temperature, shrinkage)
    aligned = metric(y, aligned_weight @ predictions, task)
    null = np.mean([
        metric(y, competence_weights(np.roll(cv_loss, shift), temperature, shrinkage) @ predictions, task)
        for shift in range(1, 6)
    ])
    return aligned, float(null)


def main() -> None:
    development_meta = json.loads(unique(
        "results/raw/fallback_loss_router_*_development.metadata.json"
    ).read_text())
    tuning = development_meta["tuning"]
    records, parity = [], []

    synthetic = load("results/raw/fallback_loss_router_*_test.npz")
    synthetic_parent = pd.read_csv(unique("results/processed/fallback_loss_router_*_test_cells.csv"))
    synthetic_parent = synthetic_parent[synthetic_parent["method"] == "competence"].set_index("episode_index")
    for index in range(len(synthetic["rho"])):
        task = str(synthetic["task_type"][index]); aligned, null = episode_gain(synthetic, index, tuning, task)
        parity.append(abs(aligned - float(synthetic_parent.loc[index, "loss"])))
        records.append({
            "domain": "synthetic", "dataset": "PriorDial", "task_type": task,
            "episode_uid": f"s-{index}", "context_size": int(synthetic["context_size"][index]),
            "feature_count": int(synthetic["feature_count"][index]),
            "rho": float(synthetic["rho"][index]), "aligned_loss": aligned,
            "permutation_null_loss": null, "gain": null - aligned,
        })

    real_specs = (
        ("real_panel_competence", "r0"),
        ("openml_breadth_competence", "r1"),
        ("regression_confirmation", "r2"),
    )
    for pattern, tag in real_specs:
        arrays = load(f"results/raw/{pattern}_*.npz")
        parent = pd.read_csv(unique(f"results/processed/{pattern}_*_cells.csv"))
        parent = parent[parent["method"] == "competence"].set_index("episode_index")
        for index in range(len(arrays["dataset"])):
            task = str(arrays["task_type"][index]) if "task_type" in arrays else "regression"
            aligned, null = episode_gain(arrays, index, tuning, task)
            parity.append(abs(aligned - float(parent.loc[index, "loss"])))
            records.append({
                "domain": "real", "dataset": str(arrays["dataset"][index]),
                "task_type": task, "episode_uid": f"{tag}-{index}",
                "context_size": 96, "feature_count": int(arrays["feature_count"][index]),
                "rho": np.nan, "aligned_loss": aligned, "permutation_null_loss": null,
                "gain": null - aligned,
            })
    max_error = float(max(parity))
    if max_error > 1e-5: raise AssertionError(f"parent mismatch {max_error}")
    frame = pd.DataFrame(records)
    audit = {
        "protocol": "EXPERT_ASSIGNMENT_CONTROL_PROTOCOL.md",
        "parent_max_abs_loss_error": max_error,
        "synthetic": {}, "real": {},
    }
    for task_index, task in enumerate(("classification", "regression")):
        audit["synthetic"][task] = bootstrap_equal_cells(
            frame[(frame["domain"] == "synthetic") & (frame["task_type"] == task)], 26_000 + task_index
        )
        real = frame[(frame["domain"] == "real") & (frame["task_type"] == task)]
        audit["real"][task] = {
            **bootstrap_datasets(real, 26_100 + task_index),
            "dataset_count": int(real["dataset"].nunique()),
        }
    audit["all_intervals_positive"] = all(
        audit[domain][task]["ci_low"] > 0
        for domain in ("synthetic", "real") for task in ("classification", "regression")
    )
    detail_path = ROOT / "results/processed/expert_assignment_control_detail_v1.csv"
    audit_path = ROOT / "results/processed/expert_assignment_control_audit_v1.json"
    for output in (detail_path, audit_path):
        if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    frame.to_csv(detail_path, index=False); audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
