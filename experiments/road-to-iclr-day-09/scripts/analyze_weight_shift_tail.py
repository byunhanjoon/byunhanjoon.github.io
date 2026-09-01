#!/usr/bin/env python3
"""Associate episode-level weight movement with immutable prediction tails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, weighted_prediction


SOURCES = {
    "small_panel": "real_panel_competence_55553b7ffd",
    "breadth_panel": "openml_breadth_competence_48170161d0",
    "regression_confirmation": "regression_confirmation_1e4911698d",
}
N_BOOT = 20_000


def tail_mean(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        loss = -(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
    else:
        loss = (prediction - y) ** 2
    count = len(loss) - int(np.floor(0.9 * len(loss)))
    return float(np.partition(loss, len(loss) - count)[-count:].mean())


def dataset_bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    records: list[dict[str, object]] = []
    for panel, stem in SOURCES.items():
        with np.load(ROOT / "results" / "raw" / f"{stem}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        metadata = json.loads((ROOT / "results" / "raw" / f"{stem}.metadata.json").read_text())
        for task_type in ("classification", "regression"):
            if task_type == "classification" and panel != "breadth_panel":
                continue
            tuning = metadata["synthetic_tuning"]
            if task_type in tuning:
                tuning = tuning[task_type]
            fixed = np.asarray(tuning["fixed_weights"], dtype=float)
            for index in range(len(arrays["dataset"])):
                if str(arrays["task_type"][index]) != task_type:
                    continue
                cv_loss = arrays["cv_expert_loss"][index].astype(float)
                competence = competence_weights(
                    cv_loss, float(tuning["temperature"]),
                    float(tuning["uniform_shrinkage"]),
                )
                experts = arrays["expert_prediction"][index].astype(float)
                y = arrays["query_y"][index].astype(float)
                fixed_tail = tail_mean(y, weighted_prediction(experts, fixed), task_type)
                competence_tail = tail_mean(y, weighted_prediction(experts, competence), task_type)
                positive = competence > 0
                kl_to_fixed = float(np.sum(
                    competence[positive] * np.log(competence[positive] / fixed[positive])
                ))
                records.append({
                    "panel": panel,
                    "dataset": str(arrays["dataset"][index]),
                    "task_type": task_type,
                    "source_episode_index": index,
                    "kl_to_fixed": kl_to_fixed,
                    "tv_to_fixed": float(0.5 * np.abs(competence - fixed).sum()),
                    "max_competence_weight": float(competence.max()),
                    "tail_gain": fixed_tail - competence_tail,
                })

    detail = pd.DataFrame(records)
    detail_path = ROOT / "results" / "processed" / "weight_shift_tail_detail_v1.csv"
    audit_path = ROOT / "results" / "processed" / "weight_shift_tail_audit_v1.json"
    detail.to_csv(detail_path, index=False)

    tasks: dict[str, object] = {}
    for offset, (task_type, task) in enumerate(detail.groupby("task_type", sort=True)):
        dataset_records = []
        for dataset, group in task.groupby("dataset", sort=True):
            ordered = group.sort_values("kl_to_fixed").reset_index(drop=True)
            quintile = pd.qcut(ordered.index, 5, labels=False)
            low = ordered.loc[quintile == 0, "tail_gain"].mean()
            high = ordered.loc[quintile == 4, "tail_gain"].mean()
            correlation = float(spearmanr(group["kl_to_fixed"], group["tail_gain"]).statistic)
            dataset_records.append({
                "dataset": dataset,
                "spearman": correlation,
                "high_minus_low_tail_gain": float(high - low),
                "low_shift_tail_gain": float(low),
                "high_shift_tail_gain": float(high),
            })
        dataset_frame = pd.DataFrame(dataset_records)
        tasks[str(task_type)] = {
            "datasets": len(dataset_frame),
            "spearman": dataset_bootstrap(dataset_frame["spearman"].to_numpy(), 205001 + offset),
            "high_minus_low_tail_gain": dataset_bootstrap(
                dataset_frame["high_minus_low_tail_gain"].to_numpy(), 205101 + offset
            ),
            "mean_kl_to_fixed": float(task["kl_to_fixed"].mean()),
            "mean_tv_to_fixed": float(task["tv_to_fixed"].mean()),
            "mean_max_competence_weight": float(task["max_competence_weight"].mean()),
            "per_dataset": dataset_records,
        }

    audit = {
        "protocol": "WEIGHT_SHIFT_TAIL_PROTOCOL.md",
        "bootstrap_replicates": N_BOOT,
        "tasks": tasks,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
