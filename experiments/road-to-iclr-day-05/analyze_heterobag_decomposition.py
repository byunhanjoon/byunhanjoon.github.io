"""Exact squared-loss quality/diversity decomposition for HeteroBag."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def values(predictions: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return 1 / (1 + np.exp(-np.clip(predictions.astype(np.float64), -40, 40)))
    return predictions.astype(np.float64)


def ensemble_terms(members: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    ensemble = members.mean(axis=0)
    mean_member_loss = float(np.mean((members - target[None, ...]) ** 2))
    ensemble_loss = float(np.mean((ensemble - target) ** 2))
    ambiguity = float(np.mean((members - ensemble[None, ...]) ** 2))
    if not np.isclose(mean_member_loss, ensemble_loss + ambiguity, rtol=1e-10, atol=1e-12):
        raise AssertionError("ambiguity identity failed")
    return mean_member_loss, ensemble_loss, ambiguity


def main() -> None:
    config = json.loads((HERE / "heterobag_phase1_config.json").read_text())
    mechanism = pd.concat([
        pd.read_csv(RESULTS / "heterobag_mechanism_classification.csv"),
        pd.read_csv(RESULTS / "heterobag_mechanism_regression.csv"),
    ], ignore_index=True)
    prediction_dir = RESULTS / "heterobag_mechanism_predictions"
    rows = []
    maximum_error = 0.0
    for record in mechanism.itertuples(index=False):
        archive = np.load(prediction_dir / f"{record.dataset}__{record.model}.npz")
        task = record.task
        target = archive["test_target"].astype(np.float64)
        t = np.stack([values(archive[f"t_{letter}_test"], task) for letter in ("a", "b", "c")])
        alternate = values(archive["alternate_c_test"], task)
        transformed = values(archive["transformed_t_c_test"], task)
        systems = {
            "ttt": t,
            "heterobag": np.stack((t[0], t[1], alternate)),
            "coordinate_placebo": np.stack((t[0], t[1], transformed)),
        }
        terms = {name: ensemble_terms(member, target) for name, member in systems.items()}
        control_member, control_ensemble, control_ambiguity = terms["ttt"]
        row = {"dataset": record.dataset, "task": task, "model": record.model}
        for name, (member_loss, ensemble_loss, ambiguity) in terms.items():
            row[f"{name}_mean_member_squared_loss"] = member_loss
            row[f"{name}_ensemble_squared_loss"] = ensemble_loss
            row[f"{name}_ambiguity"] = ambiguity
            if name != "ttt":
                quality_gain = control_member - member_loss
                diversity_gain = ambiguity - control_ambiguity
                ensemble_gain = control_ensemble - ensemble_loss
                error = abs(ensemble_gain - quality_gain - diversity_gain)
                maximum_error = max(maximum_error, error)
                row[f"{name}_member_quality_gain"] = quality_gain
                row[f"{name}_diversity_gain"] = diversity_gain
                row[f"{name}_ensemble_squared_loss_gain"] = ensemble_gain
                row[f"{name}_reconstruction_error"] = error
        row["primary_relative_gain_pct"] = record.heterobag_relative_test_gain_vs_ttt_pct
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != len(config["development_datasets"]) * len(config["architectures"]):
        raise RuntimeError(f"incomplete mechanism panel: {len(frame)} rows")
    task_summary = frame.groupby("task", as_index=False).agg(
        cells=("dataset", "size"),
        mean_heterobag_squared_gain=("heterobag_ensemble_squared_loss_gain", "mean"),
        mean_member_quality_gain=("heterobag_member_quality_gain", "mean"),
        mean_diversity_gain=("heterobag_diversity_gain", "mean"),
        cells_positive_squared_gain=("heterobag_ensemble_squared_loss_gain", lambda values: int((values > 0).sum())),
        cells_positive_quality_gain=("heterobag_member_quality_gain", lambda values: int((values > 0).sum())),
        cells_positive_diversity_gain=("heterobag_diversity_gain", lambda values: int((values > 0).sum())),
    )
    summary = {
        "status": "complete", "evidence_label": "descriptive_phase2_mechanism",
        "cells": len(frame), "maximum_ambiguity_reconstruction_error": maximum_error,
        "mean_heterobag_squared_loss_gain": float(frame.heterobag_ensemble_squared_loss_gain.mean()),
        "mean_member_quality_gain": float(frame.heterobag_member_quality_gain.mean()),
        "mean_diversity_gain": float(frame.heterobag_diversity_gain.mean()),
        "cells_positive_squared_loss_gain": int((frame.heterobag_ensemble_squared_loss_gain > 0).sum()),
        "cells_positive_member_quality_gain": int((frame.heterobag_member_quality_gain > 0).sum()),
        "cells_positive_diversity_gain": int((frame.heterobag_diversity_gain > 0).sum()),
        "spearman_squared_gain_with_primary_gain": float(
            frame.heterobag_ensemble_squared_loss_gain.corr(frame.primary_relative_gain_pct, method="spearman")
        ),
        "coordinate_placebo_mean_squared_loss_gain": float(frame.coordinate_placebo_ensemble_squared_loss_gain.mean()),
    }
    frame.to_csv(RESULTS / "heterobag_decomposition_cells.csv", index=False)
    task_summary.to_csv(RESULTS / "heterobag_decomposition_tasks.csv", index=False)
    (RESULTS / "heterobag_decomposition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

