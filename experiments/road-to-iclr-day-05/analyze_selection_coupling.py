"""Shared versus independent nuisance-coordinate model-selection ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PANELS = RMS.PANELS + ((
    "openml_taskbalanced", "openml_taskbalanced_cover_config.json",
    "openml_taskbalanced_cover",
),)
ORIGINAL_PANELS = {panel for panel, _, _ in RMS.PANELS}


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path, coupling: str) -> list[dict]:
    validation, test = [], []
    val_y = test_y = None
    shape = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        current = tuple(archive["validation_predictions"].shape[:4])
        shape = current if shape is None else shape
        if current != shape:
            raise AssertionError("factor shape mismatch")
        val_y = archive["validation_y"] if val_y is None else val_y
        test_y = archive["test_y"] if test_y is None else test_y
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and val_y is not None and test_y is not None
    quotient_val = np.asarray([proper_loss(val_y, values.mean(axis=0)) for values in validation])
    quotient_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(quotient_val))
    if coupling == "shared":
        shared = RMS.action_ids(shape, RMS.stable_seed("coupling", panel, dataset, coupling))
        actions = [shared] * len(models)
    else:
        actions = [
            RMS.action_ids(shape, RMS.stable_seed("coupling", panel, dataset, coupling, model))
            for model in models
        ]
    rows = []
    for method in RMS.METHODS:
        val_losses = np.stack([
            RMS.batched_losses(val_y, values, actions[index][method])
            for index, values in enumerate(validation)
        ], axis=1)
        selected = np.argmin(val_losses, axis=1)
        test_losses = np.stack([
            RMS.batched_losses(test_y, values, actions[index][method])
            for index, values in enumerate(test)
        ], axis=1)
        rows.append({
            "panel": panel, "dataset": dataset, "coupling": coupling, "method": method,
            "selected_realized_test_loss": float(np.mean(test_losses[np.arange(RMS.DRAWS), selected])),
            "selected_quotient_test_loss": float(np.mean(quotient_test[selected])),
            "validation_quotient_regret": float(np.mean(quotient_val[selected] - quotient_val[winner])),
            "selection_agreement": float(np.mean(selected == winner)),
            "selection_entropy_bits": RMS.entropy(selected, len(models)),
        })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            for coupling in ("shared", "independent"):
                rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory, coupling))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "selection_coupling_cells.csv", index=False)
    summary = {"status": "complete", "draws_per_dataset": RMS.DRAWS, "panels": {}}
    independent_passes = 0
    for (panel, coupling), current in frame.groupby(["panel", "coupling"]):
        pivot = current.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
        differences = pivot.strength2 - pivot.iid16
        result = {
            "datasets": len(pivot), "mean_strength2_minus_iid16": float(differences.mean()),
            "datasets_strength2_lower_iid16": int((differences < 0).sum()),
            "mean_metrics_by_method": current.groupby("method").mean(numeric_only=True).to_dict(orient="index"),
        }
        result["independent_gate_passed"] = bool(
            coupling == "independent" and differences.mean() < 0 and np.mean(differences < 0) >= 0.6
        )
        if result["independent_gate_passed"] and panel in ORIGINAL_PANELS:
            independent_passes += 1
        summary["panels"].setdefault(panel, {})[coupling] = result
    summary["independent_panels_passing"] = independent_passes
    summary["frozen_independent_coupling_gate_passed"] = independent_passes >= 2
    extension = summary["panels"]["openml_taskbalanced"]["independent"]
    summary["taskbalanced_postgate_addendum_passed"] = bool(
        extension["mean_strength2_minus_iid16"] < 0
        and extension["datasets_strength2_lower_iid16"] >= 5
    )
    (RESULTS / "selection_coupling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
