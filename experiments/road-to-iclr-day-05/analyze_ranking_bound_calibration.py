"""Calibrate Proposition 16 against observed pairwise ranking inversions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
PANELS = RMS.PANELS + (
    ("openml_external", "openml_external_cover_config.json", "openml_external_cover"),
    ("openml_taskbalanced", "openml_taskbalanced_cover_config.json", "openml_taskbalanced_cover"),
)
EXACT_FILES = {
    "confirmation": "strength2_confirmation_cells.csv",
    "menu_repeat": "strength2_menu_repeat_cells.csv",
    "subsample_repeat": "strength2_subsample_repeat_cells.csv",
    "openml_external": "strength2_openml_external_cells.csv",
    "openml_taskbalanced": "strength2_openml_taskbalanced_cells.csv",
}
METHODS = {"strength2": "strength2_residual", "iid16": "iid16_residual"}


def main() -> None:
    rows: list[dict[str, object]] = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        exact = pd.read_csv(RESULTS / EXACT_FILES[panel])
        exact = exact[exact.split == "validation"].set_index(["dataset", "model"])
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            models = config["models"]
            predictions = []
            y = None
            shape = None
            task = None
            for model in models:
                archive = np.load(directory / f"{dataset}__{model}.npz")
                manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
                task = manifest["task"]
                if task not in {"binclass", "classification"}:
                    break
                y = archive["validation_y"] if y is None else y
                shape = tuple(archive["validation_predictions"].shape[:4]) if shape is None else shape
                predictions.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
            if task not in {"binclass", "classification"}:
                continue
            assert y is not None and shape is not None
            truth = np.asarray([proper_loss(y, values.mean(axis=0)) for values in predictions])
            actions = RMS.action_ids(shape, RMS.stable_seed("ranking-bound", panel, dataset))
            for method, residual_column in METHODS.items():
                candidate_losses = np.stack([
                    RMS.batched_losses(y, values, actions[method][:DRAWS])
                    for values in predictions
                ], axis=1)
                errors = candidate_losses - truth[None]
                for left in range(len(models)):
                    for right in range(left + 1, len(models)):
                        if truth[left] == truth[right]:
                            continue
                        better, worse = (left, right) if truth[left] < truth[right] else (right, left)
                        gap = float(truth[worse] - truth[better])
                        inversion = float(np.mean(candidate_losses[:, better] > candidate_losses[:, worse]))
                        r_better = float(exact.loc[(dataset, models[better]), residual_column])
                        r_worse = float(exact.loc[(dataset, models[worse]), residual_column])
                        moment_bound_better = (8 * truth[better] + 4) * r_better
                        moment_bound_worse = (8 * truth[worse] + 4) * r_worse
                        theoretical = 4 * (moment_bound_better + moment_bound_worse) / gap ** 2
                        empirical = 4 * (
                            np.mean(errors[:, better] ** 2) + np.mean(errors[:, worse] ** 2)
                        ) / gap ** 2
                        rows.append({
                            "panel": panel, "dataset": dataset, "method": method,
                            "better_model": models[better], "worse_model": models[worse],
                            "quotient_loss_gap": gap, "empirical_inversion_rate": inversion,
                            "raw_theoretical_bound": float(theoretical),
                            "clipped_theoretical_bound": float(min(1, theoretical)),
                            "raw_empirical_markov_bound": float(empirical),
                            "clipped_empirical_markov_bound": float(min(1, empirical)),
                            "bound_covers_observed_rate": bool(inversion <= min(1, theoretical) + 1e-12),
                        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "ranking_bound_calibration_pairs.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "draws_per_pair": DRAWS, "panels": {}}
    panel_wins = 0
    for panel, panel_frame in frame.groupby("panel"):
        records = {}
        method_rates = {}
        for method, current in panel_frame.groupby("method"):
            method_rates[method] = float(current.empirical_inversion_rate.mean())
            records[method] = {
                "pairs": len(current),
                "mean_empirical_inversion_rate": method_rates[method],
                "pairs_with_nonvacuous_theoretical_bound": int((current.raw_theoretical_bound < 1).sum()),
                "pairs_with_nonvacuous_empirical_markov_bound": int((current.raw_empirical_markov_bound < 1).sum()),
                "pairs_observed_rate_covered": int(current.bound_covers_observed_rate.sum()),
                "median_raw_theoretical_bound": float(current.raw_theoretical_bound.median()),
            }
        lower = method_rates["strength2"] < method_rates["iid16"]
        panel_wins += lower
        summary["panels"][panel] = {
            "strength2_mean_inversion_below_iid": bool(lower), "methods": records
        }
    summary["panels_strength2_mean_inversion_below_iid"] = int(panel_wins)
    summary["interpretation"] = (
        "Finite bounds are reported for validity and informativeness; clipped-one bounds are vacuous."
    )
    (RESULTS / "ranking_bound_calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
