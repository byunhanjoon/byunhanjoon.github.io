"""Additional equal-budget unbiased cross-score controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_cross_quotient_selection import PANELS, cross_and_mean_scores
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTROL_MAP = {
    "srswor_cross32": "srswor16",
    "strength1_cross32": "four_strength1",
    "seed_cross32": "four_seed_blocks",
    "sobol_cross32": "sobol16",
    "lhs_cross32": "lhs16",
}


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path) -> list[dict[str, object]]:
    validation, test = [], []
    val_y = test_y = None
    shape = None
    task = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        val_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(archive["validation_predictions"].shape[:4])
        task = manifest["task"]
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert val_y is not None and test_y is not None and shape is not None
    first = RMS.action_ids(shape, RMS.stable_seed("cross-control-a", panel, dataset))
    second = RMS.action_ids(shape, RMS.stable_seed("cross-control-b", panel, dataset))
    quotient_val = np.asarray([proper_loss(val_y, values.mean(axis=0)) for values in validation])
    quotient_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(quotient_val))
    rows = []
    for output_name, base_name in CONTROL_MAP.items():
        validation_scores, test_scores = [], []
        for val_flat, test_flat in zip(validation, test):
            cross, _ = cross_and_mean_scores(val_y, val_flat, first[base_name], second[base_name])
            _, test_mean = cross_and_mean_scores(test_y, test_flat, first[base_name], second[base_name])
            validation_scores.append(cross)
            test_scores.append(test_mean)
        scores = np.stack(validation_scores, axis=1)
        realized = np.stack(test_scores, axis=1)
        selected = np.argmin(scores, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": output_name, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(realized[draw, chosen]),
            })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "cross_quotient_control_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "cross_quotient_control_cells.csv", index=False)
    primary = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    primary = primary[primary.method == "strength2_cross32"]
    combined = pd.concat([primary, cells], ignore_index=True)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    pass_counts = {control: 0 for control in CONTROL_MAP}
    for panel, current in combined.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        records = {}
        for control in CONTROL_MAP:
            lower_regret = bool(
                means.loc["strength2_cross32", "validation_quotient_regret"]
                < means.loc[control, "validation_quotient_regret"]
            )
            pass_counts[control] += lower_regret
            pivot = current.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
            difference = pivot.strength2_cross32 - pivot[control]
            records[control] = {
                "validation_regret_lower": lower_regret,
                "strength2_minus_control_realized_test_loss_mean": float(difference.mean()),
                "sources_strength2_test_lower": int((difference < 0).sum()),
                "sources": len(difference),
            }
        summary["panels"][panel] = {
            "means": means.reset_index().to_dict(orient="records"),
            "comparisons": records,
        }
    summary["panels_lower_validation_regret_by_control"] = pass_counts
    summary["postgate_control_addendum_passed"] = bool(
        all(count >= 4 for count in pass_counts.values())
    )
    (RESULTS / "cross_quotient_control_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
