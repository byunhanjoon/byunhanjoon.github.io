"""Conditional repeated-partition audit of validation-to-test rank transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 4_096
PANELS = (
    ("openml_external", "openml_external_cover_config.json", "openml_external_cover"),
    ("openml_taskbalanced", "openml_taskbalanced_cover_config.json", "openml_taskbalanced_cover"),
)


def row_losses(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-row Brier or squared-error loss for a quotient prediction."""
    if prediction.shape[-1] == 1:
        return np.square(y.astype(np.float64) - prediction[:, 0])
    target = np.eye(prediction.shape[-1], dtype=np.float64)[y.astype(int)]
    return np.sum(np.square(target - prediction), axis=-1)


def exact_candidate_losses(
    directory: Path, dataset: str, models: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    validation_rows, test_rows = [], []
    validation_y = test_y = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        if validation_y is not None and not np.array_equal(validation_y, archive["validation_y"]):
            raise AssertionError("validation labels differ across candidates")
        if test_y is not None and not np.array_equal(test_y, archive["test_y"]):
            raise AssertionError("test labels differ across candidates")
        validation_y = archive["validation_y"]
        test_y = archive["test_y"]
        task = str(manifest["task"])
        val_prediction = archive["validation_predictions"].astype(np.float64).mean(axis=(0, 1, 2, 3))
        test_prediction = archive["test_predictions"].astype(np.float64).mean(axis=(0, 1, 2, 3))
        validation_rows.append(row_losses(validation_y, val_prediction))
        test_rows.append(row_losses(test_y, test_prediction))
    assert validation_y is not None and test_y is not None
    return (
        np.stack(validation_rows, axis=1), np.stack(test_rows, axis=1),
        validation_y, test_y, task,
    )


def split_metrics(val: np.ndarray, test: np.ndarray) -> dict[str, float | int | bool]:
    val_mean = val.mean(axis=0)
    test_mean = test.mean(axis=0)
    val_winner = int(np.argmin(val_mean))
    test_winner = int(np.argmin(test_mean))
    ordered = np.sort(val_mean)
    val_rank = rankdata(val_mean, method="average")
    test_rank = rankdata(test_mean, method="average")
    correlation = float(np.corrcoef(val_rank, test_rank)[0, 1])
    return {
        "winner_agreement": bool(val_winner == test_winner),
        "validation_winner": val_winner,
        "test_winner": test_winner,
        "target_shift_floor": float(test_mean[val_winner] - test_mean[test_winner]),
        "rank_correlation": correlation,
        "validation_margin": float(ordered[1] - ordered[0]),
    }


def stratified_indices(
    y: np.ndarray, validation_y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    classes, counts = np.unique(validation_y, return_counts=True)
    for label, count in zip(classes, counts):
        eligible = np.flatnonzero(y == label)
        selected.append(rng.choice(eligible, size=int(count), replace=False))
    return np.concatenate(selected)


def analyze_dataset(
    panel: str, dataset: str, models: list[str], directory: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    val_rows, test_rows, val_y, test_y, task = exact_candidate_losses(directory, dataset, models)
    original = split_metrics(val_rows, test_rows)
    pooled_rows = np.concatenate((val_rows, test_rows), axis=0)
    pooled_y = np.concatenate((val_y, test_y), axis=0)
    rng = np.random.default_rng(RMS.stable_seed("repeated-heldout", panel, dataset))
    all_indices = np.arange(len(pooled_y))
    rows: list[dict[str, object]] = []
    classification = task in {"binclass", "multiclass"}
    for draw in range(DRAWS):
        if classification:
            validation_indices = stratified_indices(pooled_y, val_y, rng)
        else:
            validation_indices = rng.choice(all_indices, size=len(val_y), replace=False)
        test_mask = np.ones(len(pooled_y), dtype=bool)
        test_mask[validation_indices] = False
        metrics = split_metrics(pooled_rows[validation_indices], pooled_rows[test_mask])
        rows.append({
            "panel": panel, "dataset": dataset, "task": task, "draw": draw,
            **metrics,
        })
    frame = pd.DataFrame(rows)
    floor_values = frame.target_shift_floor.to_numpy()
    original_floor = float(original["target_shift_floor"])
    lower = float(np.quantile(floor_values, .025))
    upper = float(np.quantile(floor_values, .975))
    tied = np.isclose(floor_values, original_floor, rtol=0.0, atol=1e-15)
    mid_percentile = float(np.mean(floor_values < original_floor) + .5 * np.mean(tied))
    summary = {
        "panel": panel, "dataset": dataset, "task": task, "models": models,
        "evaluation_rows": int(len(pooled_y)), "validation_rows": int(len(val_y)),
        "original": original,
        "repartition": {
            "winner_agreement_probability": float(frame.winner_agreement.mean()),
            "mean_target_shift_floor": float(frame.target_shift_floor.mean()),
            "target_shift_floor_interval_95": [lower, upper],
            "mean_rank_correlation": float(frame.rank_correlation.mean()),
            "rank_correlation_interval_95": [float(x) for x in np.quantile(frame.rank_correlation, [.025, .975])],
            "mean_validation_margin": float(frame.validation_margin.mean()),
            "original_floor_mid_percentile": mid_percentile,
            "original_floor_inside_central_95": bool(lower <= original_floor <= upper),
            "original_floor_above_97_5_quantile": bool(original_floor > upper),
        },
    }
    return rows, summary


def main() -> None:
    rows: list[dict[str, object]] = []
    datasets: list[dict[str, object]] = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            current_rows, current_summary = analyze_dataset(
                panel, dataset, config["models"], directory
            )
            rows.extend(current_rows)
            datasets.append(current_summary)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "repeated_holdout_shift_draws.csv", index=False)
    dataset_frame = pd.DataFrame([
        {
            "panel": item["panel"], "dataset": item["dataset"], "task": item["task"],
            "original_winner_agreement": item["original"]["winner_agreement"],
            "original_target_shift_floor": item["original"]["target_shift_floor"],
            "original_rank_correlation": item["original"]["rank_correlation"],
            **item["repartition"],
        }
        for item in datasets
    ])
    dataset_frame.to_csv(RESULTS / "repeated_holdout_shift_cells.csv", index=False)
    panels: dict[str, object] = {}
    for panel, current in dataset_frame.groupby("panel"):
        finite_count = int((
            current.original_floor_inside_central_95
            & (current.winner_agreement_probability < .95)
        ).sum())
        exceptional_count = int(current.original_floor_above_97_5_quantile.sum())
        if finite_count >= len(current) / 2:
            label = "finite_partition_explanation_supported"
        elif exceptional_count >= len(current) / 2:
            label = "original_split_exceptional"
        else:
            label = "mixed"
        panels[panel] = {
            "datasets": int(len(current)),
            "original_winner_agreement": float(current.original_winner_agreement.mean()),
            "mean_repartition_winner_agreement_probability": float(current.winner_agreement_probability.mean()),
            "mean_original_target_shift_floor": float(current.original_target_shift_floor.mean()),
            "mean_repartition_target_shift_floor": float(current.mean_target_shift_floor.mean()),
            "original_floor_inside_central_95_and_agreement_below_95_percent": finite_count,
            "original_floor_above_97_5_percentile": exceptional_count,
            "frozen_interpretation": label,
        }
    summary = {"status": "complete", "draws_per_dataset": DRAWS, "panels": panels, "datasets": datasets}
    (RESULTS / "repeated_holdout_shift_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "panels": panels}, indent=2))


if __name__ == "__main__":
    main()
