"""Approximate two-cover jackknife for nonlinear quotient log loss."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
BATCH = 16
EPS = 1e-12
METHODS = (
    "strength2_jackknife32", "iid_jackknife32",
    "strength2_mean32", "iid_mean32",
)
PANELS = CQS.PANELS + (
    ("openml_multiclass", "openml_multiclass_cover_config.json", "openml_multiclass_cover"),
)


def log_loss(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Log loss for one prediction or a leading batch of predictions."""
    clipped = np.clip(prediction, EPS, 1.0)
    if clipped.ndim == 2:
        return np.asarray(-np.log(clipped[np.arange(len(y)), y.astype(int)]).mean())
    row = np.arange(len(y))[None]
    batch = np.arange(clipped.shape[0])[:, None]
    return -np.log(clipped[batch, row, y.astype(int)[None]]).mean(axis=1)


def block_scores(
    y: np.ndarray, flat: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ordinary = np.empty(len(left), dtype=np.float64)
    jackknife = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), BATCH):
        stop = min(start + BATCH, len(left))
        pred_a = flat[left[start:stop]].mean(axis=1)
        pred_b = flat[right[start:stop]].mean(axis=1)
        mean_score = log_loss(y, (pred_a + pred_b) / 2)
        ordinary[start:stop] = mean_score
        jackknife[start:stop] = 2 * mean_score - .5 * (
            log_loss(y, pred_a) + log_loss(y, pred_b)
        )
    return ordinary, jackknife


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        task = str(manifest["task"])
        if task not in {"binclass", "multiclass"}:
            return [], []
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(int(x) for x in archive["validation_predictions"].shape[:4])
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert validation_y is not None and test_y is not None and shape is not None
    ids = CQS.action_ids(shape, panel, dataset)
    exact_validation = np.asarray([log_loss(validation_y, value.mean(axis=0)) for value in validation])
    exact_test = np.asarray([log_loss(test_y, value.mean(axis=0)) for value in test])
    exact_winner = int(np.argmin(exact_validation))
    scores = {method: [] for method in METHODS}
    calibration = []
    for model, candidate in zip(models, validation):
        s2_mean, s2_jack = block_scores(validation_y, candidate, *ids["strength2"])
        iid_mean, iid_jack = block_scores(validation_y, candidate, *ids["iid"])
        values_by_method = {
            "strength2_jackknife32": s2_jack, "iid_jackknife32": iid_jack,
            "strength2_mean32": s2_mean, "iid_mean32": iid_mean,
        }
        model_index = models.index(model)
        for method, values in values_by_method.items():
            scores[method].append(values)
            bias = float(values.mean() - exact_validation[model_index])
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task,
                "model": model, "method": method, "score_bias": bias,
                "score_rmse": float(np.sqrt(values.var(ddof=1) + bias ** 2)),
                "mc_standard_error": float(values.std(ddof=1) / np.sqrt(DRAWS)),
            })
    rows = []
    for method in METHODS:
        matrix = np.stack(scores[method], axis=1)
        selected = np.argmin(matrix, axis=1)
        for draw, candidate in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(candidate == exact_winner),
                "validation_log_quotient_regret": float(
                    exact_validation[candidate] - exact_validation[exact_winner]
                ),
                "selected_test_log_quotient_loss": float(exact_test[candidate]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration = [], []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current_rows, current_calibration = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current_rows); calibration.extend(current_calibration)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "log_quotient_jackknife_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_log_quotient_regret=("validation_log_quotient_regret", "mean"),
        selected_test_log_quotient_loss=("selected_test_log_quotient_loss", "mean"),
    )
    cells.to_csv(RESULTS / "log_quotient_jackknife_cells.csv", index=False)
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(RESULTS / "log_quotient_jackknife_calibration.csv", index=False)
    clauses = {"rmse": 0, "regret": 0, "bias": 0}
    panels: dict[str, object] = {}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cal = calibration_frame[calibration_frame.panel == panel]
        cal_means = cal.groupby("method").agg(
            score_rmse=("score_rmse", "mean"),
            mean_absolute_bias=("score_bias", lambda x: float(np.mean(np.abs(x)))),
        )
        conditions = {
            "rmse": cal_means.loc["strength2_jackknife32", "score_rmse"] < cal_means.loc["iid_jackknife32", "score_rmse"],
            "regret": means.loc["strength2_jackknife32", "validation_log_quotient_regret"] <= means.loc["iid_jackknife32", "validation_log_quotient_regret"],
            "bias": cal_means.loc["strength2_jackknife32", "mean_absolute_bias"] < cal_means.loc["strength2_mean32", "mean_absolute_bias"],
        }
        for key, value in conditions.items():
            clauses[key] += bool(value)
        panels[panel] = {
            "clauses": {key: bool(value) for key, value in conditions.items()},
            "calibration_means": cal_means.reset_index().to_dict(orient="records"),
            "selection_means": means.reset_index().to_dict(orient="records"),
        }
    if clauses["rmse"] >= 5 and clauses["regret"] >= 5:
        interpretation = "full_pass" if clauses["bias"] >= 4 else "efficiency_only"
    else:
        interpretation = "fail"
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "panels_passing_by_clause": clauses,
        "frozen_interpretation": interpretation, "panels": panels,
    }
    (RESULTS / "log_quotient_jackknife_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
