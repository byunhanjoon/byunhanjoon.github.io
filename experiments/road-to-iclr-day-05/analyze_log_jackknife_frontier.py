"""Four-block approximate log-quotient jackknife at 64 fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_log_quotient_jackknife as LQJ
import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
BATCH = 8
BLOCKS = 4
METHODS = ("strength2_jackknife64", "iid_jackknife64")
PANELS = LQJ.PANELS


def multiblock_scores(
    y: np.ndarray, flat: np.ndarray, block_ids: np.ndarray
) -> np.ndarray:
    output = np.empty(len(block_ids), dtype=np.float64)
    for start in range(0, len(block_ids), BATCH):
        stop = min(start + BATCH, len(block_ids))
        predictions = np.stack([
            flat[block_ids[start:stop, block]].mean(axis=1) for block in range(BLOCKS)
        ], axis=1)
        full_loss = LQJ.log_loss(y, predictions.mean(axis=1))
        block_loss = np.stack([
            LQJ.log_loss(y, predictions[:, block]) for block in range(BLOCKS)
        ], axis=1).mean(axis=1)
        output[start:stop] = (BLOCKS * full_loss - block_loss) / (BLOCKS - 1)
    return output


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
    actions = [
        RMS.action_ids(shape, RMS.stable_seed("log-jackknife64", panel, dataset, str(block)))
        for block in range(BLOCKS)
    ]
    cover_ids = np.stack([action["strength2"][:DRAWS] for action in actions], axis=1)
    iid_ids = np.stack([action["iid16"][:DRAWS] for action in actions], axis=1)
    exact_validation = np.asarray([LQJ.log_loss(validation_y, value.mean(axis=0)) for value in validation])
    exact_test = np.asarray([LQJ.log_loss(test_y, value.mean(axis=0)) for value in test])
    exact_winner = int(np.argmin(exact_validation))
    scores = {method: [] for method in METHODS}
    calibration = []
    for model_index, (model, candidate) in enumerate(zip(models, validation)):
        values_by_method = {
            METHODS[0]: multiblock_scores(validation_y, candidate, cover_ids),
            METHODS[1]: multiblock_scores(validation_y, candidate, iid_ids),
        }
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
                "validation_log_quotient_regret": float(exact_validation[candidate] - exact_validation[exact_winner]),
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
    draws.to_csv(RESULTS / "log_jackknife_frontier_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_log_quotient_regret=("validation_log_quotient_regret", "mean"),
        selected_test_log_quotient_loss=("selected_test_log_quotient_loss", "mean"),
    )
    cells.to_csv(RESULTS / "log_jackknife_frontier_cells.csv", index=False)
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(RESULTS / "log_jackknife_frontier_calibration.csv", index=False)
    prior = pd.read_csv(RESULTS / "log_quotient_jackknife_calibration.csv")
    prior = prior[prior.method == "strength2_jackknife32"]
    clauses = {"rmse": 0, "regret": 0, "frontier": 0}
    panels: dict[str, object] = {}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cal = calibration_frame[calibration_frame.panel == panel]
        cal_means = cal.groupby("method").agg(
            score_rmse=("score_rmse", "mean"),
            mean_absolute_bias=("score_bias", lambda x: float(np.mean(np.abs(x)))),
        )
        prior_rmse = float(prior[prior.panel == panel].score_rmse.mean())
        conditions = {
            "rmse": cal_means.loc[METHODS[0], "score_rmse"] < cal_means.loc[METHODS[1], "score_rmse"],
            "regret": means.loc[METHODS[0], "validation_log_quotient_regret"] <= means.loc[METHODS[1], "validation_log_quotient_regret"],
            "frontier": cal_means.loc[METHODS[0], "score_rmse"] <= prior_rmse,
        }
        for key, value in conditions.items():
            clauses[key] += bool(value)
        panels[panel] = {
            "clauses": {key: bool(value) for key, value in conditions.items()},
            "cover_jackknife32_rmse": prior_rmse,
            "calibration_means": cal_means.reset_index().to_dict(orient="records"),
            "selection_means": means.reset_index().to_dict(orient="records"),
        }
    if clauses["rmse"] >= 5 and clauses["regret"] >= 5:
        interpretation = "frontier_pass" if clauses["frontier"] >= 5 else "qualified"
    else:
        interpretation = "fail"
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "panels_passing_by_clause": clauses,
        "frozen_interpretation": interpretation, "panels": panels,
    }
    (RESULTS / "log_jackknife_frontier_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
