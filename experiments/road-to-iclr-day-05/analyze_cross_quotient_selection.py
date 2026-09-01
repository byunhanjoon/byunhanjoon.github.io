"""Equal-budget unbiased cross-score model selection over nuisance quotients."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
BATCH = 16
METHODS = (
    "strength2_cross32", "iid_cross32", "iid_u32",
    "strength2_mean32", "iid_mean32",
)
PANELS = RMS.PANELS + (
    ("openml_external", "openml_external_cover_config.json", "openml_external_cover"),
    ("openml_taskbalanced", "openml_taskbalanced_cover_config.json", "openml_taskbalanced_cover"),
)


def residuals(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if predictions.shape[-1] == 1:
        return y[None, :, None] - predictions
    targets = np.eye(predictions.shape[-1], dtype=np.float64)[y.astype(int)]
    return targets[None] - predictions


def cross_and_mean_scores(
    y: np.ndarray, flat: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.empty(len(left), dtype=np.float64)
    mean_score = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), BATCH):
        stop = min(start + BATCH, len(left))
        pred_left = flat[left[start:stop]].mean(axis=1)
        pred_right = flat[right[start:stop]].mean(axis=1)
        r_left = residuals(y, pred_left)
        r_right = residuals(y, pred_right)
        cross[start:stop] = np.mean(np.sum(r_left * r_right, axis=-1), axis=1)
        mean_prediction = (pred_left + pred_right) / 2
        mean_score[start:stop] = np.mean(
            np.sum(residuals(y, mean_prediction) ** 2, axis=-1), axis=1
        )
    return cross, mean_score


def iid_u_scores(y: np.ndarray, flat: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Complete ordered-pair U-statistic without materializing all pairs."""
    output = np.empty(len(ids), dtype=np.float64)
    members = ids.shape[1]
    for start in range(0, len(ids), BATCH):
        stop = min(start + BATCH, len(ids))
        prediction_members = flat[ids[start:stop]]
        if prediction_members.shape[-1] == 1:
            target = y[None, None, :, None]
        else:
            target = np.eye(prediction_members.shape[-1], dtype=np.float64)[y.astype(int)][None, None]
        member_residuals = target - prediction_members
        summed = member_residuals.sum(axis=1)
        numerator = np.sum(summed ** 2, axis=-1) - np.sum(
            member_residuals ** 2, axis=(1, 3)
        )
        output[start:stop] = np.mean(numerator / (members * (members - 1)), axis=1)
    return output


def action_ids(shape: tuple[int, ...], panel: str, dataset: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    first = RMS.action_ids(shape, RMS.stable_seed("cross-score-a", panel, dataset))
    second = RMS.action_ids(shape, RMS.stable_seed("cross-score-b", panel, dataset))
    return {
        "strength2": (first["strength2"], second["strength2"]),
        "iid": (first["iid16"], second["iid16"]),
    }


def analyze_dataset(
    panel: str, dataset: str, models: list[str], directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        current_shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        if shape is not None and current_shape != shape:
            raise AssertionError("factor shape mismatch")
        shape = current_shape
        if validation_y is not None and not np.array_equal(validation_y, archive["validation_y"]):
            raise AssertionError("validation labels differ across candidates")
        if test_y is not None and not np.array_equal(test_y, archive["test_y"]):
            raise AssertionError("test labels differ across candidates")
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        task = manifest["task"]
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and validation_y is not None and test_y is not None
    ids = action_ids(shape, panel, dataset)
    quotient_val = np.asarray([proper_loss(validation_y, prediction.mean(axis=0)) for prediction in validation])
    quotient_test = np.asarray([proper_loss(test_y, prediction.mean(axis=0)) for prediction in test])
    validation_winner = int(np.argmin(quotient_val))

    validation_scores = {method: [] for method in METHODS}
    test_mean_scores = {method: [] for method in METHODS}
    for val_flat, test_flat in zip(validation, test):
        s2_cross, s2_mean = cross_and_mean_scores(validation_y, val_flat, *ids["strength2"])
        iid_cross, iid_mean = cross_and_mean_scores(validation_y, val_flat, *ids["iid"])
        iid32 = np.concatenate(ids["iid"], axis=1)
        validation_scores["strength2_cross32"].append(s2_cross)
        validation_scores["iid_cross32"].append(iid_cross)
        validation_scores["iid_u32"].append(iid_u_scores(validation_y, val_flat, iid32))
        validation_scores["strength2_mean32"].append(s2_mean)
        validation_scores["iid_mean32"].append(iid_mean)

        _, s2_test_mean = cross_and_mean_scores(test_y, test_flat, *ids["strength2"])
        _, iid_test_mean = cross_and_mean_scores(test_y, test_flat, *ids["iid"])
        test_mean_scores["strength2_cross32"].append(s2_test_mean)
        test_mean_scores["strength2_mean32"].append(s2_test_mean)
        test_mean_scores["iid_cross32"].append(iid_test_mean)
        test_mean_scores["iid_u32"].append(iid_test_mean)
        test_mean_scores["iid_mean32"].append(iid_test_mean)

    rows: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    for method in METHODS:
        scores = np.stack(validation_scores[method], axis=1)
        for model_index, model in enumerate(models):
            values = scores[:, model_index]
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
            bias = float(values.mean() - quotient_val[model_index])
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task,
                "model": model, "method": method,
                "mean_selection_score": float(values.mean()),
                "exact_quotient_validation_loss": float(quotient_val[model_index]),
                "score_bias": bias, "mc_standard_error": standard_error,
                "standardized_bias": bias / standard_error if standard_error > 1e-15 else np.nan,
            })
        selected = np.argmin(scores, axis=1)
        realized_test = np.stack(test_mean_scores[method], axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task, "method": method,
                "draw": draw, "selected_model": models[int(chosen)],
                "selection_score": float(scores[draw, chosen]),
                "selection_agreement": bool(chosen == validation_winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[validation_winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(realized_test[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows = []
    calibration_rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            current_rows, current_calibration = analyze_dataset(
                panel, dataset, config["models"], directory
            )
            rows.extend(current_rows)
            calibration_rows.extend(current_calibration)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "cross_quotient_selection_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "task", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "cross_quotient_selection_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "cross_quotient_score_calibration.csv", index=False)
    score_bias = calibration.groupby("method").agg(
        mean_bias=("score_bias", "mean"),
        mean_absolute_bias=("score_bias", lambda values: float(np.mean(np.abs(values)))),
        median_standardized_bias=("standardized_bias", "median"),
        cells=("score_bias", "size"),
    ).reset_index().to_dict(orient="records")
    summary: dict[str, object] = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "score_bias_calibration": score_bias, "panels": {},
    }
    clauses = {"agreement": 0, "regret": 0, "test": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        action, control = means.loc["strength2_cross32"], means.loc["iid_u32"]
        panel_clauses = {
            "agreement_above_iid_u32": bool(action.selection_agreement > control.selection_agreement),
            "validation_regret_below_iid_u32": bool(action.validation_quotient_regret < control.validation_quotient_regret),
            "selected_quotient_test_loss_below_iid_u32": bool(action.selected_quotient_test_loss < control.selected_quotient_test_loss),
        }
        clauses["agreement"] += panel_clauses["agreement_above_iid_u32"]
        clauses["regret"] += panel_clauses["validation_regret_below_iid_u32"]
        clauses["test"] += panel_clauses["selected_quotient_test_loss_below_iid_u32"]
        summary["panels"][panel] = {
            "clauses": panel_clauses,
            "means": means.reset_index().to_dict(orient="records"),
        }
    summary["panels_passing_by_clause"] = clauses
    summary["frozen_primary_gate_passed"] = bool(
        clauses["agreement"] >= 4 and clauses["regret"] >= 4 and clauses["test"] >= 3
    )
    (RESULTS / "cross_quotient_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
