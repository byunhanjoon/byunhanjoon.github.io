"""Equal-budget packed versus independent two-cover prediction means."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_disjoint_pair_cross import DRAWS, cover_graph
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def prediction_residuals(
    flat: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    quotient = flat.mean(axis=0)
    output = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), 16):
        stop = min(start + 16, len(left))
        prediction = (
            flat[left[start:stop]].mean(axis=1)
            + flat[right[start:stop]].mean(axis=1)
        ) / 2
        output[start:stop] = np.mean(
            np.sum((prediction - quotient) ** 2, axis=-1), axis=1
        )
    return output


def paired_ids(shape: tuple[int, ...], panel: str, dataset: str):
    ids, _, _, neighbors = cover_graph(shape)
    rng = np.random.default_rng(RMS.stable_seed("disjoint-pair32", panel, dataset))
    first = rng.integers(0, len(ids), size=DRAWS)
    independent = rng.integers(0, len(ids), size=DRAWS)
    if len(ids) == 1:
        disjoint = first
    else:
        positions = rng.integers(0, len(neighbors[0]), size=DRAWS)
        disjoint = np.fromiter(
            (neighbors[index][position] for index, position in zip(first, positions)),
            dtype=int, count=DRAWS,
        )
    return (ids[first], ids[disjoint]), (ids[first], ids[independent])


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        task = manifest["task"]
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert validation_y is not None and test_y is not None and shape is not None
    packed_ids, independent_ids = paired_ids(shape, panel, dataset)
    quotient_val = np.asarray([
        proper_loss(validation_y, values.mean(axis=0)) for values in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, values.mean(axis=0)) for values in test
    ])
    winner = int(np.argmin(quotient_val))
    methods = ("disjoint_pair_mean32", "independent_pair_mean32")
    scores = {method: [] for method in methods}
    test_losses = {method: [] for method in methods}
    calibration = []
    for model, val_flat, test_flat, exact in zip(models, validation, test, quotient_val):
        current = []
        for method, action in zip(methods, (packed_ids, independent_ids)):
            _, val_loss = CQS.cross_and_mean_scores(validation_y, val_flat, *action)
            _, test_loss = CQS.cross_and_mean_scores(test_y, test_flat, *action)
            residual = prediction_residuals(val_flat, *action)
            scores[method].append(val_loss); test_losses[method].append(test_loss)
            bias = float(val_loss.mean() - exact)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "product_cells": int(np.prod(shape)),
                "score_bias": bias,
                "score_rmse": float(np.sqrt(val_loss.var(ddof=1) + bias ** 2)),
                "prediction_residual": float(residual.mean()),
                "max_absolute_score_error": float(np.max(np.abs(val_loss - exact))),
            })
    rows = []
    for method in methods:
        score_matrix = np.stack(scores[method], axis=1)
        test_matrix = np.stack(test_losses[method], axis=1)
        selected = np.argmin(score_matrix, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(test_matrix[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration_rows = [], []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current, calibration = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current); calibration_rows.extend(calibration)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "disjoint_pair32_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_pair32_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "disjoint_pair32_calibration.csv", index=False)

    exact = calibration[
        (calibration.method == "disjoint_pair_mean32") &
        (calibration.product_cells <= 32)
    ]
    summary: dict[str, object] = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "exact_partition_candidates": len(exact),
        "exact_partition_max_absolute_error": float(exact.max_absolute_score_error.max()),
        "panels": {},
    }
    counts = {"rmse": 0, "residual": 0, "agreement": 0, "regret": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cal = calibration[calibration.panel == panel].groupby("method").mean(numeric_only=True)
        packed, control = means.loc["disjoint_pair_mean32"], means.loc["independent_pair_mean32"]
        clauses = {
            "score_rmse_lower": bool(cal.loc["disjoint_pair_mean32", "score_rmse"] < cal.loc["independent_pair_mean32", "score_rmse"]),
            "prediction_residual_lower": bool(cal.loc["disjoint_pair_mean32", "prediction_residual"] < cal.loc["independent_pair_mean32", "prediction_residual"]),
            "agreement_nolower": bool(packed.selection_agreement >= control.selection_agreement),
            "regret_nohigher": bool(packed.validation_quotient_regret <= control.validation_quotient_regret),
        }
        for name, value in clauses.items():
            counts[{"score_rmse_lower": "rmse", "prediction_residual_lower": "residual",
                    "agreement_nolower": "agreement", "regret_nohigher": "regret"}[name]] += int(value)
        summary["panels"][panel] = {
            "clauses": clauses,
            "score_rmse": cal.score_rmse.to_dict(),
            "prediction_residual": cal.prediction_residual.to_dict(),
            "method_means": means.reset_index().to_dict(orient="records"),
        }
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(
        counts["rmse"] == 5 and counts["residual"] == 5
        and counts["agreement"] >= 4 and counts["regret"] >= 4
        and summary["exact_partition_max_absolute_error"] < 1e-12
    )
    (RESULTS / "disjoint_pair32_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
