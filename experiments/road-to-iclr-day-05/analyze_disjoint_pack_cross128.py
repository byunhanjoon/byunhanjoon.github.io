"""Unbiased 128-fit score: cross two packs versus eight-cover U-statistic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_disjoint_pair_cross import cover_graph
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
BATCH = 4
METHODS = ("disjoint_pack_cross128", "independent_cover_u128")


def action_blocks(shape: tuple[int, ...], panel: str, dataset: str):
    first, _, _ = sample_pack_and_pairs(shape, panel + "-cross128-a", dataset)
    second, _, _ = sample_pack_and_pairs(shape, panel + "-cross128-b", dataset)
    ids = cover_graph(shape)[0]
    rng = np.random.default_rng(RMS.stable_seed("independent-cover-u128", panel, dataset))
    independent = ids[rng.integers(0, len(ids), size=(DRAWS, 8))]
    return first[:DRAWS], second[:DRAWS], independent


def block_u_scores(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(blocks), dtype=np.float64)
    mean_losses = np.empty(len(blocks), dtype=np.float64)
    count = blocks.shape[1]
    for start in range(0, len(blocks), BATCH):
        stop = min(start + BATCH, len(blocks))
        predictions = np.stack([
            flat[blocks[start:stop, block]].mean(axis=1) for block in range(count)
        ], axis=1)
        residuals = np.stack([
            CQS.residuals(y, predictions[:, block]) for block in range(count)
        ], axis=1)
        summed = residuals.sum(axis=1)
        numerator = np.sum(summed ** 2, axis=-1) - np.sum(residuals ** 2, axis=(1, 3))
        scores[start:stop] = np.mean(numerator / (count * (count - 1)), axis=1)
        mean_residual = CQS.residuals(y, predictions.mean(axis=1))
        mean_losses[start:stop] = np.mean(np.sum(mean_residual ** 2, axis=-1), axis=1)
    return scores, mean_losses


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    val_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        val_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        task = str(manifest["task"])
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert val_y is not None and test_y is not None and shape is not None
    first, second, independent = action_blocks(shape, panel, dataset)
    left, right = first.reshape(DRAWS, -1), second.reshape(DRAWS, -1)
    exact_val = np.asarray([proper_loss(val_y, values.mean(axis=0)) for values in validation])
    exact_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(exact_val))
    scores = {method: [] for method in METHODS}
    test_losses = {method: [] for method in METHODS}
    calibration = []
    for model, val_flat, test_flat, target in zip(models, validation, test, exact_val):
        pack_score, _ = CQS.cross_and_mean_scores(val_y, val_flat, left, right)
        control_score, _ = block_u_scores(val_y, val_flat, independent)
        _, pack_test = CQS.cross_and_mean_scores(test_y, test_flat, left, right)
        _, control_test = block_u_scores(test_y, test_flat, independent)
        for method, values, realized in (
            (METHODS[0], pack_score, pack_test), (METHODS[1], control_score, control_test)
        ):
            scores[method].append(values); test_losses[method].append(realized)
            bias = float(values.mean() - target)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "product_cells": int(np.prod(shape)),
                "score_bias": bias,
                "score_rmse": float(np.sqrt(np.mean((values - target) ** 2))),
                "max_absolute_score_error": float(np.max(np.abs(values - target))),
            })
    rows = []
    for method in METHODS:
        matrix = np.stack(scores[method], axis=1)
        realized = np.stack(test_losses[method], axis=1)
        selected = np.argmin(matrix, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task, "method": method,
                "draw": draw, "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(exact_val[chosen] - exact_val[winner]),
                "selected_quotient_test_loss": float(exact_test[chosen]),
                "selected_realized_test_loss": float(realized[draw, chosen]),
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
    draws.to_csv(RESULTS / "disjoint_pack_cross128_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_pack_cross128_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "disjoint_pack_cross128_calibration.csv", index=False)

    counts = {"rmse": 0, "agreement": 0, "regret": 0}
    panels = {}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        rmses = calibration[calibration.panel == panel].groupby("method").score_rmse.mean()
        clauses = {
            "score_rmse_lower": bool(rmses[METHODS[0]] < rmses[METHODS[1]]),
            "agreement_nolower": bool(means.loc[METHODS[0], "selection_agreement"] >= means.loc[METHODS[1], "selection_agreement"]),
            "regret_nohigher": bool(means.loc[METHODS[0], "validation_quotient_regret"] <= means.loc[METHODS[1], "validation_quotient_regret"]),
        }
        counts["rmse"] += int(clauses["score_rmse_lower"])
        counts["agreement"] += int(clauses["agreement_nolower"])
        counts["regret"] += int(clauses["regret_nohigher"])
        panels[panel] = {"clauses": clauses, "score_rmse": rmses.to_dict(),
                         "method_means": means.reset_index().to_dict(orient="records")}
    full = calibration[calibration.product_cells == 128].pivot(
        index=["panel", "dataset", "model"], columns="method", values="score_rmse"
    )
    exact = calibration[(calibration.method == METHODS[0]) & (calibration.product_cells <= 64)]
    mean_bias = calibration.groupby("method").score_bias.mean().to_dict()
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS, "panels": panels,
        "panels_passing_by_clause": counts,
        "full_product_candidates": int(len(full)),
        "full_product_rmse_wins": int((full[METHODS[0]] < full[METHODS[1]]).sum()),
        "exact_partition_candidates": int(len(exact)),
        "exact_partition_max_absolute_error": float(exact.max_absolute_score_error.max()),
        "overall_mean_score_bias": mean_bias,
    }
    summary["frozen_gate_passed"] = bool(
        counts["rmse"] == 5 and summary["full_product_rmse_wins"] >= 20
        and counts["agreement"] >= 4 and counts["regret"] >= 4
        and all(abs(value) < 1e-5 for value in mean_bias.values())
        and summary["exact_partition_max_absolute_error"] < 1e-12
    )
    (RESULTS / "disjoint_pack_cross128_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
