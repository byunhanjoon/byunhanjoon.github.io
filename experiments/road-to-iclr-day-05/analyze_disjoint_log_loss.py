"""Scope test for disjoint cover packing under nonlinear log loss."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_disjoint_pair_cross import DRAWS, cover_graph
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_log_quotient_jackknife import EPS, PANELS, log_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BATCH = 8


def sample_pair_actions(shape: tuple[int, ...], panel: str, dataset: str):
    ids, _, _, neighbors = cover_graph(shape)
    rng = np.random.default_rng(RMS.stable_seed("disjoint-log-pair32", panel, dataset))
    first = rng.integers(0, len(ids), size=DRAWS)
    if len(ids) == 1:
        second = first.copy()
    else:
        second = np.fromiter(
            (neighbors[index][rng.integers(0, len(neighbors[index]))] for index in first),
            dtype=int, count=DRAWS,
        )
    disjoint = ids[np.stack((first, second), axis=1)]
    independent = ids[rng.integers(0, len(ids), size=(DRAWS, 2))]
    return disjoint, independent


def packed_log_scores(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    output = np.empty(len(blocks), dtype=np.float64)
    for start in range(0, len(blocks), BATCH):
        stop = min(start + BATCH, len(blocks))
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1)
            for block in range(blocks.shape[1])
        ], axis=1), axis=1)
        output[start:stop] = log_loss(y, prediction)
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
        shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert validation_y is not None and test_y is not None and shape is not None

    disjoint, independent = sample_pair_actions(shape, panel, dataset)
    pack, two_pairs, _ = sample_pack_and_pairs(shape, panel + "-log", dataset)
    actions = {
        "disjoint_pair32": disjoint,
        "independent_pair32": independent,
        "mutually_disjoint_pack64": pack,
        "two_disjoint_pairs64": two_pairs,
    }
    exact_val = np.asarray([log_loss(validation_y, values.mean(axis=0)) for values in validation])
    exact_test = np.asarray([log_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(exact_val))
    score_lists = {method: [] for method in actions}
    test_lists = {method: [] for method in actions}
    calibration = []
    product_cells = int(np.prod(shape))
    for model, val_flat, test_flat, target in zip(models, validation, test, exact_val):
        for method, blocks in actions.items():
            values = packed_log_scores(validation_y, val_flat, blocks)
            test_values = packed_log_scores(test_y, test_flat, blocks)
            score_lists[method].append(values)
            test_lists[method].append(test_values)
            bias = float(values.mean() - target)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "product_cells": product_cells,
                "score_bias": bias, "absolute_bias": abs(bias),
                "score_rmse": float(np.sqrt(np.mean((values - target) ** 2))),
                "max_absolute_score_error": float(np.max(np.abs(values - target))),
                "clip_epsilon": EPS,
            })

    rows = []
    for method in actions:
        matrix = np.stack(score_lists[method], axis=1)
        test_matrix = np.stack(test_lists[method], axis=1)
        selected = np.argmin(matrix, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_log_quotient_regret": float(exact_val[chosen] - exact_val[winner]),
                "selected_test_log_quotient_loss": float(exact_test[chosen]),
                "selected_realized_test_log_loss": float(test_matrix[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration_rows = [], []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current, calibration = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current); calibration_rows.extend(calibration)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "disjoint_log_loss_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_log_loss_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "disjoint_log_loss_calibration.csv", index=False)

    comparisons = {
        "pair32": ("disjoint_pair32", "independent_pair32", 32),
        "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64", 64),
    }
    summary: dict[str, object] = {"status": "complete", "draws_per_dataset": DRAWS, "comparisons": {}}
    for name, (action, control, budget) in comparisons.items():
        counts = {"rmse_lower": 0, "rmse_nohigher": 0, "bias_lower": 0,
                  "agreement_nolower": 0, "regret_nohigher": 0}
        panel_results = {}
        for panel, current in cells.groupby("panel"):
            means = current.groupby("method").mean(numeric_only=True)
            cal = calibration[calibration.panel == panel].groupby("method").mean(numeric_only=True)
            clauses = {
                "score_rmse_lower": bool(cal.loc[action, "score_rmse"] < cal.loc[control, "score_rmse"] - 1e-15),
                "score_rmse_nohigher": bool(cal.loc[action, "score_rmse"] <= cal.loc[control, "score_rmse"] + 1e-15),
                "absolute_bias_lower": bool(cal.loc[action, "absolute_bias"] < cal.loc[control, "absolute_bias"] - 1e-15),
                "agreement_nolower": bool(means.loc[action, "selection_agreement"] >= means.loc[control, "selection_agreement"]),
                "regret_nohigher": bool(means.loc[action, "validation_log_quotient_regret"] <= means.loc[control, "validation_log_quotient_regret"]),
            }
            for key, value in clauses.items():
                counts[{"score_rmse_lower": "rmse_lower", "score_rmse_nohigher": "rmse_nohigher",
                        "absolute_bias_lower": "bias_lower", "agreement_nolower": "agreement_nolower",
                        "regret_nohigher": "regret_nohigher"}[key]] += int(value)
            panel_results[panel] = {
                "clauses": clauses,
                "score_rmse": {action: float(cal.loc[action, "score_rmse"]), control: float(cal.loc[control, "score_rmse"])},
                "absolute_bias": {action: float(cal.loc[action, "absolute_bias"]), control: float(cal.loc[control, "absolute_bias"])},
                "method_means": means.loc[[action, control]].reset_index().to_dict(orient="records"),
            }
        exact = calibration[(calibration.method == action) & (calibration.product_cells <= budget)]
        max_error = float(exact.max_absolute_score_error.max())
        if name == "pair32":
            passed = counts["rmse_lower"] >= 5 and counts["regret_nohigher"] >= 4 and max_error < 1e-12
        else:
            passed = counts["rmse_nohigher"] == 6 and counts["rmse_lower"] >= 3 and counts["regret_nohigher"] >= 4 and max_error < 1e-12
        summary["comparisons"][name] = {
            "action": action, "control": control, "panels_passing_by_clause": counts,
            "exact_partition_candidates": int(len(exact)),
            "exact_partition_max_absolute_error": max_error,
            "frozen_gate_passed": bool(passed), "panels": panel_results,
        }
    (RESULTS / "disjoint_log_loss_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
