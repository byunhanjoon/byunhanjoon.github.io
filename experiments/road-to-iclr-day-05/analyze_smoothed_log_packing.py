"""Interior-supported log-loss packing sensitivity experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_disjoint_log_loss import sample_pair_actions
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_log_quotient_jackknife import PANELS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ALPHAS = (1e-6, 1e-4, 1e-2)
BATCH = 8


def smoothed_log_loss(y: np.ndarray, prediction: np.ndarray, alpha: float) -> np.ndarray:
    if prediction.ndim == 2:
        prediction = prediction[None]
        scalar = True
    else:
        scalar = False
    classes = prediction.shape[-1]
    smoothed = (1.0 - alpha) * prediction + alpha / classes
    values = -np.log(smoothed[:, np.arange(len(y)), y.astype(int)]).mean(axis=1)
    return values[0] if scalar else values


def packed_scores(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray) -> dict[float, np.ndarray]:
    output = {alpha: np.empty(len(blocks), dtype=np.float64) for alpha in ALPHAS}
    for start in range(0, len(blocks), BATCH):
        stop = min(start + BATCH, len(blocks))
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1)
            for block in range(blocks.shape[1])
        ], axis=1), axis=1)
        for alpha in ALPHAS:
            output[alpha][start:stop] = smoothed_log_loss(y, prediction, alpha)
    return output


def main() -> None:
    calibration_rows, selection_rows = [], []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            validation, models = [], []
            y = None
            shape = None
            for model in config["models"]:
                archive = np.load(directory / f"{dataset}__{model}.npz")
                manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
                if manifest["task"] not in {"binclass", "multiclass"}:
                    continue
                y = archive["validation_y"]
                predictions = archive["validation_predictions"]
                shape = tuple(int(value) for value in predictions.shape[:4])
                validation.append(predictions.reshape(
                    (-1,) + predictions.shape[-2:]
                ).astype(np.float64))
                models.append(model)
            if not models:
                continue
            assert y is not None and shape is not None
            disjoint, independent = sample_pair_actions(shape, panel, dataset)
            pack, pairs, _ = sample_pack_and_pairs(shape, panel + "-log", dataset)
            actions = {
                "disjoint_pair32": disjoint,
                "independent_pair32": independent,
                "mutually_disjoint_pack64": pack,
                "two_disjoint_pairs64": pairs,
            }
            exact = {
                alpha: np.asarray([
                    smoothed_log_loss(y, values.mean(axis=0), alpha)
                    for values in validation
                ])
                for alpha in ALPHAS
            }
            scores = {
                method: [packed_scores(y, values, blocks) for values in validation]
                for method, blocks in actions.items()
            }
            product_cells = int(np.prod(shape))
            for alpha in ALPHAS:
                winner = int(np.argmin(exact[alpha]))
                for method in actions:
                    matrix = np.stack([item[alpha] for item in scores[method]], axis=1)
                    selected = np.argmin(matrix, axis=1)
                    for index, model in enumerate(models):
                        error = matrix[:, index] - exact[alpha][index]
                        calibration_rows.append({
                            "panel": panel, "dataset": dataset, "model": model,
                            "method": method, "alpha": alpha,
                            "product_cells": product_cells,
                            "score_bias": float(error.mean()),
                            "absolute_bias": float(abs(error.mean())),
                            "score_rmse": float(np.sqrt(np.mean(error ** 2))),
                            "max_absolute_score_error": float(np.max(np.abs(error))),
                            "support_floor": float(alpha / validation[index].shape[-1]),
                        })
                    selection_rows.append({
                        "panel": panel, "dataset": dataset, "method": method,
                        "alpha": alpha,
                        "selection_agreement": float(np.mean(selected == winner)),
                        "validation_regret": float(np.mean(exact[alpha][selected] - exact[alpha][winner])),
                    })

    calibration = pd.DataFrame(calibration_rows)
    selection = pd.DataFrame(selection_rows)
    calibration.to_csv(RESULTS / "smoothed_log_packing_calibration.csv", index=False)
    selection.to_csv(RESULTS / "smoothed_log_packing_selection.csv", index=False)
    comparisons = {
        "pair32": ("disjoint_pair32", "independent_pair32", 32),
        "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64", 64),
    }
    summary: dict[str, object] = {"status": "complete", "alphas": ALPHAS, "comparisons": {}}
    all_pass = True
    for name, (action, control, budget) in comparisons.items():
        alpha_results = {}
        for alpha in ALPHAS:
            cal = calibration[calibration.alpha == alpha]
            sel = selection[selection.alpha == alpha]
            rmse_wins = regret_wins = 0
            panel_rows = {}
            for panel in sorted(cal.panel.unique()):
                cal_means = cal[cal.panel == panel].groupby("method").mean(numeric_only=True)
                sel_means = sel[sel.panel == panel].groupby("method").mean(numeric_only=True)
                rmse_strict = bool(cal_means.loc[action, "score_rmse"] < cal_means.loc[control, "score_rmse"] - 1e-15)
                regret_nohigher = bool(sel_means.loc[action, "validation_regret"] <= sel_means.loc[control, "validation_regret"] + 1e-15)
                rmse_wins += int(rmse_strict); regret_wins += int(regret_nohigher)
                panel_rows[panel] = {
                    "rmse_strict": rmse_strict,
                    "regret_nohigher": regret_nohigher,
                    "action_rmse": float(cal_means.loc[action, "score_rmse"]),
                    "control_rmse": float(cal_means.loc[control, "score_rmse"]),
                }
            exact = cal[(cal.method == action) & (cal.product_cells <= budget)]
            max_error = float(exact.max_absolute_score_error.max())
            represented = len(panel_rows)
            passed = bool(
                (rmse_wins >= 5 if name == "pair32" else rmse_wins >= 3)
                and (rmse_wins == represented if name == "pack64" else True)
                and regret_wins >= 4 and max_error < 1e-12
            )
            all_pass &= passed
            alpha_results[f"{alpha:g}"] = {
                "represented_panels": represented,
                "panels_with_strictly_lower_rmse": rmse_wins,
                "panels_with_nohigher_regret": regret_wins,
                "exact_partition_candidates": int(len(exact)),
                "exact_partition_max_absolute_error": max_error,
                "gate_passed": passed,
                "panels": panel_rows,
            }
        summary["comparisons"][name] = alpha_results
    summary["all_smoothing_levels_passed"] = bool(all_pass)
    summary["interpretation"] = "interior_supported_robustness_pass" if all_pass else "smoothing_boundary"
    (RESULTS / "smoothed_log_packing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
