"""Four-cover U-statistic versus IID-U at 64 fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_cross_quotient_selection import PANELS, iid_u_scores, residuals
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
BATCH = 8
BLOCKS = 4
METHODS = ("cover_block_u64", "iid_u64")


def cover_block_scores(
    y: np.ndarray, flat: np.ndarray, block_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(block_ids), dtype=np.float64)
    mean_losses = np.empty(len(block_ids), dtype=np.float64)
    for start in range(0, len(block_ids), BATCH):
        stop = min(start + BATCH, len(block_ids))
        block_predictions = np.stack([
            flat[block_ids[start:stop, block]].mean(axis=1)
            for block in range(BLOCKS)
        ], axis=1)
        block_residuals = np.stack([
            residuals(y, block_predictions[:, block]) for block in range(BLOCKS)
        ], axis=1)
        summed = block_residuals.sum(axis=1)
        numerator = np.sum(summed ** 2, axis=-1) - np.sum(
            block_residuals ** 2, axis=(1, 3)
        )
        scores[start:stop] = np.mean(
            numerator / (BLOCKS * (BLOCKS - 1)), axis=1
        )
        mean_residual = residuals(y, block_predictions.mean(axis=1))
        mean_losses[start:stop] = np.mean(
            np.sum(mean_residual ** 2, axis=-1), axis=1
        )
    return scores, mean_losses


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
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
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert val_y is not None and test_y is not None and shape is not None
    actions = [
        RMS.action_ids(
            shape, RMS.stable_seed("block-u64", panel, dataset, str(block))
        )
        for block in range(BLOCKS)
    ]
    cover_ids = np.stack([
        action["strength2"][:DRAWS] for action in actions
    ], axis=1)
    iid_ids = np.concatenate([
        action["iid16"][:DRAWS] for action in actions
    ], axis=1)
    quotient_val = np.asarray([
        proper_loss(val_y, values.mean(axis=0)) for values in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, values.mean(axis=0)) for values in test
    ])
    winner = int(np.argmin(quotient_val))
    val_scores = {method: [] for method in METHODS}
    test_scores = {method: [] for method in METHODS}
    calibration = []
    for model, val_flat, test_flat, exact in zip(
        models, validation, test, quotient_val
    ):
        cover_val, _ = cover_block_scores(val_y, val_flat, cover_ids)
        iid_val = iid_u_scores(val_y, val_flat, iid_ids)
        _, cover_test = cover_block_scores(test_y, test_flat, cover_ids)
        iid_test = RMS.batched_losses(test_y, test_flat, iid_ids, batch=BATCH)
        for method, values in (
            ("cover_block_u64", cover_val), ("iid_u64", iid_val)
        ):
            val_scores[method].append(values)
            bias = float(values.mean() - exact)
            variance = float(values.var(ddof=1))
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task,
                "model": model, "method": method, "score_bias": bias,
                "score_rmse": float(np.sqrt(variance + bias ** 2)),
                "mc_standard_error": float(np.sqrt(variance / DRAWS)),
            })
        test_scores["cover_block_u64"].append(cover_test)
        test_scores["iid_u64"].append(iid_test)
    rows = []
    for method in METHODS:
        scores = np.stack(val_scores[method], axis=1)
        realized = np.stack(test_scores[method], axis=1)
        selected = np.argmin(scores, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(
                    quotient_val[chosen] - quotient_val[winner]
                ),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(realized[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration_rows = [], []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current_rows, current_calibration = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current_rows)
            calibration_rows.extend(current_calibration)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "cross_score_budget_frontier_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "cross_score_budget_frontier_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(
        RESULTS / "cross_score_budget_frontier_calibration.csv", index=False
    )
    primary = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    primary = primary[primary.method == "strength2_cross32"]
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    clauses = {"rmse": 0, "agreement": 0, "regret": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        rmses = calibration[
            calibration.panel == panel
        ].groupby("method").score_rmse.mean()
        earlier = primary[primary.panel == panel].mean(numeric_only=True)
        panel_clauses = {
            "cover_rmse_below_iid_u64": bool(
                rmses.cover_block_u64 < rmses.iid_u64
            ),
            "cover_agreement_nondecreasing_from_32": bool(
                means.loc["cover_block_u64", "selection_agreement"]
                >= earlier.selection_agreement
            ),
            "cover_regret_nonincreasing_from_32": bool(
                means.loc["cover_block_u64", "validation_quotient_regret"]
                <= earlier.validation_quotient_regret
            ),
        }
        clauses["rmse"] += panel_clauses["cover_rmse_below_iid_u64"]
        clauses["agreement"] += panel_clauses[
            "cover_agreement_nondecreasing_from_32"
        ]
        clauses["regret"] += panel_clauses[
            "cover_regret_nonincreasing_from_32"
        ]
        summary["panels"][panel] = {
            "clauses": panel_clauses,
            "score_rmse": {
                method: float(value) for method, value in rmses.items()
            },
            "method_means": means.reset_index().to_dict(orient="records"),
            "strength2_cross32_means": {
                "selection_agreement": float(earlier.selection_agreement),
                "validation_quotient_regret": float(
                    earlier.validation_quotient_regret
                ),
            },
        }
    summary["panels_passing_by_clause"] = clauses
    summary["frontier_gate_passed"] = bool(
        clauses["rmse"] >= 4
        and clauses["agreement"] >= 4
        and clauses["regret"] >= 4
    )
    (RESULTS / "cross_score_budget_frontier_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
