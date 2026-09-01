"""Independent pilot screening followed by fresh cover cross-scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
KEEP = 2


def cover_pair_ids(
    shape: tuple[int, ...], panel: str, dataset: str, stage: str
) -> tuple[np.ndarray, np.ndarray]:
    left = RMS.action_ids(
        shape, RMS.stable_seed("screen-cross", stage, "left", panel, dataset)
    )["strength2"][:DRAWS]
    right = RMS.action_ids(
        shape, RMS.stable_seed("screen-cross", stage, "right", panel, dataset)
    )["strength2"][:DRAWS]
    return left, right


def analyze_dataset(
    panel: str, dataset: str, models: list[str], directory: Path
) -> list[dict[str, object]]:
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    validation_y = test_y = None
    shape: tuple[int, ...] | None = None
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

    pilot_ids = cover_pair_ids(shape, panel, dataset, "pilot")
    deploy_ids = cover_pair_ids(shape, panel, dataset, "deployment")
    pilot_scores, deploy_scores, deploy_test_losses = [], [], []
    for val_flat, test_flat in zip(validation, test):
        pilot, _ = CQS.cross_and_mean_scores(validation_y, val_flat, *pilot_ids)
        deploy, _ = CQS.cross_and_mean_scores(validation_y, val_flat, *deploy_ids)
        _, test_loss = CQS.cross_and_mean_scores(test_y, test_flat, *deploy_ids)
        pilot_scores.append(pilot)
        deploy_scores.append(deploy)
        deploy_test_losses.append(test_loss)
    pilot_matrix = np.stack(pilot_scores, axis=1)
    deploy_matrix = np.stack(deploy_scores, axis=1)
    deploy_test_matrix = np.stack(deploy_test_losses, axis=1)
    quotient_val = np.asarray([
        proper_loss(validation_y, prediction.mean(axis=0))
        for prediction in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, prediction.mean(axis=0)) for prediction in test
    ])
    winner = int(np.argmin(quotient_val))

    active = np.argpartition(pilot_matrix, kth=KEEP - 1, axis=1)[:, :KEEP]
    active_deploy = np.take_along_axis(deploy_matrix, active, axis=1)
    chosen_position = np.argmin(active_deploy, axis=1)
    selected = active[np.arange(DRAWS), chosen_position]
    winner_included = np.any(active == winner, axis=1)
    total_fits = 32 * len(models) + 32 * KEEP
    baseline_fits = 64 * len(models)

    rows: list[dict[str, object]] = []
    for draw, chosen in enumerate(selected):
        rows.append({
            "panel": panel, "dataset": dataset, "task": task, "draw": draw,
            "candidates": len(models), "retained": KEEP,
            "total_fits": total_fits, "equal64_fits": baseline_fits,
            "fit_saving_fraction": 1 - total_fits / baseline_fits,
            "pilot_winner_inclusion": bool(winner_included[draw]),
            "selection_agreement": bool(chosen == winner),
            "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
            "selected_quotient_test_loss": float(quotient_test[chosen]),
            "selected_realized_test_loss": float(deploy_test_matrix[draw, chosen]),
            "selected_model": models[int(chosen)],
        })
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "screen_then_cross_draws.csv", index=False)
    cells = frame.groupby(
        ["panel", "dataset", "task", "candidates"], as_index=False
    ).agg(
        pilot_winner_inclusion=("pilot_winner_inclusion", "mean"),
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        fit_saving_fraction=("fit_saving_fraction", "first"),
        total_fits=("total_fits", "first"),
        equal64_fits=("equal64_fits", "first"),
    )
    cells.to_csv(RESULTS / "screen_then_cross_cells.csv", index=False)

    equal32 = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    equal32 = equal32[equal32.method == "strength2_cross32"]
    equal64 = pd.read_csv(RESULTS / "cross_score_budget_frontier_cells.csv")
    equal64 = equal64[equal64.method == "cover_block_u64"]
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    counts = {"saving": 0, "inclusion": 0, "agreement32": 0, "regret32": 0, "regret64": 0}
    for panel, current in cells.groupby("panel"):
        proposed = current.mean(numeric_only=True)
        base32 = equal32[equal32.panel == panel].mean(numeric_only=True)
        base64 = equal64[equal64.panel == panel].mean(numeric_only=True)
        clauses = {
            "saving_at_least_15_percent": bool(current.fit_saving_fraction.min() >= .15),
            "pilot_inclusion_at_least_98_percent": bool(proposed.pilot_winner_inclusion >= .98),
            "agreement_at_least_equal32": bool(proposed.selection_agreement >= base32.selection_agreement),
            "regret_at_most_equal32": bool(proposed.validation_quotient_regret <= base32.validation_quotient_regret),
            "regret_within_25_percent_equal64": bool(
                proposed.validation_quotient_regret <= 1.25 * base64.validation_quotient_regret + 1e-15
            ),
        }
        for name, value in clauses.items():
            key = {"saving_at_least_15_percent": "saving",
                   "pilot_inclusion_at_least_98_percent": "inclusion",
                   "agreement_at_least_equal32": "agreement32",
                   "regret_at_most_equal32": "regret32",
                   "regret_within_25_percent_equal64": "regret64"}[name]
            counts[key] += int(value)
        summary["panels"][panel] = {
            "clauses": clauses,
            "screen_then_cross": proposed.to_dict(),
            "equal32": base32.to_dict(), "equal64": base64.to_dict(),
        }
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(
        counts["saving"] == 5 and counts["inclusion"] >= 4
        and counts["agreement32"] >= 4 and counts["regret32"] >= 4
        and counts["regret64"] >= 4
    )
    (RESULTS / "screen_then_cross_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
