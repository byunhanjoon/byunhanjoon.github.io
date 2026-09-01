"""Single-cover screening followed by independent four-cover U scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_cross_score_budget_frontier import cover_block_scores
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
KEEP = 2


def action_ids(shape: tuple[int, ...], panel: str, dataset: str):
    pilot = RMS.action_ids(
        shape, RMS.stable_seed("cheap-screen", panel, dataset)
    )["strength2"][:DRAWS]
    blocks = np.stack([
        RMS.action_ids(
            shape, RMS.stable_seed("precise-deploy", panel, dataset, str(block))
        )["strength2"][:DRAWS]
        for block in range(4)
    ], axis=1)
    return pilot, blocks


def analyze_dataset(
    panel: str, dataset: str, models: list[str], directory: Path
) -> list[dict[str, object]]:
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
    pilot_ids, deploy_ids = action_ids(shape, panel, dataset)

    pilot_scores, deployment_scores, test_losses = [], [], []
    for val_flat, test_flat in zip(validation, test):
        pilot_scores.append(RMS.batched_losses(
            validation_y, val_flat, pilot_ids, batch=8
        ))
        deployment, _ = cover_block_scores(validation_y, val_flat, deploy_ids)
        _, test_loss = cover_block_scores(test_y, test_flat, deploy_ids)
        deployment_scores.append(deployment)
        test_losses.append(test_loss)
    pilot_matrix = np.stack(pilot_scores, axis=1)
    deployment_matrix = np.stack(deployment_scores, axis=1)
    test_matrix = np.stack(test_losses, axis=1)
    quotient_val = np.asarray([
        proper_loss(validation_y, prediction.mean(axis=0)) for prediction in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, prediction.mean(axis=0)) for prediction in test
    ])
    winner = int(np.argmin(quotient_val))

    active = np.argpartition(pilot_matrix, kth=KEEP - 1, axis=1)[:, :KEEP]
    allocated = active[np.arange(DRAWS), np.argmin(
        np.take_along_axis(deployment_matrix, active, axis=1), axis=1
    )]
    paired = np.argmin(deployment_matrix, axis=1)
    included = np.any(active == winner, axis=1)
    total_fits = 16 * len(models) + 64 * KEEP
    equal_fits = 64 * len(models)

    rows: list[dict[str, object]] = []
    for draw in range(DRAWS):
        for method, chosen in (
            ("cheap_screen_precise_deploy", allocated[draw]),
            ("paired_all_candidate_u64", paired[draw]),
        ):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "draw": draw, "method": method, "candidates": len(models),
                "pilot_winner_inclusion": bool(included[draw]),
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(test_matrix[draw, chosen]),
                "charged_fits": total_fits if method == "cheap_screen_precise_deploy" else equal_fits,
                "equal64_fits": equal_fits,
                "fit_saving_fraction": 1 - total_fits / equal_fits if method == "cheap_screen_precise_deploy" else 0.0,
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
    frame.to_csv(RESULTS / "cheap_screen_precise_deploy_draws.csv", index=False)
    cells = frame.groupby(
        ["panel", "dataset", "task", "method", "candidates"], as_index=False
    ).agg(
        pilot_winner_inclusion=("pilot_winner_inclusion", "mean"),
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        charged_fits=("charged_fits", "first"),
        equal64_fits=("equal64_fits", "first"),
        fit_saving_fraction=("fit_saving_fraction", "first"),
    )
    cells.to_csv(RESULTS / "cheap_screen_precise_deploy_cells.csv", index=False)

    summary: dict[str, object] = {"status": "complete", "panels": {}}
    counts = {"saving": 0, "inclusion": 0, "agreement": 0, "regret": 0}
    for panel, current in cells.groupby("panel"):
        proposal = current[current.method == "cheap_screen_precise_deploy"]
        control = current[current.method == "paired_all_candidate_u64"]
        pmean = proposal.mean(numeric_only=True)
        cmean = control.mean(numeric_only=True)
        clauses = {
            "saving_at_least_20_percent": bool(proposal.fit_saving_fraction.min() >= .20),
            "pilot_inclusion_at_least_98_percent": bool(pmean.pilot_winner_inclusion >= .98),
            "agreement_at_least_paired_u64": bool(pmean.selection_agreement >= cmean.selection_agreement),
            "regret_at_most_paired_u64": bool(pmean.validation_quotient_regret <= cmean.validation_quotient_regret),
        }
        counts["saving"] += int(clauses["saving_at_least_20_percent"])
        counts["inclusion"] += int(clauses["pilot_inclusion_at_least_98_percent"])
        counts["agreement"] += int(clauses["agreement_at_least_paired_u64"])
        counts["regret"] += int(clauses["regret_at_most_paired_u64"])
        summary["panels"][panel] = {
            "clauses": clauses, "proposed": pmean.to_dict(),
            "paired_all_candidate_u64": cmean.to_dict(),
        }
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(
        counts["saving"] >= 4 and counts["inclusion"] >= 4
        and counts["agreement"] >= 4 and counts["regret"] >= 4
    )
    (RESULTS / "cheap_screen_precise_deploy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
