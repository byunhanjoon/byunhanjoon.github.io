"""Validation-only adaptive stopping on literal nested nuisance covers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_anytime_nested_cover import nested_family
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512


def flat_ids(designs: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.ravel_multi_index(np.moveaxis(designs, -1, 0), shape)


def losses(y: np.ndarray, candidates: list[np.ndarray], ids: np.ndarray) -> np.ndarray:
    return np.stack([RMS.batched_losses(y, predictions, ids) for predictions in candidates], axis=1)


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path) -> list[dict[str, object]]:
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    shape = None
    validation_y = test_y = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        current_shape = tuple(map(int, archive["validation_predictions"].shape[:4]))
        if shape is not None and current_shape != shape:
            raise AssertionError("candidate factor shapes differ")
        shape = current_shape
        validation_y = archive["validation_y"] if validation_y is None else validation_y
        test_y = archive["test_y"] if test_y is None else test_y
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and validation_y is not None and test_y is not None

    quotient_validation = np.asarray([proper_loss(validation_y, values.mean(axis=0)) for values in validation])
    quotient_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    quotient_winner = int(np.argmin(quotient_validation))
    rng = np.random.default_rng(RMS.stable_seed("adaptive_nested", panel, dataset))
    family = nested_family(shape[1], shape[2],)
    schedules = family[rng.integers(0, len(family), size=DRAWS)]
    ids64 = flat_ids(schedules, shape)

    fixed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for budget in (4, 16, 64):
        ids = ids64[:, :budget]
        validation_losses = losses(validation_y, validation, ids)
        test_losses = losses(test_y, test, ids)
        selected = np.argmin(validation_losses, axis=1)
        fixed[budget] = selected, validation_losses, test_losses

    loo_winners = []
    for omitted in range(4):
        keep = np.asarray([index for index in range(4) if index != omitted])
        loo_winners.append(np.argmin(losses(validation_y, validation, ids64[:, keep]), axis=1))
    loo_winners = np.stack(loo_winners, axis=1)
    stop4 = np.all(loo_winners == loo_winners[:, :1], axis=1)

    block_winners = []
    for block in range(4):
        block_winners.append(np.argmin(
            losses(validation_y, validation, ids64[:, 4 * block:4 * (block + 1)]), axis=1
        ))
    block_winners = np.stack(block_winners, axis=1)
    stop16 = (~stop4) & np.all(block_winners == block_winners[:, :1], axis=1)
    budgets = np.where(stop4, 4, np.where(stop16, 16, 64))
    selected = np.where(stop4, fixed[4][0], np.where(stop16, fixed[16][0], fixed[64][0]))
    realized_test = np.empty(DRAWS)
    for budget in (4, 16, 64):
        mask = budgets == budget
        realized_test[mask] = fixed[budget][2][np.arange(DRAWS)[mask], selected[mask]]

    rows: list[dict[str, object]] = []
    for budget in (4, 16, 64):
        current_selected, _, current_test = fixed[budget]
        rows.append({
            "panel": panel, "dataset": dataset, "method": f"fixed_nested_b{budget}",
            "mean_budget": float(budget), "stop_at_4": float(budget == 4),
            "stop_at_16": float(budget == 16), "stop_at_64": float(budget == 64),
            "selection_agreement": float(np.mean(current_selected == quotient_winner)),
            "validation_quotient_regret": float(np.mean(quotient_validation[current_selected] - quotient_validation[quotient_winner])),
            "selected_quotient_test_loss": float(np.mean(quotient_test[current_selected])),
            "selected_realized_test_loss": float(np.mean(current_test[np.arange(DRAWS), current_selected])),
        })
    rows.append({
        "panel": panel, "dataset": dataset, "method": "adaptive_nested",
        "mean_budget": float(budgets.mean()), "stop_at_4": float(np.mean(budgets == 4)),
        "stop_at_16": float(np.mean(budgets == 16)), "stop_at_64": float(np.mean(budgets == 64)),
        "selection_agreement": float(np.mean(selected == quotient_winner)),
        "validation_quotient_regret": float(np.mean(quotient_validation[selected] - quotient_validation[quotient_winner])),
        "selected_quotient_test_loss": float(np.mean(quotient_test[selected])),
        "selected_realized_test_loss": float(realized_test.mean()),
    })
    conservative_stop16 = fixed[4][0] == fixed[16][0]
    conservative_budgets = np.where(conservative_stop16, 16, 64)
    conservative_selected = np.where(conservative_stop16, fixed[16][0], fixed[64][0])
    conservative_realized = np.where(
        conservative_stop16,
        fixed[16][2][np.arange(DRAWS), conservative_selected],
        fixed[64][2][np.arange(DRAWS), conservative_selected],
    )
    rows.append({
        "panel": panel, "dataset": dataset,
        "method": "conservative_adaptive_postfailure",
        "mean_budget": float(conservative_budgets.mean()), "stop_at_4": 0.0,
        "stop_at_16": float(conservative_stop16.mean()),
        "stop_at_64": float((~conservative_stop16).mean()),
        "selection_agreement": float(np.mean(conservative_selected == quotient_winner)),
        "validation_quotient_regret": float(np.mean(
            quotient_validation[conservative_selected] - quotient_validation[quotient_winner]
        )),
        "selected_quotient_test_loss": float(np.mean(quotient_test[conservative_selected])),
        "selected_realized_test_loss": float(conservative_realized.mean()),
    })
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    for panel, config_name, directory in RMS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "adaptive_selection_schedule_cells.csv", index=False)
    means = frame.groupby(["panel", "method"], as_index=False).mean(numeric_only=True)
    means.to_csv(RESULTS / "adaptive_selection_schedule_means.csv", index=False)
    panels: dict[str, object] = {}
    for panel, current in means.groupby("panel"):
        indexed = current.set_index("method")
        adaptive = indexed.loc["adaptive_nested"]
        conservative = indexed.loc["conservative_adaptive_postfailure"]
        fixed16 = indexed.loc["fixed_nested_b16"]
        clauses = {
            "mean_budget_below_64": bool(adaptive.mean_budget < 64),
            "agreement_at_least_fixed16": bool(adaptive.selection_agreement >= fixed16.selection_agreement),
            "regret_no_larger_than_fixed16": bool(adaptive.validation_quotient_regret <= fixed16.validation_quotient_regret),
        }
        panels[panel] = {
            "passed": bool(all(clauses.values())), "clauses": clauses,
            "postfailure_conservative_clauses": {
                "mean_budget_below_64": bool(conservative.mean_budget < 64),
                "agreement_at_least_fixed16": bool(
                    conservative.selection_agreement >= fixed16.selection_agreement
                ),
                "regret_no_larger_than_fixed16": bool(
                    conservative.validation_quotient_regret <= fixed16.validation_quotient_regret
                ),
            },
            "method_means": indexed.reset_index().to_dict(orient="records"),
        }
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "exploratory_posthoc": True, "panels": panels,
        "panels_passing_exploratory_gate": int(sum(value["passed"] for value in panels.values())),
    }
    (RESULTS / "adaptive_selection_schedule_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
