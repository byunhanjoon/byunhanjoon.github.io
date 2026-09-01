"""Two-replicate unbiased-selector stability sets at a 64-fit budget."""

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
DRAWS = 512
METHODS = ("cover_cross_union64", "iid_u_union64")
PANELS = CQS.PANELS


def block_actions(shape: tuple[int, ...], panel: str, dataset: str):
    return [
        RMS.action_ids(shape, RMS.stable_seed("stability-set", panel, dataset, str(block)))
        for block in range(4)
    ]


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        if validation_y is not None and not np.array_equal(validation_y, archive["validation_y"]):
            raise AssertionError("validation labels differ")
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(int(x) for x in archive["validation_predictions"].shape[:4])
        task = str(manifest["task"])
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert validation_y is not None and test_y is not None and shape is not None
    actions = block_actions(shape, panel, dataset)
    score_replicates = {method: [[], []] for method in METHODS}
    quotient_validation = np.asarray([
        proper_loss(validation_y, candidate.mean(axis=0)) for candidate in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, candidate.mean(axis=0)) for candidate in test
    ])
    validation_winner = int(np.argmin(quotient_validation))
    test_winner = int(np.argmin(quotient_test))
    for candidate in validation:
        for replicate, (left_block, right_block) in enumerate(((0, 1), (2, 3))):
            cover, _ = CQS.cross_and_mean_scores(
                validation_y, candidate,
                actions[left_block]["strength2"][:DRAWS],
                actions[right_block]["strength2"][:DRAWS],
            )
            iid_ids = np.concatenate((
                actions[left_block]["iid16"][:DRAWS],
                actions[right_block]["iid16"][:DRAWS],
            ), axis=1)
            iid = CQS.iid_u_scores(validation_y, candidate, iid_ids)
            score_replicates["cover_cross_union64"][replicate].append(cover)
            score_replicates["iid_u_union64"][replicate].append(iid)
    rows = []
    for method in METHODS:
        first = np.argmin(np.stack(score_replicates[method][0], axis=1), axis=1)
        second = np.argmin(np.stack(score_replicates[method][1], axis=1), axis=1)
        for draw, (winner_a, winner_b) in enumerate(zip(first, second)):
            members = np.unique([int(winner_a), int(winner_b)])
            validation_regrets = quotient_validation[members] - quotient_validation[validation_winner]
            test_regrets = quotient_test[members] - quotient_test[test_winner]
            singleton = len(members) == 1
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "set_size": int(len(members)),
                "validation_winner_covered": bool(validation_winner in members),
                "test_winner_covered": bool(test_winner in members),
                "wrong_singleton": bool(singleton and validation_winner not in members),
                "correct_singleton": bool(singleton and validation_winner in members),
                "best_validation_regret": float(validation_regrets.min()),
                "worst_validation_regret": float(validation_regrets.max()),
                "best_test_regret": float(test_regrets.min()),
                "worst_test_regret": float(test_regrets.max()),
                "mean_member_test_loss": float(quotient_test[members].mean()),
            })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "stability_set_draws.csv", index=False)
    metrics = [
        "set_size", "validation_winner_covered", "test_winner_covered",
        "wrong_singleton", "correct_singleton", "best_validation_regret",
        "worst_validation_regret", "best_test_regret", "worst_test_regret",
        "mean_member_test_loss",
    ]
    cells = frame.groupby(["panel", "dataset", "task", "method"], as_index=False)[metrics].mean()
    cells.to_csv(RESULTS / "stability_set_cells.csv", index=False)
    clauses = {"coverage": 0, "size": 0, "wrong_singleton": 0}
    strict = {key: 0 for key in clauses}
    panels: dict[str, object] = {}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cover, iid = means.loc[METHODS[0]], means.loc[METHODS[1]]
        conditions = {
            "coverage": cover.validation_winner_covered >= iid.validation_winner_covered,
            "size": cover.set_size <= iid.set_size,
            "wrong_singleton": cover.wrong_singleton <= iid.wrong_singleton,
        }
        strict_conditions = {
            "coverage": cover.validation_winner_covered > iid.validation_winner_covered,
            "size": cover.set_size < iid.set_size,
            "wrong_singleton": cover.wrong_singleton < iid.wrong_singleton,
        }
        for key in clauses:
            clauses[key] += bool(conditions[key])
            strict[key] += bool(strict_conditions[key])
        panels[panel] = {
            "clauses": {key: bool(value) for key, value in conditions.items()},
            "strict": {key: bool(value) for key, value in strict_conditions.items()},
            "means": means.reset_index().to_dict(orient="records"),
        }
    passed = bool(all(clauses[key] >= 4 and strict[key] >= 1 for key in clauses))
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "panels_passing_by_clause": clauses,
        "panels_strict_by_clause": strict,
        "frozen_gate_passed": passed, "panels": panels,
    }
    (RESULTS / "stability_set_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
