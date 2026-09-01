"""Stability-set repeat with independent action streams for every candidate."""

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
METHODS = ("cover_cross_union64_independent", "iid_u_union64_independent")
PANELS = CQS.PANELS


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
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
    exact_validation = np.asarray([
        proper_loss(validation_y, candidate.mean(axis=0)) for candidate in validation
    ])
    exact_test = np.asarray([
        proper_loss(test_y, candidate.mean(axis=0)) for candidate in test
    ])
    validation_winner = int(np.argmin(exact_validation))
    test_winner = int(np.argmin(exact_test))
    scores = {method: [[], []] for method in METHODS}
    for model, candidate in zip(models, validation):
        actions = [
            RMS.action_ids(
                shape, RMS.stable_seed("stability-set-independent", panel, dataset, model, str(block))
            )
            for block in range(4)
        ]
        for replicate, (left, right) in enumerate(((0, 1), (2, 3))):
            cover, _ = CQS.cross_and_mean_scores(
                validation_y, candidate,
                actions[left]["strength2"][:DRAWS], actions[right]["strength2"][:DRAWS],
            )
            iid_ids = np.concatenate((
                actions[left]["iid16"][:DRAWS], actions[right]["iid16"][:DRAWS]
            ), axis=1)
            iid = CQS.iid_u_scores(validation_y, candidate, iid_ids)
            scores[METHODS[0]][replicate].append(cover)
            scores[METHODS[1]][replicate].append(iid)
    rows = []
    for method in METHODS:
        winner_a = np.argmin(np.stack(scores[method][0], axis=1), axis=1)
        winner_b = np.argmin(np.stack(scores[method][1], axis=1), axis=1)
        for draw, (first, second) in enumerate(zip(winner_a, winner_b)):
            members = np.unique([int(first), int(second)])
            singleton = len(members) == 1
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw, "set_size": int(len(members)),
                "validation_winner_covered": bool(validation_winner in members),
                "wrong_singleton": bool(singleton and validation_winner not in members),
                "test_winner_covered": bool(test_winner in members),
                "best_validation_regret": float((exact_validation[members] - exact_validation[validation_winner]).min()),
                "worst_validation_regret": float((exact_validation[members] - exact_validation[validation_winner]).max()),
            })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory_name))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "stability_set_independent_draws.csv", index=False)
    metrics = [
        "set_size", "validation_winner_covered", "wrong_singleton",
        "test_winner_covered", "best_validation_regret", "worst_validation_regret",
    ]
    cells = frame.groupby(["panel", "dataset", "task", "method"], as_index=False)[metrics].mean()
    cells.to_csv(RESULTS / "stability_set_independent_cells.csv", index=False)
    clauses = {"coverage": 0, "size": 0, "wrong_singleton": 0}
    strict = {key: 0 for key in clauses}
    panels: dict[str, object] = {}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cover, iid = means.loc[METHODS[0]], means.loc[METHODS[1]]
        comparisons = {
            "coverage": (cover.validation_winner_covered >= iid.validation_winner_covered,
                         cover.validation_winner_covered > iid.validation_winner_covered),
            "size": (cover.set_size <= iid.set_size, cover.set_size < iid.set_size),
            "wrong_singleton": (cover.wrong_singleton <= iid.wrong_singleton,
                                cover.wrong_singleton < iid.wrong_singleton),
        }
        for key, (weak, strong) in comparisons.items():
            clauses[key] += bool(weak); strict[key] += bool(strong)
        panels[panel] = {
            "clauses": {key: bool(value[0]) for key, value in comparisons.items()},
            "strict": {key: bool(value[1]) for key, value in comparisons.items()},
            "means": means.reset_index().to_dict(orient="records"),
        }
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "panels_passing_by_clause": clauses, "panels_strict_by_clause": strict,
        "frozen_gate_passed": bool(all(clauses[key] >= 4 and strict[key] >= 1 for key in clauses)),
        "panels": panels,
    }
    (RESULTS / "stability_set_independent_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
