"""Small multiclass scope addendum for quotient model selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "openml_multiclass_cover_config.json").read_text())
    directory = RESULTS / "openml_multiclass_cover"
    rows = []
    winner_alignment = {}
    for dataset in config["datasets"]:
        rows.extend(RMS.analyze_dataset(
            "openml_multiclass", dataset, config["models"], directory
        ))
        validation, test = [], []
        val_y = test_y = None
        for model in config["models"]:
            archive = np.load(directory / f"{dataset}__{model}.npz")
            val_y, test_y = archive["validation_y"], archive["test_y"]
            validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).mean(axis=0))
            test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).mean(axis=0))
        val_losses = [proper_loss(val_y, value) for value in validation]
        test_losses = [proper_loss(test_y, value) for value in test]
        winner_alignment[dataset] = {
            "validation_winner": config["models"][int(np.argmin(val_losses))],
            "test_winner": config["models"][int(np.argmin(test_losses))],
            "aligned": bool(np.argmin(val_losses) == np.argmin(test_losses)),
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "multiclass_model_selection_draws.csv", index=False)
    cells = frame.groupby(["dataset", "method"], as_index=False).agg(
        selection_agreement=("agrees_validation_quotient_winner", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "multiclass_model_selection_cells.csv", index=False)
    means = cells.groupby("method").mean(numeric_only=True)
    test_pivot = cells.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
    differences = test_pivot.strength2 - test_pivot.iid16
    clauses = {
        "agreement_above_iid": bool(means.loc["strength2", "selection_agreement"] > means.loc["iid16", "selection_agreement"]),
        "validation_regret_below_iid": bool(means.loc["strength2", "validation_quotient_regret"] < means.loc["iid16", "validation_quotient_regret"]),
        "realized_test_loss_lower_on_at_least_one": bool((differences < 0).sum() >= 1),
    }
    summary = {
        "status": "complete", "datasets": len(config["datasets"]),
        "draws_per_dataset": RMS.DRAWS, "winner_alignment": winner_alignment,
        "means": means.reset_index().to_dict(orient="records"),
        "strength2_minus_iid_test_loss_by_dataset": {
            dataset: float(value) for dataset, value in differences.items()
        },
        "clauses": clauses, "scope_addendum_passed": bool(all(clauses.values())),
    }
    (RESULTS / "multiclass_model_selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
