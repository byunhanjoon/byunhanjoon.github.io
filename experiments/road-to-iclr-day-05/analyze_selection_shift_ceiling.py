"""Exact-quotient ceiling diagnostic for validation-selected model choice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PANELS = (
    ("confirmation", "tier1_confirmation_config.json", "tier1_confirmation"),
    ("menu_repeat", "tier1_menu_repeat_config.json", "tier1_menu_repeat"),
    ("subsample_repeat", "tier1_subsample_repeat_config.json", "tier1_subsample_repeat"),
    ("openml_external", "openml_external_cover_config.json", "openml_external_cover"),
    ("openml_taskbalanced", "openml_taskbalanced_cover_config.json", "openml_taskbalanced_cover"),
)


def main() -> None:
    rows = []
    for panel, config_name, directory in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            validation_losses, test_losses = [], []
            for model in config["models"]:
                archive = np.load(RESULTS / directory / f"{dataset}__{model}.npz")
                validation_losses.append(proper_loss(
                    archive["validation_y"], archive["validation_predictions"].mean(axis=(0, 1, 2, 3))
                ))
                test_losses.append(proper_loss(
                    archive["test_y"], archive["test_predictions"].mean(axis=(0, 1, 2, 3))
                ))
            validation_losses = np.asarray(validation_losses)
            test_losses = np.asarray(test_losses)
            val_winner = int(np.argmin(validation_losses))
            test_winner = int(np.argmin(test_losses))
            correlation = spearmanr(validation_losses, test_losses).statistic
            rows.append({
                "panel": panel, "dataset": dataset,
                "validation_winner": config["models"][val_winner],
                "test_winner": config["models"][test_winner],
                "same_winner": val_winner == test_winner,
                "validation_winner_test_loss": test_losses[val_winner],
                "test_oracle_loss": test_losses[test_winner],
                "validation_winner_test_regret": test_losses[val_winner] - test_losses[test_winner],
                "candidate_rank_spearman": correlation,
                "validation_margin": float(np.partition(validation_losses, 1)[1] - validation_losses[val_winner]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "selection_shift_ceiling_datasets.csv", index=False)
    panels = {}
    for panel, current in frame.groupby("panel"):
        panels[panel] = {
            "datasets": len(current),
            "validation_winner_equals_test_winner": int(current.same_winner.sum()),
            "mean_exact_validation_winner_test_regret": float(current.validation_winner_test_regret.mean()),
            "median_candidate_rank_spearman": float(current.candidate_rank_spearman.median()),
            "datasets_with_zero_exact_selection_ceiling": int((current.validation_winner_test_regret <= 1e-15).sum()),
        }
    summary = {"status": "complete", "post_outcome_diagnostic": True, "panels": panels}
    (RESULTS / "selection_shift_ceiling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
