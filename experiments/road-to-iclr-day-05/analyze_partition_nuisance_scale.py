"""Compare validation/test gap movement to unbiased nuisance-score RMSE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLITS = (2026082831, 2026082901, 2026082911, 2026082921)
DATASETS = (
    "openml-kr-vs-kp-3", "openml-credit-approval-29", "openml-sick-38",
    "openml-tic-tac-toe-50", "openml-mushroom-24", "openml-colic-25",
    "openml-hepatitis-55", "openml-vote-56",
)
MODELS = ("native_histgb", "catboost_native")


def prefix(split_seed: int) -> tuple[Path, str]:
    if split_seed == SPLITS[0]:
        return RESULTS / "openml_modern_model_cover", "modern_model"
    return RESULTS / f"openml_modern_model_split_{split_seed}", f"modern_split_{split_seed}"


def quotient_loss(directory: Path, dataset: str, model: str, split: str) -> float:
    archive = np.load(directory / f"{dataset}__{model}.npz")
    predictions = archive[f"{split}_predictions"]
    quotient = predictions.reshape((-1,) + predictions.shape[-2:]).mean(axis=0)
    return proper_loss(archive[f"{split}_y"], quotient)


def main() -> None:
    rows = []
    for split_seed in SPLITS:
        directory, output_prefix = prefix(split_seed)
        calibration = pd.read_csv(RESULTS / f"{output_prefix}_packing_calibration.csv")
        calibration = calibration[
            (calibration.family == "pair_cross64")
            & (calibration.method == "disjoint_pair_cross64")
        ].set_index(["dataset", "model"])
        for dataset in DATASETS:
            losses = {}
            for split in ("validation", "test"):
                losses[split] = {
                    model: quotient_loss(directory, dataset, model, split)
                    for model in MODELS
                }
            validation_gap = losses["validation"][MODELS[1]] - losses["validation"][MODELS[0]]
            test_gap = losses["test"][MODELS[1]] - losses["test"][MODELS[0]]
            scale = float(np.sqrt(sum(
                calibration.loc[(dataset, model), "score_rmse"] ** 2
                for model in MODELS
            )))
            shift = abs(test_gap - validation_gap)
            rows.append({
                "split_seed": split_seed, "dataset": dataset,
                "validation_gap_catboost_minus_histgb": validation_gap,
                "test_gap_catboost_minus_histgb": test_gap,
                "winner_flip": bool(validation_gap * test_gap < 0),
                "absolute_partition_gap_movement": shift,
                "pair_cross64_gap_rmse_quadrature_scale": scale,
                "partition_to_nuisance_scale_ratio": shift / max(scale, 1e-18),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "partition_nuisance_scale_cells.csv", index=False)
    alternate = frame[frame.split_seed != SPLITS[0]]
    summary = {
        "status": "complete", "evidence_status": "post_outcome_mechanism",
        "dataset_split_pairs": int(len(frame)),
        "winner_flips": int(frame.winner_flip.sum()),
        "median_partition_to_nuisance_scale_ratio": float(
            frame.partition_to_nuisance_scale_ratio.median()
        ),
        "minimum_partition_to_nuisance_scale_ratio": float(
            frame.partition_to_nuisance_scale_ratio.min()
        ),
        "maximum_partition_to_nuisance_scale_ratio": float(
            frame.partition_to_nuisance_scale_ratio.max()
        ),
        "winner_flip_median_ratio": float(
            frame.loc[frame.winner_flip, "partition_to_nuisance_scale_ratio"].median()
        ),
        "alternate_splits": {
            "dataset_split_pairs": int(len(alternate)),
            "winner_flips": int(alternate.winner_flip.sum()),
            "median_ratio": float(alternate.partition_to_nuisance_scale_ratio.median()),
        },
        "interpretation": "partition_gap_movement_dominates_conditional_nuisance_rmse",
        "formal_inference_claimed": False,
    }
    (RESULTS / "partition_nuisance_scale_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
