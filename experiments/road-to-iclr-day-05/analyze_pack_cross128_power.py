"""Gap-calibrated pairwise ranking power for unbiased 128-fit scores."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_disjoint_pack_cross128 import DRAWS, METHODS, action_blocks, block_u_scores
from analyze_mixed_resolvable_packing import SHAPE
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GAPS = (.25, .5, 1.0, 2.0)


def dataset_errors(panel: str, dataset: str, models: list[str], directory: Path):
    predictions, y = [], None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        if tuple(archive["validation_predictions"].shape[:4]) != SHAPE:
            return None
        y = archive["validation_y"]
        predictions.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
    assert y is not None
    first, second, independent = action_blocks(SHAPE, panel, dataset)
    left, right = first.reshape(DRAWS, -1), second.reshape(DRAWS, -1)
    errors = {method: [] for method in METHODS}
    for flat in predictions:
        target = proper_loss(y, flat.mean(axis=0))
        pack, _ = CQS.cross_and_mean_scores(y, flat, left, right)
        control, _ = block_u_scores(y, flat, independent)
        errors[METHODS[0]].append(pack - target)
        errors[METHODS[1]].append(control - target)
    return {method: np.stack(values, axis=1) for method, values in errors.items()}


def main() -> None:
    rows = []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            errors = dataset_errors(panel, dataset, config["models"], RESULTS / directory_name)
            if errors is None:
                continue
            independent_errors = {}
            for method, matrix in errors.items():
                shuffled = np.empty_like(matrix)
                for candidate in range(matrix.shape[1]):
                    rng = np.random.default_rng(RMS.stable_seed(
                        "pack128-power-permute", panel, dataset, method, str(candidate)
                    ))
                    shuffled[:, candidate] = matrix[rng.permutation(DRAWS), candidate]
                independent_errors[method] = shuffled
            for left_index, right_index in itertools.combinations(range(len(config["models"])), 2):
                control_difference = (
                    errors[METHODS[1]][:, left_index] - errors[METHODS[1]][:, right_index]
                )
                scale = float(control_difference.std(ddof=1))
                if scale <= 1e-15:
                    continue
                for coupling, current_errors in (
                    ("common", errors), ("candidate_independent", independent_errors)
                ):
                    for gap_multiplier in GAPS:
                        gap = gap_multiplier * scale
                        for method, matrix in current_errors.items():
                            difference = matrix[:, left_index] - matrix[:, right_index]
                            inversion = .5 * (np.mean(difference > gap) + np.mean(difference < -gap))
                            rows.append({
                                "panel": panel, "dataset": dataset,
                                "left_model": config["models"][left_index],
                                "right_model": config["models"][right_index],
                                "coupling": coupling, "gap_multiplier": gap_multiplier,
                                "absolute_gap": gap, "method": method,
                                "inversion_probability": float(inversion),
                            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "pack_cross128_power_cells.csv", index=False)
    pivot = frame.pivot(
        index=["panel", "dataset", "left_model", "right_model", "coupling", "gap_multiplier"],
        columns="method", values="inversion_probability"
    )
    pivot["pack_lower"] = pivot[METHODS[0]] < pivot[METHODS[1]]
    pivot["difference"] = pivot[METHODS[0]] - pivot[METHODS[1]]
    pivot.reset_index().to_csv(RESULTS / "pack_cross128_power_comparisons.csv", index=False)
    clauses, panels = 0, {}
    grouped = pivot.reset_index().groupby(["panel", "coupling", "gap_multiplier"])
    for (panel, coupling, gap), current in grouped:
        action = float(current[METHODS[0]].mean())
        control = float(current[METHODS[1]].mean())
        passed = action < control
        clauses += int(passed)
        panels.setdefault(panel, []).append({
            "coupling": coupling, "gap_multiplier": gap,
            "pack_mean_inversion": action, "control_mean_inversion": control,
            "strictly_lower": bool(passed),
        })
    strict = int(pivot.pack_lower.sum())
    total = int(len(pivot))
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "nondegenerate_candidate_pairs": int(
            pivot.reset_index()[["panel", "dataset", "left_model", "right_model"]].drop_duplicates().shape[0]
        ),
        "panel_gap_coupling_clauses": int(len(grouped)),
        "clauses_passing": clauses,
        "pair_gap_coupling_cells": total, "strict_pair_cells": strict,
        "strict_pair_cell_fraction": strict / total,
        "panels": panels,
        "frozen_gate_passed": bool(clauses == len(grouped) and strict / total >= .8),
    }
    (RESULTS / "pack_cross128_power_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
