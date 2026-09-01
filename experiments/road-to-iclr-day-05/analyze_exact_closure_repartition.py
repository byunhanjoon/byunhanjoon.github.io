"""Exact packed-closure selection under fresh validation/test repartitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_repartitioned_cross_selection as RCS
import analyze_repeated_holdout_shift as RHS
import analyze_robust_model_selection as RMS
from analyze_disjoint_pair_cross import cover_graph


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = RCS.DRAWS
BATCH = 4
METHODS = ("exact_packed_closure128", "independent_cover_u128")


def cover_u_partition_scores(y: np.ndarray, flat: np.ndarray, masks: np.ndarray,
                             blocks: np.ndarray) -> np.ndarray:
    output = np.empty(DRAWS, dtype=np.float64)
    target = RCS.target_array(y, flat.shape[-1])
    counts = masks.sum(axis=1)
    for start in range(0, DRAWS, BATCH):
        stop = min(start + BATCH, DRAWS)
        predictions = np.stack([
            flat[blocks[start:stop, block]].mean(axis=1) for block in range(8)
        ], axis=1)
        residual = target[None, None] - predictions
        summed = residual.sum(axis=1)
        numerator = np.sum(summed ** 2, axis=-1) - np.sum(residual ** 2, axis=(1, 3))
        row_score = numerator / (8 * 7)
        output[start:stop] = np.sum(row_score * masks[start:stop], axis=1) / counts[start:stop]
    return output


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    candidates, validation_y, test_y, task, shape = RCS.load_dataset(directory, dataset, models)
    if int(np.prod(shape)) > 64:
        return []
    pooled_y = np.concatenate((validation_y, test_y))
    masks = RCS.make_masks(
        pooled_y, validation_y, task in {"binclass", "multiclass"},
        np.random.default_rng(RMS.stable_seed("exact-closure-repartition", panel, dataset)),
    )
    ids = cover_graph(shape)[0]
    rng = np.random.default_rng(RMS.stable_seed("closure-cover-u128", panel, dataset))
    blocks = ids[rng.integers(0, len(ids), size=(DRAWS, 8))]
    quotient_rows = np.stack([RHS.row_losses(pooled_y, flat.mean(axis=0)) for flat in candidates], axis=1)
    validation_counts = masks.sum(axis=1)
    test_counts = len(pooled_y) - validation_counts
    exact_validation = masks.astype(float) @ quotient_rows / validation_counts[:, None]
    exact_test = (quotient_rows.sum(axis=0)[None] - masks.astype(float) @ quotient_rows) / test_counts[:, None]
    scores = np.stack([
        cover_u_partition_scores(pooled_y, flat, masks, blocks) for flat in candidates
    ], axis=1)
    validation_winner = np.argmin(exact_validation, axis=1)
    test_winner = np.argmin(exact_test, axis=1)
    selections = {
        METHODS[0]: validation_winner,
        METHODS[1]: np.argmin(scores, axis=1),
    }
    rows = []
    for method, selected in selections.items():
        for draw, candidate in enumerate(selected):
            candidate = int(candidate)
            rows.append({
                "panel": panel, "dataset": dataset, "task": task, "method": method,
                "draw": draw, "validation_agreement": bool(candidate == validation_winner[draw]),
                "test_agreement": bool(candidate == test_winner[draw]),
                "validation_quotient_regret": float(exact_validation[draw, candidate] - exact_validation[draw, validation_winner[draw]]),
                "test_quotient_regret": float(exact_test[draw, candidate] - exact_test[draw, test_winner[draw]]),
                "selected_test_quotient_loss": float(exact_test[draw, candidate]),
                "validation_test_winner_agreement": bool(validation_winner[draw] == test_winner[draw]),
            })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory_name in RHS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory_name))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "exact_closure_repartition_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "exact_closure_repartition_cells.csv", index=False)
    panels, validation_passes, transfer_passes = {}, 0, 0
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        action, control = means.loc[METHODS[0]], means.loc[METHODS[1]]
        pivot = current.pivot(index="dataset", columns="method", values="test_quotient_regret")
        difference = pivot[METHODS[0]] - pivot[METHODS[1]]
        interval = RMS.cluster_interval(
            difference.to_numpy(), RMS.stable_seed("closure-repartition-source", panel)
        )
        validation_better = bool(action.validation_quotient_regret < control.validation_quotient_regret)
        transfer = bool(action.test_quotient_regret <= control.test_quotient_regret and interval[1] < 0)
        validation_passes += int(validation_better); transfer_passes += int(transfer)
        panels[panel] = {
            "eligible_sources": int(len(pivot)), "method_means": means.reset_index().to_dict(orient="records"),
            "validation_regret_strictly_lower": validation_better,
            "mean_test_regret_difference": float(difference.mean()),
            "sources_with_lower_exact_test_regret": int((difference < 0).sum()),
            "test_regret_source_bootstrap_95_interval": interval,
            "transfer_clause": transfer,
        }
    method_pass = validation_passes == 2
    transfer_pass = transfer_passes == 2
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "eligible_sources": int(cells.dataset.nunique()), "panels": panels,
        "method_gate_passed": method_pass, "transfer_gate_passed": transfer_pass,
        "frozen_interpretation": "transfer_pass" if transfer_pass else (
            "validation_only_pass" if method_pass else "method_failure"
        ),
    }
    (RESULTS / "exact_closure_repartition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
