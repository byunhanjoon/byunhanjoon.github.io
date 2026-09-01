"""Batch uncertainty for scrambled Sobol/LHS nuisance baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from analyze_qmc_baselines import mapped, seed
from analyze_strength2_cover import expected_residual, incidence_covariance, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BATCHES = 16
PER_BATCH = 4096


def continuous_family(kind: str, offset: int) -> np.ndarray:
    output = np.empty((PER_BATCH, 16, 4), dtype=np.float32)
    for local in range(PER_BATCH):
        index = offset + local
        if kind == "sobol16":
            output[local] = qmc.Sobol(d=4, scramble=True, seed=seed(kind, index)).random_base2(4)
        elif kind == "lhs16":
            output[local] = qmc.LatinHypercube(d=4, scramble=True, seed=seed(kind, index)).random(16)
        else:
            raise ValueError(kind)
    return output


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    chosen = screened[screened.study == "strength2_confirmation"][["dataset", "model"]]
    cells = []
    shapes = set()
    for cell in chosen.itertuples(index=False):
        predictions = np.load(RESULTS / "tier1_confirmation" / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].astype(np.float64)
        shape = tuple(map(int, predictions.shape[:4]))
        shapes.add(shape)
        cov2 = incidence_covariance(strength2_family(*shape[1:]), shape)
        cells.append((cell.dataset, cell.model, shape, predictions, expected_residual(predictions, cov2)))
    rows = []
    for batch in range(BATCHES):
        for kind in ("sobol16", "lhs16"):
            points = continuous_family(kind, batch * PER_BATCH)
            covariances = {
                shape: incidence_covariance(mapped(points, shape), shape) for shape in shapes
            }
            current = []
            for dataset, model, shape, predictions, strength2 in cells:
                control = expected_residual(predictions, covariances[shape])
                current.append((strength2, control))
                rows.append({
                    "batch": batch, "control": kind, "dataset": dataset, "model": model,
                    "strength2_residual": strength2, "control_residual": control,
                    "strength2_lower": strength2 < control,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "qmc_uncertainty_cells.csv", index=False)
    batch = frame.groupby(["control", "batch"], as_index=False).agg(
        cells_strength2_lower=("strength2_lower", "sum"),
        strength2_mean=("strength2_residual", "mean"), control_mean=("control_residual", "mean"),
    )
    batch["pooled_reduction"] = 1 - batch.strength2_mean / batch.control_mean
    batch.to_csv(RESULTS / "qmc_uncertainty_batches.csv", index=False)
    controls = {}
    gate = True
    for control, current in batch.groupby("control"):
        interval = np.quantile(current.pooled_reduction, [0.025, 0.5, 0.975])
        controls[control] = {
            "cell_win_count_min_median_max": [
                int(current.cells_strength2_lower.min()),
                float(current.cells_strength2_lower.median()),
                int(current.cells_strength2_lower.max()),
            ],
            "pooled_reduction_quantiles_025_50_975": [float(value) for value in interval],
        }
        gate &= interval[0] > 0
    summary = {
        "status": "complete", "batches": BATCHES, "designs_per_batch": PER_BATCH,
        "total_designs_per_control": BATCHES * PER_BATCH, "controls": controls,
        "original_qmc_gate_remains_failed": True,
        "frozen_pooled_uncertainty_gate_passed": bool(gate),
    }
    (RESULTS / "qmc_uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
