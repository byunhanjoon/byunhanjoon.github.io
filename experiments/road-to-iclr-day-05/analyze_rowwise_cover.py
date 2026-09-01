"""Exact row-wise risk profile of the confirmed strength-2 cover."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import (
    incidence_covariance,
    strength1_family,
    strength2_family,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def row_residual(predictions: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    return np.einsum("ij,irk,jrk->r", covariance, centered, centered, optimize=True)


def summaries(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(values.max()),
    }


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    material = screened[screened.study == "strength2_confirmation"]
    input_dir = RESULTS / "tier1_confirmation"
    cache = {}
    cell_rows = []
    equal_cell_profiles = []
    for cell in material.itertuples():
        predictions = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].astype(np.float64)
        cardinalities = tuple(predictions.shape[:4])
        key = cardinalities[1:]
        if key not in cache:
            family1 = strength1_family(*key)
            family2 = strength2_family(*key)
            cache[key] = (
                incidence_covariance(family1, cardinalities),
                incidence_covariance(family2, cardinalities),
            )
        covariance1, covariance2 = cache[key]
        strength1 = row_residual(predictions, covariance1) / 4
        strength2 = row_residual(predictions, covariance2)
        flat = predictions.reshape((-1,) + predictions.shape[-2:])
        centroid = flat.mean(axis=0)
        iid = np.mean(np.sum((flat - centroid) ** 2, axis=-1), axis=0) / 16
        seed_average = predictions.mean(axis=3)
        seed_flat = seed_average.reshape((-1,) + seed_average.shape[-2:])
        seed_blocks = np.mean(np.sum((seed_flat - centroid) ** 2, axis=-1), axis=0) / 4
        comparators = {"iid16": iid, "four_strength1": strength1, "four_seed_blocks": seed_blocks}
        row = {"dataset": cell.dataset, "model": cell.model, "rows": len(strength2), **summaries(strength2, "strength2")}
        for name, values in comparators.items():
            row.update(summaries(values, name))
            row[f"fraction_rows_strength2_lower_than_{name}"] = float(np.mean(strength2 < values))
            row[f"p95_reduction_vs_{name}"] = 1 - row["strength2_p95"] / row[f"{name}_p95"] if row[f"{name}_p95"] else 0.0
            row[f"p99_reduction_vs_{name}"] = 1 - row["strength2_p99"] / row[f"{name}_p99"] if row[f"{name}_p99"] else 0.0
            row[f"max_reduction_vs_{name}"] = 1 - row["strength2_max"] / row[f"{name}_max"] if row[f"{name}_max"] else 0.0
        cell_rows.append(row)
        quantile_grid = np.linspace(0, 1, 1001)
        equal_cell_profiles.append({
            "strength2": np.quantile(strength2, quantile_grid),
            **{name: np.quantile(values, quantile_grid) for name, values in comparators.items()},
        })
    cells = pd.DataFrame(cell_rows)
    profile_means = {
        name: np.mean([profile[name] for profile in equal_cell_profiles], axis=0)
        for name in ("strength2", "iid16", "four_strength1", "four_seed_blocks")
    }
    summary = {
        "status": "complete", "cells": len(cells),
        "mean_fraction_rows_strength2_lower": {
            name: float(cells[f"fraction_rows_strength2_lower_than_{name}"].mean())
            for name in ("iid16", "four_strength1", "four_seed_blocks")
        },
        "cells_with_lower_p95": {
            name: int((cells.strength2_p95 < cells[f"{name}_p95"]).sum())
            for name in ("iid16", "four_strength1", "four_seed_blocks")
        },
        "equal_cell_weight_p95_reduction": {
            name: float(1 - profile_means["strength2"][950] / profile_means[name][950])
            for name in ("iid16", "four_strength1", "four_seed_blocks")
        },
        "equal_cell_weight_p99_reduction": {
            name: float(1 - profile_means["strength2"][990] / profile_means[name][990])
            for name in ("iid16", "four_strength1", "four_seed_blocks")
        },
        "cells_with_lower_max": {
            name: int((cells.strength2_max < cells[f"{name}_max"]).sum())
            for name in ("iid16", "four_strength1", "four_seed_blocks")
        },
    }
    cells.to_csv(RESULTS / "rowwise_cover_cells.csv", index=False)
    (RESULTS / "rowwise_cover_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

