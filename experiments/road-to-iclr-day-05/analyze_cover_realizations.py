"""Monte Carlo deployment-realization reliability for the confirmed cover."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import strength1_family, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 4096
BASE_SEED = 2026082801


def cell_seed(dataset: str, model: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{dataset}:{model}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def flat_ids(designs: np.ndarray, cardinalities: tuple[int, ...]) -> np.ndarray:
    return np.ravel_multi_index(np.moveaxis(designs, -1, 0), cardinalities)


def design_risks(gram: np.ndarray, ids: np.ndarray, batch: int = 128) -> np.ndarray:
    output = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), batch):
        current = ids[start : start + batch]
        pairwise = gram[current[:, :, None], current[:, None, :]]
        output[start : start + len(current)] = pairwise.mean(axis=(1, 2))
    return output


def gram_matrix(predictions: np.ndarray) -> np.ndarray:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    return np.einsum("irk,jrk->ij", centered, centered, optimize=True) / predictions.shape[-2]


def sample_cell(predictions: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cardinalities = tuple(predictions.shape[:4])
    category, label, seeds = cardinalities[1], cardinalities[2], cardinalities[3]
    if seeds != 4:
        raise ValueError("confirmation realization audit expects four model seeds")
    gram = gram_matrix(predictions)
    family2 = strength2_family(category, label, seeds)
    chosen2 = family2[rng.integers(0, len(family2), size=DRAWS)]
    ids2 = flat_ids(chosen2, cardinalities)
    iid = rng.integers(0, np.prod(cardinalities), size=(DRAWS, 16))
    family1 = strength1_family(category, label, seeds)
    selected1 = family1[rng.integers(0, len(family1), size=(DRAWS, 4))]
    ids1 = flat_ids(selected1.reshape(DRAWS, 16, 4), cardinalities)
    schema_shape = cardinalities[:3]
    schema = np.column_stack([
        rng.integers(0, size, size=DRAWS * 4) for size in schema_shape
    ]).reshape(DRAWS, 4, 3)
    blocks = np.empty((DRAWS, 4, 4, 4), dtype=int)
    blocks[..., :3] = schema[:, :, None, :]
    blocks[..., 3] = np.arange(4)[None, None, :]
    block_ids = flat_ids(blocks.reshape(DRAWS, 16, 4), cardinalities)
    return {
        "strength2": design_risks(gram, ids2),
        "iid16": design_risks(gram, iid),
        "four_strength1": design_risks(gram, ids1),
        "four_seed_blocks": design_risks(gram, block_ids),
    }


def summarize(action: np.ndarray, comparator: np.ndarray) -> dict[str, float]:
    reduction = 1 - action / comparator
    return {
        "probability_action_lower": float(np.mean(action < comparator)),
        "mean_reduction": float(1 - action.mean() / comparator.mean()),
        "median_reduction": float(np.median(reduction)),
        "reduction_q05": float(np.quantile(reduction, 0.05)),
        "reduction_q95": float(np.quantile(reduction, 0.95)),
    }


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    material = screened[screened.study == "strength2_confirmation"]
    input_dir = RESULTS / "tier1_confirmation"
    config = json.loads((HERE / "tier1_confirmation_config.json").read_text())
    pooled = {name: np.zeros(DRAWS) for name in ("strength2", "iid16", "four_strength1", "four_seed_blocks")}
    grouped = {
        group: {name: np.zeros(DRAWS) for name in pooled}
        for group in sorted(set(config["source_groups"].values()))
    }
    cell_rows = []
    for row in material.itertuples():
        archive = np.load(input_dir / f"{row.dataset}__{row.model}.npz")
        risks = sample_cell(archive["test_predictions"].astype(np.float64), np.random.default_rng(cell_seed(row.dataset, row.model)))
        group = config["source_groups"][row.dataset]
        for name, values in risks.items():
            pooled[name] += values
            grouped[group][name] += values
        cell_rows.append({
            "dataset": row.dataset, "model": row.model,
            **{
                f"probability_strength2_lower_than_{name}": float(np.mean(risks["strength2"] < risks[name]))
                for name in ("iid16", "four_strength1", "four_seed_blocks")
            },
        })
    comparisons = {
        name: summarize(pooled["strength2"], pooled[name])
        for name in ("iid16", "four_strength1", "four_seed_blocks")
    }
    beats_all = np.ones(DRAWS, dtype=bool)
    for name in ("iid16", "four_strength1", "four_seed_blocks"):
        beats_all &= pooled["strength2"] < pooled[name]
    group_probabilities = {}
    for group, values in grouped.items():
        group_beats = np.ones(DRAWS, dtype=bool)
        for name in ("iid16", "four_strength1", "four_seed_blocks"):
            group_beats &= values["strength2"] < values[name]
        group_probabilities[group] = float(group_beats.mean())
    summary = {
        "status": "complete", "draws": DRAWS, "validation_material_cells": len(material),
        "pooled_probability_strength2_beats_all": float(beats_all.mean()),
        "comparisons": comparisons,
        "source_group_probability_strength2_beats_all": group_probabilities,
    }
    pd.DataFrame(cell_rows).to_csv(RESULTS / "cover_realization_cells.csv", index=False)
    (RESULTS / "cover_realization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

