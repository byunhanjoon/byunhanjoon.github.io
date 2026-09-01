#!/usr/bin/env python3
"""Day 6 metric-partition hour encoder on the established UCI Bike panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY4 = HERE.parents[1] / "road-to-iclr-day-04"
sys.path.insert(0, str(DAY4))
sys.path.insert(0, str(HERE))

from bike_cyclic_pilot import load_bike  # noqa: E402
from support_heat_pilot import (  # noqa: E402
    PARTS,
    base_schema,
    clean_numeric,
    combine,
    linear_basis,
    make_prepared,
    quantile_nodes,
    read_rows,
    train_model,
    write_rows,
)
from metric_partition_benchmark import farthest_landmarks, normalized_kernel  # noqa: E402


METHODS = ("qple", "periodic", "code_rbf", "mpe_ring", "mmpe_ring", "mpe_corrupt")


def partitions(hour: dict[str, np.ndarray], metric: np.ndarray, multiscale: bool) -> dict[str, np.ndarray]:
    anchors = farthest_landmarks(metric, np.arange(24), 16)
    distance = {part: metric[np.ix_(hour[part].astype(np.int64), anchors)] for part in PARTS}
    cover = max(float(np.max(np.min(distance["train"], axis=1))), 1.0)
    scales = (0.5, 1.0, 2.0) if multiscale else (1.0,)
    return {part: np.mean([normalized_kernel(d, cover * s) for s in scales], axis=0) for part, d in distance.items()}


def representations(seed: int = 20260930):
    dataset = load_bike()
    assert dataset.x_num is not None
    clean = clean_numeric(dataset.x_num)
    hour = {part: clean[part][:, 0] for part in PARTS}
    qblocks = []
    for column in range(1, clean["train"].shape[1]):
        nodes = quantile_nodes(clean["train"][:, column], 16)
        block = {part: linear_basis(clean[part][:, column], nodes) for part in PARTS}
        qblocks.append(block)

    hour_nodes = quantile_nodes(hour["train"], 16)
    qple = {part: linear_basis(hour[part], hour_nodes) for part in PARTS}
    assert qple["train"].shape[1] == 16
    phase_freq = np.arange(1, 9, dtype=np.float64)
    periodic = {
        part: np.concatenate(
            [
                np.sin(2 * np.pi * hour[part][:, None] * phase_freq[None, :] / 24.0),
                np.cos(2 * np.pi * hour[part][:, None] * phase_freq[None, :] / 24.0),
            ],
            axis=1,
        )
        for part in PARTS
    }
    indices = np.arange(24)
    path = np.abs(indices[:, None] - indices[None, :]).astype(np.float64)
    ring = np.minimum(path, 24 - path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(24)
    corrupt = ring[np.ix_(perm, perm)]
    hour_variants = {
        "qple": qple,
        "periodic": periodic,
        "code_rbf": partitions(hour, path, False),
        "mpe_ring": partitions(hour, ring, False),
        "mmpe_ring": partitions(hour, ring, True),
        "mpe_corrupt": partitions(hour, corrupt, False),
    }
    nonnumeric = base_schema(dataset, seed=seed, include_num=False)
    variants = {name: combine([block, *qblocks, nonnumeric]) for name, block in hour_variants.items()}
    dimensions = {name: int(x["train"].shape[1]) for name, x in variants.items()}
    assert len(set(dimensions.values())) == 1, dimensions
    return dataset, variants, dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results/bike_confirmation.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dataset, variants, dimensions = representations()
    rows = list(read_rows(args.output))
    completed = {(r["model"], int(r["seed"]), r["method"]) for r in rows}
    for model in ("mlp", "resnet"):
        for seed in (20260930, 20260931, 20260932):
            for method in METHODS:
                if (model, seed, method) in completed:
                    continue
                result, _ = train_model(
                    make_prepared(dataset, variants[method], {"method": method}),
                    seed=seed,
                    device=args.device,
                    model_name=model,
                    width=128,
                    depth=2,
                    dropout=0.1,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    batch_size=512,
                    max_epochs=40,
                    patience=6,
                )
                row = {"dataset": dataset.name, "model": model, "seed": seed, "method": method, **result}
                rows.append(row)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "protocol": "BIKE_CONFIRMATION_FREEZE.md",
        "dimensions": dimensions,
        "rows": len(rows),
        "methods": METHODS,
        "seeds": [20260930, 20260931, 20260932],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
