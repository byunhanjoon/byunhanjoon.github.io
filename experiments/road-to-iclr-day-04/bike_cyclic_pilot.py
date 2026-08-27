#!/usr/bin/env python3
"""Independent cyclic-field test on the UCI Bike Sharing hourly dataset.

Only the declared Hour field changes.  Quantile PLE, mass normalization, a
path, the correct 24-hour ring, and a permuted ring all span the same 23-
dimensional function space.  Rows are split chronologically and no test label
is used for representation fitting or model selection.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from support_heat_pilot import (
    HERE,
    Dataset,
    PARTS,
    base_schema,
    clean_numeric,
    combine,
    hat_basis,
    linear_basis,
    make_prepared,
    mass_power_basis,
    quantile_nodes,
    read_rows,
    riesz_basis,
    train_model,
    write_rows,
)


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike+sharing+dataset.zip"
)
METHODS = (
    "quantile_ple",
    "hour_mass",
    "hour_path",
    "hour_ring",
    "hour_wrong_ring",
)


def load_bike() -> Dataset:
    with urlopen(UCI_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("hour.csv") as handle:
        frame = pd.read_csv(handle)

    # UCI supplies rows in time order.  The task is prospective prediction on
    # the final 20%, with the preceding 20% reserved for validation.
    n = len(frame)
    boundaries = (0, int(0.6 * n), int(0.8 * n), n)
    indices = {
        part: np.arange(boundaries[i], boundaries[i + 1])
        for i, part in enumerate(PARTS)
    }

    numeric = ["hr", "temp", "atemp", "hum", "windspeed"]
    binary = ["yr", "holiday", "workingday"]
    categorical = ["season", "mnth", "weekday", "weathersit"]

    def split(columns: list[str], *, strings: bool = False) -> dict[str, np.ndarray]:
        values = frame[columns].to_numpy()
        if strings:
            values = values.astype(str)
        else:
            values = values.astype(np.float32)
        return {part: np.asarray(values[idx]) for part, idx in indices.items()}

    target = np.log1p(frame["cnt"].to_numpy(dtype=np.float32))
    return Dataset(
        name="uci-bike-hourly",
        task="regression",
        x_num=split(numeric),
        x_bin=split(binary),
        x_cat=split(categorical, strings=True),
        y={part: target[idx] for part, idx in indices.items()},
        n_classes=1,
        split_fingerprint="chronological-60-20-20",
    )


def representations(
    dataset: Dataset, bins: int, strength: float
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    assert dataset.x_num is not None
    clean = clean_numeric(dataset.x_num)
    qblocks: list[dict[str, np.ndarray]] = []
    for column in range(clean["train"].shape[1]):
        nodes = quantile_nodes(clean["train"][:, column], bins)
        qblocks.append(
            {
                part: linear_basis(clean[part][:, column], nodes)
                for part in PARTS
            }
        )

    hour_nodes = np.arange(24, dtype=np.float64)
    hour_hats = {
        part: hat_basis(clean[part][:, 0], hour_nodes) for part in PARTS
    }
    hour_mass, mass_rank = mass_power_basis(hour_hats, 1.0)
    hour_path, path_meta = riesz_basis(
        clean, 0, hour_nodes, strength, topology="path"
    )
    hour_ring, ring_meta = riesz_basis(
        clean, 0, hour_nodes, strength, topology="ring"
    )
    hour_wrong_ring, wrong_meta = riesz_basis(
        clean, 0, hour_nodes, strength, topology="ring", permuted=True
    )
    nonnumeric = base_schema(dataset, seed=20260826, include_num=False)

    hour_variants = {
        "quantile_ple": qblocks[0],
        "hour_mass": hour_mass,
        "hour_path": hour_path,
        "hour_ring": hour_ring,
        "hour_wrong_ring": hour_wrong_ring,
    }
    variants = {
        name: combine([hour, *qblocks[1:], nonnumeric])
        for name, hour in hour_variants.items()
    }
    dimensions = {name: int(x["train"].shape[1]) for name, x in variants.items()}
    assert len(set(dimensions.values())) == 1, dimensions
    return variants, {
        "source": UCI_URL,
        "target": "log1p(cnt)",
        "split": dataset.split_fingerprint,
        "declared_field": "hr",
        "declared_topology": "24-hour ring",
        "hour_mass_rank": mass_rank,
        "dimensions": dimensions,
        "path": path_meta,
        "ring": ring_meta,
        "wrong_ring": wrong_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet"])
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20260844, 20260845, 20260846],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/bike_cyclic_geometry.csv",
    )
    args = parser.parse_args()

    dataset = load_bike()
    variants, metadata = representations(dataset, args.bins, args.strength)
    methods = args.methods or list(METHODS)
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (str(r["model"]), int(r["seed"]), str(r["method"])) for r in rows
    }
    for model in args.models:
        for seed in args.seeds:
            for method in methods:
                if (model, seed, method) in completed:
                    continue
                result, _ = train_model(
                    make_prepared(dataset, variants[method], {"method": method}),
                    seed=seed,
                    device=args.device,
                    model_name=model,
                    width=args.width,
                    depth=args.depth,
                    dropout=0.1,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    batch_size=512,
                    max_epochs=args.epochs,
                    patience=args.patience,
                )
                row = {
                    "dataset": dataset.name,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "strength": args.strength,
                    **result,
                }
                rows.append(row)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)

    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
