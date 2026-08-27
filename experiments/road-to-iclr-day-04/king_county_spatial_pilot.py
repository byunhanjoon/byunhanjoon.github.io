#!/usr/bin/env python3
"""Predeclared spatial replication on temporally split King County sales.

The semantic intervention is fixed before training: only latitude and longitude
receive residual representers. The data are OpenML house_sales (id 42092),
sorted by sale date into 60/20/20 train/validation/test partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from residual_riesz_pilot import METHODS, build_features
from support_heat_pilot import (
    HERE,
    PARTS,
    Dataset,
    make_prepared,
    parameter_count,
    parameter_matched_width,
    read_rows,
    train_model,
    write_rows,
)
from tabm_support_pilot import FlatTabM, count_parameters, train_tabm


FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
SPATIAL_FIELDS = [FEATURES.index("lat"), FEATURES.index("long")]


def load_king_county() -> tuple[Dataset, dict[str, object]]:
    frame = fetch_openml(data_id=42092, as_frame=True, parser="auto").frame
    order = np.argsort(frame["date"].astype(str).to_numpy(), kind="stable")
    n = len(order)
    boundaries = (int(0.6 * n), int(0.8 * n))
    split_indices = {
        "train": order[: boundaries[0]],
        "val": order[boundaries[0] : boundaries[1]],
        "test": order[boundaries[1] :],
    }
    numeric = frame[FEATURES].to_numpy(dtype=np.float32)
    zipcode = frame["zipcode"].astype(str).to_numpy()[:, None]
    target = np.log(frame["price"].to_numpy(dtype=np.float64)).astype(np.float32)
    dates = frame["date"].astype(str).to_numpy()
    dataset = Dataset(
        name="king-county-sales",
        task="regression",
        x_num={part: numeric[index] for part, index in split_indices.items()},
        x_bin=None,
        x_cat={part: zipcode[index] for part, index in split_indices.items()},
        y={part: target[index] for part, index in split_indices.items()},
        n_classes=1,
        split_fingerprint="openml-42092-date-60-20-20",
    )
    metadata = {
        "source": "https://www.openml.org/d/42092",
        "feature_names": FEATURES,
        "spatial_fields": SPATIAL_FIELDS,
        "spatial_field_names": [FEATURES[index] for index in SPATIAL_FIELDS],
        "target": "log(price)",
        "split": "stable date order, 60/20/20",
        "rows": {part: int(len(index)) for part, index in split_indices.items()},
        "date_ranges": {
            part: [str(dates[index[0]]), str(dates[index[-1]])]
            for part, index in split_indices.items()
        },
    }
    return dataset, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet", "tabm"])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260850, 20260851, 20260852]
    )
    parser.add_argument("--wrong-permutation-seed", type=int, default=991337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/king_county_spatial.csv",
    )
    args = parser.parse_args()

    dataset, source_metadata = load_king_county()
    variants, feature_metadata = build_features(
        dataset,
        args.bins,
        args.strength,
        20260826,
        shared_raple_anchor=True,
        wrong_permutation_seed=args.wrong_permutation_seed,
        representer_fields=SPATIAL_FIELDS,
    )
    methods = args.methods or [method for method in METHODS if method in variants]
    rows = list(read_rows(args.output))
    complete = {
        (str(row["model"]), int(row["seed"]), str(row["method"])) for row in rows
    }

    for model in args.models:
        if model == "tabm":
            target_parameters = count_parameters(
                FlatTabM(
                    variants["quantile_ple"]["train"].shape[1],
                    1,
                    args.width,
                    args.depth,
                    args.members,
                )
            )
        else:
            target_parameters = parameter_count(
                model,
                variants["quantile_ple"]["train"].shape[1],
                1,
                args.width,
                args.depth,
            )
        for seed in args.seeds:
            for method in methods:
                key = (model, seed, method)
                if key in complete:
                    continue
                features = variants[method]
                prepared = make_prepared(dataset, features, {"method": method})
                if model == "tabm":
                    result = train_tabm(
                        prepared,
                        seed=seed,
                        device=args.device,
                        width=args.width,
                        depth=args.depth,
                        ensemble_size=args.members,
                        epochs=args.epochs,
                        patience=args.patience,
                        target_parameters=target_parameters,
                    )
                else:
                    width, parameters = parameter_matched_width(
                        model,
                        features["train"].shape[1],
                        1,
                        args.depth,
                        target_parameters,
                    )
                    result, _ = train_model(
                        prepared,
                        seed=seed,
                        device=args.device,
                        model_name=model,
                        width=width,
                        depth=args.depth,
                        dropout=0.1,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        batch_size=512,
                        max_epochs=args.epochs,
                        patience=args.patience,
                    )
                    result.update(
                        {
                            "width": width,
                            "parameters": parameters,
                            "parameter_error_fraction": (
                                parameters - target_parameters
                            )
                            / target_parameters,
                        }
                    )
                row = {
                    "dataset": dataset.name,
                    "task": dataset.task,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "strength": args.strength,
                    "target_parameters": target_parameters,
                    **result,
                }
                rows.append(row)
                complete.add(key)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)

    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {"dataset": source_metadata, "features": feature_metadata}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
