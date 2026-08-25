"""Rank-deficiency, ridge, and canonicalization scalability stress tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .broad_data import Representation, config, controlled_representation, load_broad_dataset
from .core import geometry, make_prepared
from .run_broad_benchmark import _transform, train_cell


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def duplicate(rep: Representation, fraction: float) -> Representation:
    dimension = rep.parts["train"].shape[1]
    count = int(round(dimension * fraction))
    if count == 0:
        return rep
    rng = np.random.default_rng(77123)
    columns = np.sort(rng.choice(dimension, count, replace=count > dimension))
    values = {
        part: np.ascontiguousarray(np.column_stack((matrix, matrix[:, columns])), dtype=np.float64)
        for part, matrix in rep.parts.items()
    }
    return Representation(
        name=rep.name + f"_duplicate_{fraction:g}",
        parts=values,
        metadata={**rep.metadata, "duplicate_fraction": fraction, "duplicate_columns": columns.tolist()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=RESULTS / "robustness.csv")
    args = parser.parse_args()
    cfg = config()
    selected = json.loads((RESULTS / "selected_hyperparameters.json").read_text())["selected"]
    datasets = [
        name
        for index, name in enumerate(cfg["robustness"]["datasets"])
        if index % args.num_shards == args.shard
    ]
    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    complete = {
        (row["dataset"], row["duplicate_fraction"], row["target_kappa"], row["remedy"], row["ridge"], row["seed"])
        for row in rows
    }
    remedies = ["adamw", "anchor_whiten_adamw", "sketch_anchor_whiten_adamw", "input_natural"]
    for dataset_name in datasets:
        dataset = load_broad_dataset(dataset_name)
        for fraction in cfg["robustness"]["duplicate_fractions"]:
            for kappa in cfg["kappas"]:
                rep = duplicate(controlled_representation(dataset, kappa), float(fraction))
                for remedy in remedies:
                    ridges = cfg["robustness"]["ridge_relative"] if remedy == "input_natural" else [1e-8]
                    setting = selected.get(remedy, selected.get("anchor_whiten_adamw", selected["adamw"]))
                    for ridge in ridges:
                        started = time.perf_counter()
                        try:
                            transformed, transform_meta = _transform(rep, remedy)
                            preprocessing_seconds = time.perf_counter() - started
                            prepared = make_prepared(dataset, transformed, {})
                            transform_failure = ""
                        except Exception as error:
                            transformed = None
                            transform_meta = {}
                            prepared = None
                            preprocessing_seconds = time.perf_counter() - started
                            transform_failure = f"{type(error).__name__}: {error}"
                        for seed in cfg["broad_seeds"]:
                            key = (dataset_name, fraction, kappa, remedy, ridge, seed)
                            if key in complete:
                                continue
                            failure = transform_failure
                            fit = {}
                            if not failure:
                                try:
                                    fit, _ = train_cell(
                                        prepared,
                                        model_name="mlp",
                                        remedy=remedy,
                                        seed=seed,
                                        device=args.device,
                                        learning_rate=float(setting["learning_rate"]),
                                        ridge=float(ridge),
                                        precondition_frequency=int(setting.get("precondition_frequency", 10)),
                                    )
                                except Exception as error:
                                    failure = f"{type(error).__name__}: {error}"
                            row = {
                                "experiment": "rank_scalability_robustness",
                                "dataset": dataset_name,
                                "task": dataset.task,
                                "duplicate_fraction": fraction,
                                "target_kappa": kappa,
                                "remedy": remedy,
                                "ridge": ridge,
                                "seed": seed,
                                "failure": failure,
                                "preprocessing_seconds": preprocessing_seconds,
                                "transform_metadata": json.dumps(transform_meta, sort_keys=True),
                                **(geometry(transformed["train"]) if transformed is not None else {}),
                                **fit,
                            }
                            rows.append(row)
                            write(args.output, rows)
                            status = failure or f"test={fit['test_primary']:.6f}"
                            print(
                                f"{dataset_name:34s} dup={fraction:.2f} κ={kappa:4g} "
                                f"{remedy:28s} ridge={ridge:g} s{seed} {status}",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
