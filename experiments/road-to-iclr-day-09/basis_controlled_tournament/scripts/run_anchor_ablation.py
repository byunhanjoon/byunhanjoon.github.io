#!/usr/bin/env python3
"""Training-only anchor-count, selection, normalization, and rank ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    bd,
    development_specs,
    load_blocks,
    load_prior_protocol,
    load_protocol,
    orthogonal_all_orbit,
    read_prediction_bundle,
    save_prediction_bundle,
    task_error,
)
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


VARIANTS = [
    {"anchors": anchors, "selection": selection, "normalize": True}
    for anchors in (8, 16, 32)
    for selection in ("random_index", "gram_pivot")
] + [{"anchors": 16, "selection": "gram_pivot", "normalize": False}]


def run(spec, seed: int, device: str) -> None:
    protocol = load_protocol()
    prior = load_prior_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    rows = []
    for parameters in VARIANTS:
        label = f"m{parameters['anchors']}-{parameters['selection']}-norm{int(parameters['normalize'])}"
        mapped = [
            build_interface(rep, "gram_anchor", blocks.dataset.key, **parameters) for rep in orbit
        ]
        audit = audit_orbit_coordinates(mapped[0], mapped[1:])
        maximum = max(
            max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"])
            for row in audit
        )
        if maximum >= 1e-8:
            raise RuntimeError(f"non-invariant anchor variant {label}: {maximum}")
        path = (
            ROOT
            / "results"
            / "raw"
            / "anchor_ablation"
            / blocks.dataset.key
            / f"seed_{seed}"
            / f"{label}.npz"
        )
        if path.exists():
            validation, test, metadata = read_prediction_bundle(path)
        else:
            validation, test, telemetry = bd.fit_predict(
                "controlled_mlp",
                blocks.dataset.problem_type,
                mapped[0],
                blocks.dataset.y_train,
                blocks.dataset.y_validation,
                seed,
                device,
                prior,
            )
            metadata = {"parameters": parameters, "telemetry": telemetry, "max_coordinate_error": maximum}
            save_prediction_bundle(path, validation, test, metadata)
        block_audits = mapped[0].metadata["block_audits"]
        ranks = [int(value["empirical_rank"]) for value in block_audits.values()]
        pivot_ranks = [int(value["pivot_or_anchor_rank"]) for value in block_audits.values()]
        rows.append(
            {
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": "controlled_mlp",
                "seed": seed,
                "variant": label,
                **parameters,
                "min_empirical_rank": min(ranks),
                "median_empirical_rank": float(np.median(ranks)),
                "min_anchor_rank": min(pivot_ranks),
                "max_coordinate_error": maximum,
                "validation_task_error": task_error(
                    blocks.dataset.problem_type, blocks.dataset.y_validation, validation
                ),
                "test_task_error": task_error(blocks.dataset.problem_type, blocks.dataset.y_test, test),
                "fit_seconds": float(metadata["telemetry"]["fit_seconds"]),
            }
        )
        print(f"[anchor] {blocks.dataset.key} seed={seed} {label}", flush=True)
    destination = ROOT / "results" / "processed" / "anchor_ablation"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / f"{blocks.dataset.key}__seed_{seed}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run(spec, int(seed), args.device)


if __name__ == "__main__":
    main()
