#!/usr/bin/env python3
"""Parameter-matched TabM transport of the shared-anchor Residual-Riesz control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_riesz_pilot import METHODS, build_features
from support_heat_pilot import HERE, load_dataset, load_tabred, make_prepared
from tabm_support_pilot import (
    FlatTabM,
    count_parameters,
    read_rows,
    train_tabm,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["california", "tabred-weather"])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260850, 20260851, 20260852])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--wrong-permutation-seed", type=int, default=991337)
    parser.add_argument("--max-representer-fields", type=int, default=0)
    parser.add_argument("--representer-fields", nargs="+", type=int)
    parser.add_argument(
        "--field-selector",
        choices=(
            "mass_energy", "mass_oof_gain", "mass_oof_optimal_gain",
            "mass_oof_alignment_z", "energy", "energy_gap",
            "isospectral_gap", "isospectral_retention_gap", "oof_gain",
            "oof_optimal_gain", "oof_alignment_z"
        ),
        default="mass_energy",
    )
    parser.add_argument("--positive-selector-only", action="store_true")
    parser.add_argument("--selector-min-score", type=float, default=0.0)
    parser.add_argument("--selector-fdr", type=float, default=0.0)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/tabm_residual_riesz_vs_raple.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    complete = {
        (str(row["dataset"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    all_metadata: dict[str, object] = {}
    for name in args.datasets:
        if name.startswith("tabred-"):
            dataset = load_tabred(
                name.removeprefix("tabred-"),
                args.max_train_rows,
                args.max_eval_rows,
                20260826,
            )
        else:
            dataset = load_dataset(
                name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
        variants, metadata = build_features(
            dataset,
            args.bins,
            args.strength,
            20260826,
            shared_raple_anchor=True,
            wrong_permutation_seed=args.wrong_permutation_seed,
            max_representer_fields=args.max_representer_fields,
            representer_fields=args.representer_fields,
            field_selector=args.field_selector,
            positive_selector_only=args.positive_selector_only,
            selector_min_score=args.selector_min_score,
            selector_fdr=args.selector_fdr,
        )
        all_metadata[name] = metadata
        methods = args.methods or [method for method in METHODS if method in variants]
        target_parameters = count_parameters(
            FlatTabM(
                variants["quantile_ple"]["train"].shape[1],
                1,
                args.width,
                args.depth,
                args.members,
            )
        )
        for seed in args.seeds:
            for method in methods:
                key = (name, seed, method)
                if key in complete:
                    continue
                result = train_tabm(
                    make_prepared(dataset, variants[method], {"method": method}),
                    seed=seed,
                    device=args.device,
                    width=args.width,
                    depth=args.depth,
                    ensemble_size=args.members,
                    epochs=args.epochs,
                    patience=args.patience,
                    target_parameters=target_parameters,
                )
                row = {
                    "dataset": name,
                    "task": dataset.task,
                    "model": "tabm",
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
        json.dumps(all_metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
