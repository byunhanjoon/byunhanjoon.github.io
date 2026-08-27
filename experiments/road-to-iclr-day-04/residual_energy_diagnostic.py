#!/usr/bin/env python3
"""Chart-invariant residual-energy diagnostics under a shared RAPLE anchor."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from residual_riesz_pilot import field_forms, fit_shared_raple, representer_values
from support_heat_pilot import (
    HERE,
    PARTS,
    clean_numeric,
    hat_basis,
    load_dataset,
    load_tabred,
    support_nodes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=[
            "california", "tabred-cooking-time", "tabred-delivery-eta",
            "tabred-maps-routing", "tabred-weather",
        ],
    )
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/residual_riesz_energy.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
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
        _, _, residual, folds, _ = fit_shared_raple(dataset, 20260826)
        clean = clean_numeric(dataset.x_num)
        for column in range(clean["train"].shape[1]):
            nodes = support_nodes(clean["train"][:, column], args.bins, 0.35)
            hats = {
                part: hat_basis(clean[part][:, column], nodes) for part in PARTS
            }
            mean = hats["train"].mean(axis=0)
            phi = {part: values - mean for part, values in hats.items()}
            covector = phi["train"].T @ residual / len(residual)
            operators = field_forms(
                phi["train"], nodes, column, args.strength
            )
            for kind, operator in zip(
                ("mass", "riesz", "wrong", "isospectral"), operators
            ):
                coefficients = np.linalg.pinv(operator, rcond=1e-10) @ covector
                energy = float(covector @ coefficients)
                oof = representer_values(
                    phi, residual, folds, operator
                )["train"].reshape(-1)
                oof_gain = float(np.mean(2 * residual * oof - oof * oof))
                rows.append(
                    {
                        "dataset": name,
                        "column": column,
                        "kind": kind,
                        "nodes": len(nodes),
                        "residual_energy": energy,
                        "oof_squared_loss_gain": oof_gain,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
