#!/usr/bin/env python3
"""Cross-fitted signal selection for zero-start exact-support tokens."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from semantic_multiview_pilot import PARTS, load_tabred
from support_identity_transfer_pilot import (
    Encodings,
    METHODS,
    exact_support_codes,
    prepare_encodings,
    quantile_bin_codes,
    read_rows,
    train_one,
    write_rows,
)


HERE = Path(__file__).resolve().parent


def smoothed_map(
    fit_key: np.ndarray,
    fit_target: np.ndarray,
    query: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    levels, inverse = np.unique(fit_key, return_inverse=True)
    counts = np.bincount(inverse)
    prior = float(fit_target.mean())
    means = (
        np.bincount(inverse, weights=fit_target) + smoothing * prior
    ) / (counts + smoothing)
    positions = np.searchsorted(levels, query)
    clipped = np.minimum(positions, len(levels) - 1)
    known = (positions < len(levels)) & (levels[clipped] == query)
    output = np.full(len(query), prior, dtype=np.float64)
    output[known] = means[clipped[known]]
    return output


def signal_scores(
    x: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> list[dict[str, object]]:
    cards = np.array([len(np.unique(x[:, field])) for field in range(x.shape[1])])
    candidates = np.flatnonzero(
        (cards >= 2) & (cards <= config["support_cardinality_max"])
    )
    splitter = KFold(
        config["selector_folds"], shuffle=True, random_state=2701
    )
    folds = list(splitter.split(x))
    rows = []
    for field in candidates:
        edges = np.unique(
            np.quantile(x[:, field], np.linspace(0.0, 1.0, config["qple_bins"] + 1))
        )
        bin_code = np.searchsorted(edges[1:-1], x[:, field], side="right")
        fold_gains = []
        for fit, holdout in folds:
            exact_prediction = smoothed_map(
                x[fit, field], y[fit], x[holdout, field], config["selector_smoothing"]
            )
            bin_prediction = smoothed_map(
                bin_code[fit], y[fit], bin_code[holdout], config["selector_smoothing"]
            )
            exact_loss = float(np.mean((exact_prediction - y[holdout]) ** 2))
            bin_loss = float(np.mean((bin_prediction - y[holdout]) ** 2))
            fold_gains.append(100 * (bin_loss - exact_loss) / max(bin_loss, 1e-12))
        mean_gain = float(np.mean(fold_gains))
        positive = int(np.sum(np.asarray(fold_gains) > 0))
        rows.append(
            {
                "column": int(field),
                "cardinality": int(cards[field]),
                "mean_relative_gain_pct": mean_gain,
                "positive_folds": positive,
                "fold_gains_pct": ";".join(f"{value:.6f}" for value in fold_gains),
                "selected": bool(
                    positive >= config["selector_min_positive_folds"]
                    and mean_gain >= config["selector_min_relative_gain_pct"]
                ),
            }
        )
    return rows


def subset_encoding(
    full: Encodings,
    x_num: dict[str, np.ndarray],
    selected: list[int],
) -> Encodings:
    positions = [full.selected_columns.index(column) for column in selected]
    cardinalities = [full.cardinalities[position] for position in positions]
    exact = {
        part: values[:, positions].copy()
        for part, values in full.exact_codes.items()
    }
    binned = quantile_bin_codes(x_num, full.qple_edges, selected, cardinalities)
    return Encodings(
        full.qple_edges,
        full.tple_edges,
        selected,
        cardinalities,
        exact,
        binned,
    )


def prediction_path(
    output: Path, dataset: str, model: str, seed: int, method: str
) -> Path:
    return output.parent / f"{output.stem}_predictions" / f"{dataset}__{model}__{seed}__{method}.npz"


def analyze(config: dict, output: Path) -> dict[str, object]:
    frame = pd.read_csv(output)
    cells = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        scores = group.set_index("method")
        if not set(METHODS).issubset(scores.index):
            continue
        q_gain = 100 * (scores.loc["qple", "val_rmse"] - scores.loc["qple_support", "val_rmse"]) / scores.loc["qple", "val_rmse"]
        control_gain = 100 * (scores.loc["qple_bin_control", "val_rmse"] - scores.loc["qple_support", "val_rmse"]) / scores.loc["qple_bin_control", "val_rmse"]
        t_gain = 100 * (scores.loc["tple", "val_rmse"] - scores.loc["tple_support", "val_rmse"]) / scores.loc["tple", "val_rmse"]
        cells.append({
            "dataset": dataset, "model": model,
            "q_support_val_gain_pct": q_gain,
            "exact_vs_bin_val_gain_pct": control_gain,
            "t_support_val_gain_pct": t_gain,
            "q_support_test_gain_pct": 100 * (scores.loc["qple", "test_rmse"] - scores.loc["qple_support", "test_rmse"]) / scores.loc["qple", "test_rmse"],
            "t_support_test_gain_pct": 100 * (scores.loc["tple", "test_rmse"] - scores.loc["tple_support", "test_rmse"]) / scores.loc["tple", "test_rmse"],
            "cell_gate": bool(q_gain > 0 and control_gain > 0 and t_gain > 0),
        })
    cells_frame = pd.DataFrame(cells)
    cells_frame.to_csv(output.with_name(output.stem + "_cells.csv"), index=False)
    development = cells_frame[cells_frame.dataset.isin(config["development_datasets"])]
    dataset_gates = {
        dataset: int(group.cell_gate.sum()) >= 2
        for dataset, group in development.groupby("dataset")
    }
    passed = any(dataset_gates.values())
    decision = {
        "architecture_gate_passed": passed,
        "dataset_gates": dataset_gates,
        "passing_development_cells": int(development.cell_gate.sum()),
        "development_cells": int(len(development)),
        "transfer_authorized": passed,
    }
    output.with_name(output.stem + "_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(cells_frame.to_string(index=False)); print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "signal_gated_support_config.json")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results/signal_gated_support.csv")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.analyze_only: analyze(config, args.output); return
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {(row["dataset"], row["model"], row["method"]) for row in rows}
    selector_metadata = {}
    for dataset_name in args.datasets or config["development_datasets"]:
        data = load_tabred(dataset_name, max_train_rows=config["max_train_rows"], max_eval_rows=config["max_eval_rows"], sample_seed=config["sample_seed"])
        full = prepare_encodings(data, config)
        diagnostics = signal_scores(data.x_num["train"], data.y["train"], config)
        selected = [int(row["column"]) for row in diagnostics if row["selected"]]
        if not selected:
            raise RuntimeError(f"signal selector chose no fields for {dataset_name}")
        encoding = subset_encoding(full, data.x_num, selected)
        selector_metadata[dataset_name] = diagnostics
        for model in args.models or config["architectures"]:
            for method in args.methods or config["methods"]:
                key = (dataset_name, model, method)
                if key in completed: continue
                result, val_prediction, test_prediction = train_one(data, encoding, config, method, model, config["seed"], args.device)
                path = prediction_path(args.output, dataset_name, model, config["seed"], method)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle: np.savez_compressed(handle, validation=val_prediction, test=test_prediction)
                row = {"dataset": dataset_name, "model": model, "method": method, "seed": config["seed"], "selected_columns": ";".join(map(str, selected)), "n_selected": len(selected), **result}
                rows.append(row); completed.add(key); write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({"config": config, "selector": selector_metadata}, indent=2, sort_keys=True) + "\n")
    required = {(dataset, model, method) for dataset in config["development_datasets"] for model in config["architectures"] for method in config["methods"]}
    if required.issubset(completed): analyze(config, args.output)


if __name__ == "__main__": main()
