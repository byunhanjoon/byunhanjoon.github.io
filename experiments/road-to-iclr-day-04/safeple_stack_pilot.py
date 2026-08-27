#!/usr/bin/env python3
"""Prospective validation-selected convex stack of two T-PLE models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from openml_external_data import load_openml
from support_identity_transfer_pilot import codes_for_method, make_loader, prepare_encodings
from trichart_external_compute_controls import evaluate, fit_t_model, metrics
from trichart_frozen_anchor_classification import (
    fit_anchor as fit_classification_anchor,
    prepare_classification_encodings,
)
from trichart_frozen_anchor_pilot import fit_anchor as fit_regression_anchor


HERE = Path(__file__).resolve().parent


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=HERE / "safeple_stack_prospective_config.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/safeple_stack_prospective.csv",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    for dataset_name in config["development_datasets"]:
        task = config["dataset_tasks"][dataset_name]
        data = load_openml(dataset_name, config)
        if task == "classification":
            encoding = prepare_classification_encodings(data, config)
            fit_anchor = fit_classification_anchor
        else:
            encoding = prepare_encodings(data, config)
            fit_anchor = fit_regression_anchor
        for architecture in config["architectures"]:
            key = (dataset_name, architecture)
            if key in done:
                continue
            anchor, anchor_result = fit_anchor(
                data, encoding, config, architecture, args.device
            )
            resolved = torch.device(args.device)
            codes = codes_for_method(encoding, "tple")
            batch = (
                min(config["batch_size"], 256)
                if architecture == "ft_transformer"
                else config["batch_size"]
            )
            streams = {
                part: make_loader(
                    data, codes, part, 2 * batch, False, config["seed"]
                )
                for part in ("val", "test")
            }
            _, anchor_val, val_target = evaluate(
                anchor, streams["val"], resolved, task, data.y_scale
            )
            _, anchor_test, test_target = evaluate(
                anchor, streams["test"], resolved, task, data.y_scale
            )
            del anchor
            _, second_result, second_val, second_test, _, _ = fit_t_model(
                data,
                encoding,
                config,
                architecture,
                int(anchor_result["anchor_parameters"]),
                config["seed"] + 101,
                task,
                args.device,
            )
            metric = "log_loss" if task == "classification" else "rmse"
            candidates = []
            for alpha in config["alpha_grid"]:
                prediction = (1.0 - alpha) * anchor_val + alpha * second_val
                score = metrics(prediction, val_target, task, data.y_scale)
                candidates.append((score[metric], float(alpha), score))
            _, selected_alpha, selected_val = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            selected_test = metrics(
                (1.0 - selected_alpha) * anchor_test
                + selected_alpha * second_test,
                test_target,
                task,
                data.y_scale,
            )
            fixed_val = metrics(
                0.5 * (anchor_val + second_val), val_target, task, data.y_scale
            )
            fixed_test = metrics(
                0.5 * (anchor_test + second_test), test_target, task, data.y_scale
            )
            row = {
                "task": task,
                "dataset": dataset_name,
                "model": architecture,
                "method": config["method"],
                "seed": config["seed"],
                "selected_alpha": selected_alpha,
                **anchor_result,
                **{f"second_{name}": value for name, value in second_result.items()},
                **{f"stack_val_{name}": value for name, value in selected_val.items()},
                **{f"stack_test_{name}": value for name, value in selected_test.items()},
                **{f"fixed_val_{name}": value for name, value in fixed_val.items()},
                **{f"fixed_test_{name}": value for name, value in fixed_test.items()},
            }
            rows.append(row)
            done.add(key)
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
