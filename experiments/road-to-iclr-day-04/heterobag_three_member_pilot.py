#!/usr/bin/env python3
"""Prospective equal-compute T+T+alternate versus T+T+T experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from multiview_equal_compute_pilot import fit_rank
from openml_external_data import load_openml
from support_identity_transfer_pilot import codes_for_method, make_loader, prepare_encodings
from trichart_external_compute_controls import evaluate, fit_t_model, metrics
from trichart_frozen_anchor_classification import (
    fit_anchor as fit_classification_anchor,
    prepare_classification_encodings,
)
from trichart_frozen_anchor_pilot import fit_anchor as fit_regression_anchor
from universal_mass_identity_pilot import prepare


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
    parser.add_argument("--config", type=Path, default=HERE / "heterobag_three_member_config.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/heterobag_three_member.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    for dataset_name in config["development_datasets"]:
        task = config["dataset_tasks"][dataset_name]
        data = load_openml(dataset_name, config)
        universal = prepare(data, config)
        if task == "classification":
            encoding = prepare_classification_encodings(data, config)
            fit_anchor = fit_classification_anchor
        else:
            encoding = prepare_encodings(data, config)
            fit_anchor = fit_regression_anchor
        for architecture in config["architectures"]:
            if (dataset_name, architecture) in done:
                continue
            anchor, anchor_result = fit_anchor(
                data, encoding, config, architecture, args.device
            )
            target_parameters = int(anchor_result["anchor_parameters"])
            batch = (
                min(config["batch_size"], 256)
                if architecture in {"ft_transformer", "hybrid"}
                else config["batch_size"]
            )
            streams = {
                part: make_loader(
                    data, codes_for_method(encoding, "tple"), part,
                    2 * batch, False, config["seed"],
                )
                for part in ("val", "test")
            }
            resolved = torch.device(args.device)
            _, a_val, val_target = evaluate(
                anchor, streams["val"], resolved, task, data.y_scale
            )
            _, a_test, test_target = evaluate(
                anchor, streams["test"], resolved, task, data.y_scale
            )
            del anchor
            _, t1_result, t1_val, t1_test, _, _ = fit_t_model(
                data, encoding, config, architecture, target_parameters,
                config["seed"] + 101, task, args.device, method="tple",
            )
            _, t2_result, t2_val, t2_test, _, _ = fit_t_model(
                data, encoding, config, architecture, target_parameters,
                config["seed"] + 202, task, args.device, method="tple",
            )
            view = config["fixed_view_by_task"][task]
            if view == "qple":
                _, alt_result, alt_val, alt_test, _, _ = fit_t_model(
                    data, encoding, config, architecture, target_parameters,
                    config["seed"] + 202, task, args.device, method="qple",
                )
                alt_parameters = alt_result["parameters"]
            elif view == "midrank":
                alt_result, alt_val, alt_test, _, _ = fit_rank(
                    universal, config, architecture, target_parameters,
                    config["seed"] + 202, task, args.device,
                )
                alt_parameters = alt_result["parameters"]
            else:
                raise KeyError(view)
            candidate_val = (a_val + t1_val + alt_val) / 3.0
            candidate_test = (a_test + t1_test + alt_test) / 3.0
            control_val = (a_val + t1_val + t2_val) / 3.0
            control_test = (a_test + t1_test + t2_test) / 3.0
            primary = "log_loss" if task == "classification" else "rmse"
            candidate_val_metrics = metrics(
                candidate_val, val_target, task, data.y_scale
            )
            candidate_test_metrics = metrics(
                candidate_test, test_target, task, data.y_scale
            )
            control_val_metrics = metrics(control_val, val_target, task, data.y_scale)
            control_test_metrics = metrics(control_test, test_target, task, data.y_scale)
            row: dict[str, object] = {
                "task": task,
                "dataset": dataset_name,
                "model": architecture,
                "alternate_view": view,
                "seed_a": config["seed"],
                "seed_b": config["seed"] + 101,
                "seed_c": config["seed"] + 202,
                "anchor_parameters": target_parameters,
                "t1_parameters": t1_result["parameters"],
                "t2_parameters": t2_result["parameters"],
                "alternate_parameters": alt_parameters,
            }
            for split, candidate_metrics, control_metrics in (
                ("val", candidate_val_metrics, control_val_metrics),
                ("test", candidate_test_metrics, control_test_metrics),
            ):
                for name, value in candidate_metrics.items():
                    row[f"heterobag_{split}_{name}"] = value
                for name, value in control_metrics.items():
                    row[f"ttt_{split}_{name}"] = value
                row[f"relative_{split}_{primary}_gain_pct"] = (
                    100.0 * (control_metrics[primary] - candidate_metrics[primary])
                    / control_metrics[primary]
                )
            rows.append(row)
            done.add((dataset_name, architecture))
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
