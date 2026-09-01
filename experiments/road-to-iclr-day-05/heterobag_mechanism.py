#!/usr/bin/env python3
"""Frozen HeteroBag triplet with homogeneous and coordinate-placebo controls."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import sys


HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "road-to-iclr-day-04"
sys.path.insert(0, str(DAY4))

from heterobag_three_member_pilot import (  # noqa: E402
    fit_classification_anchor,
    fit_regression_anchor,
)
from multiview_equal_compute_pilot import fit_rank  # noqa: E402
from openml_external_data import load_openml  # noqa: E402
from support_identity_transfer_pilot import (  # noqa: E402
    codes_for_method,
    make_loader,
    prepare_encodings,
)
from trichart_external_compute_controls import evaluate, fit_t_model, metrics  # noqa: E402
from trichart_frozen_anchor_classification import (  # noqa: E402
    prepare_classification_encodings,
)
from universal_mass_identity_pilot import prepare  # noqa: E402


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


def transformed_t_inputs(data, encoding):
    """Reverse numeric-field coordinates without adding a semantic chart."""

    transformed_data = replace(
        data,
        x_num={part: np.ascontiguousarray(values[:, ::-1]) for part, values in data.x_num.items()},
    )
    transformed_encoding = replace(
        encoding,
        qple_edges=np.ascontiguousarray(encoding.qple_edges[::-1]),
        tple_edges=np.ascontiguousarray(encoding.tple_edges[::-1]),
    )
    return transformed_data, transformed_encoding


def probability_or_value(prediction: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return 1.0 / (1.0 + np.exp(-np.clip(prediction, -40.0, 40.0)))
    return prediction


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def pair_diagnostics(left, right, target, task: str) -> dict[str, float]:
    left_value = probability_or_value(left, task)
    right_value = probability_or_value(right, task)
    return {
        "prediction_correlation": correlation(left_value, right_value),
        "error_correlation": correlation(left_value - target, right_value - target),
        "mean_absolute_disagreement": float(np.mean(np.abs(left_value - right_value))),
        "mean_squared_disagreement": float(np.mean((left_value - right_value) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if "base_config" in config:
        base = json.loads((args.config.parent / config["base_config"]).read_text())
        config = {**base, **{key: value for key, value in config.items() if key != "base_config"}}
    selected = args.dataset or config["development_datasets"]
    unknown = sorted(set(selected) - set(config["development_datasets"]))
    if unknown:
        raise ValueError(f"datasets are not in frozen config: {unknown}")
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in selected:
        task = config["dataset_tasks"][dataset_name]
        data = load_openml(dataset_name, config)
        universal = prepare(data, config)
        if task == "classification":
            encoding = prepare_classification_encodings(data, config)
            fit_anchor = fit_classification_anchor
        else:
            encoding = prepare_encodings(data, config)
            fit_anchor = fit_regression_anchor
        transformed_data, transformed_encoding = transformed_t_inputs(data, encoding)

        for architecture in config["architectures"]:
            if (dataset_name, architecture) in done:
                continue
            anchor, anchor_result = fit_anchor(data, encoding, config, architecture, args.device)
            target_parameters = int(anchor_result["anchor_parameters"])
            batch = min(config["batch_size"], 256) if architecture == "ft_transformer" else config["batch_size"]
            streams = {
                part: make_loader(
                    data, codes_for_method(encoding, "tple"), part,
                    2 * batch, False, config["seed"],
                )
                for part in ("val", "test")
            }
            resolved = torch.device(args.device)
            _, t_a_val, val_target = evaluate(anchor, streams["val"], resolved, task, data.y_scale)
            _, t_a_test, test_target = evaluate(anchor, streams["test"], resolved, task, data.y_scale)
            del anchor

            t_members = [(t_a_val, t_a_test, target_parameters)]
            for seed in (config["seed"] + 101, config["seed"] + 202):
                _, result, val, test, current_val_target, current_test_target = fit_t_model(
                    data, encoding, config, architecture, target_parameters,
                    seed, task, args.device, method="tple",
                )
                np.testing.assert_array_equal(current_val_target, val_target)
                np.testing.assert_array_equal(current_test_target, test_target)
                t_members.append((val, test, int(result["parameters"])))

            alternate_members = []
            alternate_parameters = []
            for seed in (config["seed"], config["seed"] + 101, config["seed"] + 202):
                if task == "classification":
                    _, result, val, test, current_val_target, current_test_target = fit_t_model(
                        data, encoding, config, architecture, target_parameters,
                        seed, task, args.device, method="qple",
                    )
                    parameters = int(result["parameters"])
                else:
                    result, val, test, current_val_target, current_test_target = fit_rank(
                        universal, config, architecture, target_parameters,
                        seed, task, args.device,
                    )
                    parameters = int(result["parameters"])
                np.testing.assert_array_equal(current_val_target, val_target)
                np.testing.assert_array_equal(current_test_target, test_target)
                alternate_members.append((val, test))
                alternate_parameters.append(parameters)

            _, transformed_result, transformed_val, transformed_test, transformed_val_target, transformed_test_target = fit_t_model(
                transformed_data, transformed_encoding, config, architecture,
                target_parameters, config["seed"] + 202, task, args.device,
                method="tple",
            )
            np.testing.assert_array_equal(transformed_val_target, val_target)
            np.testing.assert_array_equal(transformed_test_target, test_target)

            t_val = [item[0] for item in t_members]
            t_test = [item[1] for item in t_members]
            alt_val = [item[0] for item in alternate_members]
            alt_test = [item[1] for item in alternate_members]
            ensembles = {
                "heterobag": (
                    (t_val[0] + t_val[1] + alt_val[2]) / 3.0,
                    (t_test[0] + t_test[1] + alt_test[2]) / 3.0,
                ),
                "ttt": (np.mean(t_val, axis=0), np.mean(t_test, axis=0)),
                "alternate_homogeneous": (np.mean(alt_val, axis=0), np.mean(alt_test, axis=0)),
                "transformed_t_placebo": (
                    (t_val[0] + t_val[1] + transformed_val) / 3.0,
                    (t_test[0] + t_test[1] + transformed_test) / 3.0,
                ),
            }
            primary = "log_loss" if task == "classification" else "rmse"
            row: dict[str, object] = {
                "dataset": dataset_name,
                "task": task,
                "model": architecture,
                "alternate_view": "qple" if task == "classification" else "midrank",
                "seed_a": config["seed"],
                "seed_b": config["seed"] + 101,
                "seed_c": config["seed"] + 202,
                "target_parameters": target_parameters,
                "maximum_alternate_parameter_relative_mismatch": max(
                    abs(value - target_parameters) / target_parameters
                    for value in alternate_parameters
                ),
                "transformed_t_parameters": int(transformed_result["parameters"]),
            }
            for name, (val_prediction, test_prediction) in ensembles.items():
                for split, prediction, target in (
                    ("val", val_prediction, val_target),
                    ("test", test_prediction, test_target),
                ):
                    for metric_name, value in metrics(prediction, target, task, data.y_scale).items():
                        row[f"{name}_{split}_{metric_name}"] = value
            for prefix, left, right in (
                ("same_representation", t_test[1], t_test[2]),
                ("cross_representation", t_test[1], alt_test[2]),
                ("coordinate_placebo", t_test[1], transformed_test),
            ):
                for name, value in pair_diagnostics(left, right, test_target, task).items():
                    row[f"{prefix}_{name}"] = value
            reference = float(row[f"ttt_test_{primary}"])
            for name in ("heterobag", "alternate_homogeneous", "transformed_t_placebo"):
                row[f"{name}_relative_test_gain_vs_ttt_pct"] = (
                    100.0 * (reference - float(row[f"{name}_test_{primary}"])) / reference
                )

            np.savez_compressed(
                args.prediction_dir / f"{dataset_name}__{architecture}.npz",
                t_a_test=t_test[0], t_b_test=t_test[1], t_c_test=t_test[2],
                alternate_a_test=alt_test[0], alternate_b_test=alt_test[1],
                alternate_c_test=alt_test[2], transformed_t_c_test=transformed_test,
                test_target=test_target,
            )
            rows.append(row)
            done.add((dataset_name, architecture))
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
