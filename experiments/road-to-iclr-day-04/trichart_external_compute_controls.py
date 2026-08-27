#!/usr/bin/env python3
"""Independent-ensemble and wide-T-PLE controls for the external cascade."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openml_external_data import load_openml
from semantic_multiview_pilot import PARTS
from support_identity_transfer_pilot import (
    SupportModel,
    build_matched_model,
    codes_for_method,
    make_loader,
    parameter_count,
    prepare_encodings,
)
from trichart_frozen_anchor_classification import (
    _classification_metrics,
    fit_anchor as fit_classification_anchor,
    prepare_classification_encodings,
)
from trichart_frozen_anchor_pilot import fit_anchor as fit_regression_anchor


HERE = Path(__file__).resolve().parent


def build_external_matched_model(
    data,
    encoding,
    config: dict,
    architecture: str,
    target_parameters: int,
    method: str = "tple",
):
    if architecture != "ft_transformer":
        return build_matched_model(
            data, encoding, config, method, architecture, target_parameters
        )

    def build(ff_width: int):
        return SupportModel(
            data=data,
            encoding=encoding,
            method=method,
            architecture=architecture,
            d_token=config["d_token"],
            width=config["width"],
            depth=config["depth"],
            ft_feedforward_width=ff_width,
            dropout=config["dropout"],
        )

    low, high = 4, 2048
    while low < high:
        middle = (low + high) // 2
        if parameter_count(build(middle)) < target_parameters:
            low = middle + 1
        else:
            high = middle
    candidates = {max(4, low - 1), low}
    chosen = min(
        candidates,
        key=lambda value: abs(parameter_count(build(value)) - target_parameters),
    )
    return build(chosen), config["width"], chosen


@torch.inference_mode()
def evaluate(model, stream, device: torch.device, task: str, scale: float):
    model.eval()
    predictions, targets = [], []
    for x_num, x_bin, x_cat, codes, target in stream:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                x_num.to(device),
                x_bin.to(device),
                x_cat.to(device),
                codes.to(device),
            )
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions).numpy()
    truth = torch.cat(targets).numpy()
    if task == "classification":
        metrics = _classification_metrics(prediction, truth)
    else:
        mse = float(np.mean((prediction - truth) ** 2))
        metrics = {"loss": mse, "rmse": math.sqrt(mse) * scale}
    return metrics, prediction, truth


def metrics(prediction: np.ndarray, target: np.ndarray, task: str, scale: float):
    if task == "classification":
        return _classification_metrics(prediction, target)
    mse = float(np.mean((prediction - target) ** 2))
    return {"loss": mse, "rmse": math.sqrt(mse) * scale}


def fit_t_model(
    data,
    encoding,
    config: dict,
    architecture: str,
    target_parameters: int,
    training_seed: int,
    task: str,
    device: str,
    method: str = "tple",
):
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    resolved = torch.device(device)
    model, width, ff_width = build_external_matched_model(
        data, encoding, config, architecture, target_parameters, method
    )
    model = model.to(resolved)
    codes = codes_for_method(encoding, method)
    batch = (
        min(config["batch_size"], 256)
        if architecture == "ft_transformer"
        else config["batch_size"]
    )
    streams = {
        part: make_loader(
            data,
            codes,
            part,
            batch if part == "train" else 2 * batch,
            part == "train",
            training_seed,
        )
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    metric = "log_loss" if task == "classification" else "loss"
    best, best_epoch, stale, state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for x_num, x_bin, x_cat, support_codes, target in streams["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction = model(
                    x_num.to(resolved),
                    x_bin.to(resolved),
                    x_cat.to(resolved),
                    support_codes.to(resolved),
                )
                loss = criterion(prediction, target.to(resolved))
            loss.backward()
            optimizer.step()
        validation, _, _ = evaluate(
            model, streams["val"], resolved, task, data.y_scale
        )
        if validation[metric] < best:
            best, best_epoch, stale = validation[metric], epoch, 0
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > config["patience"]:
            break
    assert state is not None
    model.load_state_dict(state)
    validation, val_prediction, val_target = evaluate(
        model, streams["val"], resolved, task, data.y_scale
    )
    test, test_prediction, test_target = evaluate(
        model, streams["test"], resolved, task, data.y_scale
    )
    return model, {
        "parameters": parameter_count(model),
        "matched_width": width,
        "matched_ft_feedforward_width": ff_width,
        "best_epoch": best_epoch,
        "train_seconds": time.perf_counter() - started,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test.items()},
    }, val_prediction, test_prediction, val_target, test_target


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    candidate = {
        (row["dataset"], row["model"]): row
        for row in csv.DictReader(args.candidate.open())
    }
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    for dataset_name in config["development_datasets"]:
        data = load_openml(dataset_name, config)
        if args.task == "classification":
            encoding = prepare_classification_encodings(data, config)
            fit_anchor = fit_classification_anchor
        else:
            encoding = prepare_encodings(data, config)
            fit_anchor = fit_regression_anchor
        for architecture in config["architectures"]:
            key = (dataset_name, architecture)
            if key in done:
                continue
            candidate_row = candidate[key]
            residual_budget = int(candidate_row["residual_target_parameters"])
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
            eval_streams = {
                part: make_loader(
                    data, codes, part, 2 * batch, False, config["seed"]
                )
                for part in ("val", "test")
            }
            _, anchor_val, val_target = evaluate(
                anchor, eval_streams["val"], resolved, args.task, data.y_scale
            )
            _, anchor_test, test_target = evaluate(
                anchor, eval_streams["test"], resolved, args.task, data.y_scale
            )
            del anchor
            _, second_result, second_val, second_test, _, _ = fit_t_model(
                data,
                encoding,
                config,
                architecture,
                residual_budget,
                config["seed"] + 101,
                args.task,
                args.device,
            )
            ensemble_val = metrics(
                0.5 * (anchor_val + second_val), val_target, args.task, data.y_scale
            )
            ensemble_test = metrics(
                0.5 * (anchor_test + second_test), test_target, args.task, data.y_scale
            )
            wide_budget = int(anchor_result["anchor_parameters"]) + residual_budget
            _, wide_result, _, _, _, _ = fit_t_model(
                data,
                encoding,
                config,
                architecture,
                wide_budget,
                config["seed"],
                args.task,
                args.device,
            )
            row = {
                "task": args.task,
                "dataset": dataset_name,
                "model": architecture,
                "seed": config["seed"],
                "anchor_parameters": anchor_result["anchor_parameters"],
                "residual_budget": residual_budget,
                "combined_budget": wide_budget,
                **{f"second_{key}": value for key, value in second_result.items()},
                **{f"ensemble_val_{key}": value for key, value in ensemble_val.items()},
                **{f"ensemble_test_{key}": value for key, value in ensemble_test.items()},
                **{f"wide_{key}": value for key, value in wide_result.items()},
            }
            rows.append(row)
            done.add(key)
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
