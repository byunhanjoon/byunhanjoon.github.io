#!/usr/bin/env python3
"""Equal-compute heterogeneous versus homogeneous two-member PLE ensembles."""
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
from support_identity_transfer_pilot import codes_for_method, make_loader, prepare_encodings
from trichart_external_compute_controls import evaluate, fit_t_model, metrics
from trichart_frozen_anchor_classification import (
    fit_anchor as fit_classification_anchor,
    prepare_classification_encodings,
)
from trichart_frozen_anchor_pilot import fit_anchor as fit_regression_anchor
from universal_mass_identity_pilot import UniversalModel, loader as rank_loader, prepare


HERE = Path(__file__).resolve().parent
PARTS = ("train", "val", "test")


def active_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_rank_model(universal, config: dict, architecture: str, target: int):
    def build(width: int, ff_width: int):
        model = UniversalModel(
            universal, config, "rank_only", architecture, width, ff_width
        )
        for name, parameter in model.tokenizer.named_parameters():
            if name not in {"rank_weight", "field_bias"}:
                parameter.requires_grad_(False)
        return model

    if architecture == "ft_transformer":
        low, high = 4, 2048
        while low < high:
            middle = (low + high) // 2
            if active_count(build(config["width"], middle)) < target:
                low = middle + 1
            else:
                high = middle
        candidates = {max(4, low - 1), low}
        ff_width = min(
            candidates,
            key=lambda value: abs(
                active_count(build(config["width"], value)) - target
            ),
        )
        return build(config["width"], ff_width), config["width"], ff_width
    low, high = 8, 1024
    while low < high:
        middle = (low + high) // 2
        if active_count(build(middle, config["ft_feedforward_width"])) < target:
            low = middle + 1
        else:
            high = middle
    candidates = {max(8, low - 1), low}
    width = min(
        candidates,
        key=lambda value: abs(
            active_count(build(value, config["ft_feedforward_width"])) - target
        ),
    )
    return build(width, config["ft_feedforward_width"]), width, config["ft_feedforward_width"]


@torch.inference_mode()
def evaluate_rank(model, stream, device: torch.device, task: str, scale: float):
    model.eval()
    predictions, targets = [], []
    for rank, lower, upper, code, information, target in stream:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                rank.to(device),
                lower.to(device),
                upper.to(device),
                code.to(device),
                information.to(device),
            )
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return metrics(prediction, target, task, scale), prediction, target


def fit_rank(
    universal,
    config: dict,
    architecture: str,
    target_parameters: int,
    training_seed: int,
    task: str,
    device: str,
):
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    model, width, ff_width = build_rank_model(
        universal, config, architecture, target_parameters
    )
    resolved = torch.device(device)
    model = model.to(resolved)
    batch = (
        min(config["batch_size"], 256)
        if architecture == "ft_transformer"
        else config["batch_size"]
    )
    streams = {
        part: rank_loader(
            universal,
            "rank_only",
            part,
            batch if part == "train" else 2 * batch,
            part == "train",
            training_seed,
        )
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    metric = "log_loss" if task == "classification" else "loss"
    best, best_epoch, stale, state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for rank, lower, upper, code, information, target in streams["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction = model(
                    rank.to(resolved),
                    lower.to(resolved),
                    upper.to(resolved),
                    code.to(resolved),
                    information.to(resolved),
                )
                loss = criterion(prediction, target.to(resolved))
            loss.backward()
            optimizer.step()
        validation, _, _ = evaluate_rank(
            model, streams["val"], resolved, task, universal.y_scale
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
    validation, val, val_target = evaluate_rank(
        model, streams["val"], resolved, task, universal.y_scale
    )
    test_metrics, test_prediction, test_target = evaluate_rank(
        model, streams["test"], resolved, task, universal.y_scale
    )
    return {
        "parameters": active_count(model),
        "matched_width": width,
        "matched_ft_feedforward_width": ff_width,
        "best_epoch": best_epoch,
        "train_seconds": time.perf_counter() - started,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }, val, test_prediction, val_target, test_target


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
        "--data-config", type=Path, default=HERE / "safeple_stack_prospective_config.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/multiview_equal_compute.csv",
    )
    args = parser.parse_args()
    config = json.loads(args.data_config.read_text())
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
            key = (dataset_name, architecture)
            if key in done:
                continue
            anchor, anchor_result = fit_anchor(
                data, encoding, config, architecture, args.device
            )
            target_parameters = int(anchor_result["anchor_parameters"])
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
            _, t_val, val_target = evaluate(
                anchor, streams["val"], resolved, task, data.y_scale
            )
            _, t_test, test_target = evaluate(
                anchor, streams["test"], resolved, task, data.y_scale
            )
            del anchor
            _, tt_result, tt_val, tt_test, _, _ = fit_t_model(
                data,
                encoding,
                config,
                architecture,
                target_parameters,
                config["seed"] + 101,
                task,
                args.device,
                method="tple",
            )
            fixed_pair = config.get("fixed_view_by_task", {}).get(task)
            q_result = rank_result = None
            pairs = {"tt": (tt_val, tt_test)}
            if fixed_pair in {None, "tq"}:
                _, q_result, q_val, q_test, _, _ = fit_t_model(
                    data,
                    encoding,
                    config,
                    architecture,
                    target_parameters,
                    config["seed"] + 101,
                    task,
                    args.device,
                    method="qple",
                )
                pairs["tq"] = (q_val, q_test)
            if fixed_pair in {None, "trank"}:
                rank_result, rank_val, rank_test, _, _ = fit_rank(
                    universal,
                    config,
                    architecture,
                    target_parameters,
                    config["seed"] + 101,
                    task,
                    args.device,
                )
                pairs["trank"] = (rank_val, rank_test)
            pair_metrics = {}
            for name, (val_member, test_member) in pairs.items():
                pair_metrics.update(
                    {
                        **{
                            f"{name}_val_{metric_name}": value
                            for metric_name, value in metrics(
                                0.5 * (t_val + val_member),
                                val_target,
                                task,
                                data.y_scale,
                            ).items()
                        },
                        **{
                            f"{name}_test_{metric_name}": value
                            for metric_name, value in metrics(
                                0.5 * (t_test + test_member),
                                test_target,
                                task,
                                data.y_scale,
                            ).items()
                        },
                    }
                )
            primary = "log_loss" if task == "classification" else "rmse"
            selected = fixed_pair or min(
                ("tq", "trank"),
                key=lambda name: pair_metrics[f"{name}_val_{primary}"],
            )
            row = {
                "task": task,
                "dataset": dataset_name,
                "model": architecture,
                "seed": config["seed"],
                "selected_heterogeneous_pair": selected,
                "anchor_parameters": target_parameters,
                "tt_parameters": tt_result["parameters"],
                "q_parameters": (
                    q_result["parameters"] if q_result is not None else ""
                ),
                "rank_active_parameters": (
                    rank_result["parameters"] if rank_result is not None else ""
                ),
                **pair_metrics,
            }
            rows.append(row)
            done.add(key)
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
