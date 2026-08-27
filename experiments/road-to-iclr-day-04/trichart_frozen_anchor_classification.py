#!/usr/bin/env python3
"""Binary-classification replication of the frozen TriChart anchor residual."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn

from adult_identity_mechanism_pilot import classifier_tree_edges
from semantic_multiview_pilot import PARTS, SplitData, _encode_categories, quantile_edges
from support_identity_transfer_pilot import (
    Encodings,
    build_matched_model,
    codes_for_method,
    exact_support_codes,
    parameter_count,
    quantile_bin_codes,
)
from trichart_frozen_anchor_pilot import FrozenAnchorResidual
from trichart_shared_pilot import make_loader, move
from universal_mass_identity_pilot import prepare


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY1))

import real_data_benchmark as day1  # noqa: E402


def _subset_parts(
    parts: dict[str, np.ndarray], limits: dict[str, int], seed: int
) -> dict[str, np.ndarray]:
    output = {}
    for offset, part in enumerate(PARTS):
        values = parts[part]
        if len(values) > limits[part]:
            index = np.sort(
                np.random.default_rng(seed + offset).choice(
                    len(values), limits[part], replace=False
                )
            )
            output[part] = values[index]
        else:
            output[part] = values
    return output


def load_binary(name: str, config: dict) -> SplitData:
    source = day1.load_dataset(DAY1 / "data", name)
    if source.task != "binclass":
        raise ValueError(f"{name} is {source.task}, not binary classification")
    if source.x_num is None:
        raise ValueError(f"{name} has no numerical fields")
    limits = {
        "train": config["max_train_rows"],
        "val": config["max_eval_rows"],
        "test": config["max_eval_rows"],
    }
    full_sizes = {part: len(source.y[part]) for part in PARTS}
    selected_y = _subset_parts(source.y, limits, config["sample_seed"])
    # Recreate the same deterministic row indices for every feature block.
    def subset(values: dict[str, np.ndarray] | None):
        return (
            None
            if values is None
            else _subset_parts(values, limits, config["sample_seed"])
        )

    x_num = {
        part: np.ascontiguousarray(values, dtype=np.float32)
        for part, values in day1._clean_numeric(subset(source.x_num)).items()
    }
    raw_bin = subset(source.x_bin)
    x_bin = (
        None
        if raw_bin is None
        else {
            part: np.ascontiguousarray(values, dtype=np.float32)
            for part, values in day1._clean_numeric(raw_bin).items()
        }
    )
    raw_cat = subset(source.x_cat)
    if raw_cat is None:
        x_cat, cardinalities = None, []
    else:
        x_cat, cardinalities = _encode_categories(raw_cat)
    y = {
        part: np.ascontiguousarray(values, dtype=np.float32)
        for part, values in selected_y.items()
    }
    return SplitData(
        x_num=x_num,
        x_bin=x_bin,
        x_cat=x_cat,
        y=y,
        y_mean=0.0,
        y_scale=1.0,
        category_cardinalities=cardinalities,
        cyclic_columns=[],
        cyclic_names=[],
        cyclic_periods=[],
        cyclic_origins=[],
        split_sizes_full=full_sizes,
    )


def prepare_classification_encodings(data: SplitData, config: dict) -> Encodings:
    qple = quantile_edges(data.x_num["train"], config["qple_bins"])
    tple = classifier_tree_edges(
        data.x_num["train"],
        data.y["train"],
        config["tple_bins"],
        config["tple_min_samples_leaf"],
    )
    columns, cardinalities, exact = exact_support_codes(
        data.x_num, config["support_cardinality_max"]
    )
    binned = quantile_bin_codes(data.x_num, qple, columns, cardinalities)
    return Encodings(qple, tple, columns, cardinalities, exact, binned)


def _classification_metrics(logits: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return {
        "log_loss": float(log_loss(target, probability)),
        "auc": float(roc_auc_score(target, probability)),
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
    }


@torch.inference_mode()
def evaluate_anchor(model, stream, device: torch.device):
    model.eval()
    logits, targets = [], []
    for x_num, x_bin, x_cat, codes, target in stream:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                x_num.to(device), x_bin.to(device), x_cat.to(device), codes.to(device)
            )
        logits.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    logit = torch.cat(logits).numpy()
    truth = torch.cat(targets).numpy()
    return _classification_metrics(logit, truth), logit


def fit_anchor(
    data,
    encoding,
    config: dict,
    architecture: str,
    device: str,
    evaluate_test: bool = True,
):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    reference, _, _ = build_matched_model(
        data, encoding, config, "qple_support", architecture, None
    )
    target_parameters = parameter_count(reference)
    del reference
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    anchor, width, ff_width = build_matched_model(
        data, encoding, config, "tple", architecture, target_parameters
    )
    anchor = anchor.to(resolved)
    codes = codes_for_method(encoding, "tple")
    batch = (
        min(config["batch_size"], 256)
        if architecture == "ft_transformer"
        else config["batch_size"]
    )
    from adult_identity_mechanism_pilot import loader

    streams = {
        part: loader(
            data,
            codes,
            part,
            batch if part == "train" else 2 * batch,
            part == "train",
            seed,
        )
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(
        anchor.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    best, best_epoch, stale, state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        anchor.train()
        for x_num, x_bin, x_cat, support_codes, target in streams["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction = anchor(
                    x_num.to(resolved),
                    x_bin.to(resolved),
                    x_cat.to(resolved),
                    support_codes.to(resolved),
                )
                loss = criterion(prediction, target.to(resolved))
            loss.backward()
            optimizer.step()
        validation, _ = evaluate_anchor(anchor, streams["val"], resolved)
        if validation["log_loss"] < best:
            best, best_epoch, stale = validation["log_loss"], epoch, 0
            state = {
                key: value.detach().cpu().clone()
                for key, value in anchor.state_dict().items()
            }
        else:
            stale += 1
        if stale > config["patience"]:
            break
    assert state is not None
    anchor.load_state_dict(state)
    validation, _ = evaluate_anchor(anchor, streams["val"], resolved)
    test = None
    if evaluate_test:
        test, _ = evaluate_anchor(anchor, streams["test"], resolved)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    result = {
        "anchor_parameters": parameter_count(anchor),
        "anchor_target_parameters": target_parameters,
        "anchor_width": width,
        "anchor_ft_feedforward_width": ff_width,
        "anchor_best_epoch": best_epoch,
        "anchor_val_log_loss": validation["log_loss"],
        "anchor_val_auc": validation["auc"],
        "anchor_val_accuracy": validation["accuracy"],
        "anchor_train_seconds": time.perf_counter() - started,
    }
    if test is not None:
        result.update(
            anchor_test_log_loss=test["log_loss"],
            anchor_test_auc=test["auc"],
            anchor_test_accuracy=test["accuracy"],
        )
    return anchor, result


@torch.inference_mode()
def evaluate_residual(model: FrozenAnchorResidual, stream, device: torch.device):
    model.eval()
    predictions, anchors, targets = [], [], []
    for batch in stream:
        *features, target = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction, anchor = model(*features)
        predictions.append(prediction.float().cpu())
        anchors.append(anchor.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions).numpy()
    anchor = torch.cat(anchors).numpy()
    truth = torch.cat(targets).numpy()
    return (
        _classification_metrics(prediction, truth),
        _classification_metrics(anchor, truth),
        prediction,
    )


def train_residual(
    data, universal, encoding, anchor, config: dict, architecture: str, device: str
):
    seed = config["seed"]
    random.seed(seed + 101)
    np.random.seed(seed + 101)
    torch.manual_seed(seed + 101)
    torch.cuda.manual_seed_all(seed + 101)
    resolved = torch.device(device)
    model = FrozenAnchorResidual(
        anchor, data, universal, encoding, config, architecture
    ).to(resolved)
    batch = (
        min(config["batch_size"], 256)
        if architecture == "ft_transformer"
        else config["batch_size"]
    )
    streams = {
        part: make_loader(
            data, universal, part, {**config, "batch_size": batch}, part == "train"
        )
        for part in PARTS
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    initial, _, _ = evaluate_residual(model, streams["val"], resolved)
    best, best_epoch, stale = initial["log_loss"], 0, 0
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch_values in streams["train"]:
            *features, target = move(batch_values, resolved)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction, _ = model(*features)
                loss = criterion(prediction, target)
                loss = loss + config["residual_gate_l1_weight"] * model.residual_gate.abs()
            loss.backward()
            optimizer.step()
        validation, _, _ = evaluate_residual(model, streams["val"], resolved)
        if validation["log_loss"] < best:
            best, best_epoch, stale = validation["log_loss"], epoch, 0
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > config["patience"]:
            break
    model.load_state_dict(state)
    validation, val_anchor, val_prediction = evaluate_residual(
        model, streams["val"], resolved
    )
    test, test_anchor, test_prediction = evaluate_residual(
        model, streams["test"], resolved
    )
    return {
        "residual_parameters": sum(parameter.numel() for parameter in trainable),
        "residual_best_epoch": best_epoch,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"val_anchor_{key}": value for key, value in val_anchor.items()},
        **{f"test_{key}": value for key, value in test.items()},
        **{f"test_anchor_{key}": value for key, value in test_anchor.items()},
        "residual_gate": float(model.residual_gate.detach().cpu()),
        "residual_train_seconds": time.perf_counter() - started,
    }, val_prediction, test_prediction


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
        "--config",
        type=Path,
        default=HERE / "trichart_frozen_anchor_classification_config.json",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/trichart_frozen_anchor_classification.csv",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    metadata = {"config": config, "datasets": {}}
    for dataset_name in args.datasets or config["development_datasets"]:
        data = load_binary(dataset_name, config)
        universal = prepare(data, config)
        encoding = prepare_classification_encodings(data, config)
        metadata["datasets"][dataset_name] = {
            "n_fields": universal.n_fields,
            "full_split_sizes": data.split_sizes_full,
            "sampled_split_sizes": {
                part: len(data.y[part]) for part in PARTS
            },
        }
        for model_name in args.models or config["architectures"]:
            key = (dataset_name, model_name)
            if key in done:
                continue
            started = time.perf_counter()
            anchor, anchor_result = fit_anchor(
                data, encoding, config, model_name, args.device
            )
            residual_result, val, test = train_residual(
                data, universal, encoding, anchor, config, model_name, args.device
            )
            directory = args.output.parent / f"{args.output.stem}_predictions"
            directory.mkdir(parents=True, exist_ok=True)
            with (
                directory / f"{dataset_name}__{model_name}__{config['seed']}.npz"
            ).open("wb") as handle:
                np.savez_compressed(handle, validation=val, test=test)
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "method": config["method"],
                "seed": config["seed"],
                "n_fields": universal.n_fields,
                **anchor_result,
                **residual_result,
                "total_train_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            done.add(key)
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
