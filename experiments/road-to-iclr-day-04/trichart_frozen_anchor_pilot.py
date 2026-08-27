#!/usr/bin/env python3
"""Frozen T-PLE anchor plus a zero-start Q/midrank chart residual."""
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
from torch import Tensor, nn

from semantic_multiview_pilot import PARTS, MLPBackbone, ResNetBackbone, load_tabred
from support_identity_transfer_pilot import (
    MatchedFTTransformer,
    SupportModel,
    SupportTokenizer,
    build_matched_model,
    codes_for_method,
    evaluate as evaluate_typed,
    make_loader as make_typed_loader,
    parameter_count,
    prepare_encodings,
)
from trichart_shared_pilot import make_loader, move
from universal_mass_identity_pilot import UniversalTokenizer, prepare


HERE = Path(__file__).resolve().parent


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
    batch_size = min(config["batch_size"], 256) if architecture == "ft_transformer" else config["batch_size"]
    streams = {
        part: make_typed_loader(
            data,
            codes,
            part,
            batch_size if part == "train" else 2 * batch_size,
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
    best, best_epoch, stale, state = math.inf, 0, 0, None
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
                loss = torch.nn.functional.mse_loss(prediction, target.to(resolved))
            loss.backward()
            optimizer.step()
        validation, _ = evaluate_typed(anchor, streams["val"], resolved, data.y_scale)
        if validation["loss"] < best:
            best, best_epoch, stale = validation["loss"], epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in anchor.state_dict().items()}
        else:
            stale += 1
        if stale > config["patience"]:
            break
    assert state is not None
    anchor.load_state_dict(state)
    validation, _ = evaluate_typed(anchor, streams["val"], resolved, data.y_scale)
    test = None
    if evaluate_test:
        test, _ = evaluate_typed(anchor, streams["test"], resolved, data.y_scale)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    result = {
        "anchor_parameters": parameter_count(anchor),
        "anchor_target_parameters": target_parameters,
        "anchor_width": width,
        "anchor_ft_feedforward_width": ff_width,
        "anchor_best_epoch": best_epoch,
        "anchor_val_loss": validation["loss"],
        "anchor_val_rmse": validation["rmse"],
    }
    if test is not None:
        result.update(
            anchor_test_loss=test["loss"],
            anchor_test_rmse=test["rmse"],
        )
    return anchor, result


class FrozenAnchorResidual(nn.Module):
    def __init__(self, anchor: SupportModel, data, universal, encoding, config: dict, architecture: str) -> None:
        super().__init__()
        self.anchor = anchor
        n_bin = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
        common = {
            "n_bin_fields": n_bin,
            "category_cardinalities": data.category_cardinalities,
            "support_columns": encoding.selected_columns,
            "support_cardinalities": encoding.cardinalities,
            "d_token": config["d_token"],
            "use_support": False,
        }
        self.q_tokenizer = SupportTokenizer(edges=encoding.qple_edges, **common)
        self.rank_tokenizer = UniversalTokenizer(universal.n_fields, config, "rank_only")
        n_fields = universal.n_fields
        if architecture == "mlp":
            self.residual_backbone = MLPBackbone(
                n_fields, config["d_token"], config["width"], config["depth"]
            )
        elif architecture == "resnet":
            self.residual_backbone = ResNetBackbone(
                n_fields, config["d_token"], config["width"], config["depth"]
            )
        elif architecture == "ft_transformer":
            self.residual_backbone = MatchedFTTransformer(
                n_fields,
                config["d_token"],
                config["depth"],
                config["ft_feedforward_width"],
                config["dropout"],
            )
        else:
            raise KeyError(architecture)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self,
        x_num: Tensor,
        x_bin: Tensor,
        x_cat: Tensor,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> tuple[Tensor, Tensor]:
        empty_codes = code[:, :0]
        with torch.no_grad():
            t_tokens = self.anchor.tokenizer(x_num, x_bin, x_cat, empty_codes)
            anchor_prediction = self.anchor.backbone(t_tokens)[0]
        q_tokens = self.q_tokenizer(x_num, x_bin, x_cat, empty_codes)
        rank_tokens = self.rank_tokenizer(
            rank, rank_lower, rank_upper, code, information
        )
        residual_tokens = 0.5 * (
            (q_tokens - t_tokens.detach()) + (rank_tokens - t_tokens.detach())
        )
        residual_prediction = self.residual_backbone(residual_tokens)[0]
        return anchor_prediction + self.residual_gate * residual_prediction, anchor_prediction


@torch.inference_mode()
def evaluate(model: FrozenAnchorResidual, stream, device: torch.device, scale: float):
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
    loss = float(np.mean((prediction - truth) ** 2))
    anchor_loss = float(np.mean((anchor - truth) ** 2))
    return {
        "loss": loss,
        "rmse": math.sqrt(loss) * scale,
        "anchor_loss": anchor_loss,
        "anchor_rmse": math.sqrt(anchor_loss) * scale,
    }, prediction


def train_residual(data, universal, encoding, anchor, config: dict, architecture: str, device: str):
    seed = config["seed"]
    random.seed(seed + 101)
    np.random.seed(seed + 101)
    torch.manual_seed(seed + 101)
    torch.cuda.manual_seed_all(seed + 101)
    resolved = torch.device(device)
    model = FrozenAnchorResidual(anchor, data, universal, encoding, config, architecture).to(resolved)
    batch_size = min(config["batch_size"], 256) if architecture == "ft_transformer" else config["batch_size"]
    local_config = {**config, "batch_size": batch_size}
    streams = {
        part: make_loader(data, universal, part, local_config, part == "train")
        for part in PARTS
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    initial, _ = evaluate(model, streams["val"], resolved, data.y_scale)
    best, best_epoch, stale = initial["loss"], 0, 0
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch in streams["train"]:
            *features, target = move(batch, resolved)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction, _ = model(*features)
                loss = torch.nn.functional.mse_loss(prediction, target)
                loss = loss + config["residual_gate_l1_weight"] * model.residual_gate.abs()
            loss.backward()
            optimizer.step()
        validation, _ = evaluate(model, streams["val"], resolved, data.y_scale)
        if validation["loss"] < best:
            best, best_epoch, stale = validation["loss"], epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > config["patience"]:
            break
    model.load_state_dict(state)
    validation, val_prediction = evaluate(model, streams["val"], resolved, data.y_scale)
    test, test_prediction = evaluate(model, streams["test"], resolved, data.y_scale)
    return {
        "residual_parameters": sum(parameter.numel() for parameter in trainable),
        "residual_best_epoch": best_epoch,
        "val_loss": validation["loss"],
        "val_rmse": validation["rmse"],
        "val_anchor_rmse": validation["anchor_rmse"],
        "test_loss": test["loss"],
        "test_rmse": test["rmse"],
        "test_anchor_rmse": test["anchor_rmse"],
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
    parser.add_argument("--config", type=Path, default=HERE / "trichart_frozen_anchor_config.json")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results/trichart_frozen_anchor.csv")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    metadata = {"config": config, "datasets": {}}
    for dataset_name in args.datasets or config["development_datasets"]:
        data = load_tabred(
            dataset_name,
            max_train_rows=config["max_train_rows"],
            max_eval_rows=config["max_eval_rows"],
            sample_seed=config["sample_seed"],
        )
        universal = prepare(data, config)
        encoding = prepare_encodings(data, config)
        metadata["datasets"][dataset_name] = {
            "n_fields": universal.n_fields,
            "full_split_sizes": data.split_sizes_full,
        }
        models = args.models or (
            config["maps_architectures"]
            if dataset_name == "maps-routing"
            else config["architectures"]
        )
        for model_name in models:
            key = (dataset_name, model_name)
            if key in done:
                continue
            started = time.perf_counter()
            anchor, anchor_result = fit_anchor(
                data, encoding, config, model_name, args.device
            )
            residual_result, val, test = train_residual(
                data,
                universal,
                encoding,
                anchor,
                config,
                model_name,
                args.device,
            )
            directory = args.output.parent / f"{args.output.stem}_predictions"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{dataset_name}__{model_name}__{config['seed']}.npz").open("wb") as handle:
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
