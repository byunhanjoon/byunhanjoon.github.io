#!/usr/bin/env python3
"""Shared-backbone fusion of Q-PLE, T-PLE, and universal midrank views."""
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
from torch.utils.data import DataLoader, TensorDataset

from semantic_multiview_pilot import PARTS, MLPBackbone, ResNetBackbone, load_tabred
from support_identity_transfer_pilot import (
    MatchedFTTransformer,
    SupportTokenizer,
    prepare_encodings,
)
from universal_mass_identity_pilot import UniversalTokenizer, prepare


HERE = Path(__file__).resolve().parent
METHODS = ("shared_noalign", "shared_consistency")


class TriChartModel(nn.Module):
    def __init__(self, data, universal, encoding, config: dict, architecture: str) -> None:
        super().__init__()
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
        self.t_tokenizer = SupportTokenizer(edges=encoding.tple_edges, **common)
        self.rank_tokenizer = UniversalTokenizer(
            universal.n_fields, config, "rank_only"
        )
        n_fields = universal.n_fields
        if architecture == "mlp":
            self.backbone = MLPBackbone(
                n_fields, config["d_token"], config["width"], config["depth"]
            )
        elif architecture == "resnet":
            self.backbone = ResNetBackbone(
                n_fields, config["d_token"], config["width"], config["depth"]
            )
        elif architecture == "ft_transformer":
            self.backbone = MatchedFTTransformer(
                n_fields,
                config["d_token"],
                config["depth"],
                config["ft_feedforward_width"],
                config["dropout"],
            )
        else:
            raise KeyError(architecture)

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
    ) -> Tensor:
        empty_codes = code[:, :0]
        q_tokens = self.q_tokenizer(x_num, x_bin, x_cat, empty_codes)
        t_tokens = self.t_tokenizer(x_num, x_bin, x_cat, empty_codes)
        rank_tokens = self.rank_tokenizer(
            rank, rank_lower, rank_upper, code, information
        )
        predictions = [
            self.backbone(tokens)[0]
            for tokens in (q_tokens, t_tokens, rank_tokens)
        ]
        return torch.stack(predictions, dim=1)


def make_loader(data, universal, part: str, config: dict, shuffle: bool) -> DataLoader:
    rows = len(data.y[part])
    x_bin = (
        data.x_bin[part]
        if data.x_bin is not None
        else np.empty((rows, 0), dtype=np.float32)
    )
    x_cat = (
        data.x_cat[part]
        if data.x_cat is not None
        else np.empty((rows, 0), dtype=np.int64)
    )
    batch = config["batch_size"] if part == "train" else 2 * config["batch_size"]
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data.x_num[part]),
            torch.from_numpy(x_bin),
            torch.from_numpy(x_cat),
            torch.from_numpy(universal.rank[part]),
            torch.from_numpy(universal.rank_lower[part]),
            torch.from_numpy(universal.rank_upper[part]),
            torch.from_numpy(universal.exact_code[part]),
            torch.from_numpy(universal.information[part]),
            torch.from_numpy(data.y[part]),
        ),
        batch_size=batch,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(config["seed"]),
        pin_memory=True,
    )


def move(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.inference_mode()
def evaluate(model: TriChartModel, stream: DataLoader, device: torch.device, scale: float):
    model.eval()
    predictions, targets = [], []
    for batch in stream:
        *features, target = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            views = model(*features)
        predictions.append(views.float().cpu())
        targets.append(target.float().cpu())
    views = torch.cat(predictions).numpy()
    truth = torch.cat(targets).numpy()
    mean = views.mean(axis=1)
    losses = ((views - truth[:, None]) ** 2).mean(axis=0)
    loss = float(np.mean((mean - truth) ** 2))
    return {
        "loss": loss,
        "rmse": math.sqrt(loss) * scale,
        "q_rmse": math.sqrt(float(losses[0])) * scale,
        "t_rmse": math.sqrt(float(losses[1])) * scale,
        "rank_rmse": math.sqrt(float(losses[2])) * scale,
        "disagreement": float(np.mean(np.var(views, axis=1))),
    }, mean


def train_one(data, universal, encoding, config: dict, method: str, architecture: str, device: str):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    model = TriChartModel(data, universal, encoding, config, architecture).to(resolved)
    batch_size = min(config["batch_size"], 256) if architecture == "ft_transformer" else config["batch_size"]
    local_config = {**config, "batch_size": batch_size}
    streams = {
        part: make_loader(data, universal, part, local_config, part == "train")
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best, best_epoch, stale, state = math.inf, 0, 0, None
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
                views = model(*features)
                supervised = ((views - target[:, None]) ** 2).mean()
                centered = views - views.mean(dim=1, keepdim=True)
                consistency = centered.square().mean()
                weight = config["consistency_weight"] if method == "shared_consistency" else 0.0
                loss = supervised + weight * consistency
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
    assert state is not None
    model.load_state_dict(state)
    validation, val_prediction = evaluate(model, streams["val"], resolved, data.y_scale)
    test, test_prediction = evaluate(model, streams["test"], resolved, data.y_scale)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "train_seconds": time.perf_counter() - started,
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
    parser.add_argument("--config", type=Path, default=HERE / "trichart_shared_config.json")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results/trichart_shared.csv")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"], row["method"]) for row in rows}
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
        for model in args.models or config["architectures"]:
            for method in args.methods or config["methods"]:
                key = (dataset_name, model, method)
                if key in done:
                    continue
                result, val, test = train_one(
                    data, universal, encoding, config, method, model, args.device
                )
                directory = args.output.parent / f"{args.output.stem}_predictions"
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / f"{dataset_name}__{model}__{config['seed']}__{method}.npz").open("wb") as handle:
                    np.savez_compressed(handle, validation=val, test=test)
                row = {
                    "dataset": dataset_name,
                    "model": model,
                    "method": method,
                    "seed": config["seed"],
                    "n_fields": universal.n_fields,
                    **result,
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
