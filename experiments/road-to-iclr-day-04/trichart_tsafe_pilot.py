#!/usr/bin/env python3
"""T-PLE-anchored, zero-start token fusion for three tabular charts."""
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
    SupportTokenizer,
    prepare_encodings,
)
from trichart_shared_pilot import make_loader, move
from universal_mass_identity_pilot import UniversalTokenizer, prepare


HERE = Path(__file__).resolve().parent
METHODS = ("scalar_gate", "field_gate")


class TAnchoredTriChart(nn.Module):
    """One-pass token fusion that is exactly T-PLE when its gates are zero."""

    def __init__(self, data, universal, encoding, config: dict, architecture: str, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise KeyError(method)
        self.method = method
        n_bin = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
        common = {
            "n_bin_fields": n_bin,
            "category_cardinalities": data.category_cardinalities,
            "support_columns": encoding.selected_columns,
            "support_cardinalities": encoding.cardinalities,
            "d_token": config["d_token"],
            "use_support": False,
        }

        # Keep this order paired with SupportModel(method="tple"): tokenizer,
        # then backbone. Extra charts are initialized only after the exact
        # fallback path exists.
        self.t_tokenizer = SupportTokenizer(edges=encoding.tple_edges, **common)
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

        self.q_tokenizer = SupportTokenizer(edges=encoding.qple_edges, **common)
        self.rank_tokenizer = UniversalTokenizer(universal.n_fields, config, "rank_only")
        gate_fields = 1 if method == "scalar_gate" else n_fields
        self.q_gate = nn.Parameter(torch.zeros(gate_fields))
        self.rank_gate = nn.Parameter(torch.zeros(gate_fields))

    def chart_tokens(
        self,
        x_num: Tensor,
        x_bin: Tensor,
        x_cat: Tensor,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        empty_codes = code[:, :0]
        t_tokens = self.t_tokenizer(x_num, x_bin, x_cat, empty_codes)
        q_tokens = self.q_tokenizer(x_num, x_bin, x_cat, empty_codes)
        rank_tokens = self.rank_tokenizer(
            rank, rank_lower, rank_upper, code, information
        )
        return t_tokens, q_tokens, rank_tokens

    def fused_tokens(self, *features: Tensor) -> tuple[Tensor, Tensor]:
        t_tokens, q_tokens, rank_tokens = self.chart_tokens(*features)
        q_gate = self.q_gate.view(1, -1, 1)
        rank_gate = self.rank_gate.view(1, -1, 1)
        fused = (
            t_tokens
            + q_gate * (q_tokens - t_tokens)
            + rank_gate * (rank_tokens - t_tokens)
        )
        return fused, t_tokens

    def forward(self, *features: Tensor) -> Tensor:
        fused, _ = self.fused_tokens(*features)
        return self.backbone(fused)[0]

    def t_prediction(self, *features: Tensor) -> Tensor:
        _, t_tokens = self.fused_tokens(*features)
        return self.backbone(t_tokens)[0]


@torch.inference_mode()
def evaluate(model: TAnchoredTriChart, stream, device: torch.device, scale: float):
    model.eval()
    predictions, t_predictions, targets = [], [], []
    for batch in stream:
        *features, target = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(*features)
            t_prediction = model.t_prediction(*features)
        predictions.append(prediction.float().cpu())
        t_predictions.append(t_prediction.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions).numpy()
    t_prediction = torch.cat(t_predictions).numpy()
    truth = torch.cat(targets).numpy()
    loss = float(np.mean((prediction - truth) ** 2))
    t_loss = float(np.mean((t_prediction - truth) ** 2))
    return {
        "loss": loss,
        "rmse": math.sqrt(loss) * scale,
        "t_branch_rmse": math.sqrt(t_loss) * scale,
    }, prediction


def train_one(data, universal, encoding, config: dict, method: str, architecture: str, device: str):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    model = TAnchoredTriChart(data, universal, encoding, config, architecture, method).to(resolved)
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
                prediction = model(*features)
                supervised = torch.nn.functional.mse_loss(prediction, target)
                gate_penalty = model.q_gate.abs().mean() + model.rank_gate.abs().mean()
                loss = supervised + config["gate_l1_weight"] * gate_penalty
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
    gates = torch.cat((model.q_gate.detach().abs(), model.rank_gate.detach().abs()))
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "val_loss": validation["loss"],
        "val_rmse": validation["rmse"],
        "val_t_branch_rmse": validation["t_branch_rmse"],
        "test_loss": test["loss"],
        "test_rmse": test["rmse"],
        "test_t_branch_rmse": test["t_branch_rmse"],
        "mean_abs_chart_gate": float(gates.mean().cpu()),
        "max_abs_chart_gate": float(gates.max().cpu()),
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
    parser.add_argument("--config", type=Path, default=HERE / "trichart_tsafe_config.json")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results/trichart_tsafe.csv")
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
        models = args.models or (
            config["maps_architectures"]
            if dataset_name == "maps-routing"
            else config["architectures"]
        )
        for model in models:
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
