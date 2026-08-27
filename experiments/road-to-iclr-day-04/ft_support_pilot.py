"""Field-token Transformer test for measured-support representations."""

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
from torch.utils.data import DataLoader, TensorDataset

import support_heat_pilot as support
from experiments.day3.core import (
    base_schema,
    clean_numeric,
    combine,
    loss_numpy,
    make_prepared,
    metric,
)


HERE = Path(__file__).resolve().parent


def block_dimensions(method: str, metadata: list[dict[str, object]], dataset) -> list[int]:
    dimensions = []
    for field in metadata:
        quantile = int(field["quantile_nodes"]) - 1
        active = bool(field["adaptive"])
        if not active or method == "quantile_ple":
            dimensions.append(quantile)
        elif method in ("adaptive_support_ple",):
            dimensions.append(int(field["support_nodes"]) - 1)
        elif method == "adaptive_support_whitened":
            dimensions.append(int(field["support_whitened_rank"]))
        elif method in ("adaptive_support_riesz", "adaptive_support_wrong_riesz"):
            dimensions.append(int(field["riesz_rank"]))
        else:
            raise ValueError(method)

    if dataset.x_bin is not None:
        clean = clean_numeric(dataset.x_bin)["train"]
        dimensions.extend([1] * int(np.sum(clean.std(axis=0) > 1e-12)))
    if dataset.x_cat is not None:
        for column in range(dataset.x_cat["train"].shape[1]):
            dimensions.append(len(np.unique(dataset.x_cat["train"][:, column].astype(str))) - 1)
    return [dimension for dimension in dimensions if dimension > 0]


class FieldTokenTransformer(nn.Module):
    def __init__(
        self,
        block_dims: list[int],
        output_size: int,
        d_token: int = 64,
        d_ff: int = 128,
        n_blocks: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_dims = block_dims
        self.projections = nn.ModuleList(
            nn.Linear(dimension, d_token, bias=False) for dimension in block_dims
        )
        self.token_bias = nn.Parameter(torch.empty(len(block_dims), d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.token_bias, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        start = 0
        for index, (dimension, projection) in enumerate(zip(self.block_dims, self.projections)):
            tokens.append(projection(x[:, start : start + dimension]) + self.token_bias[index])
            start += dimension
        if start != x.shape[1]:
            raise RuntimeError(f"Block dimensions sum to {start}, input has {x.shape[1]}")
        fields = torch.stack(tokens, dim=1)
        cls = self.cls.expand(len(x), -1, -1)
        return self.head(self.encoder(torch.cat((cls, fields), dim=1))[:, 0])


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_d_ff(block_dims: list[int], output_size: int, target: int) -> tuple[int, int]:
    low, high = 32, 512
    while low < high:
        middle = (low + high) // 2
        count = parameter_count(FieldTokenTransformer(block_dims, output_size, d_ff=middle))
        if count < target:
            low = middle + 1
        else:
            high = middle
    choices = []
    for d_ff in (max(32, low - 1), low):
        count = parameter_count(FieldTokenTransformer(block_dims, output_size, d_ff=d_ff))
        choices.append((d_ff, count))
    return min(choices, key=lambda pair: abs(pair[1] - target))


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            output.append(model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(output)


def train_ft(
    data,
    block_dims: list[int],
    *,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    target_parameters: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    output_size = data.n_classes if data.task == "multiclass" else 1
    d_ff, parameters = matched_d_ff(block_dims, output_size, target_parameters)
    model = FieldTokenTransformer(block_dims, output_size, d_ff=d_ff).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    if data.task == "binclass":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    elif data.task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
        batch_size=512,
        shuffle=True,
        generator=generator,
        pin_memory=resolved.type == "cuda",
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(resolved, non_blocking=True), y.to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if data.task != "multiclass":
                prediction = prediction.squeeze(-1)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
        val_prediction = predict(model, data.x["val"], resolved, 1024)
        val_loss = loss_numpy(data.task, val_prediction, data.y["val"])
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_prediction = predict(model, data.x["val"], resolved, 1024)
    test_prediction = predict(model, data.x["test"], resolved, 1024)
    return {
        "input_features": data.x["train"].shape[1],
        "tokens": len(block_dims),
        "d_ff": d_ff,
        "parameters": parameters,
        "parameter_error_fraction": (parameters - target_parameters) / target_parameters,
        "best_epoch": best_epoch,
        "val_loss": loss_numpy(data.task, val_prediction, data.y["val"]),
        "test_loss": loss_numpy(data.task, test_prediction, data.y["test"]),
        "val_metric": metric(data, val_prediction, data.y["val"]),
        "test_metric": metric(data, test_prediction, data.y["test"]),
        "train_seconds": time.perf_counter() - started,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "black-friday", "california"])
    parser.add_argument("--methods", nargs="+", default=[
        "quantile_ple", "adaptive_support_ple", "adaptive_support_whitened",
        "adaptive_support_riesz", "adaptive_support_wrong_riesz",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260834])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE / "results/ft_support.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    complete = {(row["dataset"], int(row["seed"]), row["method"]) for row in rows}
    for name in args.datasets:
        dataset = support.load_dataset(
            name, max_train_rows=50000, max_eval_rows=15000, sample_seed=20260826
        )
        numeric, metadata = support.numeric_representations(
            dataset.x_num, bins=32, spike_fraction=0.35,
            heat_strength=1.0, minimum_excess_mass=0.02,
        )
        nonnumeric = base_schema(dataset, seed=20260826, include_num=False)
        variants = {method: combine([numeric[method], nonnumeric]) for method in args.methods}
        dimensions = {method: block_dimensions(method, metadata, dataset) for method in args.methods}
        for method in args.methods:
            assert sum(dimensions[method]) == variants[method]["train"].shape[1]
        output_size = dataset.n_classes if dataset.task == "multiclass" else 1
        target_parameters = parameter_count(
            FieldTokenTransformer(dimensions["quantile_ple"], output_size)
        )
        for seed in args.seeds:
            for method, features in variants.items():
                key = (name, seed, method)
                if key in complete:
                    continue
                result = train_ft(
                    make_prepared(dataset, features, {"method": method}),
                    dimensions[method], seed=seed, device=args.device,
                    epochs=args.epochs, patience=args.patience,
                    target_parameters=target_parameters,
                )
                row = {
                    "dataset": name, "task": dataset.task,
                    "model": "field_token_transformer", "seed": seed,
                    "method": method, "target_parameters": target_parameters,
                    **result,
                }
                rows.append(row)
                complete.add(key)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
