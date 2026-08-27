"""Small FT-Transformer falsification panel for Day 4 innovation views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import innovation_pilot as pilot
from experiments.day3.core import (
    PARTS,
    base_schema,
    load_dataset,
    loss_numpy,
    make_prepared,
    metric,
)


HERE = Path(__file__).resolve().parent


class ScalarFTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        output_size: int,
        d_token: int = 64,
        n_blocks: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(d_token))
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x[:, :, None] * self.weight[None] + self.bias[None]
        cls = self.cls.expand(len(x), -1, -1)
        return self.head(self.encoder(torch.cat((cls, tokens), dim=1))[:, 0])


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            output.append(model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(output)


def train_ft(data, seed: int, device: str, epochs: int, patience: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    output_size = data.n_classes if data.task == "multiclass" else 1
    model = ScalarFTTransformer(data.x["train"].shape[1], output_size).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    if data.task == "binclass":
        criterion = nn.BCEWithLogitsLoss()
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
    best_loss, stale, best_epoch, best_state = math.inf, 0, 0, None
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
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "val_loss": loss_numpy(data.task, val_prediction, data.y["val"]),
        "test_loss": loss_numpy(data.task, test_prediction, data.y["test"]),
        "val_metric": metric(data, val_prediction, data.y["val"]),
        "test_metric": metric(data, test_prediction, data.y["test"]),
    }


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["diamond", "higgs-small", "tabred-weather"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument("--output", type=Path, default=HERE / "results/ft_innovation_pilot.csv")
    args = parser.parse_args()
    rows: list[dict[str, object]] = list(read(args.output))
    completed = {(r["dataset"], int(r["seed"]), r["method"]) for r in rows}
    for name in args.datasets:
        dataset = (
            pilot.load_weather(args.max_train_rows, args.max_eval_rows, 20260826)
            if name == "tabred-weather"
            else load_dataset(
                name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
        )
        baseline = base_schema(dataset, seed=20260826)
        global_views, metadata = pilot.factorize(
            baseline, max_rank=8, variance_fraction=0.5
        )
        adaptive, adaptive_metadata = pilot.measure_adaptive_innovation(
            dataset, baseline, max_rank=8, variance_fraction=0.5
        )
        variants = {
            "baseline": baseline,
            "common_skip": global_views["common_skip"],
            "innovation_sum": global_views["innovation_sum"],
            "measure_innovation": adaptive,
        }
        for seed in args.seeds:
            for method, x in variants.items():
                key = (name, seed, method)
                if key in completed:
                    continue
                prepared = make_prepared(dataset, x, {"method": method})
                result = train_ft(prepared, seed, args.device, args.epochs, args.patience)
                row = {
                    "dataset": name,
                    "task": dataset.task,
                    "model": "scalar_ft_transformer",
                    "seed": seed,
                    "method": method,
                    "factor_rank": metadata["rank"],
                    "measure_rank": adaptive_metadata["rank"],
                    **result,
                }
                rows.append(row)
                completed.add(key)
                write(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
