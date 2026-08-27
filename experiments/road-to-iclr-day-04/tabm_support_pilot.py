"""TabM transport test for measured-support FieldRiesz representations."""

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
    PARTS,
    base_schema,
    combine,
    loss_numpy,
    make_prepared,
    metric,
)


HERE = Path(__file__).resolve().parent


class FlatTabM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        from tabm import TabM

        self.model = TabM.make(
            n_num_features=input_size,
            d_out=output_size,
            d_block=width,
            n_blocks=depth,
            dropout=0.1,
            activation="ReLU",
            k=ensemble_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_width(
    input_size: int,
    output_size: int,
    depth: int,
    ensemble_size: int,
    target: int,
) -> tuple[int, int]:
    low, high = 16, 768
    while low < high:
        middle = (low + high) // 2
        count = count_parameters(
            FlatTabM(input_size, output_size, middle, depth, ensemble_size)
        )
        if count < target:
            low = middle + 1
        else:
            high = middle
    candidates = (max(16, low - 1), low)
    choices = [
        (
            width,
            count_parameters(
                FlatTabM(input_size, output_size, width, depth, ensemble_size)
            ),
        )
        for width in candidates
    ]
    return min(choices, key=lambda item: abs(item[1] - target))


def member_loss(task: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "binclass":
        return nn.functional.binary_cross_entropy_with_logits(
            prediction.squeeze(-1), target[:, None].expand_as(prediction.squeeze(-1))
        )
    if task == "multiclass":
        expanded = target[:, None].expand(-1, prediction.shape[1]).reshape(-1)
        return nn.functional.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]), expanded
        )
    return nn.functional.mse_loss(
        prediction.squeeze(-1), target[:, None].expand_as(prediction.squeeze(-1))
    )


def ensemble_prediction(task: str, prediction: torch.Tensor) -> torch.Tensor:
    if task == "binclass":
        probability = prediction.squeeze(-1).sigmoid().mean(dim=1).clamp(1e-6, 1 - 1e-6)
        return torch.logit(probability)[:, None]
    if task == "multiclass":
        probability = prediction.softmax(dim=-1).mean(dim=1).clamp_min(1e-8)
        return probability.log()
    return prediction.squeeze(-1).mean(dim=1)[:, None]


def predict(
    model: nn.Module,
    x: np.ndarray,
    task: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            members = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            output.append(ensemble_prediction(task, members).cpu().numpy())
    return np.concatenate(output)


def train_tabm(
    data,
    *,
    seed: int,
    device: str,
    width: int,
    depth: int,
    ensemble_size: int,
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
    train_width, parameters = matched_width(
        data.x["train"].shape[1], output_size, depth, ensemble_size, target_parameters
    )
    model = FlatTabM(
        data.x["train"].shape[1], output_size, train_width, depth, ensemble_size
    ).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            loss = member_loss(data.task, model(x), y)
            loss.backward()
            optimizer.step()
        val_prediction = predict(model, data.x["val"], data.task, resolved, 1024)
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
    val_prediction = predict(model, data.x["val"], data.task, resolved, 1024)
    test_prediction = predict(model, data.x["test"], data.task, resolved, 1024)
    return {
        "input_features": data.x["train"].shape[1],
        "width": train_width,
        "parameters": parameters,
        "parameter_error_fraction": (parameters - target_parameters) / target_parameters,
        "members": ensemble_size,
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260833])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=HERE / "results/tabm_support.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    complete = {(row["dataset"], int(row["seed"]), row["method"]) for row in rows}
    for name in args.datasets:
        dataset = support.load_dataset(
            name, max_train_rows=50000, max_eval_rows=15000, sample_seed=20260826
        )
        numeric, _ = support.numeric_representations(
            dataset.x_num,
            bins=32,
            spike_fraction=0.35,
            heat_strength=1.0,
            minimum_excess_mass=0.02,
        )
        nonnumeric = base_schema(dataset, seed=20260826, include_num=False)
        variants = {
            method: combine([numeric[method], nonnumeric]) for method in args.methods
        }
        output_size = dataset.n_classes if dataset.task == "multiclass" else 1
        target_parameters = count_parameters(
            FlatTabM(
                variants["quantile_ple"]["train"].shape[1],
                output_size,
                args.width,
                args.depth,
                args.members,
            )
        )
        for seed in args.seeds:
            for method, features in variants.items():
                key = (name, seed, method)
                if key in complete:
                    continue
                prepared = make_prepared(dataset, features, {"method": method})
                result = train_tabm(
                    prepared,
                    seed=seed,
                    device=args.device,
                    width=args.width,
                    depth=args.depth,
                    ensemble_size=args.members,
                    epochs=args.epochs,
                    patience=args.patience,
                    target_parameters=target_parameters,
                )
                row = {
                    "dataset": name,
                    "task": dataset.task,
                    "model": "tabm",
                    "seed": seed,
                    "method": method,
                    "target_parameters": target_parameters,
                    **result,
                }
                rows.append(row)
                complete.add(key)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
