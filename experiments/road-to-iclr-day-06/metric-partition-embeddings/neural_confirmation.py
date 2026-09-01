#!/usr/bin/env python3
"""Frozen Stage C neural confirmation for the single-scale MPE winner."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import basis_control
import metric_partition_benchmark as mpb


METHODS = ("ple", "periodic", "code_rbf", "u_ple", "mpe_native", "mpe_corrupt")


class TokenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Linear(16, 16, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.token(x)).squeeze(-1)


def arrays(domain: mpb.DomainData, schema: int, seed: int, method: str) -> dict[str, np.ndarray]:
    stored = mpb.stored_values(domain, schema, seed)
    if method == "u_ple":
        feature = basis_control.uniform_ple(stored)
    else:
        feature = mpb.feature_map(domain, stored, method, seed)
    return mpb.standardize(feature["train"], feature)


def train_one(
    feature: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> dict[str, float | int]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TokenMLP().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = {p: torch.as_tensor(v, dtype=torch.float32, device=device) for p, v in feature.items()}
    y = {p: torch.as_tensor(v, dtype=torch.float32, device=device) for p, v in target.items()}
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x["train"]) - y["train"]) ** 2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(x["val"]) - y["val"]) ** 2).cpu())
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        losses = {
            f"{part}_mse": float(torch.mean((model(x[part]) - y[part]) ** 2).cpu())
            for part in ("train", "val", "test")
        }
    return {**losses, "best_epoch": best_epoch, "epochs_ran": epoch + 1, "parameter_count": parameter_count}


def run(output: Path, device_name: str, max_epochs: int, patience: int) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    seeds = (20260920, 20260921, 20260922)
    rows = []
    started = time.time()
    for domain_name in ("cycle", "tree"):
        for seed in seeds:
            domain = mpb.make_domain(domain_name, seed)
            target = mpb.standardize_target(domain)
            for schema in range(4):
                for method in METHODS:
                    result = train_one(arrays(domain, schema, seed, method), target, seed, device, max_epochs, patience)
                    row = {"domain": domain_name, "seed": seed, "schema": schema, "method": method, **result}
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "PROTOCOL_FREEZE.md Stage C plus POSTHOC_CAPACITY_CONTROL.md",
        "methods": METHODS,
        "seeds": seeds,
        "schemas": 4,
        "expected_rows": 144,
        "actual_rows": len(rows),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "parameter_counts": sorted(set(int(r["parameter_count"]) for r in rows)),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=mpb.RESULTS / "neural_confirmation.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    run(args.output, args.device, args.max_epochs, args.patience)
