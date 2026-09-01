"""Adversarial controls for the real-data projective follow-up."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

import run_projective_real as experiment


HERE = Path(__file__).resolve().parent
OUT = HERE / "projective_real"


class JointDiagNet(nn.Module):
    def __init__(self):
        super().__init__()
        width = 256
        self.trunk = nn.Sequential(
            nn.Linear(experiment.HISTORY_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2 * experiment.OUTPUT_DIM),
        )

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        output = self.trunk(history)
        mean = output[:, : experiment.OUTPUT_DIM]
        diagonal = nn.functional.softplus(output[:, experiment.OUTPUT_DIM :]) + 1e-4
        query_mean = torch.sum(query * mean, dim=-1)
        variance = torch.sum(query.square() * diagonal.square(), dim=-1)
        return query_mean, variance


def broad_queries(rng: np.random.Generator, batch: int) -> np.ndarray:
    query = rng.normal(size=(batch, experiment.OUTPUT_DIM)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
    kinds = rng.integers(0, 4, size=batch)
    original = kinds == 0
    if original.any():
        query[original] = experiment.training_queries(rng, int(original.sum()))
    for index in np.flatnonzero(kinds == 1):
        query[index] = 0
        pair = rng.choice(experiment.OUTPUT_DIM, 2, replace=False)
        query[index, pair] = (1.0, -1.0)
    scaled = kinds == 3
    query[scaled] *= rng.uniform(0.3, 2.7, size=scaled.sum()).astype(np.float32)[:, None]
    return query


def train_broad(
    model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int
) -> float:
    rng = np.random.default_rng(seed + 17)
    model.to(experiment.DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    started = time.perf_counter()
    for _ in range(3_000):
        indices = rng.integers(0, len(history), size=512)
        query_np = broad_queries(rng, len(indices))
        x = torch.from_numpy(history[indices]).to(experiment.DEVICE)
        y = torch.from_numpy(future[indices]).to(experiment.DEVICE)
        query = torch.from_numpy(query_np).to(experiment.DEVICE)
        target = torch.sum(query * y, dim=-1)
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(x, query)
        loss = experiment.gaussian_nll(mean, variance, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return time.perf_counter() - started


def main() -> None:
    started = time.perf_counter()
    rows = []
    parameter_counts = {}
    for dataset in experiment.DATASETS:
        train_history, train_future, test_history, test_future = experiment.make_windows(dataset)
        for seed in experiment.SEEDS:
            controls = (
                ("querynet_broad", experiment.QueryNet(), True),
                ("jointdiag", JointDiagNet(), False),
            )
            for model_name, model, broad in controls:
                torch.manual_seed(seed)
                # Recreate after seeding so initialization is reproducible.
                model = experiment.QueryNet() if model_name == "querynet_broad" else JointDiagNet()
                parameter_counts[model_name] = sum(parameter.numel() for parameter in model.parameters())
                seconds = (
                    train_broad(model, train_history, train_future, seed)
                    if broad
                    else experiment.train_model(model, train_history, train_future, seed)
                )
                metrics = experiment.evaluate(model, test_history, test_future, seed)
                rows.append(
                    {"dataset": dataset, "seed": seed, "model": model_name, "train_seconds": seconds, **metrics}
                )
                torch.save(model.state_dict(), OUT / f"{dataset}__{model_name}__seed-{seed}.pt")
    controls = pd.DataFrame(rows)
    controls.to_csv(OUT / "control_cells.csv", index=False)
    controls.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "control_summary.csv", index=False
    )

    original = pd.read_csv(OUT / "cells.csv")
    projective = original[original.model == "projectivenet"].set_index(["dataset", "seed"])
    broad = controls[controls.model == "querynet_broad"].set_index(["dataset", "seed"])
    diagonal = controls[controls.model == "jointdiag"].set_index(["dataset", "seed"])
    broad_nll_wins = int((projective.heldout_nll <= broad.heldout_nll).sum())
    diagonal_nll_wins = int((projective.heldout_nll <= diagonal.heldout_nll).sum())
    projective_coverage = float(projective.mean_coverage_error.mean())
    broad_coverage = float(broad.mean_coverage_error.mean())
    audit = {
        "status": "complete",
        "wall_seconds": time.perf_counter() - started,
        "parameter_counts": parameter_counts,
        "projective_vs_broad_nll_wins": broad_nll_wins,
        "projective_vs_jointdiag_nll_wins": diagonal_nll_wins,
        "projective_mean_coverage_error": projective_coverage,
        "broad_mean_coverage_error": broad_coverage,
        "coverage_error_difference": projective_coverage - broad_coverage,
        "survives_broad_query_control": bool(
            broad_nll_wins >= 6 and projective_coverage - broad_coverage <= 0.05
        ),
        "supports_full_covariance_advantage": bool(diagonal_nll_wins >= 6),
    }
    (OUT / "control_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
