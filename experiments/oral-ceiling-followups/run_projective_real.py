"""Real-data follow-up for projectively consistent temporal queries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "phasecover-confirmation" / "raw" / "data"
OUT = HERE / "projective_real"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL_SHA256 = "831ca4517303c86bde13d4211b6b2dc33f1a59e6e60933805713b078e9c299ee"
DATASETS = ("JenaWeather", "Electricity", "Traffic")
SEEDS = (20261301, 20261302, 20261303)
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
HISTORY_STEPS = 32
CHANNELS = 8
HORIZON = 4
HISTORY_DIM = HISTORY_STEPS * CHANNELS
OUTPUT_DIM = HORIZON * CHANNELS
RANK = 8


def make_windows(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(SOURCE / f"{name}.npz")
    values = payload["values"].astype(np.float32)
    train_end = int(payload["train_end"])
    validation_end = int(payload["validation_end"])

    def build(first: int, last: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        available = np.arange(max(HISTORY_STEPS, first), last - HORIZON + 1)
        locations = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64)
        starts = available[locations]
        history = np.stack([values[t - HISTORY_STEPS : t] for t in starts]).reshape(count, -1)
        future = np.stack([values[t : t + HORIZON] for t in starts]).reshape(count, -1)
        return np.ascontiguousarray(history), np.ascontiguousarray(future)

    train_history, train_future = build(HISTORY_STEPS, train_end, 16_384)
    test_history, test_future = build(validation_end, len(values), 4_096)
    return train_history, train_future, test_history, test_future


def training_queries(rng: np.random.Generator, batch: int) -> np.ndarray:
    queries = np.zeros((batch, OUTPUT_DIM), dtype=np.float32)
    kinds = rng.integers(0, 4, size=batch)
    for index, kind in enumerate(kinds):
        if kind == 0:
            queries[index, rng.integers(OUTPUT_DIM)] = 1.0
        elif kind == 1:
            channel = int(rng.integers(CHANNELS))
            queries[index, channel::CHANNELS] = 1.0 / HORIZON
        elif kind == 2:
            horizon = int(rng.integers(HORIZON))
            queries[index, horizon * CHANNELS : (horizon + 1) * CHANNELS] = 1.0 / CHANNELS
        else:
            count = int(rng.integers(2, 6))
            selected = rng.choice(OUTPUT_DIM, count, replace=False)
            queries[index, selected] = 1.0
    return queries


def heldout_queries(rng: np.random.Generator, count: int) -> np.ndarray:
    queries = rng.normal(size=(count, OUTPUT_DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
    kinds = np.arange(count) % 3
    for index in np.flatnonzero(kinds == 0):
        queries[index] = 0
        pair = rng.choice(OUTPUT_DIM, 2, replace=False)
        queries[index, pair] = (1.0, -1.0)
    queries[kinds == 2] *= rng.uniform(0.3, 2.7, size=(kinds == 2).sum())[:, None]
    return queries


class QueryNet(nn.Module):
    def __init__(self):
        super().__init__()
        width = 256
        self.network = nn.Sequential(
            nn.Linear(HISTORY_DIM + OUTPUT_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(torch.cat([history, query], dim=-1))
        return output[:, 0], nn.functional.softplus(output[:, 1]) + 1e-4


class ProjectiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        width = 192
        self.trunk = nn.Sequential(
            nn.Linear(HISTORY_DIM, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.output = nn.Linear(width, OUTPUT_DIM + OUTPUT_DIM * RANK + OUTPUT_DIM)

    def joint(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.output(self.trunk(history))
        mean = output[:, :OUTPUT_DIM]
        factor = output[:, OUTPUT_DIM : OUTPUT_DIM + OUTPUT_DIM * RANK].reshape(-1, OUTPUT_DIM, RANK)
        diagonal = nn.functional.softplus(output[:, -OUTPUT_DIM:]) + 1e-4
        return mean, factor, diagonal

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        mean, factor, diagonal = self.joint(history)
        query_mean = torch.sum(query * mean, dim=-1)
        projected_factor = torch.einsum("bd,bdr->br", query, factor)
        variance = torch.sum(projected_factor.square(), dim=-1) + torch.sum(
            query.square() * diagonal.square(), dim=-1
        )
        return query_mean, variance


def gaussian_nll(mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (torch.log(variance) + (target - mean).square() / variance).mean()


def train_model(model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed + 17)
    model.to(DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    started = time.perf_counter()
    for _ in range(3_000):
        indices = rng.integers(0, len(history), size=512)
        query_np = training_queries(rng, len(indices))
        x = torch.from_numpy(history[indices]).to(DEVICE)
        y = torch.from_numpy(future[indices]).to(DEVICE)
        query = torch.from_numpy(query_np).to(DEVICE)
        target = torch.sum(query * y, dim=-1)
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(x, query)
        loss = gaussian_nll(mean, variance, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if time.perf_counter() - started > 30 * 60:
            raise TimeoutError("one projective cell exceeded 30 minutes")
    return time.perf_counter() - started


@torch.no_grad()
def evaluate(model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 29)
    x = torch.from_numpy(history).to(DEVICE)
    y = torch.from_numpy(future).to(DEVICE)
    query = torch.from_numpy(heldout_queries(rng, len(history))).to(DEVICE)
    model.eval()
    mean, variance = model(x, query)
    target = torch.sum(query * y, dim=-1)
    nll = float(gaussian_nll(mean, variance, target))
    standard_error = torch.abs(target - mean) / torch.sqrt(variance)
    coverage_50 = float((standard_error <= 0.67448975).float().mean())
    coverage_90 = float((standard_error <= 1.64485363).float().mean())

    a = torch.from_numpy(heldout_queries(rng, len(history))).to(DEVICE)
    b = torch.from_numpy(heldout_queries(rng, len(history))).to(DEVICE)
    scale = torch.from_numpy(rng.uniform(0.3, 2.7, len(history)).astype(np.float32)).to(DEVICE)
    ma, va = model(x, a)
    mb, vb = model(x, b)
    mapb, vapb = model(x, a + b)
    mamb, vamb = model(x, a - b)
    msa, vsa = model(x, scale[:, None] * a)

    def relative(error: Tensor, reference: Tensor) -> float:
        numerator = torch.sqrt(torch.mean(error.square()))
        denominator = torch.sqrt(torch.mean(reference.square())) + 1e-8
        return float(numerator / denominator)

    scale_mean = relative(msa - scale * ma, scale * ma)
    scale_variance = relative(vsa - scale.square() * va, scale.square() * va)
    return {
        "heldout_nll": nll,
        "coverage_50": coverage_50,
        "coverage_90": coverage_90,
        "mean_coverage_error": 0.5 * (abs(coverage_50 - 0.50) + abs(coverage_90 - 0.90)),
        "mean_additivity_violation": relative(mapb - ma - mb, mapb),
        "scale_violation": 0.5 * (scale_mean + scale_variance),
        "variance_polarization_violation": relative(vapb + vamb - 2 * va - 2 * vb, vapb + vamb),
    }


def main() -> None:
    overall_started = time.perf_counter()
    rows = []
    parameter_counts = {}
    for dataset in DATASETS:
        train_history, train_future, test_history, test_future = make_windows(dataset)
        for seed in SEEDS:
            for model_name, constructor in (("querynet", QueryNet), ("projectivenet", ProjectiveNet)):
                torch.manual_seed(seed)
                model = constructor()
                parameter_counts[model_name] = sum(parameter.numel() for parameter in model.parameters())
                seconds = train_model(model, train_history, train_future, seed)
                metrics = evaluate(model, test_history, test_future, seed)
                rows.append(
                    {"dataset": dataset, "seed": seed, "model": model_name, "train_seconds": seconds, **metrics}
                )
                torch.save(model.state_dict(), OUT / f"{dataset}__{model_name}__seed-{seed}.pt")
            if time.perf_counter() - overall_started > 2 * 60 * 60:
                raise TimeoutError("projective real-data follow-up exceeded two hours")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    summary = frame.groupby(["dataset", "model"], as_index=False).agg(
        heldout_nll=("heldout_nll", "mean"),
        coverage_50=("coverage_50", "mean"),
        coverage_90=("coverage_90", "mean"),
        mean_coverage_error=("mean_coverage_error", "mean"),
        mean_additivity_violation=("mean_additivity_violation", "mean"),
        scale_violation=("scale_violation", "mean"),
        variance_polarization_violation=("variance_polarization_violation", "mean"),
        train_seconds=("train_seconds", "mean"),
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    identities = ["mean_additivity_violation", "scale_violation", "variance_polarization_violation"]
    query = summary[summary.model == "querynet"].set_index("dataset")
    projective = summary[summary.model == "projectivenet"].set_index("dataset")
    query_dataset_passes = sum((query.loc[dataset, identities] >= 0.05).sum() >= 2 for dataset in DATASETS)
    projective_identity_passes = sum(projective.loc[dataset, identities].max() < 1e-5 for dataset in DATASETS)
    nll_dataset_wins = sum(projective.loc[dataset, "heldout_nll"] <= query.loc[dataset, "heldout_nll"] for dataset in DATASETS)
    pivot = frame.pivot(index=["dataset", "seed"], columns="model", values="heldout_nll")
    nll_cell_wins = int((pivot.projectivenet <= pivot.querynet).sum())
    query_calibration = float(query.mean_coverage_error.mean())
    projective_calibration = float(projective.mean_coverage_error.mean())
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - overall_started,
        "cells": len(frame),
        "parameter_counts": parameter_counts,
        "querynet_dataset_identity_passes": int(query_dataset_passes),
        "projectivenet_dataset_identity_passes": int(projective_identity_passes),
        "projectivenet_nll_dataset_wins": int(nll_dataset_wins),
        "projectivenet_nll_cell_wins": nll_cell_wins,
        "querynet_mean_coverage_error": query_calibration,
        "projectivenet_mean_coverage_error": projective_calibration,
        "coverage_error_difference": projective_calibration - query_calibration,
    }
    audit["passed"] = bool(
        query_dataset_passes >= 2
        and projective_identity_passes == 3
        and nll_dataset_wins >= 2
        and nll_cell_wins >= 6
        and projective_calibration - query_calibration <= 0.05
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
