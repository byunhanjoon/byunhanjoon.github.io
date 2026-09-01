"""Pilot B: projectively consistent probabilistic temporal queries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
OUT = HERE / "projective"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (20261231, 20261232, 20261233)
PROTOCOL_SHA256 = "538f14851b6a1cf54737c3b9bc8df3cf3b227c1b33cf2ef53469f261317b3164"
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
HISTORY_DIM = 16 * 4
OUTPUT_DIM = 8
RANK = 3


def make_process(process_seed: int, sample_seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    process_rng = np.random.default_rng(process_seed)
    rng = np.random.default_rng(sample_seed)
    state = rng.normal(size=(n, 4)).astype(np.float32)
    history = [state]
    for _ in range(15):
        noise = rng.normal(scale=0.18, size=state.shape).astype(np.float32)
        state = 0.68 * state + 0.17 * np.roll(state, 1, axis=1) + 0.12 * np.sin(state) + noise
        history.append(state)
    context = np.stack(history, axis=1).astype(np.float32)
    last = context[:, -1]
    first = np.tanh(0.75 * last + 0.18 * np.roll(last, 1, axis=1) + 0.08 * context.mean(axis=1))
    second = np.tanh(0.75 * first + 0.18 * np.roll(first, 1, axis=1) + 0.08 * last)
    mean = np.concatenate([first, second], axis=1).astype(np.float32)
    base = process_rng.normal(scale=0.16, size=(OUTPUT_DIM, RANK)).astype(np.float32)
    modulation = 0.75 + 0.25 * np.tanh(last[:, :RANK])
    factor = base[None] * modulation[:, None, :]
    diagonal = 0.08 + 0.07 / (1.0 + np.exp(-np.concatenate([last, first], axis=1)))
    latent_noise = rng.normal(size=(n, RANK)).astype(np.float32)
    independent_noise = rng.normal(size=(n, OUTPUT_DIM)).astype(np.float32)
    future = mean + np.einsum("ndr,nr->nd", factor, latent_noise) + diagonal * independent_noise
    return context.reshape(n, -1), future.astype(np.float32)


def training_queries(rng: np.random.Generator, batch: int) -> np.ndarray:
    queries = np.zeros((batch, OUTPUT_DIM), dtype=np.float32)
    kinds = rng.integers(0, 3, size=batch)
    for index, kind in enumerate(kinds):
        if kind == 0:
            queries[index, rng.integers(OUTPUT_DIM)] = 1.0
        elif kind == 1:
            count = int(rng.integers(2, 5))
            selected = rng.choice(OUTPUT_DIM, count, replace=False)
            queries[index, selected] = 1.0 / count
        else:
            selected = rng.choice(OUTPUT_DIM, 2, replace=False)
            queries[index, selected] = 1.0
    return queries


def heldout_queries(rng: np.random.Generator, n: int) -> np.ndarray:
    queries = rng.normal(size=(n, OUTPUT_DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
    kinds = np.arange(n) % 3
    difference = kinds == 0
    for index in np.flatnonzero(difference):
        queries[index] = 0
        pair = rng.choice(OUTPUT_DIM, 2, replace=False)
        queries[index, pair] = (1.0, -1.0)
    queries[kinds == 1] *= 2.5
    return queries


class QueryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HISTORY_DIM + OUTPUT_DIM, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2),
        )

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(torch.cat([history, query], dim=-1))
        return output[:, 0], nn.functional.softplus(output[:, 1]) + 1e-4


class ProjectiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(HISTORY_DIM, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
        )
        self.output = nn.Linear(128, OUTPUT_DIM + OUTPUT_DIM * RANK + OUTPUT_DIM)

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
    rng = np.random.default_rng(seed + 19)
    model.to(DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    started = time.perf_counter()
    for _ in range(5_000):
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
        if time.perf_counter() - started > 29 * 60:
            raise TimeoutError("projective pilot training exceeded budget")
    return time.perf_counter() - started


@torch.no_grad()
def predict(model: nn.Module, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
    model.eval()
    return model(history, query)


@torch.no_grad()
def evaluate(model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 29)
    count = min(8_192, len(history))
    x = torch.from_numpy(history[:count]).to(DEVICE)
    y = torch.from_numpy(future[:count]).to(DEVICE)
    query = torch.from_numpy(heldout_queries(rng, count)).to(DEVICE)
    mean, variance = predict(model, x, query)
    target = torch.sum(query * y, dim=-1)
    nll = float(gaussian_nll(mean, variance, target))

    a = torch.from_numpy(heldout_queries(rng, count)).to(DEVICE)
    b = torch.from_numpy(heldout_queries(rng, count)).to(DEVICE)
    scale = torch.from_numpy(rng.uniform(0.3, 2.7, size=count).astype(np.float32)).to(DEVICE)
    ma, va = predict(model, x, a)
    mb, vb = predict(model, x, b)
    mapb, vapb = predict(model, x, a + b)
    mamb, vamb = predict(model, x, a - b)
    msa, vsa = predict(model, x, scale[:, None] * a)

    def relative(error: Tensor, reference: Tensor) -> float:
        return float(torch.sqrt(torch.mean(error.square())) / (torch.sqrt(torch.mean(reference.square())) + 1e-8))

    additivity = relative(mapb - ma - mb, mapb)
    scale_mean = relative(msa - scale * ma, scale * ma)
    scale_variance = relative(vsa - scale.square() * va, scale.square() * va)
    scale_violation = 0.5 * (scale_mean + scale_variance)
    polarization = relative(vapb + vamb - 2 * va - 2 * vb, vapb + vamb)
    return {
        "heldout_nll": nll,
        "mean_additivity_violation": additivity,
        "scale_violation": scale_violation,
        "variance_polarization_violation": polarization,
    }


def main() -> None:
    overall_started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        train_history, train_future = make_process(seed, seed + 1, 65_536)
        test_history, test_future = make_process(seed, seed + 10_000, 8_192)
        torch.manual_seed(seed)
        for name, model in (("querynet", QueryNet()), ("projectivenet", ProjectiveNet())):
            seconds = train_model(model, train_history, train_future, seed)
            metrics = evaluate(model, test_history, test_future, seed)
            rows.append({"seed": seed, "model": name, "train_seconds": seconds, **metrics})
            torch.save(model.state_dict(), OUT / f"{name}__seed-{seed}.pt")
        if time.perf_counter() - overall_started > 30 * 60:
            raise TimeoutError("projective pilot exceeded 30 minutes")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    means = frame.groupby("model", as_index=False).mean(numeric_only=True)
    means.to_csv(OUT / "summary.csv", index=False)
    query = means[means.model == "querynet"].iloc[0]
    projective = means[means.model == "projectivenet"].iloc[0]
    identities = ["mean_additivity_violation", "scale_violation", "variance_polarization_violation"]
    query_violations = sum(float(query[item]) >= 0.05 for item in identities)
    projective_max = max(float(projective[item]) for item in identities)
    pivot = frame.pivot(index="seed", columns="model", values="heldout_nll")
    nll_wins = int((pivot.projectivenet <= pivot.querynet).sum())
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - overall_started,
        "seeds": len(SEEDS),
        "querynet_identities_above_5pct": query_violations,
        "projectivenet_max_identity_violation": projective_max,
        "projectivenet_nll_wins": nll_wins,
        "querynet_mean_nll": float(query.heldout_nll),
        "projectivenet_mean_nll": float(projective.heldout_nll),
    }
    audit["passed"] = bool(query_violations >= 2 and projective_max < 1e-5 and nll_wins >= 2)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
