"""Pilot M: non-Gaussian, projectively consistent Gaussian mixtures."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent / "oral-ceiling-followups"
sys.path.insert(0, str(FOLLOWUP))
import run_projective_controls as controls  # noqa: E402
import run_projective_real as experiment  # noqa: E402


OUT = HERE / "mixture"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL_SHA256 = "b5148cca2610c49d8cca287d123d81427cc2daa1874150ca16056159d8b3daab"
COMPONENTS = 4


class ProjectiveMixtureNet(nn.Module):
    def __init__(self, components: int):
        super().__init__()
        self.components = components
        width = 192
        self.backbone = nn.Sequential(
            nn.Linear(experiment.HISTORY_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, components * (1 + 2 * experiment.OUTPUT_DIM))

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.output(self.backbone(history)).reshape(len(history), self.components, -1)
        log_weights = torch.log_softmax(output[:, :, 0], dim=-1)
        joint_mean = output[:, :, 1 : 1 + experiment.OUTPUT_DIM]
        diagonal = nn.functional.softplus(output[:, :, 1 + experiment.OUTPUT_DIM :]) + 1e-4
        component_mean = torch.einsum("bkd,bd->bk", joint_mean, query)
        component_variance = torch.einsum("bkd,bd->bk", diagonal.square(), query.square())
        return log_weights, component_mean, component_variance


class DirectMixtureNet(nn.Module):
    def __init__(self, components: int = COMPONENTS):
        super().__init__()
        self.components = components
        width = 192
        self.network = nn.Sequential(
            nn.Linear(experiment.HISTORY_DIM + experiment.OUTPUT_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3 * components),
        )

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.network(torch.cat([history, query], dim=-1)).reshape(len(history), self.components, 3)
        log_weights = torch.log_softmax(output[:, :, 0], dim=-1)
        component_mean = output[:, :, 1]
        component_variance = nn.functional.softplus(output[:, :, 2]) + 1e-4
        return log_weights, component_mean, component_variance


def mixture_nll(log_weights: Tensor, mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
    log_density = -0.5 * (
        np.log(2.0 * np.pi) + torch.log(variance) + (target[:, None] - mean).square() / variance
    )
    return -torch.logsumexp(log_weights + log_density, dim=-1).mean()


def moments(log_weights: Tensor, mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor]:
    weights = torch.exp(log_weights)
    mixture_mean = torch.sum(weights * mean, dim=-1)
    second = torch.sum(weights * (variance + mean.square()), dim=-1)
    return mixture_mean, (second - mixture_mean.square()).clamp_min(1e-8)


def train_model(model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed + 17)
    model.to(experiment.DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    started = time.perf_counter()
    for _ in range(3_000):
        indices = rng.integers(0, len(history), size=512)
        query_np = controls.broad_queries(rng, len(indices))
        x = torch.from_numpy(history[indices]).to(experiment.DEVICE)
        y = torch.from_numpy(future[indices]).to(experiment.DEVICE)
        query = torch.from_numpy(query_np).to(experiment.DEVICE)
        target = torch.sum(query * y, dim=-1)
        optimizer.zero_grad(set_to_none=True)
        log_weights, mean, variance = model(x, query)
        loss = mixture_nll(log_weights, mean, variance, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if time.perf_counter() - started > 30 * 60:
            raise TimeoutError("one mixture cell exceeded 30 minutes")
    return time.perf_counter() - started


@torch.no_grad()
def evaluate(model: nn.Module, history: np.ndarray, future: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 29)
    x = torch.from_numpy(history).to(experiment.DEVICE)
    y = torch.from_numpy(future).to(experiment.DEVICE)
    query = torch.from_numpy(experiment.heldout_queries(rng, len(history))).to(experiment.DEVICE)
    target = torch.sum(query * y, dim=-1)
    model.eval()
    log_weights, component_mean, component_variance = model(x, query)
    mean, variance = moments(log_weights, component_mean, component_variance)
    z = (target[:, None] - component_mean) / torch.sqrt(component_variance)
    pit = torch.sum(torch.exp(log_weights) * torch.special.ndtr(z), dim=-1)
    grid = torch.arange(1, 10, device=pit.device, dtype=pit.dtype) / 10
    pit_error = float(torch.mean(torch.abs((pit[:, None] <= grid[None]).float().mean(dim=0) - grid)))

    a = torch.from_numpy(experiment.heldout_queries(rng, len(history))).to(experiment.DEVICE)
    b = torch.from_numpy(experiment.heldout_queries(rng, len(history))).to(experiment.DEVICE)
    scale = torch.from_numpy(rng.uniform(0.3, 2.7, len(history)).astype(np.float32)).to(experiment.DEVICE)

    def predict_moments(q: Tensor) -> tuple[Tensor, Tensor]:
        return moments(*model(x, q))

    ma, va = predict_moments(a)
    mb, vb = predict_moments(b)
    mapb, vapb = predict_moments(a + b)
    mamb, vamb = predict_moments(a - b)
    msa, vsa = predict_moments(scale[:, None] * a)

    def relative(error: Tensor, reference: Tensor) -> float:
        return float(
            torch.sqrt(torch.mean(error.square())) / (torch.sqrt(torch.mean(reference.square())) + 1e-8)
        )

    return {
        "heldout_nll": float(mixture_nll(log_weights, component_mean, component_variance, target)),
        "rmse": float(torch.sqrt(torch.mean((target - mean).square()))),
        "pit_calibration_error": pit_error,
        "mean_additivity_violation": relative(mapb - ma - mb, mapb),
        "scale_violation": 0.5
        * (
            relative(msa - scale * ma, scale * ma)
            + relative(vsa - scale.square() * va, scale.square() * va)
        ),
        "variance_polarization_violation": relative(vapb + vamb - 2 * va - 2 * vb, vapb + vamb),
    }


def main() -> None:
    overall_started = time.perf_counter()
    rows = []
    parameter_counts = {}
    constructors = {
        "joint_gaussian": lambda: ProjectiveMixtureNet(1),
        "projective_mixture4": lambda: ProjectiveMixtureNet(COMPONENTS),
        "direct_mixture4": lambda: DirectMixtureNet(COMPONENTS),
    }
    for dataset in experiment.DATASETS:
        train_history, train_future, test_history, test_future = experiment.make_windows(dataset)
        for seed in experiment.SEEDS:
            for model_name, constructor in constructors.items():
                torch.manual_seed(seed)
                model = constructor()
                parameter_counts[model_name] = sum(parameter.numel() for parameter in model.parameters())
                seconds = train_model(model, train_history, train_future, seed)
                result = evaluate(model, test_history, test_future, seed)
                rows.append(
                    {"dataset": dataset, "seed": seed, "model": model_name, "train_seconds": seconds, **result}
                )
                torch.save(model.state_dict(), OUT / f"{dataset}__{model_name}__seed-{seed}.pt")
            if time.perf_counter() - overall_started > 2 * 60 * 60:
                raise TimeoutError("mixture pilot exceeded two hours")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    summary = frame.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    pivot = frame.pivot(index=["dataset", "seed"], columns="model", values="heldout_nll")
    mixture_gaussian_wins = int((pivot.projective_mixture4 <= pivot.joint_gaussian).sum())
    mixture_direct_wins = int((pivot.projective_mixture4 <= pivot.direct_mixture4).sum())
    mean_table = summary.pivot(index="dataset", columns="model", values="heldout_nll")
    dataset_improvements = {
        dataset: float(mean_table.loc[dataset, "joint_gaussian"] - mean_table.loc[dataset, "projective_mixture4"])
        for dataset in experiment.DATASETS
    }
    improvement_datasets = sum(value >= 0.05 for value in dataset_improvements.values())
    mixture_rows = summary[summary.model == "projective_mixture4"]
    identity_columns = ["mean_additivity_violation", "scale_violation", "variance_polarization_violation"]
    maximum_identity = float(mixture_rows[identity_columns].to_numpy().max())
    calibration = summary.groupby("model").pit_calibration_error.mean()
    better_comparator = min(float(calibration.joint_gaussian), float(calibration.direct_mixture4))
    calibration_difference = float(calibration.projective_mixture4 - better_comparator)
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - overall_started,
        "cells": len(frame),
        "parameter_counts": parameter_counts,
        "dataset_nll_improvements": dataset_improvements,
        "improvement_datasets": int(improvement_datasets),
        "mixture_vs_gaussian_cell_wins": mixture_gaussian_wins,
        "mixture_vs_direct_cell_wins": mixture_direct_wins,
        "maximum_projective_identity_violation": maximum_identity,
        "mixture_pit_calibration_error": float(calibration.projective_mixture4),
        "calibration_error_difference": calibration_difference,
    }
    audit["passed"] = bool(
        improvement_datasets >= 2
        and mixture_gaussian_wins >= 6
        and mixture_direct_wins >= 6
        and maximum_identity < 1e-5
        and calibration_difference <= 0.05
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
