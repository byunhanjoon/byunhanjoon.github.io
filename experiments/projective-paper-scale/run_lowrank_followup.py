"""Capacity-matched rank-4 covariance and validation-calibration follow-up."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

import run_decisive as decisive


HERE = Path(__file__).resolve().parent
OUT = HERE / "lowrank_outputs"
CHECKPOINTS = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
DEVICE = decisive.DEVICE
RANK = 4
WIDTH = 118
PROTOCOL_SHA256 = "798cfff340c2fae989c6631dd06c4292fe8428e7fc6b5d09196f648a2b20c5ea"


class LowRankProjectiveMixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.components = decisive.mixture.COMPONENTS
        self.backbone = nn.Sequential(
            nn.Linear(decisive.mixture.experiment.HISTORY_DIM, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
            nn.GELU(),
        )
        per_component = 1 + 2 * decisive.mixture.experiment.OUTPUT_DIM + decisive.mixture.experiment.OUTPUT_DIM * RANK
        self.output = nn.Linear(WIDTH, self.components * per_component)

    def joint(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dimension = decisive.mixture.experiment.OUTPUT_DIM
        output = self.output(self.backbone(history)).reshape(len(history), self.components, -1)
        log_weights = torch.log_softmax(output[:, :, 0], dim=-1)
        means = output[:, :, 1 : 1 + dimension]
        diagonal = nn.functional.softplus(output[:, :, 1 + dimension : 1 + 2 * dimension]) + 1e-4
        factors = output[:, :, 1 + 2 * dimension :].reshape(len(history), self.components, dimension, RANK)
        return log_weights, means, diagonal, factors

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_weights, means, diagonal, factors = self.joint(history)
        projected_mean = torch.einsum("bkd,bd->bk", means, query)
        projected_factor = torch.einsum("bkdr,bd->bkr", factors, query)
        variance = projected_factor.square().sum(dim=-1) + torch.einsum(
            "bkd,bd->bk", diagonal.square(), query.square()
        )
        return log_weights, projected_mean, variance.clamp_min(1e-8)


def make_validation_windows(name: str, count: int = 4_096) -> tuple[np.ndarray, np.ndarray]:
    experiment = decisive.mixture.experiment
    payload = np.load(experiment.SOURCE / f"{name}.npz")
    values = payload["values"].astype(np.float32)
    first = int(payload["train_end"])
    last = int(payload["validation_end"])
    available = np.arange(max(experiment.HISTORY_STEPS, first), last - experiment.HORIZON + 1)
    locations = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64)
    starts = available[locations]
    history = np.stack([values[t - experiment.HISTORY_STEPS : t] for t in starts]).reshape(count, -1)
    future = np.stack([values[t : t + experiment.HORIZON] for t in starts]).reshape(count, -1)
    return np.ascontiguousarray(history), np.ascontiguousarray(future)


def train_one(dataset: str, seed: int, history: np.ndarray, future: np.ndarray) -> dict[str, float]:
    path = CHECKPOINTS / f"{dataset}__lowrank4__seed-{seed}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    decisive.seed_everything(seed)
    model = LowRankProjectiveMixture()
    parameters = decisive.count_parameters(model)
    seconds = decisive.mixture.train_model(model, history, future, seed)
    metadata = {"parameters": parameters, "train_seconds": seconds, "steps": 3_000, "batch_size": 512}
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    print(f"trained lowrank4 {dataset} seed={seed}: {seconds:.1f}s parameters={parameters}", flush=True)
    return metadata


def load_one(dataset: str, seed: int) -> tuple[LowRankProjectiveMixture, dict[str, float]]:
    payload = torch.load(CHECKPOINTS / f"{dataset}__lowrank4__seed-{seed}.pt", map_location="cpu", weights_only=False)
    model = LowRankProjectiveMixture()
    model.load_state_dict(payload["state_dict"])
    return model, payload["metadata"]


@torch.no_grad()
def validation_predictions(
    model: LowRankProjectiveMixture,
    history: np.ndarray,
    future: np.ndarray,
    queries: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    log_weights_parts, mean_parts, variance_parts = [], [], []
    model.to(DEVICE).eval()
    for first in range(0, len(history), 256):
        last = min(first + 256, len(history))
        x = torch.from_numpy(history[first:last]).to(DEVICE)
        q = torch.from_numpy(queries[first:last]).to(DEVICE)
        log_weights, means, diagonal, factors = model.joint(x)
        projected_mean = torch.einsum("bkd,bqd->bqk", means, q)
        projected_factor = torch.einsum("bkdr,bqd->bqkr", factors, q)
        variance = projected_factor.square().sum(-1) + torch.einsum(
            "bkd,bqd->bqk", diagonal.square(), q.square()
        )
        log_weights_parts.append(log_weights[:, None, :].expand(-1, len(decisive.FAMILIES), -1))
        mean_parts.append(projected_mean)
        variance_parts.append(variance.clamp_min(1e-8))
    targets = torch.from_numpy(np.einsum("bqd,bd->bq", queries, future)).to(DEVICE)
    return (
        torch.cat(log_weights_parts),
        torch.cat(mean_parts),
        torch.cat(variance_parts),
        targets,
    )


def fit_temperature(dataset: str, seed: int, model: LowRankProjectiveMixture) -> tuple[float, float, float]:
    history, future = make_validation_windows(dataset)
    queries = decisive.make_queries(seed + 10_000, len(history))
    log_weights, means, variances, targets = validation_predictions(model, history, future, queries)
    shape = (-1, model.components)
    flat_weights = log_weights.reshape(shape).detach()
    flat_means = means.reshape(shape).detach()
    flat_variances = variances.reshape(shape).detach()
    flat_targets = targets.reshape(-1).detach()
    with torch.no_grad():
        before = float(decisive.mixture.mixture_nll(flat_weights, flat_means, flat_variances, flat_targets))
    log_temperature = torch.zeros((), device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature.clamp(-2.3, 2.3))
        loss = decisive.mixture.mixture_nll(
            flat_weights, flat_means, flat_variances * temperature, flat_targets
        )
        loss.backward()
        optimizer.step()
    temperature = float(torch.exp(log_temperature.clamp(-2.3, 2.3)).detach())
    with torch.no_grad():
        after = float(
            decisive.mixture.mixture_nll(
                flat_weights, flat_means, flat_variances * temperature, flat_targets
            )
        )
    return temperature, before, after


def sample_joint(
    model: LowRankProjectiveMixture,
    history: Tensor,
    samples: int,
    temperature: float,
) -> Tensor:
    log_weights, means, diagonal, factors = model.joint(history)
    indices = torch.multinomial(torch.exp(log_weights), samples, replacement=True)
    dimension = decisive.mixture.experiment.OUTPUT_DIM
    gather_vector = indices[:, :, None].expand(-1, -1, dimension)
    gather_factor = indices[:, :, None, None].expand(-1, -1, dimension, RANK)
    selected_mean = torch.gather(means, 1, gather_vector)
    selected_diagonal = torch.gather(diagonal, 1, gather_vector)
    selected_factor = torch.gather(factors, 1, gather_factor)
    diagonal_noise = selected_diagonal * torch.randn_like(selected_diagonal)
    rank_noise = torch.einsum(
        "bsdr,bsr->bsd",
        selected_factor,
        torch.randn(len(history), samples, RANK, device=history.device, dtype=history.dtype),
    )
    draw = selected_mean + math.sqrt(temperature) * (diagonal_noise + rank_noise)
    return draw.permute(0, 2, 1)


@torch.no_grad()
def evaluate_one(
    model: LowRankProjectiveMixture,
    history: np.ndarray,
    future: np.ndarray,
    queries: np.ndarray,
    temperature: float,
) -> tuple[dict[str, float], float]:
    model.to(DEVICE).eval()
    parts = []
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    for first in range(0, len(history), 128):
        last = min(first + 128, len(history))
        x = torch.from_numpy(history[first:last]).to(DEVICE)
        q = torch.from_numpy(queries[first:last]).to(DEVICE)
        draws = sample_joint(model, x, decisive.SAMPLES, temperature)
        parts.append(torch.einsum("bds,bqd->bqs", draws, q).cpu())
    torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - started
    samples = torch.cat(parts)
    targets = torch.from_numpy(np.einsum("bqd,bd->bq", queries, future))
    return decisive.score_samples(samples, targets), 1_000.0 * elapsed / len(history)


def audit(frame: pd.DataFrame) -> dict[str, object]:
    existing = pd.read_csv(HERE / "outputs" / "evaluation_cells.csv")
    combined = pd.concat([existing, frame], ignore_index=True)
    summary = combined.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    macro = summary.pivot(index="dataset", columns="model", values="macro_crps")
    calibrated = macro["lowrank4_calibrated"]
    tactis = macro["tactis2"]
    diagonal = macro["projective_mixture4"]
    within_tactis = calibrated <= 1.02 * tactis
    electricity_improvement = float(
        (diagonal["Electricity"] - calibrated["Electricity"]) / diagonal["Electricity"]
    )
    dense_columns = ["crps_dense", "crps_scaled_dense"]
    dense = combined.groupby("model")[dense_columns].mean().mean(axis=1)
    dense_ratio = float(dense["lowrank4_calibrated"] / dense["projective_mixture4"])
    coverage = summary.pivot(index="dataset", columns="model", values="coverage_error")
    traffic_coverage_reduction = float(
        coverage.loc["Traffic", "projective_mixture4"] - coverage.loc["Traffic", "lowrank4_calibrated"]
    )
    latency = summary.groupby("model").latency_ms_per_context.mean()
    speedup = float(latency["tactis2"] / latency["lowrank4_calibrated"])
    parameter_count = int(frame.parameters.iloc[0])
    capacity_gap = abs(parameter_count - 136_580) / 136_580
    result: dict[str, object] = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "cells": len(frame),
        "all_finite": bool(np.isfinite(frame["macro_crps"]).all()),
        "parameter_count": parameter_count,
        "capacity_relative_gap": capacity_gap,
        "within_2pct_tactis_datasets": int(within_tactis.sum()),
        "electricity_crps_improvement_vs_diagonal": electricity_improvement,
        "dense_crps_ratio_vs_diagonal": dense_ratio,
        "traffic_coverage_error_reduction": traffic_coverage_reduction,
        "speedup_vs_tactis": speedup,
        "dataset_crps": {
            dataset: {
                model: float(macro.loc[dataset, model])
                for model in ("projective_mixture4", "lowrank4", "lowrank4_calibrated", "tactis2")
            }
            for dataset in decisive.DATASETS
        },
        "dataset_coverage_error": {
            dataset: {
                model: float(coverage.loc[dataset, model])
                for model in ("projective_mixture4", "lowrank4", "lowrank4_calibrated", "tactis2")
            }
            for dataset in decisive.DATASETS
        },
    }
    gates = {
        "tactis_quality": result["within_2pct_tactis_datasets"] >= 2,
        "electricity_improvement": electricity_improvement >= 0.05,
        "dense_quality": dense_ratio <= 1.02,
        "traffic_calibration": traffic_coverage_reduction >= 0.03,
        "speed": speedup >= 50.0,
        "capacity": capacity_gap <= 0.01,
        "finite": result["all_finite"],
    }
    result["gates"] = gates
    result["passed"] = bool(all(gates.values()))
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    digest = hashlib.sha256((HERE / "LOWRANK_PROTOCOL.md").read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol changed: expected {PROTOCOL_SHA256}, found {digest}")
    started = time.perf_counter()
    training_rows, evaluation_rows, calibration_rows = [], [], []
    for dataset in decisive.DATASETS:
        train_history, train_future, test_history, test_future = decisive.mixture.experiment.make_windows(dataset)
        test_history = test_history[: decisive.EVAL_CONTEXTS]
        test_future = test_future[: decisive.EVAL_CONTEXTS]
        for seed in decisive.SEEDS:
            metadata = train_one(dataset, seed, train_history, train_future)
            training_rows.append({"dataset": dataset, "seed": seed, "model": "lowrank4", **metadata})
            model, _ = load_one(dataset, seed)
            temperature, before, after = fit_temperature(dataset, seed, model)
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "temperature": temperature,
                    "validation_nll_before": before,
                    "validation_nll_after": after,
                }
            )
            queries = decisive.make_queries(seed, decisive.EVAL_CONTEXTS)
            for name, value in (("lowrank4", 1.0), ("lowrank4_calibrated", temperature)):
                decisive.seed_everything(seed + 449)
                metrics, latency = evaluate_one(model, test_history, test_future, queries, value)
                evaluation_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": name,
                        "parameters": metadata["parameters"],
                        "train_seconds": metadata["train_seconds"],
                        "temperature": value,
                        "latency_ms_per_context": latency,
                        **metrics,
                    }
                )
                print(
                    f"evaluated {name} {dataset} seed={seed}: CRPS={metrics['macro_crps']:.4f} "
                    f"coverage_error={metrics['coverage_error']:.4f} temperature={value:.3f}",
                    flush=True,
                )
            del model
            torch.cuda.empty_cache()
    training = pd.DataFrame(training_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    calibration = pd.DataFrame(calibration_rows)
    training.to_csv(OUT / "training_cells.csv", index=False)
    evaluation.to_csv(OUT / "evaluation_cells.csv", index=False)
    calibration.to_csv(OUT / "calibration_cells.csv", index=False)
    evaluation.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "evaluation_summary.csv", index=False
    )
    result = audit(evaluation)
    result["wall_seconds"] = time.perf_counter() - started
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
