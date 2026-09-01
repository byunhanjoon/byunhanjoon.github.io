"""Compute-limited closest-baseline test for projective linear queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
NOVELTY = HERE.parent / "projective-novelty-pilots"
FOLLOWUP = HERE.parent / "oral-ceiling-followups"
MOSES_ROOT = HERE / "vendor" / "moses"
TACTIS_ROOT = HERE / "vendor" / "tactis"
for path in (NOVELTY, FOLLOWUP, MOSES_ROOT, TACTIS_ROOT):
    sys.path.insert(0, str(path))

import run_mixture as mixture  # noqa: E402
from run_mixture_capacity_controls import MatchedDirectMixture  # noqa: E402
from core.model import Moses  # noqa: E402
from tactis.model.tactis import TACTiS  # noqa: E402


OUT = HERE / "outputs"
CHECKPOINTS = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
DEVICE = mixture.experiment.DEVICE
DATASETS = mixture.experiment.DATASETS
SEEDS = mixture.experiment.SEEDS
FAMILIES = ("point", "difference", "dense", "scaled_dense")
EVAL_CONTEXTS = 1_024
SAMPLES = 256
PROTOCOL_SHA256 = "204e6745f71ba719776cf8ce0bbb829a3d9c2f897571e4c3aa1af488c57d35df"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_moses() -> Moses:
    return Moses(
        n_inputs=mixture.experiment.CHANNELS,
        n_heads=2,
        num_components=4,
        latent_dim=64,
        num_flow_layers=2,
        num_encoder_layers=1,
        num_bins=16,
        bounds=20.0,
        encoder_model="transformer",
    )


def make_tactis() -> TACTiS:
    temporal_encoder = {
        "attention_layers": 3,
        "attention_heads": 3,
        "attention_dim": 16,
        "attention_feedforward_dim": 16,
        "dropout": 0.0,
    }
    return TACTiS(
        num_series=mixture.experiment.CHANNELS,
        flow_series_embedding_dim=5,
        copula_series_embedding_dim=5,
        flow_input_encoder_layers=3,
        copula_input_encoder_layers=3,
        input_encoding_normalization=True,
        data_normalization="standardization",
        loss_normalization="series",
        positional_encoding={"dropout": 0.0},
        flow_temporal_encoder=temporal_encoder.copy(),
        copula_temporal_encoder=temporal_encoder.copy(),
        copula_decoder={
            "min_u": 0.01,
            "max_u": 0.99,
            "attentional_copula": {
                "attention_heads": 3,
                "attention_layers": 3,
                "attention_dim": 16,
                "mlp_layers": 3,
                "mlp_dim": 16,
                "resolution": 50,
            },
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 8,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
    )


def moses_coordinates(batch: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    history_time = torch.arange(mixture.experiment.HISTORY_STEPS, device=device, dtype=torch.float32)
    future_time = torch.arange(
        mixture.experiment.HISTORY_STEPS,
        mixture.experiment.HISTORY_STEPS + mixture.experiment.HORIZON,
        device=device,
        dtype=torch.float32,
    )
    scale = float(mixture.experiment.HISTORY_STEPS + mixture.experiment.HORIZON - 1)
    tobs = history_time.repeat_interleave(mixture.experiment.CHANNELS).expand(batch, -1) / scale
    tqry = future_time.repeat_interleave(mixture.experiment.CHANNELS).expand(batch, -1) / scale
    channels = torch.arange(mixture.experiment.CHANNELS, device=device, dtype=torch.float32)
    cobs = channels.repeat(mixture.experiment.HISTORY_STEPS).expand(batch, -1)
    cqry = channels.repeat(mixture.experiment.HORIZON).expand(batch, -1)
    obs_mask = torch.ones_like(tobs)
    qry_mask = torch.ones_like(tqry)
    return tobs, cobs, obs_mask, tqry, cqry, qry_mask


def tactis_coordinates(batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
    history_time = torch.arange(mixture.experiment.HISTORY_STEPS, device=device).expand(batch, -1).float()
    future_time = torch.arange(
        mixture.experiment.HISTORY_STEPS,
        mixture.experiment.HISTORY_STEPS + mixture.experiment.HORIZON,
        device=device,
    ).expand(batch, -1).float()
    return history_time, future_time


def train_moses(dataset: str, seed: int, history: np.ndarray, future: np.ndarray) -> dict[str, float]:
    destination = CHECKPOINTS / f"{dataset}__moses__seed-{seed}.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        return payload["metadata"]
    seed_everything(seed)
    model = make_moses().to(DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 113)
    started = time.perf_counter()
    final_loss = math.nan
    for step in range(3_000):
        indices = rng.integers(0, len(history), size=32)
        x = torch.from_numpy(history[indices]).to(DEVICE)
        y = torch.from_numpy(future[indices]).to(DEVICE)
        tobs, cobs, obs_mask, tqry, cqry, qry_mask = moses_coordinates(len(indices), DEVICE)
        optimizer.zero_grad(set_to_none=True)
        model(
            tobs=tobs,
            cobs=cobs,
            obs_mask=obs_mask,
            xobs=x,
            tqry=tqry,
            cqry=cqry,
            qry_mask=qry_mask,
        )
        loss = model.compute_njnll(y, qry_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite MOSES loss for {dataset}/{seed} at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    seconds = time.perf_counter() - started
    metadata = {
        "train_seconds": seconds,
        "final_train_loss": final_loss,
        "parameters": count_parameters(model),
        "steps": 3_000,
        "batch_size": 32,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, destination)
    print(f"trained moses {dataset} seed={seed}: {seconds:.1f}s loss={final_loss:.4f}", flush=True)
    return metadata


def stage2_parameters(model: TACTiS) -> list[nn.Parameter]:
    prefixes = (
        "copula_series_encoder",
        "copula_time_encoding",
        "copula_input_encoder",
        "copula_encoder",
        "decoder.copula",
    )
    return [parameter for name, parameter in model.named_parameters() if name.startswith(prefixes)]


def train_tactis(dataset: str, seed: int, history: np.ndarray, future: np.ndarray) -> dict[str, float]:
    destination = CHECKPOINTS / f"{dataset}__tactis2__seed-{seed}.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        return payload["metadata"]
    seed_everything(seed)
    model = make_tactis().to(DEVICE).train()
    rng = np.random.default_rng(seed + 227)
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    final_stage1 = math.nan
    for step in range(1_500):
        indices = rng.integers(0, len(history), size=32)
        x = torch.from_numpy(history[indices].reshape(-1, 32, 8)).permute(0, 2, 1).to(DEVICE)
        y = torch.from_numpy(future[indices].reshape(-1, 4, 8)).permute(0, 2, 1).to(DEVICE)
        history_time, future_time = tactis_coordinates(len(indices), DEVICE)
        optimizer.zero_grad(set_to_none=True)
        marginal_logdet, _ = model.loss(history_time, x, future_time, y)
        loss = -marginal_logdet.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite TACTiS stage-1 loss for {dataset}/{seed} at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_stage1 = float(loss.detach())

    model.initialize_stage2()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(stage2_parameters(model), lr=4e-4, weight_decay=1e-5)
    final_stage2 = math.nan
    for step in range(1_500):
        indices = rng.integers(0, len(history), size=32)
        x = torch.from_numpy(history[indices].reshape(-1, 32, 8)).permute(0, 2, 1).to(DEVICE)
        y = torch.from_numpy(future[indices].reshape(-1, 4, 8)).permute(0, 2, 1).to(DEVICE)
        history_time, future_time = tactis_coordinates(len(indices), DEVICE)
        optimizer.zero_grad(set_to_none=True)
        _, copula_loss = model.loss(history_time, x, future_time, y)
        loss = copula_loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite TACTiS stage-2 loss for {dataset}/{seed} at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(stage2_parameters(model), 1.0)
        optimizer.step()
        final_stage2 = float(loss.detach())
    seconds = time.perf_counter() - started
    metadata = {
        "train_seconds": seconds,
        "final_stage1_loss": final_stage1,
        "final_stage2_loss": final_stage2,
        "parameters": count_parameters(model),
        "stage1_steps": 1_500,
        "stage2_steps": 1_500,
        "batch_size": 32,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, destination)
    print(
        f"trained tactis2 {dataset} seed={seed}: {seconds:.1f}s "
        f"losses={final_stage1:.4f}/{final_stage2:.4f}",
        flush=True,
    )
    return metadata


def make_queries(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 331)
    dimension = mixture.experiment.OUTPUT_DIM
    queries = np.zeros((count, len(FAMILIES), dimension), dtype=np.float32)
    point = rng.integers(0, dimension, size=count)
    queries[np.arange(count), 0, point] = rng.choice((-1.0, 1.0), size=count)
    for row in range(count):
        pair = rng.choice(dimension, size=2, replace=False)
        queries[row, 1, pair] = (1.0, -1.0)
    dense = rng.normal(size=(count, dimension)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
    queries[:, 2] = dense
    scaled = rng.normal(size=(count, dimension)).astype(np.float32)
    scaled /= np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-8
    scaled *= rng.uniform(0.3, 2.7, size=count).astype(np.float32)[:, None]
    queries[:, 3] = scaled
    return queries


def sample_projective(model: mixture.ProjectiveMixtureNet, history: Tensor, samples: int) -> Tensor:
    output = model.output(model.backbone(history)).reshape(len(history), model.components, -1)
    weights = torch.softmax(output[:, :, 0], dim=-1)
    means = output[:, :, 1 : 1 + mixture.experiment.OUTPUT_DIM]
    scales = nn.functional.softplus(output[:, :, 1 + mixture.experiment.OUTPUT_DIM :]) + 1e-4
    indices = torch.multinomial(weights, samples, replacement=True)
    gather = indices[:, :, None].expand(-1, -1, mixture.experiment.OUTPUT_DIM)
    selected_mean = torch.gather(means, 1, gather)
    selected_scale = torch.gather(scales, 1, gather)
    draws = selected_mean + selected_scale * torch.randn_like(selected_mean)
    return draws.permute(0, 2, 1)


def sample_direct(model: MatchedDirectMixture, history: Tensor, queries: Tensor, samples: int) -> Tensor:
    batch, query_count, dimension = queries.shape
    x = history[:, None, :].expand(-1, query_count, -1).reshape(batch * query_count, -1)
    q = queries.reshape(batch * query_count, dimension)
    log_weights, means, variances = model(x, q)
    indices = torch.multinomial(torch.exp(log_weights), samples, replacement=True)
    selected_mean = torch.gather(means, 1, indices)
    selected_scale = torch.sqrt(torch.gather(variances, 1, indices))
    return (selected_mean + selected_scale * torch.randn_like(selected_mean)).reshape(batch, query_count, samples)


def ensemble_crps(samples: Tensor, target: Tensor) -> Tensor:
    sample_count = samples.shape[-1]
    first = torch.mean(torch.abs(samples - target[..., None]), dim=-1)
    ordered = torch.sort(samples, dim=-1).values
    coefficient = 2 * torch.arange(1, sample_count + 1, device=samples.device) - sample_count - 1
    second = torch.sum(ordered * coefficient, dim=-1) / (sample_count * sample_count)
    return first - second


def score_samples(samples: Tensor, targets: Tensor) -> dict[str, float]:
    crps = ensemble_crps(samples, targets)
    lower50 = torch.quantile(samples, 0.25, dim=-1)
    upper50 = torch.quantile(samples, 0.75, dim=-1)
    lower90 = torch.quantile(samples, 0.05, dim=-1)
    upper90 = torch.quantile(samples, 0.95, dim=-1)
    coverage50 = ((targets >= lower50) & (targets <= upper50)).float().mean(dim=0)
    coverage90 = ((targets >= lower90) & (targets <= upper90)).float().mean(dim=0)
    result: dict[str, float] = {}
    for index, family in enumerate(FAMILIES):
        result[f"crps_{family}"] = float(crps[:, index].mean())
        result[f"coverage50_{family}"] = float(coverage50[index])
        result[f"coverage90_{family}"] = float(coverage90[index])
    result["macro_crps"] = float(crps.mean())
    result["coverage50"] = float(coverage50.mean())
    result["coverage90"] = float(coverage90.mean())
    result["coverage_error"] = 0.5 * (abs(result["coverage50"] - 0.50) + abs(result["coverage90"] - 0.90))
    return result


@torch.no_grad()
def evaluate_one(
    model_name: str,
    model: nn.Module,
    history: np.ndarray,
    future: np.ndarray,
    queries: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float], float]:
    model.to(DEVICE).eval()
    all_samples = []
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    for first in range(0, len(history), batch_size):
        last = min(first + batch_size, len(history))
        x = torch.from_numpy(history[first:last]).to(DEVICE)
        q = torch.from_numpy(queries[first:last]).to(DEVICE)
        if model_name == "projective_mixture4":
            joint = sample_projective(model, x, SAMPLES)
            projected = torch.einsum("bds,bqd->bqs", joint, q)
        elif model_name == "direct_mixture4_matched":
            projected = sample_direct(model, x, q, SAMPLES)
        elif model_name == "moses":
            tobs, cobs, obs_mask, tqry, cqry, qry_mask = moses_coordinates(last - first, DEVICE)
            model(
                tobs=tobs,
                cobs=cobs,
                obs_mask=obs_mask,
                xobs=x,
                tqry=tqry,
                cqry=cqry,
                qry_mask=qry_mask,
            )
            joint = model.sample_joint(qry_mask, num_samples=SAMPLES)
            projected = torch.einsum("bds,bqd->bqs", joint, q)
        elif model_name == "tactis2":
            history_series = x.reshape(-1, 32, 8).permute(0, 2, 1)
            history_time, future_time = tactis_coordinates(last - first, DEVICE)
            draws = model.sample(SAMPLES, history_time, history_series, future_time)
            joint = draws[:, :, -mixture.experiment.HORIZON :, :].permute(0, 2, 1, 3).reshape(
                last - first, mixture.experiment.OUTPUT_DIM, SAMPLES
            )
            projected = torch.einsum("bds,bqd->bqs", joint, q)
        else:
            raise ValueError(model_name)
        all_samples.append(projected.cpu())
    torch.cuda.synchronize(DEVICE)
    latency = time.perf_counter() - started
    sample_tensor = torch.cat(all_samples)
    targets = torch.from_numpy(np.einsum("bqd,bd->bq", queries, future))
    return score_samples(sample_tensor, targets), 1_000.0 * latency / len(history)


def load_model(model_name: str, dataset: str, seed: int) -> tuple[nn.Module, dict[str, float]]:
    if model_name == "projective_mixture4":
        model = mixture.ProjectiveMixtureNet(mixture.COMPONENTS)
        path = NOVELTY / "mixture" / f"{dataset}__projective_mixture4__seed-{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model, {"parameters": count_parameters(model), "train_seconds": math.nan}
    if model_name == "direct_mixture4_matched":
        model = MatchedDirectMixture()
        path = NOVELTY / "mixture" / f"{dataset}__direct_mixture4_matched__seed-{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model, {"parameters": count_parameters(model), "train_seconds": math.nan}
    path = CHECKPOINTS / f"{dataset}__{model_name}__seed-{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model_name == "moses":
        model = make_moses()
    elif model_name == "tactis2":
        model = make_tactis()
        model.initialize_stage2()
    else:
        raise ValueError(model_name)
    model.load_state_dict(payload["state_dict"])
    return model, payload["metadata"]


def train_all() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        train_history, train_future, _, _ = mixture.experiment.make_windows(dataset)
        for seed in SEEDS:
            for model_name, trainer in (("moses", train_moses), ("tactis2", train_tactis)):
                metadata = trainer(dataset, seed, train_history, train_future)
                rows.append({"dataset": dataset, "seed": seed, "model": model_name, **metadata})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "training_cells.csv", index=False)
    return frame


def evaluate_all() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        _, _, test_history, test_future = mixture.experiment.make_windows(dataset)
        test_history = test_history[:EVAL_CONTEXTS]
        test_future = test_future[:EVAL_CONTEXTS]
        for seed in SEEDS:
            queries = make_queries(seed, EVAL_CONTEXTS)
            for model_name, batch_size in (
                ("projective_mixture4", 128),
                ("direct_mixture4_matched", 128),
                ("moses", 32),
                ("tactis2", 32),
            ):
                seed_everything(seed + 449)
                model, metadata = load_model(model_name, dataset, seed)
                metrics, latency = evaluate_one(
                    model_name, model, test_history, test_future, queries, batch_size
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model_name,
                        "parameters": metadata["parameters"],
                        "train_seconds": metadata.get("train_seconds", math.nan),
                        "latency_ms_per_context": latency,
                        **metrics,
                    }
                )
                print(
                    f"evaluated {model_name} {dataset} seed={seed}: "
                    f"CRPS={metrics['macro_crps']:.4f} latency={latency:.3f}ms/context",
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "evaluation_cells.csv", index=False)
    frame.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "evaluation_summary.csv", index=False
    )
    return frame


def audit(frame: pd.DataFrame) -> dict[str, object]:
    summary = frame.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    table = summary.pivot(index="dataset", columns="model", values="macro_crps")
    projective = table["projective_mixture4"]
    moses_ok = projective <= 1.02 * table["moses"]
    tactis_ok = projective <= 1.02 * table["tactis2"]
    coverage = summary.groupby("model").coverage_error.mean()
    better_joint_coverage = min(float(coverage["moses"]), float(coverage["tactis2"]))
    calibration_gap = float(coverage["projective_mixture4"] - better_joint_coverage)
    latency = summary.groupby("model").latency_ms_per_context.mean()
    fastest_joint = min(float(latency["moses"]), float(latency["tactis2"]))
    speedup = fastest_joint / float(latency["projective_mixture4"])
    result: dict[str, object] = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "cells": len(frame),
        "all_finite": bool(np.isfinite(frame["macro_crps"]).all()),
        "projective_within_2pct_moses_datasets": int(moses_ok.sum()),
        "projective_within_2pct_tactis_datasets": int(tactis_ok.sum()),
        "projective_coverage_error": float(coverage["projective_mixture4"]),
        "better_joint_coverage_error": better_joint_coverage,
        "coverage_error_gap": calibration_gap,
        "four_query_speedup_vs_fastest_joint": speedup,
        "speed_gate_passed": bool(speedup >= 5.0),
        "dataset_crps": {
            dataset: {model: float(table.loc[dataset, model]) for model in table.columns}
            for dataset in table.index
        },
    }
    result["predictive_gate_passed"] = bool(
        result["all_finite"]
        and result["projective_within_2pct_moses_datasets"] >= 2
        and result["projective_within_2pct_tactis_datasets"] >= 2
        and calibration_gap <= 0.03
    )
    result["promising"] = result["predictive_gate_passed"]
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def verify_protocol() -> None:
    digest = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol changed: expected {PROTOCOL_SHA256}, found {digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    verify_protocol()
    started = time.perf_counter()
    if not args.skip_train:
        train_all()
    if args.skip_eval:
        return
    frame = evaluate_all()
    result = audit(frame)
    result["wall_seconds_this_run"] = time.perf_counter() - started
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
