"""Post-hoc aggregation-aware corrective experiment for Day 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from day8_core import ArrayData, ModernNCAModel, TabRModel, evaluate_predictions, make_synthetic, train_model
from risk_retrieval_followup import (
    DATASETS,
    MODEL_SEEDS,
    PROCESSED,
    REAL_OUTPUT,
    SPLIT_SEEDS,
    SYNTH_SEEDS,
    SYNTH_OUTPUT,
    atomic_json,
    cross_fitted_proxy,
    exact_synthetic_proxy,
    fit_or_load,
    load_processed,
    slug,
    squared_distances,
)


HERE = Path(__file__).resolve().parent
REAL_AGG = HERE / "raw" / "posthoc_aggregate_risk"
SYNTH_AGG = HERE / "raw" / "posthoc_aggregate_risk_synthetic"
K_GRID = (16, 32, 64)


def project_simplex(values: Tensor) -> Tensor:
    """Euclidean projection of each row onto the probability simplex."""

    ordered, _ = torch.sort(values, dim=1, descending=True)
    cssv = ordered.cumsum(dim=1) - 1.0
    indices = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)[None]
    positive = ordered - cssv / indices > 0
    rho = positive.sum(dim=1).clamp_min(1) - 1
    theta = torch.gather(cssv / indices, 1, rho[:, None])
    return (values - theta).clamp_min(0.0)


def risk_objective(weights: Tensor, discrepancies: Tensor, variances: Tensor) -> Tensor:
    aggregate = torch.einsum("bk,bkc->bc", weights, discrepancies)
    return aggregate.square().sum(dim=1) + (weights.square() * variances).sum(dim=1)


def solve_weights(
    discrepancies: Tensor,
    variances: Tensor,
    mode: str,
    iterations: int = 384,
) -> tuple[Tensor, dict[str, float]]:
    """Solve the special PSD simplex QP with a conservative fixed step."""

    batch, candidates, _ = discrepancies.shape
    if mode == "mismatch":
        variances = variances.mean(dim=1, keepdim=True).expand_as(variances)
    elif mode == "reliability":
        weights = variances.clamp_min(1e-7).reciprocal()
        weights = weights / weights.sum(dim=1, keepdim=True)
        zeros = torch.zeros_like(discrepancies)
        uniform = torch.full_like(weights, 1.0 / candidates)
        return weights, {
            "uniform_objective": float(risk_objective(uniform, zeros, variances).mean()),
            "optimized_objective": float(risk_objective(weights, zeros, variances).mean()),
            "max_simplex_error": float((weights.sum(dim=1) - 1).abs().max()),
            "minimum_weight": float(weights.min()),
        }
    elif mode != "full":
        raise ValueError(mode)

    weights = torch.full(
        (batch, candidates),
        1.0 / candidates,
        device=discrepancies.device,
        dtype=discrepancies.dtype,
    )
    uniform = weights.clone()
    accelerated = weights.clone()
    momentum = torch.ones((batch, 1), device=weights.device, dtype=weights.dtype)
    # 2 * lambda_max(D D^T + diag(v)) is bounded by this value.
    lipschitz = 2.0 * (
        discrepancies.square().sum(dim=(1, 2)) + variances.max(dim=1).values
    ).clamp_min(1e-6)
    step = 0.95 / lipschitz
    for _ in range(iterations):
        aggregate = torch.einsum("bk,bkc->bc", accelerated, discrepancies)
        gradient = 2.0 * torch.einsum("bc,bkc->bk", aggregate, discrepancies)
        gradient = gradient + 2.0 * accelerated * variances
        next_weights = project_simplex(accelerated - step[:, None] * gradient)
        next_momentum = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * momentum.square()))
        accelerated = next_weights + (momentum - 1.0) / next_momentum * (next_weights - weights)
        weights, momentum = next_weights, next_momentum
    uniform_objective = risk_objective(uniform, discrepancies, variances)
    optimized_objective = risk_objective(weights, discrepancies, variances)
    return weights, {
        "uniform_objective": float(uniform_objective.mean()),
        "optimized_objective": float(optimized_objective.mean()),
        "objective_nonincrease_fraction": float((optimized_objective <= uniform_objective + 1e-7).float().mean()),
        "max_simplex_error": float((weights.sum(dim=1) - 1).abs().max()),
        "minimum_weight": float(weights.min()),
    }


def discrepancy_tensor(
    proxy: dict[str, np.ndarray],
    part: str,
    query_slice: slice,
    indices: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    m_train = torch.as_tensor(proxy["m_train"], device=device, dtype=torch.float32)
    m_query = torch.as_tensor(proxy[f"m_{part}"][query_slice], device=device, dtype=torch.float32)
    if m_train.ndim == 1:
        discrepancies = m_train[indices] - m_query[:, None]
        discrepancies = discrepancies[..., None]
    else:
        discrepancies = m_train[indices] - m_query[:, None, :]
    sigma = torch.as_tensor(proxy["sigma_train"], device=device, dtype=torch.float32)[indices]
    return discrepancies, sigma.clamp_min(1e-7)


@torch.no_grad()
def aggregate_prediction_grid(
    model: nn.Module,
    data: ArrayData,
    part: str,
    device: torch.device,
    configs: dict[str, tuple[dict[str, np.ndarray], str]],
    k_values: tuple[int, ...],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict[str, float]]]:
    if not isinstance(model, (TabRModel, ModernNCAModel)):
        raise TypeError(type(model))
    model.eval()
    q_num = torch.as_tensor(data.x_num[part], device=device)
    q_cat = torch.as_tensor(data.x_cat[part], device=device)
    c_num = torch.as_tensor(data.x_num["train"], device=device)
    c_cat = torch.as_tensor(data.x_cat["train"], device=device)
    c_y = torch.as_tensor(data.y["train"], device=device)
    c_key = model.keys(c_num, c_cat)
    outputs: dict[tuple[str, int], list[Tensor]] = {
        (name, k): [] for name in configs for k in k_values
    }
    diagnostics: dict[tuple[str, int], list[dict[str, float]]] = {
        (name, k): [] for name in configs for k in k_values
    }
    max_k = min(max(k_values), len(c_y))
    for start in range(0, len(q_num), 256):
        stop = min(start + 256, len(q_num))
        q_key = model.keys(q_num[start:stop], q_cat[start:stop])
        distances = squared_distances(q_key, c_key)
        shortlist = torch.topk(distances, k=max_k, largest=False).indices
        for k in k_values:
            chosen = shortlist[:, : min(k, max_k)]
            if data.task == "classification":
                candidate_target = nn.functional.one_hot(c_y[chosen].long(), data.n_classes).float()
            else:
                candidate_target = c_y[chosen].float()[..., None]
            for name, (proxy, mode) in configs.items():
                discrepancies, variances = discrepancy_tensor(
                    proxy, part, slice(start, stop), chosen, device
                )
                weights, diag = solve_weights(discrepancies, variances, mode)
                prediction = torch.einsum("bk,bko->bo", weights, candidate_target)
                outputs[(name, k)].append(prediction.cpu())
                diagnostics[(name, k)].append(diag)
    arrays = {key: torch.cat(value).numpy() for key, value in outputs.items()}
    merged_diagnostics = {}
    for key, chunks in diagnostics.items():
        merged_diagnostics[key] = {
            metric: float(np.mean([chunk[metric] for chunk in chunks]))
            for metric in chunks[0]
        }
    return arrays, merged_diagnostics


def direct_proxy_metrics(data: ArrayData, proxy: dict[str, np.ndarray], part: str) -> dict[str, float]:
    prediction = proxy[f"m_{part}"]
    if data.task == "regression":
        prediction = prediction[:, None]
    return evaluate_predictions(prediction, data.y[part], data)


def tune_aggregate_methods(
    model: nn.Module,
    data: ArrayData,
    device: torch.device,
    configs: dict[str, tuple[dict[str, np.ndarray], str]],
) -> list[dict[str, Any]]:
    validation_predictions, validation_diag = aggregate_prediction_grid(
        model, data, "validation", device, configs, K_GRID
    )
    choices: dict[str, tuple[int, dict[str, float]]] = {}
    for name in configs:
        candidates = []
        for k in K_GRID:
            metrics = evaluate_predictions(validation_predictions[(name, k)], data.y["validation"], data)
            candidates.append((k, metrics))
        choices[name] = min(candidates, key=lambda item: (item[1]["loss"], item[0]))
    chosen_ks = tuple(sorted(set(k for k, _ in choices.values())))
    test_predictions, test_diag = aggregate_prediction_grid(
        model, data, "test", device, configs, chosen_ks
    )
    rows = []
    for name, (k, validation_metrics) in choices.items():
        test_metrics = evaluate_predictions(test_predictions[(name, k)], data.y["test"], data)
        rows.append({
            "method": name,
            "k": k,
            "validation_loss": validation_metrics["loss"],
            **test_metrics,
            **{f"validation_{key}": value for key, value in validation_diag[(name, k)].items()},
            **{f"test_{key}": value for key, value in test_diag[(name, k)].items()},
        })
    return rows


def primary_baseline(dataset: str, split_seed: int, model: str, model_seed: int) -> dict[str, Any]:
    path = REAL_OUTPUT / f"{slug(dataset)}__split-{split_seed}__{model}__seed-{model_seed}.json"
    payload = json.loads(path.read_text())
    row = next(item for item in payload["methods"] if item["method"] == "distance")
    return {key: row[key] for key in ("loss", "metric", "score", "metric_name")}


def run_real(device: torch.device, shard: int, n_shards: int) -> None:
    cells = [
        (name, split_seed, model, model_seed)
        for name, _, _ in DATASETS
        for split_seed in SPLIT_SEEDS
        for model in ("TabR", "ModernNCA")
        for model_seed in MODEL_SEEDS
    ]
    for cell_index, (name, split_seed, model_name, model_seed) in enumerate(cells):
        if cell_index % n_shards != shard:
            continue
        output = REAL_AGG / f"{slug(name)}__split-{split_seed}__{model_name}__seed-{model_seed}.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "complete":
            continue
        print(f"aggregate real shard={shard} {name} split={split_seed} {model_name} seed={model_seed}", flush=True)
        data = load_processed(name, split_seed)
        proxy_path = PROCESSED / f"{slug(name)}__split-{split_seed}__proxy.npz"
        proxy = cross_fitted_proxy(data, split_seed, proxy_path)
        model, _ = fit_or_load(data, name, split_seed, model_name, model_seed, device)
        configs = {
            "aggregate_full": (proxy, "full"),
            "aggregate_mismatch": (proxy, "mismatch"),
            "aggregate_reliability": (proxy, "reliability"),
        }
        rows = tune_aggregate_methods(model, data, device, configs)
        rows.insert(0, {"method": "distance_model", "k": None, **primary_baseline(name, split_seed, model_name, model_seed)})
        rows.append({"method": "direct_proxy", "k": None, **direct_proxy_metrics(data, proxy, "test")})
        atomic_json(output, {
            "status": "complete",
            "posthoc": True,
            "dataset": name,
            "task": data.task,
            "split_seed": split_seed,
            "model": model_name,
            "model_seed": model_seed,
            "methods": rows,
        })


def synthetic_primary_baseline(task: str, model: str, seed: int) -> dict[str, Any]:
    path = SYNTH_OUTPUT / f"{task}__{model}__seed-{seed}.json"
    payload = json.loads(path.read_text())
    row = next(item for item in payload["methods"] if item["method"] == "distance")
    return {key: row[key] for key in ("loss", "metric", "score", "metric_name")}


def run_synthetic(device: torch.device, shard: int, n_shards: int) -> None:
    cells = [
        (task, seed, model)
        for task in ("S1_rotating", "S2_global", "S3_noise", "S4_warp")
        for seed in SYNTH_SEEDS
        for model in ("TabR", "ModernNCA")
    ]
    for cell_index, (task, seed, model_name) in enumerate(cells):
        if cell_index % n_shards != shard:
            continue
        output = SYNTH_AGG / f"{task}__{model_name}__seed-{seed}.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "complete":
            continue
        print(f"aggregate synthetic shard={shard} {task} {model_name} seed={seed}", flush=True)
        data, meta = make_synthetic(task, seed, n_train=4096, n_val=1024, n_test=1024)
        model, _ = train_model(data, model_name, "raw", "raw", seed, device, "standard", max_epochs=48)
        exact = exact_synthetic_proxy(data, meta)
        estimated = cross_fitted_proxy(data, seed)
        configs = {
            "aggregate_exact": (exact, "full"),
            "aggregate_estimated": (estimated, "full"),
        }
        rows = tune_aggregate_methods(model, data, device, configs)
        rows.insert(0, {"method": "distance_model", "k": None, **synthetic_primary_baseline(task, model_name, seed)})
        rows.append({"method": "direct_estimated_proxy", "k": None, **direct_proxy_metrics(data, estimated, "test")})
        atomic_json(output, {
            "status": "complete",
            "posthoc": True,
            "task": task,
            "seed": seed,
            "model": model_name,
            "methods": rows,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("real", "synthetic"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.stage == "real":
        run_real(device, args.shard, args.n_shards)
    else:
        run_synthetic(device, args.shard, args.n_shards)


if __name__ == "__main__":
    main()
