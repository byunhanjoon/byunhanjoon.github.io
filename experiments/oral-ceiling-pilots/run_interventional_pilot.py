"""Pilot C: meta-pretraining for interventions in confounded temporal tables."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
OUT = HERE / "interventional"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (20261241, 20261242, 20261243)
K_VALUES = (0, 2, 4, 8)
PROTOCOL_SHA256 = "538f14851b6a1cf54737c3b9bc8df3cf3b227c1b33cf2ef53469f261317b3164"
DEVICE = torch.device("cuda:0")
N_OBS = 48
MAX_INT = 8
CONTEXT = N_OBS + MAX_INT


def sample_training_batch(batch: int, causal: bool, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def uniform(low: float, high: float, shape: tuple[int, ...]) -> Tensor:
        return torch.rand(shape, generator=generator, device=DEVICE) * (high - low) + low

    b = uniform(-1.5, 1.5, (batch, 1))
    tau = uniform(-2.0, 2.0, (batch, 1))
    c = uniform(-1.5, 1.5, (batch, 1))
    alpha = uniform(-1.5, 1.5, (batch, 1))
    delta = uniform(-1.5, 1.5, (batch, 1))
    u_obs = torch.randn((batch, N_OBS), generator=generator, device=DEVICE)
    x_obs = u_obs + 0.5 * torch.randn((batch, N_OBS), generator=generator, device=DEVICE)
    a_obs = alpha * x_obs + delta * u_obs + 0.5 * torch.randn(
        (batch, N_OBS), generator=generator, device=DEVICE
    )
    y_obs = b * x_obs + tau * a_obs + c * u_obs + 0.3 * torch.randn(
        (batch, N_OBS), generator=generator, device=DEVICE
    )
    obs = torch.stack([x_obs, a_obs, y_obs, torch.zeros_like(x_obs)], dim=-1)

    u_int = torch.randn((batch, MAX_INT), generator=generator, device=DEVICE)
    x_int = u_int + 0.5 * torch.randn((batch, MAX_INT), generator=generator, device=DEVICE)
    a_int = 1.5 * torch.randn((batch, MAX_INT), generator=generator, device=DEVICE)
    y_int = b * x_int + tau * a_int + c * u_int + 0.3 * torch.randn(
        (batch, MAX_INT), generator=generator, device=DEVICE
    )
    intervention = torch.stack([x_int, a_int, y_int, torch.ones_like(x_int)], dim=-1)
    k = torch.randint(0, MAX_INT + 1, (batch,), generator=generator, device=DEVICE) if causal else torch.zeros(
        batch, dtype=torch.long, device=DEVICE
    )
    intervention_valid = torch.arange(MAX_INT, device=DEVICE)[None] < k[:, None]
    context = torch.cat([obs, intervention], dim=1)
    valid = torch.cat([torch.ones((batch, N_OBS), dtype=torch.bool, device=DEVICE), intervention_valid], dim=1)

    u_query = torch.randn((batch, 1), generator=generator, device=DEVICE)
    x_query = u_query + 0.5 * torch.randn((batch, 1), generator=generator, device=DEVICE)
    a_query = 1.5 * torch.randn((batch, 1), generator=generator, device=DEVICE)
    query = torch.cat([x_query, a_query], dim=1)
    target = (b * x_query + tau * a_query + c * (x_query / 1.25)).squeeze(1)
    return context, valid, query, target


class ContextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 96
        self.row = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.query = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=192, dropout=0.05, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, context: Tensor, valid: Tensor, query: Tensor) -> Tensor:
        row_tokens = self.row(context)
        query_token = self.query(query)[:, None]
        tokens = torch.cat([row_tokens, query_token], dim=1)
        mask = torch.cat([~valid, torch.zeros((len(valid), 1), dtype=torch.bool, device=valid.device)], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        return self.head(encoded[:, -1]).squeeze(-1)


def train_model(seed: int, causal: bool) -> tuple[ContextTransformer, float]:
    torch.manual_seed(seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + (17 if causal else 23))
    model = ContextTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    started = time.perf_counter()
    model.train()
    for _ in range(5_000):
        context, valid, query, target = sample_training_batch(128, causal, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(context, valid, query)
            loss = nn.functional.mse_loss(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if time.perf_counter() - started > 14 * 60:
            raise TimeoutError("one interventional model exceeded half-budget")
    return model, time.perf_counter() - started


def make_test_environments(seed: int, k: int, environments: int = 256, queries: int = 32) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = rng.uniform(
        low=np.array([-1.5, -2.0, -1.5, -1.5, -1.5]),
        high=np.array([1.5, 2.0, 1.5, 1.5, 1.5]),
        size=(environments, 5),
    ).astype(np.float32)
    b, tau, c, alpha, delta = [params[:, i : i + 1] for i in range(5)]
    u_obs = rng.normal(size=(environments, N_OBS)).astype(np.float32)
    x_obs = u_obs + 0.5 * rng.normal(size=u_obs.shape).astype(np.float32)
    a_obs = alpha * x_obs + delta * u_obs + 0.5 * rng.normal(size=u_obs.shape).astype(np.float32)
    y_obs = b * x_obs + tau * a_obs + c * u_obs + 0.3 * rng.normal(size=u_obs.shape).astype(np.float32)
    obs = np.stack([x_obs, a_obs, y_obs, np.zeros_like(x_obs)], axis=-1)
    u_int = rng.normal(size=(environments, MAX_INT)).astype(np.float32)
    x_int = u_int + 0.5 * rng.normal(size=u_int.shape).astype(np.float32)
    a_int = 1.5 * rng.normal(size=u_int.shape).astype(np.float32)
    y_int = b * x_int + tau * a_int + c * u_int + 0.3 * rng.normal(size=u_int.shape).astype(np.float32)
    intervention = np.stack([x_int, a_int, y_int, np.ones_like(x_int)], axis=-1)
    context = np.concatenate([obs, intervention], axis=1).astype(np.float32)
    valid = np.zeros((environments, CONTEXT), dtype=bool)
    valid[:, : N_OBS + k] = True
    u_query = rng.normal(size=(environments, queries)).astype(np.float32)
    x_query = u_query + 0.5 * rng.normal(size=u_query.shape).astype(np.float32)
    a_query = 1.5 * rng.normal(size=u_query.shape).astype(np.float32)
    target = b * x_query + tau * a_query + c * (x_query / 1.25)
    return {"context": context, "valid": valid, "x": x_query, "a": a_query, "target": target.astype(np.float32)}


@torch.no_grad()
def meta_predict(model: ContextTransformer, data: dict[str, np.ndarray], ignore_interventions: bool) -> np.ndarray:
    model.eval()
    environments, queries = data["x"].shape
    predictions = np.empty((environments, queries), dtype=np.float32)
    for start in range(0, environments, 32):
        end = min(start + 32, environments)
        context = torch.from_numpy(data["context"][start:end]).to(DEVICE)
        valid_np = data["valid"][start:end].copy()
        if ignore_interventions:
            valid_np[:, N_OBS:] = False
        valid = torch.from_numpy(valid_np).to(DEVICE)
        for query_index in range(queries):
            query = torch.from_numpy(
                np.stack([data["x"][start:end, query_index], data["a"][start:end, query_index]], axis=1)
            ).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions[start:end, query_index] = model(context, valid, query).float().cpu().numpy()
    return predictions


def ridge_predictions(data: dict[str, np.ndarray], balanced: bool) -> np.ndarray:
    environments, queries = data["x"].shape
    output = np.empty((environments, queries), dtype=np.float32)
    for environment in range(environments):
        valid = data["valid"][environment]
        rows = data["context"][environment, valid]
        design = np.stack([np.ones(len(rows)), rows[:, 0], rows[:, 1]], axis=1)
        target = rows[:, 2]
        weights = np.ones(len(rows), dtype=np.float64)
        randomized = rows[:, 3] > 0.5
        if balanced and randomized.any():
            weights[~randomized] = 0.5 / (~randomized).sum()
            weights[randomized] = 0.5 / randomized.sum()
        root = np.sqrt(weights)[:, None]
        gram = (design * root).T @ (design * root) + 0.1 * np.eye(3)
        coefficient = np.linalg.solve(gram, (design * root).T @ (target * root[:, 0]))
        query_design = np.stack(
            [np.ones(queries), data["x"][environment], data["a"][environment]], axis=1
        )
        output[environment] = query_design @ coefficient
    return output


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    environment_rmse = np.sqrt(np.mean(np.square(prediction - target), axis=1))
    return float(np.sqrt(np.mean(np.square(prediction - target)))), environment_rmse


def main() -> None:
    overall_started = time.perf_counter()
    rows = []
    environment_rows = []
    for seed in SEEDS:
        causal, causal_seconds = train_model(seed, causal=True)
        observational, observational_seconds = train_model(seed, causal=False)
        torch.save(causal.state_dict(), OUT / f"causalpfn__seed-{seed}.pt")
        torch.save(observational.state_dict(), OUT / f"obspfn__seed-{seed}.pt")
        for k in K_VALUES:
            data = make_test_environments(seed + 10_000 + k, k)
            predictions = {
                "causalpfn": meta_predict(causal, data, ignore_interventions=False),
                "obspfn": meta_predict(observational, data, ignore_interventions=True),
                "ridge_naive": ridge_predictions(data, balanced=False),
                "ridge_balanced": ridge_predictions(data, balanced=True),
            }
            env_metrics = {}
            for model_name, prediction in predictions.items():
                rmse, environment_rmse = metrics(prediction, data["target"])
                env_metrics[model_name] = environment_rmse
                rows.append({
                    "seed": seed,
                    "k": k,
                    "model": model_name,
                    "rmse": rmse,
                    "train_seconds": causal_seconds if model_name == "causalpfn" else (
                        observational_seconds if model_name == "obspfn" else 0.0
                    ),
                })
            best_ridge = np.minimum(env_metrics["ridge_naive"], env_metrics["ridge_balanced"])
            causal_env = env_metrics["causalpfn"]
            for environment, (causal_error, obs_error, ridge_error) in enumerate(
                zip(causal_env, env_metrics["obspfn"], best_ridge)
            ):
                environment_rows.append({
                    "seed": seed,
                    "k": k,
                    "environment": environment,
                    "causalpfn_rmse": causal_error,
                    "obspfn_rmse": obs_error,
                    "best_ridge_rmse": ridge_error,
                    "beats_both": causal_error < obs_error and causal_error < ridge_error,
                })
        if time.perf_counter() - overall_started > 30 * 60:
            raise TimeoutError("interventional pilot exceeded 30 minutes")
    frame = pd.DataFrame(rows)
    environments = pd.DataFrame(environment_rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    environments.to_csv(OUT / "environment_metrics.csv", index=False)
    summary = frame.groupby(["k", "model"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    gates = {}
    passed = True
    for k in (4, 8):
        group = summary[summary.k == k].set_index("model")
        causal_rmse = float(group.loc["causalpfn", "rmse"])
        obs_reduction = 1.0 - causal_rmse / float(group.loc["obspfn", "rmse"])
        ridge_rmse = min(float(group.loc["ridge_naive", "rmse"]), float(group.loc["ridge_balanced", "rmse"]))
        ridge_reduction = 1.0 - causal_rmse / ridge_rmse
        environment_fraction = float(environments[environments.k == k].beats_both.mean())
        gate_pass = obs_reduction >= 0.20 and ridge_reduction >= 0.20 and environment_fraction >= 0.80
        gates[str(k)] = {
            "causalpfn_rmse": causal_rmse,
            "reduction_vs_obspfn": obs_reduction,
            "reduction_vs_best_ridge": ridge_reduction,
            "environment_win_fraction": environment_fraction,
            "passed": gate_pass,
        }
        passed = passed and gate_pass
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - overall_started,
        "seeds": len(SEEDS),
        "gates": gates,
        "passed": bool(passed),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
