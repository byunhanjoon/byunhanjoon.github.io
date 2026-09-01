#!/usr/bin/env python3
"""Small learned PFN for the latent useful/irrelevant geometry prior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from sklearn.metrics import roc_auc_score
from torch import nn


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "learned_pfn"
N_STATES = 32
N_CONTEXT = 20
SCALES = (0.3, 1.0, 3.0)
NOISES = (0.1, 0.3, 1.0)
PRIORS = (0.1, 0.5, 0.9)
SEEDS = (20260831, 20260832, 20260833)


def heat_covariance(scale: float) -> np.ndarray:
    adjacency = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for index in range(N_STATES):
        adjacency[index, (index - 1) % N_STATES] = 1.0
        adjacency[index, (index + 1) % N_STATES] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    kernel = expm(-scale * laplacian)
    diagonal = np.sqrt(np.diag(kernel))
    return kernel / diagonal[:, None] / diagonal[None, :]


def fourier_coordinates() -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(N_STATES) / N_STATES
    columns = []
    for frequency in range(1, 5):
        columns.extend([np.sin(frequency * angle), np.cos(frequency * angle)])
    return np.column_stack(columns).astype(np.float32)


class TaskGenerator:
    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        kernels = np.stack([heat_covariance(scale) for scale in SCALES])
        self.kernels = torch.as_tensor(kernels, dtype=torch.float32, device=device)
        self.cholesky = torch.linalg.cholesky(self.kernels)
        self.coordinates = torch.as_tensor(
            fourier_coordinates(), dtype=torch.float32, device=device
        )

    def sample(
        self,
        batch: int,
        prior: float = 0.5,
        scale_index: int | None = None,
        noise_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if scale_index is None:
            scales = torch.randint(
                len(SCALES), (batch,), generator=self.generator, device=self.device
            )
        else:
            scales = torch.full((batch,), scale_index, device=self.device, dtype=torch.long)
        if noise_index is None:
            noises = torch.randint(
                len(NOISES), (batch,), generator=self.generator, device=self.device
            )
        else:
            noises = torch.full((batch,), noise_index, device=self.device, dtype=torch.long)
        smooth = torch.rand(
            batch, generator=self.generator, device=self.device
        ) < prior
        latent = torch.randn(
            batch, N_STATES, generator=self.generator, device=self.device
        )
        smooth_effect = torch.bmm(
            self.cholesky.index_select(0, scales), latent.unsqueeze(-1)
        ).squeeze(-1)
        effect = torch.where(smooth[:, None], smooth_effect, latent)
        noise = torch.as_tensor(NOISES, device=self.device).index_select(0, noises)
        observed = effect + noise[:, None] * torch.randn(
            batch, N_STATES, generator=self.generator, device=self.device
        )
        ranking = torch.rand(
            batch, N_STATES, generator=self.generator, device=self.device
        ).argsort(dim=1)
        context_index = ranking[:, :N_CONTEXT]
        mask = torch.zeros(batch, N_STATES, device=self.device, dtype=torch.bool)
        mask.scatter_(1, context_index, True)
        scale_values = torch.as_tensor(SCALES, device=self.device).index_select(0, scales)
        return {
            "effect": effect,
            "observed": observed,
            "mask": mask,
            "smooth": smooth,
            "scale_index": scales,
            "noise_index": noises,
            "scale": scale_values,
            "noise": noise,
            "kernel": self.kernels.index_select(0, scales),
        }

    def features(self, task: dict[str, torch.Tensor], structured: bool) -> torch.Tensor:
        mask = task["mask"].float()
        columns = [
            task["observed"] * mask,
            mask,
            1.0 - mask,
            torch.log(task["scale"])[:, None].expand(-1, N_STATES),
            torch.log(task["noise"])[:, None].expand(-1, N_STATES),
        ]
        features = torch.stack(columns, dim=-1)
        if structured:
            coordinates = self.coordinates[None].expand(len(mask), -1, -1)
            features = torch.cat([features, coordinates], dim=-1)
        return features


class StructuredPFN(nn.Module):
    def __init__(self, structured: bool):
        super().__init__()
        input_size = 13 if structured else 5
        self.input = nn.Sequential(nn.Linear(input_size, 64), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(self.input(features))).squeeze(-1)


def analytic_predictions(task: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = len(task["effect"])
    context = task["mask"].float().argsort(dim=1, descending=True)[:, :N_CONTEXT]
    query = task["mask"].float().argsort(dim=1)[:, : N_STATES - N_CONTEXT]
    batch_index = torch.arange(batch, device=task["effect"].device)[:, None, None]
    kernel_tt = task["kernel"][
        batch_index,
        context[:, :, None],
        context[:, None, :],
    ]
    kernel_qt = task["kernel"][
        batch_index,
        query[:, :, None],
        context[:, None, :],
    ]
    observed_context = task["observed"].gather(1, context)
    eye = torch.eye(N_CONTEXT, device=task["effect"].device)[None]
    smooth_covariance = kernel_tt + task["noise"][:, None, None] ** 2 * eye
    solved = torch.linalg.solve(smooth_covariance, observed_context.unsqueeze(-1))
    smooth_query = torch.bmm(kernel_qt, solved).squeeze(-1)

    sign, logdet = torch.linalg.slogdet(smooth_covariance)
    if not bool((sign > 0).all()):
        raise RuntimeError("non-positive smooth covariance")
    quadratic = torch.bmm(
        observed_context[:, None],
        torch.linalg.solve(smooth_covariance, observed_context.unsqueeze(-1)),
    ).flatten()
    smooth_log = -0.5 * (N_CONTEXT * math.log(2.0 * math.pi) + logdet + quadratic)
    random_variance = 1.0 + task["noise"] ** 2
    random_log = -0.5 * (
        N_CONTEXT * math.log(2.0 * math.pi)
        + N_CONTEXT * torch.log(random_variance)
        + (observed_context**2).sum(dim=1) / random_variance
    )
    posterior = torch.sigmoid(smooth_log - random_log)

    zero = torch.zeros_like(task["effect"])
    smooth = torch.zeros_like(task["effect"]).scatter(1, query, smooth_query)
    mixture = torch.zeros_like(task["effect"]).scatter(
        1, query, posterior[:, None] * smooth_query
    )
    hard = torch.zeros_like(task["effect"]).scatter(
        1, query, (posterior > 0.5)[:, None] * smooth_query
    )
    oracle = torch.zeros_like(task["effect"]).scatter(
        1, query, task["smooth"][:, None] * smooth_query
    )
    return {
        "zero": zero,
        "always_smooth": smooth,
        "hard_route": hard,
        "bayes_mixture": mixture,
        "regime_oracle": oracle,
        "posterior": posterior,
        "smooth_query": smooth_query,
        "query_index": query,
    }


def query_mse(
    prediction: torch.Tensor, effect: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - effect) ** 2)[~mask].reshape(len(effect), -1).mean(dim=1)


def train_model(
    variant: str, seed: int, device: torch.device, steps: int, batch_size: int,
) -> tuple[StructuredPFN, list[dict]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = TaskGenerator(device, seed + 1000)
    model = StructuredPFN(variant == "structured").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    trace = []
    for step in range(1, steps + 1):
        task = generator.sample(batch_size)
        prediction = model(generator.features(task, variant == "structured"))
        loss = query_mse(prediction, task["effect"], task["mask"]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {"step": step, "loss": float(loss.detach())}
            trace.append(row)
            print(variant, seed, step, round(row["loss"], 5), flush=True)
    return model, trace


@torch.inference_mode()
def evaluate_cell(
    model: StructuredPFN,
    variant: str,
    generator: TaskGenerator,
    prior: float,
    scale_index: int,
    noise_index: int,
    tasks: int,
    batch_size: int,
) -> dict:
    losses = {name: [] for name in (
        "model", "zero", "always_smooth", "hard_route", "bayes_mixture", "regime_oracle"
    )}
    trust_values, posterior_values, regimes = [], [], []
    for start in range(0, tasks, batch_size):
        batch = min(batch_size, tasks - start)
        task = generator.sample(batch, prior, scale_index, noise_index)
        prediction = model(generator.features(task, variant == "structured"))
        analytic = analytic_predictions(task)
        losses["model"].append(query_mse(prediction, task["effect"], task["mask"]).cpu())
        for name in losses:
            if name == "model":
                continue
            losses[name].append(
                query_mse(analytic[name], task["effect"], task["mask"]).cpu()
            )
        query = analytic["query_index"]
        model_query = prediction.gather(1, query)
        smooth_query = analytic["smooth_query"]
        trust = (model_query * smooth_query).sum(dim=1) / (
            (smooth_query**2).sum(dim=1) + 1e-8
        )
        trust_values.append(trust.clamp(0.0, 1.0).cpu())
        posterior_values.append(analytic["posterior"].cpu())
        regimes.append(task["smooth"].cpu())

    row = {
        "variant": variant,
        "true_prior": prior,
        "scale": SCALES[scale_index],
        "noise": NOISES[noise_index],
        "tasks": tasks,
    }
    for name, chunks in losses.items():
        values = torch.cat(chunks).numpy()
        row[f"mse_{name}"] = float(values.mean())
        row[f"se_{name}"] = float(values.std(ddof=1) / np.sqrt(len(values)))
    trust = torch.cat(trust_values).numpy()
    posterior = torch.cat(posterior_values).numpy()
    regime = torch.cat(regimes).numpy().astype(int)
    row["trust_posterior_correlation"] = float(np.corrcoef(trust, posterior)[0, 1])
    row["trust_regime_auroc"] = float(roc_auc_score(regime, trust))
    row["trust_mean"] = float(trust.mean())
    row["posterior_mean"] = float(posterior.mean())
    return row


def run(
    variant: str, seed: int, device: torch.device, steps: int,
    batch_size: int, evaluation_tasks: int,
) -> None:
    folder = OUT / variant / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model, trace = train_model(variant, seed, device, steps, batch_size)
    pd.DataFrame(trace).to_csv(folder / "training.csv", index=False)
    torch.save(model.state_dict(), folder / "model.pt")
    generator = TaskGenerator(device, seed + 2000)
    model.eval()
    rows = []
    for prior in PRIORS:
        for scale_index in range(len(SCALES)):
            for noise_index in range(len(NOISES)):
                row = evaluate_cell(
                    model, variant, generator, prior, scale_index, noise_index,
                    evaluation_tasks, batch_size,
                )
                row["seed"] = seed
                rows.append(row)
                print(variant, seed, prior, SCALES[scale_index], NOISES[noise_index], flush=True)
    pd.DataFrame(rows).to_csv(folder / "cells.csv", index=False)


def analyze() -> dict:
    paths = [
        OUT / variant / f"seed{seed}" / "cells.csv"
        for variant in ("structured", "set") for seed in SEEDS
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing learned PFN cells: {missing}")
    cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    matched = cells[cells.true_prior == 0.5]
    phase = matched.groupby(["variant", "scale", "noise"], as_index=False).mean(numeric_only=True)
    structured = phase[phase.variant == "structured"].set_index(["scale", "noise"])
    set_model = phase[phase.variant == "set"].set_index(["scale", "noise"])
    difference = set_model.mse_model - structured.mse_model
    regret = structured.mse_model - structured.mse_bayes_mixture
    beats_zero = structured.mse_model < structured.mse_zero
    beats_smooth = structured.mse_model < structured.mse_always_smooth
    seed_advantage = matched.pivot_table(
        index=["seed", "scale", "noise"], columns="variant", values="mse_model"
    )
    seed_mean = (seed_advantage["set"] - seed_advantage["structured"]).groupby("seed").mean()
    summary = {
        "status": "complete",
        "matched_prior": {
            "structured_wins_over_set": int((difference > 0).sum()),
            "phase_cells": int(len(difference)),
            "mean_advantage_over_set": float(difference.mean()),
            "mean_regret_to_bayes": float(regret.mean()),
            "beats_zero_cells": int(beats_zero.sum()),
            "beats_always_smooth_cells": int(beats_smooth.sum()),
            "trust_posterior_correlation": float(structured.trust_posterior_correlation.mean()),
            "trust_regime_auroc": float(structured.trust_regime_auroc.mean()),
            "positive_seed_advantages": int((seed_mean > 0).sum()),
            "seed_advantage": {str(key): float(value) for key, value in seed_mean.items()},
        },
        "prior_shift": {},
        "integrity": bool(np.isfinite(cells.select_dtypes(include=[np.number])).all().all()),
    }
    for prior in (0.1, 0.9):
        frame = cells[(cells.variant == "structured") & (cells.true_prior == prior)]
        summary["prior_shift"][str(prior)] = {
            "mean_model_mse": float(frame.mse_model.mean()),
            "mean_bayes_mse": float(frame.mse_bayes_mixture.mean()),
            "mean_regret_to_bayes": float((frame.mse_model - frame.mse_bayes_mixture).mean()),
            "mean_trust": float(frame.trust_mean.mean()),
        }
    matched_summary = summary["matched_prior"]
    summary["gates"] = {
        "wins_at_least_7_of_9": matched_summary["structured_wins_over_set"] >= 7,
        "mean_advantage_at_least_0p02": matched_summary["mean_advantage_over_set"] >= 0.02,
        "regret_at_most_0p05": matched_summary["mean_regret_to_bayes"] <= 0.05,
        "trust_correlation_at_least_0p70": matched_summary["trust_posterior_correlation"] >= 0.70,
        "trust_auroc_at_least_0p75": matched_summary["trust_regime_auroc"] >= 0.75,
        "beats_fixed_rules_at_least_7_of_9": (
            matched_summary["beats_zero_cells"] >= 7
            and matched_summary["beats_always_smooth_cells"] >= 7
        ),
        "positive_in_at_least_two_seeds": matched_summary["positive_seed_advantages"] >= 2,
        "integrity": summary["integrity"],
    }
    summary["passes"] = all(summary["gates"].values())
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["structured", "set"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-tasks", type=int, default=4096)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    if args.variant is None or args.seed is None:
        parser.error("--variant and --seed are required unless --analyze is used")
    run(
        args.variant, args.seed, torch.device(args.device), args.steps,
        args.batch_size, args.evaluation_tasks,
    )


if __name__ == "__main__":
    main()
