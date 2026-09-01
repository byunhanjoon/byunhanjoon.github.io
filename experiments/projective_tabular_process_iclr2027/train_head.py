#!/usr/bin/env python3
"""Development-only calibration, HPO, and training of the projective covariance head."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import CACHE, CONFIG, atomic_json, gaussian_scores


PRIMARY = [CONFIG["query_families"].index(name) for name in CONFIG["primary_aggregate_families"]]
CONTEXT_TO_INDEX = {int(n): i for i, n in enumerate(CONFIG["context_sizes"])}


@dataclass
class Episodes:
    hidden: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    target: torch.Tensor
    coefficients: torch.Tensor
    query_numeric: list[np.ndarray]
    datasets: list[str]
    context_sizes: torch.Tensor
    paths: list[str]

    def subset(self, indices: np.ndarray) -> "Episodes":
        index = torch.as_tensor(indices, device=self.hidden.device, dtype=torch.long)
        return Episodes(
            self.hidden[index],
            self.mean[index],
            self.variance[index],
            self.target[index],
            self.coefficients[index],
            [self.query_numeric[i] for i in indices],
            [self.datasets[i] for i in indices],
            self.context_sizes[index],
            [self.paths[i] for i in indices],
        )


class ProjectiveHead(nn.Module):
    def __init__(self, input_dim: int, rank: int, hidden_dim: int):
        super().__init__()
        if hidden_dim:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, rank, bias=False),
            )
        else:
            self.projector = nn.Linear(input_dim, rank, bias=False)
        self.rho_logits = nn.Parameter(torch.full((len(CONTEXT_TO_INDEX),), -2.0))
        self.rank = rank
        self.hidden_dim = hidden_dim

    def features(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        return F.normalize(self.projector(hidden), dim=-1, eps=1e-6)

    def rhos(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_logits) * 0.999


def parse_metadata(value: np.ndarray) -> dict[str, Any]:
    return json.loads(str(value.item()))


def load_episodes(stage: str, device: torch.device, query_mode: str = "batched") -> Episodes:
    episode_root = "tabicl_singleton_episodes" if query_mode == "singleton" else "tabicl_episodes"
    root = CACHE / episode_root / stage
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards under {root}")
    hidden, mean, variance, target, coefficients = [], [], [], [], []
    query_numeric: list[np.ndarray] = []
    datasets, context_sizes = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = parse_metadata(data["metadata"])
            hidden.append(data["hidden"].astype(np.float32))
            mean.append(data["mean"].astype(np.float32))
            variance.append(data["variance"].astype(np.float32))
            target.append(data["target"].astype(np.float32))
            coefficients.append(data["coefficients"].astype(np.float32)[PRIMARY])
            query_numeric.append(data["query_numeric"].astype(np.float32))
            datasets.append(meta["dataset"])
            context_sizes.append(CONTEXT_TO_INDEX[int(meta["context_size"])])
    expected = (
        len(CONFIG["development_datasets"])
        * len(CONFIG["development_splits"])
        * int(CONFIG["development_context_replicates"])
        * len(CONFIG["context_sizes"])
        if stage == "dev"
        else None
    )
    if expected is not None and len(paths) != expected:
        raise RuntimeError(f"development cache incomplete: expected {expected}, found {len(paths)}")
    return Episodes(
        hidden=torch.from_numpy(np.stack(hidden)).to(device),
        mean=torch.from_numpy(np.stack(mean)).to(device),
        variance=torch.from_numpy(np.stack(variance)).to(device),
        target=torch.from_numpy(np.stack(target)).to(device),
        coefficients=torch.from_numpy(np.stack(coefficients)).to(device),
        query_numeric=query_numeric,
        datasets=datasets,
        context_sizes=torch.as_tensor(context_sizes, device=device, dtype=torch.long),
        paths=[str(path) for path in paths],
    )


def functional_values(data: Episodes, temperatures: torch.Tensor) -> tuple[torch.Tensor, ...]:
    coeff = data.coefficients
    target = torch.einsum("efgn,en->efg", coeff, data.target)
    mean = torch.einsum("efgn,en->efg", coeff, data.mean)
    calibrated_variance = data.variance * temperatures[data.context_sizes, None]
    diagonal = torch.einsum("efgn,en->efg", coeff.square(), calibrated_variance)
    return target, mean, calibrated_variance, diagonal


def head_variance(
    model: ProjectiveHead,
    data: Episodes,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    _, _, calibrated_variance, diagonal = functional_values(data, temperatures)
    unit = model.features(data.hidden)
    weighted = torch.einsum(
        "efgn,en,esnr->efgsr",
        data.coefficients,
        calibrated_variance.clamp_min(1e-10).sqrt(),
        unit,
    )
    kernel = weighted.square().sum(-1).mean(-1)
    rho = model.rhos()[data.context_sizes, None, None]
    return ((1.0 - rho) * diagonal + rho * kernel).clamp(1e-7, 1e5)


def nll_loss(model: ProjectiveHead, data: Episodes, temperatures: torch.Tensor) -> torch.Tensor:
    target, mean, _, _ = functional_values(data, temperatures)
    variance = head_variance(model, data, temperatures)
    return 0.5 * (torch.log(2.0 * math.pi * variance) + (target - mean).square() / variance).mean()


@torch.no_grad()
def evaluate_nll(model: ProjectiveHead, data: Episodes, temperatures: torch.Tensor) -> float:
    model.eval()
    return float(nll_loss(model, data, temperatures).cpu())


def select_temperatures(data: Episodes) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select marginal scales from point queries only, before fitting correlation."""
    # Reload point coefficients because data.coefficients contains aggregate families only.
    records = []
    selected = []
    paths = [Path(path) for path in data.paths]
    point_coefficients = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            point_coefficients.append(shard["coefficients"].astype(np.float64)[0])
    point = np.stack(point_coefficients)
    mean = data.mean.detach().cpu().numpy()
    target = data.target.detach().cpu().numpy()
    variance = data.variance.detach().cpu().numpy()
    context_indices = data.context_sizes.detach().cpu().numpy()
    datasets = np.asarray(data.datasets)
    grid = np.asarray(CONFIG["marginal_temperature_grid"], dtype=np.float64)
    for context_index, context_size in enumerate(CONFIG["context_sizes"]):
        mask = context_indices == context_index
        truth = np.einsum("egn,en->eg", point[mask], target[mask])
        prediction = np.einsum("egn,en->eg", point[mask], mean[mask])
        base_variance = np.einsum("egn,en->eg", point[mask] ** 2, variance[mask])
        scores = []
        for temperature in grid:
            values = gaussian_scores(truth, prediction, temperature * base_variance)["nll"]
            # Equal dataset weight, even if a future cache has unequal episode counts.
            dataset_means = [float(values[datasets[mask] == name].mean()) for name in sorted(set(datasets[mask]))]
            score = float(np.mean(dataset_means))
            records.append({"context_size": context_size, "temperature": temperature, "point_nll": score})
            scores.append(score)
        selected.append(float(grid[int(np.argmin(scores))]))
    return np.asarray(selected, dtype=np.float32), records


def train_one(
    train: Episodes,
    validation: Episodes,
    temperatures: torch.Tensor,
    config: dict[str, Any],
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[ProjectiveHead, int, float, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ProjectiveHead(train.hidden.shape[-1], int(config["rank"]), int(config["hidden_dim"])).to(train.hidden.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    max_epochs = int(fixed_epochs or CONFIG["head_hpo"]["max_epochs"])
    patience = int(CONFIG["head_hpo"]["patience"])
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nll_loss(model, train, temperatures)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        score = evaluate_nll(model, validation, temperatures)
        history.append(score)
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if fixed_epochs is None and epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, history


def hpo(data: Episodes, temperatures: torch.Tensor) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    grid = []
    for rank, hidden_dim, weight_decay, learning_rate in itertools.product(
        CONFIG["head_hpo"]["ranks"],
        CONFIG["head_hpo"]["hidden_dims"],
        CONFIG["head_hpo"]["weight_decays"],
        CONFIG["head_hpo"]["learning_rates"],
    ):
        grid.append(
            {
                "rank": int(rank),
                "hidden_dim": int(hidden_dim),
                "weight_decay": float(weight_decay),
                "learning_rate": float(learning_rate),
            }
        )
    datasets = sorted(set(data.datasets))
    records: list[dict[str, Any]] = []
    hpo_seed = int(CONFIG["head_hpo"]["seeds"][0])
    for config_index, candidate in enumerate(grid):
        start = time.perf_counter()
        for fold, held_out in enumerate(datasets):
            train_indices = np.flatnonzero(np.asarray(data.datasets) != held_out)
            validation_indices = np.flatnonzero(np.asarray(data.datasets) == held_out)
            _, best_epoch, best_nll, history = train_one(
                data.subset(train_indices),
                data.subset(validation_indices),
                temperatures,
                candidate,
                hpo_seed + fold,
            )
            records.append(
                {
                    "config_index": config_index,
                    **candidate,
                    "held_out_dataset": held_out,
                    "best_epoch": best_epoch,
                    "validation_nll": best_nll,
                    "epochs_run": len(history),
                }
            )
        mean_score = float(np.mean([r["validation_nll"] for r in records if r["config_index"] == config_index]))
        print(f"config {config_index + 1}/{len(grid)} {candidate} cv_nll={mean_score:.6f} time={time.perf_counter()-start:.1f}s", flush=True)

    summaries = []
    for index, candidate in enumerate(grid):
        subset = [record for record in records if record["config_index"] == index]
        summaries.append(
            (
                float(np.mean([record["validation_nll"] for record in subset])),
                int(candidate["hidden_dim"] > 0),
                int(candidate["rank"]),
                index,
                candidate,
                int(np.median([record["best_epoch"] for record in subset])),
            )
        )
    _, _, _, _, best_config, final_epochs = min(summaries)
    return best_config, records, max(final_epochs, 1)


def kernel_variance(unit: np.ndarray, coefficients: np.ndarray, variance: np.ndarray) -> np.ndarray:
    weighted = np.einsum("fgn,n,snr->fgsr", coefficients, np.sqrt(np.maximum(variance, 1e-10)), unit)
    return np.mean(np.sum(weighted**2, axis=-1), axis=-1)


def select_nonparametric_kernels(data: Episodes, temperatures: np.ndarray) -> dict[str, Any]:
    """Tune explicitly labeled untrained hidden and raw-feature kernel baselines."""
    rho_grid = np.linspace(0.0, 0.9, 19)
    length_grid = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    arrays = {
        "hidden_cosine": {int(n): [] for n in CONFIG["context_sizes"]},
        "raw_rbf": {int(n): [] for n in CONFIG["context_sizes"]},
    }
    hidden = data.hidden.detach().cpu().numpy().astype(np.float64)
    means = data.mean.detach().cpu().numpy().astype(np.float64)
    targets = data.target.detach().cpu().numpy().astype(np.float64)
    variances = data.variance.detach().cpu().numpy().astype(np.float64)
    coeffs = data.coefficients.detach().cpu().numpy().astype(np.float64)
    context_indices = data.context_sizes.detach().cpu().numpy()
    for episode in range(len(data.datasets)):
        ci = int(context_indices[episode])
        n = int(CONFIG["context_sizes"][ci])
        variance = variances[episode] * temperatures[ci]
        diagonal = np.einsum("fgn,n->fg", coeffs[episode] ** 2, variance)
        truth = np.einsum("fgn,n->fg", coeffs[episode], targets[episode])
        mean = np.einsum("fgn,n->fg", coeffs[episode], means[episode])
        h = hidden[episode]
        h = (h - h.mean(axis=-1, keepdims=True)) / np.maximum(h.std(axis=-1, keepdims=True), 1e-8)
        unit = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-8)
        hidden_kernel = kernel_variance(unit, coeffs[episode], variance)
        x = data.query_numeric[episode].astype(np.float64)
        distance2 = np.maximum(
            np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T,
            0.0,
        )
        raw_kernels = {}
        for length in length_grid:
            scale = length * math.sqrt(max(x.shape[1], 1))
            K = np.exp(-0.5 * distance2 / max(scale**2, 1e-12))
            raw_kernels[float(length)] = np.einsum(
                "fgn,nm,fgm,n,m->fg",
                coeffs[episode],
                K,
                coeffs[episode],
                np.sqrt(variance),
                np.sqrt(variance),
            )
        arrays["hidden_cosine"][n].append((truth, mean, diagonal, hidden_kernel, data.datasets[episode]))
        arrays["raw_rbf"][n].append((truth, mean, diagonal, raw_kernels, data.datasets[episode]))

    selected: dict[str, Any] = {}
    for method in arrays:
        selected[method] = {}
        for n in CONFIG["context_sizes"]:
            best = None
            lengths = [None] if method == "hidden_cosine" else list(length_grid)
            for length in lengths:
                for rho in rho_grid:
                    per_dataset: dict[str, list[float]] = {}
                    for truth, mean, diagonal, kernel_obj, dataset in arrays[method][int(n)]:
                        kernel = kernel_obj if length is None else kernel_obj[float(length)]
                        variance = (1.0 - rho) * diagonal + rho * kernel
                        score = float(gaussian_scores(truth, mean, variance)["nll"].mean())
                        per_dataset.setdefault(dataset, []).append(score)
                    objective = float(np.mean([np.mean(values) for values in per_dataset.values()]))
                    candidate = (objective, float(rho), None if length is None else float(length))
                    if best is None or candidate < best:
                        best = candidate
            assert best is not None
            selected[method][str(n)] = {"development_nll": best[0], "rho": best[1], "length": best[2]}
    return selected


def save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_episodes("dev", device, args.query_mode)
    temperatures_np, temperature_records = select_temperatures(data)
    temperatures = torch.as_tensor(temperatures_np, device=device)
    print("selected marginal temperatures", dict(zip(CONFIG["context_sizes"], temperatures_np.tolist())), flush=True)

    best_config, hpo_records, final_epochs = hpo(data, temperatures)
    print("selected head", best_config, "epochs", final_epochs, flush=True)
    artifact_dir = CACHE / ("head_singleton" if args.query_mode == "singleton" else "head")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_csv(artifact_dir / "head_hpo.csv", hpo_records)
    save_csv(artifact_dir / "marginal_temperature_hpo.csv", temperature_records)

    checkpoints = []
    for seed in CONFIG["head_hpo"]["seeds"]:
        model, _, train_nll, history = train_one(
            data,
            data,
            temperatures,
            best_config,
            int(seed),
            fixed_epochs=final_epochs,
        )
        path = artifact_dir / f"projective_head_seed{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": best_config,
                "seed": int(seed),
                "epochs": final_epochs,
                "development_nll": train_nll,
                "temperatures": temperatures_np,
                "input_dim": int(data.hidden.shape[-1]),
            },
            path,
        )
        checkpoints.append(str(path))
        print(f"saved {path} development_nll={train_nll:.6f} rhos={model.rhos().detach().cpu().numpy()}", flush=True)

    nonparametric = select_nonparametric_kernels(data, temperatures_np)
    summary = {
        "development_episode_count": len(data.datasets),
        "development_datasets": sorted(set(data.datasets)),
        "marginal_temperatures": dict(zip(map(str, CONFIG["context_sizes"]), map(float, temperatures_np))),
        "best_head_config": best_config,
        "final_epochs": final_epochs,
        "checkpoints": checkpoints,
        "nonparametric_kernels": nonparametric,
        "query_mode": args.query_mode,
    }
    atomic_json(artifact_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["batched", "singleton"], default="batched")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
