#!/usr/bin/env python3
"""Cross-fitted, context-only calibration of projective correlation strength."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CACHE, CONFIG, gaussian_scores, load_openml_task, load_spec, native_frame, slug, stable_seed
from extract_tabicl import TabICLExtractor
from score_projective import load_models


RHO_GRID = np.r_[np.linspace(0.0, 0.9, 19), 0.99]


def local_coefficients(seed: int, q: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = []
    k = int(rng.integers(2, q))
    subset = rng.choice(q, size=k, replace=False)
    a = np.zeros(q); a[subset] = 1.0 / k; vectors.append(a)
    a = np.zeros(q); a[subset] = 1.0 / math.sqrt(k); vectors.append(a)
    i, j = rng.choice(q, size=2, replace=False)
    a = np.zeros(q); a[i] = 1 / math.sqrt(2); a[j] = -1 / math.sqrt(2); vectors.append(a)
    perm = rng.permutation(q); a = np.zeros(q); a[perm[: q // 2]] = 1 / math.sqrt(q); a[perm[q // 2 :]] = -1 / math.sqrt(q); vectors.append(a)
    a = rng.normal(size=q); a /= np.linalg.norm(a); vectors.append(a)
    a = np.abs(rng.normal(size=q)); a /= np.linalg.norm(a); vectors.append(a)
    return np.stack(vectors)


@torch.no_grad()
def average_kernel(hidden: np.ndarray, models, device: torch.device) -> np.ndarray:
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    kernels = []
    for model in models:
        unit = model.features(h).cpu().numpy().astype(np.float64)
        kernels.append(np.einsum("snr,smr->nm", unit, unit) / len(unit))
    return np.mean(kernels, axis=0)


def calibration_curve(
    extractor: TabICLExtractor,
    models,
    device: torch.device,
    dataset,
    context_indices: np.ndarray,
    context_size: int,
    metric_mean: float,
    metric_scale: float,
    temperature: float,
    seed: int,
) -> list[dict[str, float]]:
    folds = 2 if context_size == 16 else 4
    permutation = np.random.default_rng(seed).permutation(context_indices)
    heldout_folds = np.array_split(permutation, folds)
    score_lists = {float(rho): {"nll": [], "crps": []} for rho in RHO_GRID}
    for fold_index, heldout in enumerate(heldout_folds):
        train = np.setdiff1d(context_indices, heldout, assume_unique=True)
        X_train = native_frame(dataset.X, train)
        X_heldout = native_frame(dataset.X, heldout)
        mean_native, variance_native, hidden, _ = extractor.fit_predict(
            X_train, dataset.y[train], X_heldout
        )
        mean = (mean_native - metric_mean) / metric_scale
        variance = np.maximum(temperature * variance_native / metric_scale**2, 1e-10)
        target = (dataset.y[heldout] - metric_mean) / metric_scale
        kernel = average_kernel(hidden, models, device)
        sd = np.sqrt(variance)
        for block_index, start in enumerate(range(0, len(heldout) - 7, 8)):
            block = slice(start, start + 8)
            a = local_coefficients(stable_seed(seed, fold_index, block_index))
            truth = a @ target[block]
            prediction = a @ mean[block]
            diagonal = np.einsum("fn,n->f", a**2, variance[block])
            kernel_variance = np.einsum(
                "fn,nm,fm,n,m->f", a, kernel[block, block], a, sd[block], sd[block]
            )
            for rho in RHO_GRID:
                functional_variance = (1.0 - rho) * diagonal + rho * kernel_variance
                scores = gaussian_scores(truth, prediction, functional_variance)
                score_lists[float(rho)]["nll"].extend(scores["nll"].tolist())
                score_lists[float(rho)]["crps"].extend(scores["crps"].tolist())
    return [
        {
            "rho": float(rho),
            "calibration_nll": float(np.mean(values["nll"])),
            "calibration_crps": float(np.mean(values["crps"])),
            "calibration_functionals": int(len(values["nll"])),
        }
        for rho, values in score_lists.items()
    ]


def load_items(stage: str, shard_index: int, num_shards: int):
    if stage == "eval":
        task_ids = [task for i, task in enumerate(CONFIG["evaluation_tasks"]) if i % num_shards == shard_index]
        return [load_openml_task(task) for task in task_ids]
    specs = CONFIG["development_datasets"] if stage == "dev" else CONFIG["application_datasets"]
    specs = [spec for i, spec in enumerate(specs) if i % num_shards == shard_index]
    return [(load_spec(spec), None) for spec in specs]


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    models, summary = load_models(device)
    extractor = TabICLExtractor(args.device)
    rows = []
    for dataset, _ in load_items(args.stage, args.shard_index, args.num_shards):
        paths = sorted((CACHE / "tabicl_episodes" / args.stage).glob(f"{slug(dataset.name)}__*.npz"))
        print(f"[{args.stage}] adaptive calibration {dataset.name}: {len(paths)} episodes", flush=True)
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as episode:
                meta = json.loads(str(episode["metadata"].item()))
                context_indices = episode["context_indices"].astype(np.int64)
            curve = calibration_curve(
                extractor,
                models,
                device,
                dataset,
                context_indices,
                int(meta["context_size"]),
                float(meta["metric_mean"]),
                float(meta["metric_scale"]),
                float(summary["marginal_temperatures"][str(meta["context_size"])]),
                stable_seed("adaptive-rho", dataset.name, path.name),
            )
            for point in curve:
                rows.append(
                    {
                        "dataset": dataset.name,
                        "episode": path.name,
                        "split": meta["split"],
                        "replicate": int(meta["replicate"]),
                        "context_size": int(meta["context_size"]),
                        **point,
                    }
                )
            if (index + 1) % 6 == 0:
                print(f"  {index + 1}/{len(paths)}", flush=True)
    out = CACHE / "adaptive_rho"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"{args.stage}_curves_shard{args.shard_index}.parquet", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "eval", "app"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
