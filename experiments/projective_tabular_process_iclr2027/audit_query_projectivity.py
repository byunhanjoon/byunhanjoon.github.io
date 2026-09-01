#!/usr/bin/env python3
"""Empirically audit query-set restriction and permutation consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CACHE, CONFIG, atomic_json, load_openml_task, native_frame, slug, stable_seed
from extract_tabicl import TabICLExtractor
from score_projective import load_models


def align_permutation(values: np.ndarray, permutation: np.ndarray, axis: int) -> np.ndarray:
    aligned = np.empty_like(values)
    index = [slice(None)] * values.ndim
    index[axis] = permutation
    aligned[tuple(index)] = values
    return aligned


def projective_covariance(
    hidden: np.ndarray,
    variance: np.ndarray,
    context_index: int,
    models,
    device: torch.device,
) -> np.ndarray:
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    sd = np.sqrt(np.maximum(variance, 1e-12))
    covariances = []
    with torch.no_grad():
        for model in models:
            unit = model.features(h).cpu().numpy().astype(np.float64)
            kernel = np.einsum("snr,smr->nm", unit, unit) / len(unit)
            rho = float(model.rhos()[context_index].cpu())
            correlation = (1.0 - rho) * np.eye(len(sd)) + rho * kernel
            covariance = sd[:, None] * correlation * sd[None, :]
            np.fill_diagonal(covariance, variance)
            covariances.append(covariance)
    return np.mean(covariances, axis=0)


def main(args: argparse.Namespace) -> None:
    singleton = args.query_mode == "singleton"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = TabICLExtractor(str(device))
    models, summary = load_models(device, "head_singleton" if singleton else "head")
    prediction_function = (
        extractor.fit_predict_singletons if singleton else extractor.fit_predict
    )
    episode_root = "tabicl_singleton_episodes" if singleton else "tabicl_episodes"
    rows = []
    for task_id in CONFIG["evaluation_tasks"]:
        dataset, _ = load_openml_task(int(task_id))
        for context_size in map(int, CONFIG["context_sizes"]):
            pattern = f"{slug(dataset.name)}__official_repeat0_fold0__rep0__n{context_size}.npz"
            path = CACHE / episode_root / "eval" / pattern
            with np.load(path, allow_pickle=False) as episode:
                meta = json.loads(str(episode["metadata"].item()))
                context_indices = episode["context_indices"].astype(np.int64)
                query_indices = episode["query_indices"].astype(np.int64)
            X_context = native_frame(dataset.X, context_indices)
            y_context = dataset.y[context_indices]
            rng = np.random.default_rng(stable_seed("projectivity-audit", dataset.name, context_size))
            subset = np.sort(rng.choice(len(query_indices), size=len(query_indices) // 2, replace=False))
            permutation = rng.permutation(len(query_indices))

            full = prediction_function(X_context, y_context, native_frame(dataset.X, query_indices))
            restricted = prediction_function(
                X_context, y_context, native_frame(dataset.X, query_indices[subset])
            )
            permuted = prediction_function(
                X_context, y_context, native_frame(dataset.X, query_indices[permutation])
            )
            full_mean, full_variance, full_hidden, _ = full
            sub_mean, sub_variance, sub_hidden, _ = restricted
            perm_mean, perm_variance, perm_hidden, _ = permuted
            aligned_mean = align_permutation(perm_mean, permutation, 0)
            aligned_variance = align_permutation(perm_variance, permutation, 0)
            aligned_hidden = align_permutation(perm_hidden, permutation, 1)

            metric_scale = float(meta["metric_scale"])
            temperature = float(summary["marginal_temperatures"][str(context_size)])
            calibrated_full = temperature * full_variance / metric_scale**2
            calibrated_sub = temperature * sub_variance / metric_scale**2
            calibrated_perm = temperature * aligned_variance / metric_scale**2
            context_index = list(map(int, CONFIG["context_sizes"])).index(context_size)
            full_covariance = projective_covariance(
                full_hidden, calibrated_full, context_index, models, device
            )
            sub_covariance = projective_covariance(
                sub_hidden, calibrated_sub, context_index, models, device
            )
            perm_covariance = projective_covariance(
                aligned_hidden, calibrated_perm, context_index, models, device
            )
            full_submatrix = full_covariance[np.ix_(subset, subset)]
            rows.append(
                {
                    "dataset": dataset.name,
                    "context_size": context_size,
                    "subset_size": len(subset),
                    "restriction_mean_max_abs": float(
                        np.max(np.abs((full_mean[subset] - sub_mean) / metric_scale))
                    ),
                    "restriction_variance_max_abs": float(
                        np.max(np.abs(calibrated_full[subset] - calibrated_sub))
                    ),
                    "restriction_hidden_max_abs": float(np.max(np.abs(full_hidden[:, subset] - sub_hidden))),
                    "restriction_covariance_max_abs": float(np.max(np.abs(full_submatrix - sub_covariance))),
                    "permutation_mean_max_abs": float(
                        np.max(np.abs((full_mean - aligned_mean) / metric_scale))
                    ),
                    "permutation_variance_max_abs": float(np.max(np.abs(calibrated_full - calibrated_perm))),
                    "permutation_hidden_max_abs": float(np.max(np.abs(full_hidden - aligned_hidden))),
                    "permutation_covariance_max_abs": float(np.max(np.abs(full_covariance - perm_covariance))),
                }
            )
        print(f"audited {dataset.name}", flush=True)

    frame = pd.DataFrame(rows)
    out = CACHE / "results" / (
        "projectivity_audit_singleton" if singleton else "projectivity_audit_batched"
    )
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "query_set_audit.csv", index=False)
    metric_columns = [column for column in frame if column.endswith("max_abs")]
    maxima = {column: float(frame[column].max()) for column in metric_columns}
    threshold = float(CONFIG["primary_gates"]["projectivity_max_abs"])
    summary_payload = {
        "datasets": int(frame["dataset"].nunique()),
        "episodes": int(len(frame)),
        "context_sizes": list(map(int, CONFIG["context_sizes"])),
        "restriction_query_size": 24,
        "maxima": maxima,
        "threshold": threshold,
        "passes": bool(max(maxima.values()) <= threshold),
        "query_mode": args.query_mode,
        "scope": "one official fold-0 replicate-0 episode per dataset and context size",
    }
    atomic_json(out / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["batched", "singleton"], default="singleton")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
