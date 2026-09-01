"""Pilot R: reconcile black-box scalar Gaussian queries into one joint Gaussian."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent / "oral-ceiling-followups"
sys.path.insert(0, str(FOLLOWUP))
import run_projective_real as experiment  # noqa: E402


OUT = HERE / "reconciliation"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS = FOLLOWUP / "projective_real"
PROTOCOL_SHA256 = "b5148cca2610c49d8cca287d123d81427cc2daa1874150ca16056159d8b3daab"
COUNT = 1_024


def reconstruction_queries() -> tuple[np.ndarray, list[tuple[int, int]]]:
    basis = np.eye(experiment.OUTPUT_DIM, dtype=np.float32)
    pairs = [(i, j) for i in range(experiment.OUTPUT_DIM) for j in range(i + 1, experiment.OUTPUT_DIM)]
    pair_queries = np.zeros((len(pairs), experiment.OUTPUT_DIM), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        pair_queries[row, (i, j)] = 1.0
    return np.concatenate([basis, pair_queries]), pairs


@torch.no_grad()
def reconstruct(model: experiment.QueryNet, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    query_np, pairs = reconstruction_queries()
    query_bank = torch.from_numpy(query_np).to(experiment.DEVICE)
    means = []
    variances = []
    started = time.perf_counter()
    for start in range(0, len(history), 16):
        batch = history[start : start + 16]
        batch_size = len(batch)
        repeated_history = batch[:, None, :].expand(batch_size, len(query_bank), -1).reshape(-1, experiment.HISTORY_DIM)
        repeated_query = query_bank[None].expand(batch_size, -1, -1).reshape(-1, experiment.OUTPUT_DIM)
        mean, variance = model(repeated_history, repeated_query)
        means.append(mean.reshape(batch_size, -1))
        variances.append(variance.reshape(batch_size, -1))
    means = torch.cat(means)
    variances = torch.cat(variances)
    mu = means[:, : experiment.OUTPUT_DIM]
    diagonal = variances[:, : experiment.OUTPUT_DIM]
    covariance = torch.diag_embed(diagonal)
    pair_variance = variances[:, experiment.OUTPUT_DIM :]
    for index, (i, j) in enumerate(pairs):
        value = 0.5 * (pair_variance[:, index] - diagonal[:, i] - diagonal[:, j])
        covariance[:, i, j] = value
        covariance[:, j, i] = value
    original_norm = torch.linalg.matrix_norm(covariance).clamp_min(1e-8)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    clipped = eigenvalues.clamp_min(1e-4)
    projected = torch.matmul(eigenvectors * clipped[:, None, :], eigenvectors.transpose(-1, -2))
    correction = float((torch.linalg.matrix_norm(projected - covariance) / original_norm).mean())
    return mu, projected, correction, time.perf_counter() - started


def summarize(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    standardized = torch.abs(target - mean) / torch.sqrt(variance)
    coverage_50 = float((standardized <= 0.67448975).float().mean())
    coverage_90 = float((standardized <= 1.64485363).float().mean())
    return {
        "nll": float(experiment.gaussian_nll(mean, variance, target)),
        "coverage_50": coverage_50,
        "coverage_90": coverage_90,
        "mean_coverage_error": 0.5 * (abs(coverage_50 - 0.50) + abs(coverage_90 - 0.90)),
    }


@torch.no_grad()
def main() -> None:
    rows = []
    diagnostics = []
    for dataset in experiment.DATASETS:
        _, _, history_np, future_np = experiment.make_windows(dataset)
        history = torch.from_numpy(history_np[:COUNT]).to(experiment.DEVICE)
        future = torch.from_numpy(future_np[:COUNT]).to(experiment.DEVICE)
        for seed in experiment.SEEDS:
            direct = experiment.QueryNet().to(experiment.DEVICE)
            projective = experiment.ProjectiveNet().to(experiment.DEVICE)
            direct.load_state_dict(
                torch.load(
                    CHECKPOINTS / f"{dataset}__querynet_broad__seed-{seed}.pt",
                    map_location=experiment.DEVICE,
                    weights_only=True,
                )
            )
            projective.load_state_dict(
                torch.load(
                    CHECKPOINTS / f"{dataset}__projectivenet__seed-{seed}.pt",
                    map_location=experiment.DEVICE,
                    weights_only=True,
                )
            )
            direct.eval()
            projective.eval()
            query_np = experiment.heldout_queries(np.random.default_rng(seed + 29), COUNT)
            query = torch.from_numpy(query_np).to(experiment.DEVICE)
            target = torch.sum(query * future, dim=-1)
            direct_mean, direct_variance = direct(history, query)
            projective_mean, projective_variance = projective(history, query)
            joint_mean, covariance, correction, seconds = reconstruct(direct, history)
            reconciled_mean = torch.sum(query * joint_mean, dim=-1)
            reconciled_variance = torch.einsum("bi,bij,bj->b", query, covariance, query)
            for model_name, mean, variance in (
                ("direct_broad", direct_mean, direct_variance),
                ("reconciled", reconciled_mean, reconciled_variance),
                ("trained_projective", projective_mean, projective_variance),
            ):
                rows.append(
                    {"dataset": dataset, "seed": seed, "model": model_name, **summarize(mean, variance, target)}
                )
            diagnostics.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "relative_psd_correction": correction,
                    "reconstruction_seconds": seconds,
                    "scalar_queries_per_history": experiment.OUTPUT_DIM * (experiment.OUTPUT_DIM + 1) // 2,
                }
            )
    frame = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    frame.to_csv(OUT / "cells.csv", index=False)
    diagnostic_frame.to_csv(OUT / "diagnostics.csv", index=False)
    summary = frame.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    gates = {}
    improvement_passes = 0
    closure_passes = 0
    coverage_passes = 0
    for dataset in experiment.DATASETS:
        group = summary[summary.dataset == dataset].set_index("model")
        direct_nll = float(group.loc["direct_broad", "nll"])
        reconciled_nll = float(group.loc["reconciled", "nll"])
        projective_nll = float(group.loc["trained_projective", "nll"])
        improvement = direct_nll - reconciled_nll
        gap = direct_nll - projective_nll
        closure = improvement / gap if gap > 0 else float("-inf")
        coverage_difference = float(
            group.loc["reconciled", "mean_coverage_error"] - group.loc["direct_broad", "mean_coverage_error"]
        )
        improvement_passes += improvement > 0
        closure_passes += closure >= 0.50
        coverage_passes += coverage_difference <= 0.05
        gates[dataset] = {
            "nll_improvement": improvement,
            "projective_gap_closed": closure,
            "coverage_error_difference": coverage_difference,
        }
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "cells": len(frame),
        "improvement_datasets": int(improvement_passes),
        "closure_datasets": int(closure_passes),
        "coverage_datasets": int(coverage_passes),
        "mean_relative_psd_correction": float(diagnostic_frame.relative_psd_correction.mean()),
        "mean_reconstruction_seconds": float(diagnostic_frame.reconstruction_seconds.mean()),
        "gates": gates,
    }
    audit["passed"] = bool(improvement_passes >= 2 and closure_passes >= 2 and coverage_passes == 3)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
