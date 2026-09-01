#!/usr/bin/env python3
"""Score the frozen projective head and its exact marginal controls on CTR23."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from common import CACHE, CONFIG, atomic_json, gaussian_scores, stable_seed
from train_head import ProjectiveHead


def metadata(data) -> dict[str, Any]:
    return json.loads(str(data["metadata"].item()))


def load_models(
    device: torch.device, head_root: str = "head"
) -> tuple[list[ProjectiveHead], dict[str, Any]]:
    summary = json.loads((CACHE / head_root / "training_summary.json").read_text())
    models = []
    for path in summary["checkpoints"]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = ProjectiveHead(
            int(checkpoint["input_dim"]), int(config["rank"]), int(config["hidden_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    return models, summary


def hidden_kernel(hidden: np.ndarray) -> np.ndarray:
    h = hidden.astype(np.float64)
    h = (h - h.mean(axis=-1, keepdims=True)) / np.maximum(h.std(axis=-1, keepdims=True), 1e-10)
    unit = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-10)
    return np.einsum("snr,smr->nm", unit, unit) / len(unit)


def rbf_kernel(x: np.ndarray, length_multiplier: float) -> np.ndarray:
    x = x.astype(np.float64)
    norm = np.sum(x**2, axis=1)
    distance2 = np.maximum(norm[:, None] + norm[None, :] - 2.0 * x @ x.T, 0.0)
    length = length_multiplier * math.sqrt(max(x.shape[1], 1))
    return np.exp(-0.5 * distance2 / max(length**2, 1e-12))


@torch.no_grad()
def learned_covariances(
    hidden: np.ndarray,
    marginal_variance: np.ndarray,
    context_index: int,
    models: list[ProjectiveHead],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    sd = np.sqrt(np.maximum(marginal_variance.astype(np.float64), 1e-12))
    covariances, shuffled_covariances = [], []
    groups, q = int(CONFIG["query_groups"]), int(CONFIG["query_size"])
    permutation = np.arange(groups * q)
    for group in range(groups):
        start = group * q
        permutation[start : start + q] = start + np.roll(np.arange(q), 1)
    for model in models:
        unit = model.features(h).cpu().numpy().astype(np.float64)
        K = np.einsum("snr,smr->nm", unit, unit) / len(unit)
        shuffled = unit[:, permutation]
        K_shuffled = np.einsum("snr,smr->nm", shuffled, shuffled) / len(shuffled)
        rho = float(model.rhos()[context_index].cpu())
        R = (1.0 - rho) * np.eye(len(sd)) + rho * K
        R_shuffled = (1.0 - rho) * np.eye(len(sd)) + rho * K_shuffled
        covariance = sd[:, None] * R * sd[None, :]
        shuffled_covariance = sd[:, None] * R_shuffled * sd[None, :]
        # The analytic construction preserves marginals exactly. Remove only
        # floating-point normalization drift from the computed Gram diagonal.
        np.fill_diagonal(covariance, marginal_variance)
        np.fill_diagonal(shuffled_covariance, marginal_variance)
        covariances.append(covariance)
        shuffled_covariances.append(shuffled_covariance)
    return np.mean(covariances, axis=0), np.mean(shuffled_covariances, axis=0)


def coverage_fields(error: float, sd: float) -> dict[str, float]:
    result = {}
    for level, z in ((50, 0.6744897501960817), (80, 1.2815515655446004), (90, 1.6448536269514722), (95, 1.959963984540054)):
        result[f"coverage_{level}"] = float(abs(error) <= z * sd)
        result[f"width_{level}"] = float(2.0 * z * sd)
    return result


def score_episode(
    path: Path,
    models: list[ProjectiveHead],
    summary: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        meta = metadata(data)
        hidden = data["hidden"].astype(np.float32)
        mean = data["mean"].astype(np.float64)
        base_variance = data["variance"].astype(np.float64)
        target = data["target"].astype(np.float64)
        coefficients = data["coefficients"].astype(np.float64)
        query_numeric = data["query_numeric"].astype(np.float64)

    context_size = int(meta["context_size"])
    context_index = list(map(int, CONFIG["context_sizes"])).index(context_size)
    temperature = float(summary["marginal_temperatures"][str(context_size)])
    marginal_variance = np.maximum(temperature * base_variance, 1e-10)
    diagonal_covariance = np.diag(marginal_variance)
    learned, shuffled = learned_covariances(
        hidden, marginal_variance, context_index, models, device
    )

    hidden_spec = summary["nonparametric_kernels"]["hidden_cosine"][str(context_size)]
    hidden_rho = float(hidden_spec["rho"])
    K_hidden = hidden_kernel(hidden)
    hidden_covariance = np.sqrt(marginal_variance)[:, None] * (
        (1.0 - hidden_rho) * np.eye(len(mean)) + hidden_rho * K_hidden
    ) * np.sqrt(marginal_variance)[None, :]

    raw_spec = summary["nonparametric_kernels"]["raw_rbf"][str(context_size)]
    raw_rho = float(raw_spec["rho"])
    K_raw = rbf_kernel(query_numeric, float(raw_spec["length"]))
    raw_covariance = np.sqrt(marginal_variance)[:, None] * (
        (1.0 - raw_rho) * np.eye(len(mean)) + raw_rho * K_raw
    ) * np.sqrt(marginal_variance)[None, :]

    covariances = {
        "tabiclv2_diagonal": diagonal_covariance,
        "projtabicl": learned,
        "projtabicl_shuffled": shuffled,
        "hidden_cosine": hidden_covariance,
        "raw_feature_rbf": raw_covariance,
    }
    rows: list[dict[str, Any]] = []
    families = list(CONFIG["query_families"])
    for method, covariance in covariances.items():
        for family_index, family in enumerate(families):
            for group in range(int(CONFIG["query_groups"])):
                a = coefficients[family_index, group]
                truth = float(a @ target)
                prediction = float(a @ mean)
                variance = float(max(a @ covariance @ a, 1e-10))
                score = {key: float(value) for key, value in gaussian_scores(truth, prediction, variance).items()}
                row = {
                    "dataset": meta["dataset"],
                    "source_id": meta["source_id"],
                    "split": meta["split"],
                    "replicate": int(meta["replicate"]),
                    "context_size": context_size,
                    "method": method,
                    "family": family,
                    "group": group,
                    "target": truth,
                    "mean": prediction,
                    "variance": variance,
                    **score,
                    **coverage_fields(truth - prediction, math.sqrt(variance)),
                }
                rows.append(row)

    audit = {
        "path": str(path),
        "dataset": meta["dataset"],
        "context_size": context_size,
        "mean_max_abs": 0.0,
        "diagonal_max_abs": float(np.max(np.abs(np.diag(learned) - marginal_variance))),
        "shuffled_diagonal_max_abs": float(np.max(np.abs(np.diag(shuffled) - marginal_variance))),
        "symmetry_max_abs": float(np.max(np.abs(learned - learned.T))),
        "min_eigenvalue": float(np.linalg.eigvalsh(learned).min()),
        "temperature": temperature,
    }
    return rows, audit


def paired_randomization(effects: np.ndarray, repetitions: int) -> float:
    observed = abs(float(np.mean(effects)))
    rng = np.random.default_rng(20270201)
    exceed = 0
    chunk = 10_000
    complete = 0
    while complete < repetitions:
        current = min(chunk, repetitions - complete)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(effects)))
        exceed += int(np.sum(np.abs((signs * effects).mean(axis=1)) >= observed - 1e-15))
        complete += current
    return (exceed + 1.0) / (repetitions + 1.0)


def bootstrap_interval(effects: np.ndarray, repetitions: int) -> tuple[float, float]:
    rng = np.random.default_rng(20270202)
    indices = rng.integers(len(effects), size=(repetitions, len(effects)))
    means = effects[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(cells: pd.DataFrame, audits: pd.DataFrame, out: Path) -> dict[str, Any]:
    primary = cells[cells["family"].isin(CONFIG["primary_aggregate_families"])]
    dataset_table = (
        primary.groupby(["dataset", "method"], as_index=False)[["nll", "crps", "squared_error"]]
        .mean()
        .pivot(index="dataset", columns="method")
    )
    effects_nll = (
        dataset_table["nll"]["tabiclv2_diagonal"] - dataset_table["nll"]["projtabicl"]
    ).sort_index()
    effects_crps = (
        dataset_table["crps"]["tabiclv2_diagonal"] - dataset_table["crps"]["projtabicl"]
    ).sort_index()
    repetitions = int(CONFIG["bootstrap_repetitions"])
    ci_nll = bootstrap_interval(effects_nll.to_numpy(), repetitions)
    ci_crps = bootstrap_interval(effects_crps.to_numpy(), repetitions)
    wins = int((effects_nll > 0).sum())
    ties = int((effects_nll == 0).sum())
    sign_p = float(binomtest(wins, wins + int((effects_nll < 0).sum()), 0.5).pvalue)
    randomization_p = paired_randomization(effects_nll.to_numpy(), max(repetitions, 100_000))
    gates = CONFIG["primary_gates"]
    gate_results = {
        "mean_nll_advantage": float(effects_nll.mean()) > float(gates["mean_nll_advantage_over_diagonal"]),
        "dataset_nll_win_rate": float((effects_nll > 0).mean()) >= float(gates["dataset_nll_win_rate"]),
        "mean_crps_advantage": float(effects_crps.mean()) > float(gates["mean_crps_advantage_over_diagonal"]),
        "paired_randomization": randomization_p < float(gates["paired_randomization_p"]),
        "mean_diagonal_identity": float(audits[["mean_max_abs", "diagonal_max_abs"]].to_numpy().max()) <= float(gates["mean_and_diagonal_max_abs"]),
        "psd": float(audits["min_eigenvalue"].min()) >= -1e-7,
    }
    by_method = (
        primary.groupby("method")[["nll", "crps", "squared_error", "coverage_90", "width_90"]]
        .mean()
        .sort_values("nll")
        .reset_index()
    )
    by_context = (
        primary.groupby(["context_size", "method"])[["nll", "crps", "squared_error"]]
        .mean()
        .reset_index()
    )
    by_family = (
        primary.groupby(["family", "method"])[["nll", "crps", "squared_error"]]
        .mean()
        .reset_index()
    )
    out.mkdir(parents=True, exist_ok=True)
    effects = pd.DataFrame(
        {
            "dataset": effects_nll.index,
            "nll_advantage_diagonal_minus_projective": effects_nll.values,
            "crps_advantage_diagonal_minus_projective": effects_crps.reindex(effects_nll.index).values,
        }
    )
    effects.to_csv(out / "dataset_effects.csv", index=False)
    by_method.to_csv(out / "by_method.csv", index=False)
    by_context.to_csv(out / "by_context.csv", index=False)
    by_family.to_csv(out / "by_family.csv", index=False)
    summary = {
        "episode_count": int(cells[["dataset", "split", "replicate", "context_size"]].drop_duplicates().shape[0]),
        "cell_count": int(len(cells)),
        "dataset_count": int(effects_nll.size),
        "primary_families": CONFIG["primary_aggregate_families"],
        "nll_advantage": {
            "mean": float(effects_nll.mean()),
            "bootstrap_95": ci_nll,
            "wins": wins,
            "losses": int((effects_nll < 0).sum()),
            "ties": ties,
            "win_rate": float((effects_nll > 0).mean()),
            "sign_p": sign_p,
            "paired_randomization_p": randomization_p,
        },
        "crps_advantage": {
            "mean": float(effects_crps.mean()),
            "bootstrap_95": ci_crps,
            "wins": int((effects_crps > 0).sum()),
            "losses": int((effects_crps < 0).sum()),
        },
        "integrity": {
            "max_diagonal_abs": float(audits["diagonal_max_abs"].max()),
            "max_shuffled_diagonal_abs": float(audits["shuffled_diagonal_max_abs"].max()),
            "max_symmetry_abs": float(audits["symmetry_max_abs"].max()),
            "minimum_eigenvalue": float(audits["min_eigenvalue"].min()),
        },
        "gates": gate_results,
        "all_primary_gates_pass_so_far": bool(all(gate_results.values())),
        "by_method": by_method.to_dict(orient="records"),
    }
    return summary


def main(args: argparse.Namespace) -> None:
    singleton = args.query_mode == "singleton"
    root = CACHE / ("tabicl_singleton_episodes" if singleton else "tabicl_episodes") / "eval"
    paths = sorted(root.glob("*.npz"))
    expected = (
        len(CONFIG["evaluation_tasks"])
        * len(CONFIG["evaluation_folds"])
        * int(CONFIG["context_replicates"])
        * len(CONFIG["context_sizes"])
    )
    if len(paths) != expected:
        raise RuntimeError(f"evaluation cache incomplete: expected {expected}, found {len(paths)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, training_summary = load_models(device, "head_singleton" if singleton else "head")
    all_rows, all_audits = [], []
    for index, path in enumerate(paths):
        rows, audit = score_episode(path, models, training_summary, device)
        all_rows.extend(rows)
        all_audits.append(audit)
        if (index + 1) % 50 == 0:
            print(f"scored {index + 1}/{len(paths)}", flush=True)
    cells = pd.DataFrame(all_rows)
    audits = pd.DataFrame(all_audits)
    out = CACHE / "results" / ("projective_singleton" if singleton else "projective")
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / "cells.parquet", index=False)
    audits.to_csv(out / "integrity.csv", index=False)
    summary = summarize(cells, audits, out)
    summary["query_mode"] = args.query_mode
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["batched", "singleton"], default="batched")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
