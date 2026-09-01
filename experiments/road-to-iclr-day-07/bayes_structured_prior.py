#!/usr/bin/env python3
"""Bayes phase diagram for a latent useful/irrelevant field geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import expit
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "pfn_prior"
N_STATES = 32
N_TASKS = 5_000
ASSUMED_PRIOR = 0.5


def heat_covariance(scale: float) -> np.ndarray:
    adjacency = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        adjacency[i, (i - 1) % N_STATES] = 1.0
        adjacency[i, (i + 1) % N_STATES] = 1.0
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    kernel = expm(-scale * laplacian)
    diagonal = np.sqrt(np.diag(kernel))
    return kernel / diagonal[:, None] / diagonal[None, :]


def log_normal(x: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance is not positive definite")
    return float(-0.5 * (len(x) * np.log(2 * np.pi) + logdet + x @ np.linalg.solve(covariance, x)))


def run_cell(true_prior: float, scale: float, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    kernel = heat_covariance(scale)
    loss = {name: [] for name in ("zero", "always_smooth", "hard_route", "bayes_mixture", "regime_oracle")}
    posterior, regime = [], []
    for _ in range(N_TASKS):
        smooth = bool(rng.random() < true_prior)
        covariance = kernel if smooth else np.eye(N_STATES)
        effect = rng.multivariate_normal(np.zeros(N_STATES), covariance)
        order = rng.permutation(N_STATES); train = np.sort(order[:20]); query = np.sort(order[20:])
        observed = effect[train] + rng.normal(0.0, noise, len(train))
        smooth_tt = kernel[np.ix_(train, train)] + noise**2 * np.eye(len(train))
        random_tt = (1.0 + noise**2) * np.eye(len(train))
        logit = log_normal(observed, smooth_tt) - log_normal(observed, random_tt)
        logit += np.log(ASSUMED_PRIOR) - np.log1p(-ASSUMED_PRIOR)
        probability = float(expit(logit))
        smooth_prediction = kernel[np.ix_(query, train)] @ np.linalg.solve(smooth_tt, observed)
        predictions = {
            "zero": np.zeros(len(query)),
            "always_smooth": smooth_prediction,
            "hard_route": smooth_prediction if probability > 0.5 else np.zeros(len(query)),
            "bayes_mixture": probability * smooth_prediction,
            "regime_oracle": smooth_prediction if smooth else np.zeros(len(query)),
        }
        for name, prediction in predictions.items():
            loss[name].append(float(np.mean((effect[query] - prediction) ** 2)))
        posterior.append(probability); regime.append(int(smooth))
    row = {"true_prior": true_prior, "scale": scale, "noise": noise, "tasks": N_TASKS}
    for name, values in loss.items():
        row[f"mse_{name}"] = float(np.mean(values)); row[f"se_{name}"] = float(np.std(values, ddof=1) / np.sqrt(N_TASKS))
    row["posterior_auroc"] = float(roc_auc_score(regime, posterior))
    row["posterior_mean"] = float(np.mean(posterior)); row["regime_rate"] = float(np.mean(regime))
    return row


def main() -> None:
    rows = []
    for prior in (0.1, 0.5, 0.9):
        for scale in (0.3, 1.0, 3.0):
            for noise in (0.1, 0.3, 1.0):
                row = run_cell(prior, scale, noise, 20260830 + len(rows)); rows.append(row)
                print(prior, scale, noise, flush=True)
    frame = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); frame.to_csv(OUT / "cells.csv", index=False)
    matched = frame[frame.true_prior == ASSUMED_PRIOR]
    comparisons = {}
    for baseline in ("zero", "always_smooth", "hard_route"):
        difference = matched[f"mse_{baseline}"] - matched["mse_bayes_mixture"]
        comparisons[baseline] = {"mean_mse_advantage": float(difference.mean()), "wins": int((difference >= 0).sum()), "cells": len(difference)}
    shifted = {}
    for prior, group in frame.groupby("true_prior"):
        shifted[str(prior)] = {
            "bayes_mixture_mse": float(group.mse_bayes_mixture.mean()),
            "best_simple_mse": float(group[["mse_zero", "mse_always_smooth", "mse_hard_route"]].min(axis=1).mean()),
            "regime_oracle_mse": float(group.mse_regime_oracle.mean()),
            "posterior_auroc": float(group.posterior_auroc.mean()),
            "posterior_mean": float(group.posterior_mean.mean()),
        }
    summary = {"status": "complete", "matched_prior_comparisons": comparisons, "deployment_prior_shift": shifted}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
