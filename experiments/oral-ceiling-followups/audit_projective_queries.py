"""Post-hoc query-family diagnostic; does not alter the frozen projective gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_projective_real as experiment


HERE = Path(__file__).resolve().parent
OUT = HERE / "projective_real"


def query_family(seed: int, family: str, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "train_style":
        return experiment.training_queries(rng, count)
    query = rng.normal(size=(count, experiment.OUTPUT_DIM)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
    if family == "difference":
        query[:] = 0
        for index in range(count):
            pair = rng.choice(experiment.OUTPUT_DIM, 2, replace=False)
            query[index, pair] = (1.0, -1.0)
    elif family == "scaled_dense":
        query *= rng.uniform(0.3, 2.7, size=count).astype(np.float32)[:, None]
    elif family != "dense":
        raise ValueError(family)
    return query


@torch.no_grad()
def main() -> None:
    rows = []
    families = ("train_style", "difference", "dense", "scaled_dense")
    for dataset in experiment.DATASETS:
        _, _, history_np, future_np = experiment.make_windows(dataset)
        history = torch.from_numpy(history_np).to(experiment.DEVICE)
        future = torch.from_numpy(future_np).to(experiment.DEVICE)
        for seed in experiment.SEEDS:
            for model_name, constructor in (("querynet", experiment.QueryNet), ("projectivenet", experiment.ProjectiveNet)):
                model = constructor().to(experiment.DEVICE)
                state = torch.load(
                    OUT / f"{dataset}__{model_name}__seed-{seed}.pt",
                    map_location=experiment.DEVICE,
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()
                for family in families:
                    query = torch.from_numpy(query_family(seed + 101, family, len(history))).to(experiment.DEVICE)
                    target = torch.sum(query * future, dim=-1)
                    mean, variance = model(history, query)
                    error = target - mean
                    standardized = torch.abs(error) / torch.sqrt(variance)
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model_name,
                            "query_family": family,
                            "nll": float(experiment.gaussian_nll(mean, variance, target)),
                            "rmse": float(torch.sqrt(torch.mean(error.square()))),
                            "mean_predicted_variance": float(variance.mean()),
                            "coverage_50": float((standardized <= 0.67448975).float().mean()),
                            "coverage_90": float((standardized <= 1.64485363).float().mean()),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "query_family_audit.csv", index=False)
    summary = frame.groupby(["dataset", "model", "query_family"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "query_family_audit_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
