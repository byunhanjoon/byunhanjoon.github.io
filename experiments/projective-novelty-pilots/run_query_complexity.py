"""Pilot Q: test whether direct-model regret grows with query composition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent / "oral-ceiling-followups"
sys.path.insert(0, str(FOLLOWUP))
import run_projective_real as experiment  # noqa: E402


OUT = HERE / "query_complexity"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS = FOLLOWUP / "projective_real"
PROTOCOL_SHA256 = "b5148cca2610c49d8cca287d123d81427cc2daa1874150ca16056159d8b3daab"
SUPPORT_SIZES = (1, 2, 4, 8, 16, 32)


def signed_queries(seed: int, count: int, support: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    queries = np.zeros((count, experiment.OUTPUT_DIM), dtype=np.float32)
    magnitude = np.float32(1.0 / np.sqrt(support))
    for row in range(count):
        selected = rng.choice(experiment.OUTPUT_DIM, support, replace=False)
        queries[row, selected] = rng.choice((-magnitude, magnitude), size=support)
    return queries


@torch.no_grad()
def metrics(model: torch.nn.Module, history: torch.Tensor, future: torch.Tensor, query: torch.Tensor) -> dict[str, float]:
    model.eval()
    mean, variance = model(history, query)
    target = torch.sum(query * future, dim=-1)
    standardized = torch.abs(target - mean) / torch.sqrt(variance)
    return {
        "nll": float(experiment.gaussian_nll(mean, variance, target)),
        "rmse": float(torch.sqrt(torch.mean((target - mean).square()))),
        "coverage_50": float((standardized <= 0.67448975).float().mean()),
        "coverage_90": float((standardized <= 1.64485363).float().mean()),
    }


def main() -> None:
    rows = []
    for dataset in experiment.DATASETS:
        _, _, history_np, future_np = experiment.make_windows(dataset)
        history = torch.from_numpy(history_np).to(experiment.DEVICE)
        future = torch.from_numpy(future_np).to(experiment.DEVICE)
        for seed in experiment.SEEDS:
            models = {
                "direct_broad": experiment.QueryNet(),
                "projective": experiment.ProjectiveNet(),
            }
            paths = {
                "direct_broad": CHECKPOINTS / f"{dataset}__querynet_broad__seed-{seed}.pt",
                "projective": CHECKPOINTS / f"{dataset}__projectivenet__seed-{seed}.pt",
            }
            for name, model in models.items():
                model.load_state_dict(torch.load(paths[name], map_location=experiment.DEVICE, weights_only=True))
                model.to(experiment.DEVICE)
            for support in SUPPORT_SIZES:
                query = torch.from_numpy(signed_queries(seed + 10_000 + support, len(history_np), support)).to(
                    experiment.DEVICE
                )
                for model_name, model in models.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "support": support,
                            "model": model_name,
                            **metrics(model, history, future, query),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    pivot = frame.pivot(index=["dataset", "seed", "support"], columns="model", values="nll")
    regret = (pivot.direct_broad - pivot.projective).rename("nll_regret").reset_index()
    regret.to_csv(OUT / "regret_cells.csv", index=False)
    summary = regret.groupby(["dataset", "support"], as_index=False).nll_regret.mean()
    summary.to_csv(OUT / "regret_summary.csv", index=False)

    gates = {}
    passes = 0
    for dataset in experiment.DATASETS:
        group = summary[summary.dataset == dataset].set_index("support").loc[list(SUPPORT_SIZES)]
        rho = float(spearmanr(np.log2(SUPPORT_SIZES), group.nll_regret).statistic)
        singleton = float(group.loc[1, "nll_regret"])
        large = float(group.loc[[16, 32], "nll_regret"].mean())
        ratio = large / max(singleton, 0.02)
        passed = rho >= 0.7 and large >= 0.2 and ratio >= 2.0
        passes += passed
        gates[dataset] = {
            "spearman_rho": rho,
            "singleton_regret": singleton,
            "large_query_regret": large,
            "large_to_singleton_ratio": ratio,
            "passed": bool(passed),
        }
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "cells": len(frame),
        "dataset_passes": int(passes),
        "gates": gates,
        "passed": bool(passes >= 2),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
