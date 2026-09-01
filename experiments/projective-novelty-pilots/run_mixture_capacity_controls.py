"""Capacity-matched controls for the non-Gaussian mixture pilot."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn

import run_mixture as mixture


HERE = Path(__file__).resolve().parent
OUT = HERE / "mixture"


class MatchedProjectiveGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        width = 241
        self.backbone = nn.Sequential(
            nn.Linear(mixture.experiment.HISTORY_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, 1 + 2 * mixture.experiment.OUTPUT_DIM)

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.output(self.backbone(history))
        log_weights = torch.zeros((len(history), 1), device=history.device, dtype=history.dtype)
        joint_mean = output[:, 1 : 1 + mixture.experiment.OUTPUT_DIM]
        diagonal = nn.functional.softplus(output[:, 1 + mixture.experiment.OUTPUT_DIM :]) + 1e-4
        component_mean = torch.sum(joint_mean * query, dim=-1, keepdim=True)
        component_variance = torch.sum(diagonal.square() * query.square(), dim=-1, keepdim=True)
        return log_weights, component_mean, component_variance


class MatchedDirectMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = mixture.COMPONENTS
        width = 249
        self.network = nn.Sequential(
            nn.Linear(mixture.experiment.HISTORY_DIM + mixture.experiment.OUTPUT_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3 * self.components),
        )

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.network(torch.cat([history, query], dim=-1)).reshape(len(history), self.components, 3)
        return (
            torch.log_softmax(output[:, :, 0], dim=-1),
            output[:, :, 1],
            nn.functional.softplus(output[:, :, 2]) + 1e-4,
        )


def main() -> None:
    started = time.perf_counter()
    rows = []
    parameter_counts = {}
    constructors = {
        "joint_gaussian_matched": MatchedProjectiveGaussian,
        "direct_mixture4_matched": MatchedDirectMixture,
    }
    for dataset in mixture.experiment.DATASETS:
        train_history, train_future, test_history, test_future = mixture.experiment.make_windows(dataset)
        for seed in mixture.experiment.SEEDS:
            for model_name, constructor in constructors.items():
                torch.manual_seed(seed)
                model = constructor()
                parameter_counts[model_name] = sum(parameter.numel() for parameter in model.parameters())
                seconds = mixture.train_model(model, train_history, train_future, seed)
                result = mixture.evaluate(model, test_history, test_future, seed)
                rows.append(
                    {"dataset": dataset, "seed": seed, "model": model_name, "train_seconds": seconds, **result}
                )
                torch.save(model.state_dict(), OUT / f"{dataset}__{model_name}__seed-{seed}.pt")
    controls = pd.DataFrame(rows)
    controls.to_csv(OUT / "capacity_control_cells.csv", index=False)
    controls.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "capacity_control_summary.csv", index=False
    )

    primary = pd.read_csv(OUT / "cells.csv")
    projective = primary[primary.model == "projective_mixture4"].set_index(["dataset", "seed"])
    gaussian = controls[controls.model == "joint_gaussian_matched"].set_index(["dataset", "seed"])
    direct = controls[controls.model == "direct_mixture4_matched"].set_index(["dataset", "seed"])
    gaussian_wins = int((projective.heldout_nll <= gaussian.heldout_nll).sum())
    direct_wins = int((projective.heldout_nll <= direct.heldout_nll).sum())
    improvements = {}
    for dataset in mixture.experiment.DATASETS:
        p = float(projective.loc[dataset].heldout_nll.mean())
        g = float(gaussian.loc[dataset].heldout_nll.mean())
        improvements[dataset] = g - p
    improvement_datasets = sum(value >= 0.05 for value in improvements.values())
    audit = {
        "status": "complete",
        "wall_seconds": time.perf_counter() - started,
        "parameter_counts": parameter_counts,
        "mixture_vs_matched_gaussian_wins": gaussian_wins,
        "mixture_vs_matched_direct_wins": direct_wins,
        "dataset_nll_improvements_vs_matched_gaussian": improvements,
        "improvement_datasets": int(improvement_datasets),
        "survives_capacity_matching": bool(
            gaussian_wins >= 6 and improvement_datasets >= 2 and direct_wins >= 6
        ),
    }
    (OUT / "capacity_control_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
