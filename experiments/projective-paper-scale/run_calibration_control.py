"""Factorial control: validation calibration on the original diagonal mixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

import run_decisive as decisive
import run_lowrank_followup as lowrank


HERE = Path(__file__).resolve().parent
OUT = HERE / "calibration_control_outputs"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL_SHA256 = "410e5c0bb77e95599bac918ddcf93f95d026837d02645135dc39cb6962e62fa4"


class DiagonalAdapter(nn.Module):
    def __init__(self, model: decisive.mixture.ProjectiveMixtureNet) -> None:
        super().__init__()
        self.model = model
        self.components = model.components

    def joint(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dimension = decisive.mixture.experiment.OUTPUT_DIM
        output = self.model.output(self.model.backbone(history)).reshape(len(history), self.components, -1)
        log_weights = torch.log_softmax(output[:, :, 0], dim=-1)
        means = output[:, :, 1 : 1 + dimension]
        diagonal = nn.functional.softplus(output[:, :, 1 + dimension :]) + 1e-4
        factors = torch.zeros(
            len(history), self.components, dimension, lowrank.RANK,
            device=history.device, dtype=history.dtype,
        )
        return log_weights, means, diagonal, factors


def load_diagonal(dataset: str, seed: int) -> DiagonalAdapter:
    model = decisive.mixture.ProjectiveMixtureNet(decisive.mixture.COMPONENTS)
    path = decisive.NOVELTY / "mixture" / f"{dataset}__projective_mixture4__seed-{seed}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return DiagonalAdapter(model)


def audit(frame: pd.DataFrame) -> dict[str, object]:
    original = pd.read_csv(HERE / "outputs" / "evaluation_cells.csv")
    rank = pd.read_csv(HERE / "lowrank_outputs" / "evaluation_cells.csv")
    combined = pd.concat([original, rank, frame], ignore_index=True)
    summary = combined.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    crps = summary.pivot(index="dataset", columns="model", values="macro_crps")
    coverage = summary.pivot(index="dataset", columns="model", values="coverage_error")
    traffic_reduction = float(
        coverage.loc["Traffic", "projective_mixture4"] - coverage.loc["Traffic", "diagonal_calibrated"]
    )
    mean_ratio = float(crps["diagonal_calibrated"].mean() / crps["projective_mixture4"].mean())
    electricity_lowrank_gain = float(
        (crps.loc["Electricity", "diagonal_calibrated"] - crps.loc["Electricity", "lowrank4_calibrated"])
        / crps.loc["Electricity", "diagonal_calibrated"]
    )
    lowrank_mean_ratio = float(crps["lowrank4_calibrated"].mean() / crps["diagonal_calibrated"].mean())
    calibration_useful = traffic_reduction >= 0.03 and mean_ratio <= 1.02
    lowrank_adds_value = electricity_lowrank_gain >= 0.02 and lowrank_mean_ratio <= 1.01
    result: dict[str, object] = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "all_finite": bool(np.isfinite(frame.macro_crps).all()),
        "traffic_coverage_error_reduction": traffic_reduction,
        "calibrated_diagonal_mean_crps_ratio": mean_ratio,
        "lowrank_electricity_gain_vs_calibrated_diagonal": electricity_lowrank_gain,
        "lowrank_mean_crps_ratio_vs_calibrated_diagonal": lowrank_mean_ratio,
        "calibration_useful": bool(calibration_useful),
        "lowrank_adds_value": bool(lowrank_adds_value),
        "dataset_crps": {
            dataset: {
                model: float(crps.loc[dataset, model])
                for model in ("projective_mixture4", "diagonal_calibrated", "lowrank4_calibrated", "tactis2")
            }
            for dataset in decisive.DATASETS
        },
        "dataset_coverage_error": {
            dataset: {
                model: float(coverage.loc[dataset, model])
                for model in ("projective_mixture4", "diagonal_calibrated", "lowrank4_calibrated", "tactis2")
            }
            for dataset in decisive.DATASETS
        },
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    digest = hashlib.sha256((HERE / "CALIBRATION_CONTROL_PROTOCOL.md").read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol changed: expected {PROTOCOL_SHA256}, found {digest}")
    rows, calibration_rows = [], []
    for dataset in decisive.DATASETS:
        _, _, test_history, test_future = decisive.mixture.experiment.make_windows(dataset)
        test_history = test_history[: decisive.EVAL_CONTEXTS]
        test_future = test_future[: decisive.EVAL_CONTEXTS]
        for seed in decisive.SEEDS:
            model = load_diagonal(dataset, seed)
            temperature, before, after = lowrank.fit_temperature(dataset, seed, model)
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "temperature": temperature,
                    "validation_nll_before": before,
                    "validation_nll_after": after,
                }
            )
            decisive.seed_everything(seed + 449)
            queries = decisive.make_queries(seed, decisive.EVAL_CONTEXTS)
            metrics, latency = lowrank.evaluate_one(model, test_history, test_future, queries, temperature)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "diagonal_calibrated",
                    "parameters": decisive.count_parameters(model.model),
                    "temperature": temperature,
                    "latency_ms_per_context": latency,
                    **metrics,
                }
            )
            print(
                f"evaluated diagonal_calibrated {dataset} seed={seed}: CRPS={metrics['macro_crps']:.4f} "
                f"coverage_error={metrics['coverage_error']:.4f} temperature={temperature:.3f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "evaluation_cells.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_cells.csv", index=False)
    frame.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "evaluation_summary.csv", index=False
    )
    print(json.dumps(audit(frame), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
