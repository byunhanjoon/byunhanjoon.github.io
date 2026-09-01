"""Electricity-first 2x2 screen: mixture count versus temporal attention."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

import run_calibration_control as calibration
import run_decisive as decisive
import run_lowrank_followup as lowrank


HERE = Path(__file__).resolve().parent
OUT = HERE / "representation_outputs"
CHECKPOINTS = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
PROTOCOL_SHA256 = "e37bf77277385647f228c64c0525e0b9a080e5787245851717cfb17351dc9cd5"
DATASET = "Electricity"
TARGET_PARAMETERS = 136_580
ADVANCEMENT_CRPS = 0.20007287
CONFIGURATIONS = ("mlp_k4", "mlp_k8", "attention_k4", "attention_k8")


class DiagonalProjectiveMixture(nn.Module):
    def __init__(self, components: int) -> None:
        super().__init__()
        self.components = components

    def distribution(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def joint(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        log_weights, means, diagonal = self.distribution(history)
        factors = torch.zeros(
            len(history), self.components, decisive.mixture.experiment.OUTPUT_DIM, lowrank.RANK,
            device=history.device, dtype=history.dtype,
        )
        return log_weights, means, diagonal, factors

    def forward(self, history: Tensor, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_weights, means, diagonal = self.distribution(history)
        projected_mean = torch.einsum("bkd,bd->bk", means, query)
        variance = torch.einsum("bkd,bd->bk", diagonal.square(), query.square())
        return log_weights, projected_mean, variance.clamp_min(1e-8)


class MLPMixture(DiagonalProjectiveMixture):
    def __init__(self, components: int, width: int) -> None:
        super().__init__(components)
        self.backbone = nn.Sequential(
            nn.Linear(decisive.mixture.experiment.HISTORY_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.output = nn.Linear(
            width, components * (1 + 2 * decisive.mixture.experiment.OUTPUT_DIM)
        )

    def distribution(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dimension = decisive.mixture.experiment.OUTPUT_DIM
        output = self.output(self.backbone(history)).reshape(len(history), self.components, -1)
        return (
            torch.log_softmax(output[:, :, 0], dim=-1),
            output[:, :, 1 : 1 + dimension],
            nn.functional.softplus(output[:, :, 1 + dimension :]) + 1e-4,
        )


class AttentionMixture(DiagonalProjectiveMixture):
    def __init__(self, components: int, width: int, heads: int) -> None:
        super().__init__(components)
        channels = decisive.mixture.experiment.CHANNELS
        steps = decisive.mixture.experiment.HISTORY_STEPS
        self.input_projection = nn.Linear(channels, width)
        self.time_embedding = nn.Parameter(torch.empty(1, steps, width))
        nn.init.normal_(self.time_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(
            width, components * (1 + 2 * decisive.mixture.experiment.OUTPUT_DIM)
        )

    def distribution(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dimension = decisive.mixture.experiment.OUTPUT_DIM
        tokens = history.reshape(
            len(history), decisive.mixture.experiment.HISTORY_STEPS, decisive.mixture.experiment.CHANNELS
        )
        encoded = self.encoder(self.input_projection(tokens) + self.time_embedding)
        representation = self.normalization(encoded[:, -1])
        output = self.output(representation).reshape(len(history), self.components, -1)
        return (
            torch.log_softmax(output[:, :, 0], dim=-1),
            output[:, :, 1 : 1 + dimension],
            nn.functional.softplus(output[:, :, 1 + dimension :]) + 1e-4,
        )


def make_model(name: str) -> DiagonalProjectiveMixture:
    if name == "mlp_k8":
        return MLPMixture(components=8, width=147)
    if name == "attention_k4":
        return AttentionMixture(components=4, width=82, heads=2)
    if name == "attention_k8":
        return AttentionMixture(components=8, width=76, heads=4)
    raise ValueError(name)


def train_one(name: str, seed: int, history: np.ndarray, future: np.ndarray) -> dict[str, float]:
    path = CHECKPOINTS / f"{DATASET}__{name}__seed-{seed}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    decisive.seed_everything(seed)
    model = make_model(name)
    parameters = decisive.count_parameters(model)
    capacity_gap = abs(parameters - TARGET_PARAMETERS) / TARGET_PARAMETERS
    if capacity_gap > 0.02:
        raise RuntimeError(f"{name} capacity gap is {capacity_gap:.3%}")
    seconds = decisive.mixture.train_model(model, history, future, seed)
    metadata = {
        "parameters": parameters,
        "capacity_relative_gap": capacity_gap,
        "train_seconds": seconds,
        "steps": 3_000,
        "batch_size": 512,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    print(f"trained {name} seed={seed}: {seconds:.1f}s parameters={parameters}", flush=True)
    return metadata


def load_model(name: str, seed: int) -> tuple[nn.Module, dict[str, float]]:
    if name == "mlp_k4":
        model = calibration.load_diagonal(DATASET, seed)
        parameters = decisive.count_parameters(model.model)
        return model, {
            "parameters": parameters,
            "capacity_relative_gap": abs(parameters - TARGET_PARAMETERS) / TARGET_PARAMETERS,
            "train_seconds": math.nan,
        }
    payload = torch.load(
        CHECKPOINTS / f"{DATASET}__{name}__seed-{seed}.pt", map_location="cpu", weights_only=False
    )
    model = make_model(name)
    model.load_state_dict(payload["state_dict"])
    return model, payload["metadata"]


def audit(frame: pd.DataFrame) -> dict[str, object]:
    existing = pd.read_csv(HERE / "outputs" / "evaluation_cells.csv")
    tactis_cells = existing[(existing.dataset == DATASET) & (existing.model == "tactis2")]
    tactis_crps = float(tactis_cells.macro_crps.mean())
    tactis_coverage = float(tactis_cells.coverage_error.mean())
    tactis_latency = float(tactis_cells.latency_ms_per_context.mean())
    means = frame.groupby("model").mean(numeric_only=True)
    baseline_cells = frame[frame.model == "mlp_k4"].set_index("seed")
    configurations: dict[str, object] = {}
    advancing = []
    for name in CONFIGURATIONS:
        cells = frame[frame.model == name].set_index("seed")
        paired_wins = int((cells.macro_crps < baseline_cells.macro_crps).sum()) if name != "mlp_k4" else 0
        speedup = tactis_latency / float(means.loc[name, "latency_ms_per_context"])
        gates = {
            "half_gap": float(means.loc[name, "macro_crps"]) <= ADVANCEMENT_CRPS,
            "paired_wins": paired_wins >= 2,
            "coverage": float(means.loc[name, "coverage_error"]) <= tactis_coverage + 0.03,
            "speed": speedup >= 100.0,
            "capacity": float(means.loc[name, "capacity_relative_gap"]) <= 0.02,
            "finite": bool(np.isfinite(cells.macro_crps).all()),
        }
        advances = name != "mlp_k4" and all(gates.values())
        if advances:
            advancing.append(name)
        configurations[name] = {
            "mean_crps": float(means.loc[name, "macro_crps"]),
            "coverage_error": float(means.loc[name, "coverage_error"]),
            "latency_ms_per_context": float(means.loc[name, "latency_ms_per_context"]),
            "speedup_vs_tactis": speedup,
            "parameters": int(round(means.loc[name, "parameters"])),
            "capacity_relative_gap": float(means.loc[name, "capacity_relative_gap"]),
            "paired_wins_vs_mlp_k4": paired_wins,
            "gates": gates,
            "advances": advances,
        }
    crps = {name: float(means.loc[name, "macro_crps"]) for name in CONFIGURATIONS}
    factor_effects = {
        "k8_minus_k4_mlp": crps["mlp_k8"] - crps["mlp_k4"],
        "k8_minus_k4_attention": crps["attention_k8"] - crps["attention_k4"],
        "attention_minus_mlp_k4": crps["attention_k4"] - crps["mlp_k4"],
        "attention_minus_mlp_k8": crps["attention_k8"] - crps["mlp_k8"],
    }
    result: dict[str, object] = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "advancement_crps": ADVANCEMENT_CRPS,
        "tactis_crps": tactis_crps,
        "tactis_coverage_error": tactis_coverage,
        "configurations": configurations,
        "factor_effects_crps": factor_effects,
        "advancing_configurations": advancing,
        "any_advance": bool(advancing),
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    digest = hashlib.sha256((HERE / "REPRESENTATION_PROTOCOL.md").read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol changed: expected {PROTOCOL_SHA256}, found {digest}")
    started = time.perf_counter()
    train_history, train_future, test_history, test_future = decisive.mixture.experiment.make_windows(DATASET)
    test_history = test_history[: decisive.EVAL_CONTEXTS]
    test_future = test_future[: decisive.EVAL_CONTEXTS]
    for name in CONFIGURATIONS[1:]:
        for seed in decisive.SEEDS:
            train_one(name, seed, train_history, train_future)
    rows, calibration_rows = [], []
    for name in CONFIGURATIONS:
        for seed in decisive.SEEDS:
            model, metadata = load_model(name, seed)
            temperature, before, after = lowrank.fit_temperature(DATASET, seed, model)
            calibration_rows.append(
                {
                    "model": name,
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
                    "dataset": DATASET,
                    "model": name,
                    "seed": seed,
                    "parameters": metadata["parameters"],
                    "capacity_relative_gap": metadata["capacity_relative_gap"],
                    "train_seconds": metadata["train_seconds"],
                    "temperature": temperature,
                    "latency_ms_per_context": latency,
                    **metrics,
                }
            )
            print(
                f"evaluated {name} seed={seed}: CRPS={metrics['macro_crps']:.4f} "
                f"coverage_error={metrics['coverage_error']:.4f} temperature={temperature:.3f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "evaluation_cells.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_cells.csv", index=False)
    frame.groupby("model", as_index=False).mean(numeric_only=True).to_csv(
        OUT / "evaluation_summary.csv", index=False
    )
    result = audit(frame)
    result["wall_seconds"] = time.perf_counter() - started
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
