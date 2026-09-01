"""Learned-consistency follow-up across lossless longitudinal views."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, dst, idct, idst
from scipy.linalg import hadamard
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "phasecover-confirmation" / "raw" / "data"
OUT = HERE / "view_consistency"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL_SHA256 = "831ca4517303c86bde13d4211b6b2dc33f1a59e6e60933805713b078e9c299ee"
DATASETS = ("JenaWeather", "Electricity", "Traffic")
SEEDS = (20261311, 20261312, 20261313)
TRAIN_VIEWS = ("levels", "reverse", "dct", "differences")
HELDOUT_VIEWS = ("dst", "hadamard", "even_odd")
ALL_VIEWS = TRAIN_VIEWS + HELDOUT_VIEWS
METHODS = ("levels_erm", "view_aug", "view_consistent")
DEVICE = torch.device("cuda:0")
HADAMARD = hadamard(32).astype(np.float32) / np.sqrt(32.0)
EVEN_ODD = np.concatenate([np.arange(0, 32, 2), np.arange(1, 32, 2)])
EVEN_ODD_INVERSE = np.argsort(EVEN_ODD)


def make_examples(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(SOURCE / f"{name}.npz")
    values = payload["values"].astype(np.float32)
    train_end = int(payload["train_end"])
    validation_end = int(payload["validation_end"])

    def build(first: int, last: int, starts_count: int) -> tuple[np.ndarray, np.ndarray]:
        available = np.arange(max(32, first), last)
        positions = np.rint(np.linspace(0, len(available) - 1, starts_count)).astype(np.int64)
        starts = available[positions]
        histories = np.stack([values[start - 32 : start] for start in starts])
        x = histories.transpose(0, 2, 1).reshape(-1, 32)
        y = values[starts].reshape(-1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    x_train, y_train = build(32, train_end, 512)
    x_test, y_test = build(validation_end, len(values), 256)
    return x_train, y_train, x_test, y_test


def transform(x: np.ndarray, view: str) -> np.ndarray:
    if view == "levels":
        output = x.copy()
    elif view == "reverse":
        output = x[:, ::-1]
    elif view == "dct":
        output = dct(x, type=2, axis=1, norm="ortho")
    elif view == "differences":
        output = np.concatenate([x[:, :1], np.diff(x, axis=1)], axis=1)
    elif view == "dst":
        output = dst(x, type=2, axis=1, norm="ortho")
    elif view == "hadamard":
        output = x @ HADAMARD.T
    elif view == "even_odd":
        output = x[:, EVEN_ODD]
    else:
        raise ValueError(view)
    return np.ascontiguousarray(output, dtype=np.float32)


def inverse(x: np.ndarray, view: str) -> np.ndarray:
    if view == "levels":
        output = x.copy()
    elif view == "reverse":
        output = x[:, ::-1]
    elif view == "dct":
        output = idct(x, type=2, axis=1, norm="ortho")
    elif view == "differences":
        output = np.cumsum(x, axis=1)
    elif view == "dst":
        output = idst(x, type=2, axis=1, norm="ortho")
    elif view == "hadamard":
        output = x @ HADAMARD
    elif view == "even_odd":
        output = x[:, EVEN_ODD_INVERSE]
    else:
        raise ValueError(view)
    return np.ascontiguousarray(output, dtype=np.float32)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


def train_model(
    method: str,
    train_views: dict[str, np.ndarray],
    target: np.ndarray,
    seed: int,
) -> tuple[Predictor, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 19)
    model = Predictor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    target_tensor = torch.from_numpy(target).to(DEVICE)
    inputs = {name: torch.from_numpy(value).to(DEVICE) for name, value in train_views.items()}
    started = time.perf_counter()
    model.train()
    for _ in range(3_000):
        indices_np = rng.integers(0, len(target), size=512)
        indices = torch.from_numpy(indices_np).to(DEVICE)
        y = target_tensor[indices]
        optimizer.zero_grad(set_to_none=True)
        if method == "levels_erm":
            predictions = model(inputs["levels"][indices])[None]
        else:
            predictions = torch.stack([model(inputs[view][indices]) for view in TRAIN_VIEWS])
        supervised = torch.mean((predictions - y[None]) ** 2)
        consistency = torch.mean((predictions - predictions.mean(dim=0, keepdim=True)) ** 2)
        loss = supervised + (consistency if method == "view_consistent" else 0.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if time.perf_counter() - started > 30 * 60:
            raise TimeoutError("one view-consistency cell exceeded 30 minutes")
    return model, time.perf_counter() - started


@torch.no_grad()
def predict(model: Predictor, x: np.ndarray) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(x), 2_048):
        batch = torch.from_numpy(x[start : start + 2_048]).to(DEVICE)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    overall_started = time.perf_counter()
    rows = []
    dispersion_rows = []
    maximum_roundtrip_error = 0.0
    for dataset in DATASETS:
        x_train, y_train, x_test, y_test = make_examples(dataset)
        normalized_train = {}
        normalized_test = {}
        normalization = {}
        transformed_test = {}
        for view in ALL_VIEWS:
            train_view = transform(x_train, view)
            test_view = transform(x_test, view)
            transformed_test[view] = test_view
            maximum_roundtrip_error = max(
                maximum_roundtrip_error,
                float(np.max(np.abs(inverse(train_view, view) - x_train))),
                float(np.max(np.abs(inverse(test_view, view) - x_test))),
            )
            mean = train_view.mean(axis=0, keepdims=True)
            scale = train_view.std(axis=0, keepdims=True)
            scale[scale < 1e-6] = 1.0
            normalization[view] = (mean, scale)
            normalized_train[view] = np.ascontiguousarray((train_view - mean) / scale, dtype=np.float32)
            normalized_test[view] = np.ascontiguousarray((test_view - mean) / scale, dtype=np.float32)

        for seed in SEEDS:
            trained = {}
            train_seconds = {}
            for method in METHODS:
                model, seconds = train_model(method, normalized_train, y_train, seed)
                trained[method] = model
                train_seconds[method] = seconds
                torch.save(model.state_dict(), OUT / f"{dataset}__{method}__seed-{seed}.pt")

            predictions_by_method = {}
            for method, model in trained.items():
                predictions_by_method[method] = {}
                for view in ALL_VIEWS:
                    prediction = predict(model, normalized_test[view])
                    predictions_by_method[method][view] = prediction
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": method,
                            "view": view,
                            "view_split": "seen" if view in TRAIN_VIEWS else "heldout",
                            "rmse": float(np.sqrt(np.mean((prediction - y_test) ** 2))),
                            "train_seconds": train_seconds[method],
                        }
                    )

            levels_model = trained["levels_erm"]
            levels_mean, levels_scale = normalization["levels"]
            oracle_predictions = {}
            for view in ALL_VIEWS:
                canonical = inverse(transformed_test[view], view)
                canonical_normalized = np.ascontiguousarray((canonical - levels_mean) / levels_scale, dtype=np.float32)
                prediction = predict(levels_model, canonical_normalized)
                oracle_predictions[view] = prediction
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": "oracle_canonical",
                        "view": view,
                        "view_split": "seen" if view in TRAIN_VIEWS else "heldout",
                        "rmse": float(np.sqrt(np.mean((prediction - y_test) ** 2))),
                        "train_seconds": train_seconds["levels_erm"],
                    }
                )

            for method, view_predictions in {**predictions_by_method, "oracle_canonical": oracle_predictions}.items():
                stack = np.stack([view_predictions[view] for view in ALL_VIEWS])
                center = stack.mean(axis=0)
                dispersion_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": method,
                        "prediction_dispersion": float(np.sqrt(np.mean((stack - center[None]) ** 2))),
                    }
                )
            if time.perf_counter() - overall_started > 2 * 60 * 60:
                raise TimeoutError("view-consistency follow-up exceeded two hours")

    frame = pd.DataFrame(rows)
    dispersion = pd.DataFrame(dispersion_rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    dispersion.to_csv(OUT / "dispersion.csv", index=False)
    summary = frame.groupby(["dataset", "model", "view", "view_split"], as_index=False).agg(
        rmse=("rmse", "mean"), train_seconds=("train_seconds", "mean")
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    gates = {}
    seen_passes = 0
    heldout_passes = 0
    canonical_passes = 0
    for dataset in DATASETS:
        group = summary[summary.dataset == dataset]
        augmented = group[group.model == "view_aug"].set_index("view")
        consistent = group[group.model == "view_consistent"].set_index("view")
        levels = group[group.model == "levels_erm"].set_index("view")
        oracle = group[group.model == "oracle_canonical"].set_index("view")
        augmented_seen = float(augmented.loc[list(TRAIN_VIEWS), "rmse"].max())
        consistent_seen = float(consistent.loc[list(TRAIN_VIEWS), "rmse"].max())
        seen_reduction = 1.0 - consistent_seen / augmented_seen
        augmented_heldout = float(augmented.loc[list(HELDOUT_VIEWS), "rmse"].mean())
        consistent_heldout = float(consistent.loc[list(HELDOUT_VIEWS), "rmse"].mean())
        heldout_reduction = 1.0 - consistent_heldout / augmented_heldout
        canonical_degradation = float(
            consistent.loc["levels", "rmse"] / levels.loc["levels", "rmse"] - 1.0
        )
        oracle_heldout = float(oracle.loc[list(HELDOUT_VIEWS), "rmse"].mean())
        oracle_gap = consistent_heldout / oracle_heldout - 1.0
        seen_passes += seen_reduction >= 0.10
        heldout_passes += heldout_reduction >= 0.05
        canonical_passes += canonical_degradation <= 0.02
        gates[dataset] = {
            "seen_worst_rmse_reduction": seen_reduction,
            "heldout_mean_rmse_reduction": heldout_reduction,
            "canonical_rmse_degradation": canonical_degradation,
            "heldout_oracle_gap": oracle_gap,
        }

    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - overall_started,
        "cells": len(frame),
        "maximum_roundtrip_error": maximum_roundtrip_error,
        "seen_gate_datasets": int(seen_passes),
        "heldout_gate_datasets": int(heldout_passes),
        "canonical_gate_datasets": int(canonical_passes),
        "gates": gates,
    }
    audit["passed"] = bool(seen_passes >= 2 and heldout_passes >= 2 and canonical_passes == 3)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
