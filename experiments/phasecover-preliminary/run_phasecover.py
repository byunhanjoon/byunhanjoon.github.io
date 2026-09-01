"""Run the frozen PhaseCover preliminary screen."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = RAW / "data"
OUTPUT = RAW / "cells"
CHECKPOINTS = RAW / "checkpoints"
PREDICTIONS = RAW / "predictions"
for directory in (DATA, OUTPUT, CHECKPOINTS, PREDICTIONS):
    directory.mkdir(parents=True, exist_ok=True)

CONTEXT = 96
HORIZON = 24
PATCH = 16
MODEL_SEEDS = (20261121, 20261122, 20261123)
IID_SEED = 20261201
PHASECOVER = (0, 4, 8, 12)
PROTOCOL_SHA256 = "ba5bade9069dbb9a02e00450044a26d69efd719d9dd3ee3bafd96f09526d1f44"
DATASETS = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "file": "ETTh1.csv",
        "kind": "csv",
        "target": "OT",
    },
    "Exchange": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        "file": "exchange_rate.txt.gz",
        "kind": "gz",
        "target": 0,
    },
    "Solar": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
        "file": "solar_AL.txt.gz",
        "kind": "gz",
        "target": "highest_train_variance",
    },
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 100:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    print(f"download {url}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, path)


def load_raw(name: str) -> tuple[np.ndarray, int]:
    config = DATASETS[name]
    path = DATA / str(config["file"])
    download(str(config["url"]), path)
    if config["kind"] == "csv":
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=np.float32)
        target = int(numeric.columns.get_loc(str(config["target"])))
    else:
        with gzip.open(path, "rt") as stream:
            values = np.loadtxt(stream, delimiter=",", dtype=np.float32)
        target = int(config["target"]) if config["target"] != "highest_train_variance" else -1
    if values.ndim != 2 or len(values) < 1000:
        raise AssertionError((name, values.shape))
    return values, target


def prepare(name: str) -> Path:
    output = DATA / f"{name}.npz"
    if output.exists():
        return output
    values, target = load_raw(name)
    train_end = int(0.60 * len(values))
    validation_end = int(0.80 * len(values))
    train = values[:train_end]
    means = np.nanmean(train, axis=0)
    values = np.where(np.isfinite(values), values, means[None])
    if target < 0:
        target = int(np.argmax(np.var(values[:train_end], axis=0)))
    scales = np.std(values[:train_end], axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    standardized = np.ascontiguousarray((values - means[None]) / scales[None], dtype=np.float32)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        values=standardized,
        target=np.asarray(target),
        train_end=np.asarray(train_end),
        validation_end=np.asarray(validation_end),
        raw_rows=np.asarray(len(values)),
        channels=np.asarray(values.shape[1]),
        target_mean=np.asarray(means[target]),
        target_scale=np.asarray(scales[target]),
    )
    os.replace(temporary, output)
    print(f"prepared {name}: rows={len(values)} channels={values.shape[1]} target={target}", flush=True)
    return output


def evenly_spaced(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, maximum).round().astype(np.int64)
    return values[positions]


class WindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, values: np.ndarray, starts: np.ndarray, target: int):
        self.values = torch.from_numpy(values)
        self.starts = torch.from_numpy(starts.astype(np.int64))
        self.target = target

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = int(self.starts[index])
        return (
            self.values[start - CONTEXT : start],
            self.values[start : start + HORIZON, self.target],
        )


def load_splits(name: str) -> tuple[dict[str, WindowDataset], dict[str, Any]]:
    payload = np.load(prepare(name))
    values = payload["values"]
    target = int(payload["target"])
    train_end = int(payload["train_end"])
    validation_end = int(payload["validation_end"])
    ranges = {
        "train": np.arange(CONTEXT, train_end - HORIZON + 1),
        "validation": np.arange(train_end, validation_end - HORIZON + 1),
        "test": np.arange(validation_end, len(values) - HORIZON + 1),
    }
    ranges["train"] = evenly_spaced(ranges["train"], 20_000)
    ranges["validation"] = evenly_spaced(ranges["validation"], 2_048)
    ranges["test"] = evenly_spaced(ranges["test"], 2_048)
    datasets = {part: WindowDataset(values, starts, target) for part, starts in ranges.items()}
    metadata = {
        "rows": int(payload["raw_rows"]),
        "channels": int(payload["channels"]),
        "target": target,
        "target_mean": float(payload["target_mean"]),
        "target_scale": float(payload["target_scale"]),
        **{f"n_{part}": len(dataset) for part, dataset in datasets.items()},
    }
    return datasets, metadata


def phase_pad(inputs: Tensor, phase: int) -> tuple[Tensor, Tensor]:
    if not 0 <= phase < PATCH:
        raise ValueError(phase)
    batch, length, channels = inputs.shape
    right = (-(phase + length)) % PATCH
    padded = inputs.new_zeros((batch, phase + length + right, channels))
    mask = inputs.new_zeros((batch, phase + length + right))
    padded[:, phase : phase + length] = inputs
    mask[:, phase : phase + length] = 1.0
    return padded, mask


class PhasePatchTransformer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        width = 96
        max_patches = (CONTEXT + 2 * PATCH - 2) // PATCH
        self.patch_projection = nn.Sequential(
            nn.Linear(PATCH * channels + PATCH, width),
            nn.LayerNorm(width),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, max_patches + 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(width))
        self.head = nn.Linear(width, HORIZON)

    def forward(self, inputs: Tensor, phase: int) -> Tensor:
        padded, mask = phase_pad(inputs, phase)
        batch, length, _ = padded.shape
        n_patches = length // PATCH
        value_patches = padded.reshape(batch, n_patches, PATCH * self.channels)
        mask_patches = mask.reshape(batch, n_patches, PATCH)
        tokens = self.patch_projection(torch.cat([value_patches, mask_patches], dim=-1))
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        return self.head(self.encoder(tokens)[:, 0])


def make_loader(dataset: Dataset, shuffle: bool, seed: int, batch_size: int = 256) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def loss_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, phase: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(inputs, phase)
            loss = nn.functional.mse_loss(prediction, targets, reduction="sum")
        total += float(loss)
        count += targets.numel()
    return total / count


def train_model(name: str, seed: int, device: torch.device) -> tuple[PhasePatchTransformer, dict[str, Any]]:
    datasets, metadata = load_splits(name)
    checkpoint = CHECKPOINTS / f"{name}__seed-{seed}.pt"
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PhasePatchTransformer(metadata["channels"]).to(device)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        return model, saved["fit"]
    train_loader = make_loader(datasets["train"], True, seed)
    validation_loader = make_loader(datasets["validation"], False, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    phase_rng = np.random.default_rng(seed + 991)
    best_loss = np.inf
    best_state = None
    stale = 0
    started = time.perf_counter()
    history = []
    for epoch in range(30):
        model.train()
        train_total, train_count = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phase = int(phase_rng.integers(0, PATCH))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(inputs, phase)
                loss = nn.functional.mse_loss(prediction, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss) * targets.numel()
            train_count += targets.numel()
        validation_loss = loss_on_loader(model, validation_loader, device, phase=0)
        history.append({"epoch": epoch + 1, "train_loss": train_total / train_count, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 5 and stale >= 5:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    fit = {
        "epochs": len(history),
        "best_validation_loss": float(best_loss),
        "wall_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    torch.save({"state_dict": best_state, "fit": fit, "metadata": metadata}, checkpoint)
    return model, fit


@torch.no_grad()
def predict_phases(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, False, seed)
    targets: list[Tensor] = []
    phase_outputs: list[list[Tensor]] = [[] for _ in range(PATCH)]
    model.eval()
    for inputs, batch_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets.append(batch_targets)
        for phase in range(PATCH):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                phase_outputs[phase].append(model(inputs, phase).float().cpu())
    return (
        torch.stack([torch.cat(parts) for parts in phase_outputs]).numpy(),
        torch.cat(targets).numpy(),
    )


def iid_subsets() -> list[tuple[int, ...]]:
    rng = np.random.default_rng(IID_SEED)
    return [tuple(sorted(rng.choice(PATCH, 4, replace=False).tolist())) for _ in range(64)]


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def evaluate(predictions: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    full = predictions.mean(axis=0)
    canonical = predictions[0]
    cover = predictions[list(PHASECOVER)].mean(axis=0)
    full_metrics = metrics(full, target)
    canonical_metrics = metrics(canonical, target)
    cover_metrics = metrics(cover, target)
    iid_rows = []
    for subset in iid_subsets():
        prediction = predictions[list(subset)].mean(axis=0)
        iid_rows.append({
            **metrics(prediction, target),
            "quotient_mse": float(np.mean(np.square(prediction - full))),
            "phases": list(subset),
        })
    iid_rmse = np.asarray([row["rmse"] for row in iid_rows])
    iid_mae = np.asarray([row["mae"] for row in iid_rows])
    iid_q = np.asarray([row["quotient_mse"] for row in iid_rows])
    phase_spread = float(np.sqrt(np.mean(np.square(predictions - full[None]))))
    return {
        "canonical": {
            **canonical_metrics,
            "quotient_mse": float(np.mean(np.square(canonical - full))),
            "phases": [0],
        },
        "iid4": {
            "rmse": float(iid_rmse.mean()),
            "rmse_sd": float(iid_rmse.std(ddof=1)),
            "rmse_q05": float(np.quantile(iid_rmse, 0.05)),
            "rmse_q95": float(np.quantile(iid_rmse, 0.95)),
            "mae": float(iid_mae.mean()),
            "quotient_mse": float(iid_q.mean()),
            "quotient_mse_sd": float(iid_q.std(ddof=1)),
            "subsets": iid_rows,
        },
        "phasecover4": {
            **cover_metrics,
            "quotient_mse": float(np.mean(np.square(cover - full))),
            "phases": list(PHASECOVER),
        },
        "full16": {**full_metrics, "quotient_mse": 0.0, "phases": list(range(PATCH))},
        "phase_spread_rms": phase_spread,
        "phase_materiality": phase_spread / max(canonical_metrics["rmse"], 1e-12),
    }


def integrity_check() -> dict[str, Any]:
    rng = np.random.default_rng(20261202)
    inputs = torch.from_numpy(rng.normal(size=(5, CONTEXT, 7)).astype(np.float32))
    maximum_error = 0.0
    mask_errors = 0
    for phase in range(PATCH):
        padded, mask = phase_pad(inputs, phase)
        reconstructed = padded[:, phase : phase + CONTEXT]
        maximum_error = max(maximum_error, float((reconstructed - inputs).abs().max()))
        mask_errors += int(not torch.equal(mask[:, phase : phase + CONTEXT], torch.ones_like(inputs[:, :, 0])))
        mask_errors += int(float(mask[:, :phase].sum()) != 0.0)
        mask_errors += int(float(mask[:, phase + CONTEXT :].sum()) != 0.0)
    subsets = iid_subsets()
    valid = all(len(set(subset)) == 4 and min(subset) >= 0 and max(subset) < PATCH for subset in subsets)
    return {
        "maximum_reconstruction_error": maximum_error,
        "mask_errors": mask_errors,
        "iid_subsets": len(subsets),
        "iid_subsets_valid": valid,
        "phasecover_unique_valid": len(set(PHASECOVER)) == 4 and min(PHASECOVER) >= 0 and max(PHASECOVER) < PATCH,
        "passed": maximum_error == 0.0 and mask_errors == 0 and valid,
    }


def run_cell(name: str, seed: int, device: torch.device) -> None:
    output = OUTPUT / f"{name}__seed-{seed}.json"
    if output.exists() and json.loads(output.read_text()).get("status") == "complete":
        return
    print(f"run {name} seed={seed} device={device}", flush=True)
    datasets, metadata = load_splits(name)
    model, fit = train_model(name, seed, device)
    predictions, target = predict_phases(model, datasets["test"], device, seed)
    prediction_path = PREDICTIONS / f"{name}__seed-{seed}.npz"
    np.savez_compressed(prediction_path, predictions=predictions, target=target)
    atomic_json(output, {
        "status": "complete",
        "dataset": name,
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "metadata": metadata,
        "fit": fit,
        "test": evaluate(predictions, target),
        "prediction_file": str(prediction_path.relative_to(HERE)),
    })
    print(f"complete {name} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "integrity"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        for name in DATASETS:
            prepare(name)
        return
    if args.stage == "integrity":
        audit = integrity_check()
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["passed"]:
            raise SystemExit(1)
        return
    cells = [(name, seed) for name in DATASETS for seed in MODEL_SEEDS]
    device = torch.device(args.device)
    for index, (name, seed) in enumerate(cells):
        if index % args.n_shards == args.shard:
            run_cell(name, seed, device)


if __name__ == "__main__":
    main()
