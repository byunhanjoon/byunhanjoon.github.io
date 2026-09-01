"""Run the frozen PhaseCover published-backbone confirmation."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTForPrediction


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = RAW / "data"
CELLS = RAW / "cells"
CHECKPOINTS = RAW / "checkpoints"
PREDICTIONS = RAW / "predictions"
for directory in (DATA, CELLS, CHECKPOINTS, PREDICTIONS):
    directory.mkdir(parents=True, exist_ok=True)

OBS_CONTEXT = 505
MODEL_CONTEXT = 512
HORIZON = 24
PATCH = 8
CHANNELS = 8
PHASECOVER = (0, 2, 4, 6)
MODEL_SEEDS = (20261211, 20261212)
BACKBONES = ("patchtst", "moment")
TRAIN_MODES = ("canonical_train", "phase_aug_train")
MOMENT_ID = "AutonLab/MOMENT-1-small"
MOMENT_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
PROTOCOL_SHA256 = "020f883bda2a4cbe02d6406eb199255b58da34b7999a28775bb4ac40fa582826"
DATASETS = {
    "JenaWeather": {
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        "file": "jena_climate_2009_2016.csv.zip",
        "kind": "zip_csv",
    },
    "Electricity": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        "file": "electricity.txt.gz",
        "kind": "gz",
    },
    "Traffic": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        "file": "traffic.txt.gz",
        "kind": "gz",
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


def load_raw(name: str) -> np.ndarray:
    config = DATASETS[name]
    path = DATA / str(config["file"])
    download(str(config["url"]), path)
    if config["kind"] == "zip_csv":
        frame = pd.read_csv(path, compression="zip")
        values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        # Jena encodes missing wind-speed readings as the finite sentinel -9999.
        values[values <= -999.0] = np.nan
    else:
        with gzip.open(path, "rt") as stream:
            values = np.loadtxt(stream, delimiter=",", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < CHANNELS or len(values) < 10_000:
        raise AssertionError((name, values.shape))
    return values


def prepare(name: str) -> Path:
    output = DATA / f"{name}.npz"
    if output.exists():
        return output
    raw = load_raw(name)
    channel_indices = np.rint(np.linspace(0, raw.shape[1] - 1, CHANNELS)).astype(np.int64)
    if len(np.unique(channel_indices)) != CHANNELS:
        raise AssertionError(channel_indices)
    values = raw[:, channel_indices]
    train_end = int(0.60 * len(values))
    validation_end = int(0.80 * len(values))
    means = np.nanmean(values[:train_end], axis=0)
    values = np.where(np.isfinite(values), values, means[None])
    scales = np.std(values[:train_end], axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    standardized = np.ascontiguousarray((values - means[None]) / scales[None], dtype=np.float32)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        values=standardized,
        train_end=np.asarray(train_end),
        validation_end=np.asarray(validation_end),
        raw_rows=np.asarray(len(raw)),
        raw_channels=np.asarray(raw.shape[1]),
        selected_channels=channel_indices,
        means=means,
        scales=scales,
    )
    os.replace(temporary, output)
    print(
        f"prepared {name}: rows={len(raw)} raw_channels={raw.shape[1]} selected={channel_indices.tolist()}",
        flush=True,
    )
    return output


def evenly_spaced(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, maximum)).astype(np.int64)
    return values[indices]


class WindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, values: np.ndarray, starts: np.ndarray):
        self.values = torch.from_numpy(values)
        self.starts = torch.from_numpy(starts.astype(np.int64))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = int(self.starts[index])
        return self.values[start - OBS_CONTEXT : start], self.values[start : start + HORIZON]


def load_splits(name: str) -> tuple[dict[str, WindowDataset], dict[str, Any]]:
    payload = np.load(prepare(name))
    values = payload["values"]
    train_end = int(payload["train_end"])
    validation_end = int(payload["validation_end"])
    starts = {
        "train": np.arange(OBS_CONTEXT, train_end - HORIZON + 1),
        "validation": np.arange(train_end, validation_end - HORIZON + 1),
        "test": np.arange(validation_end, len(values) - HORIZON + 1),
    }
    starts["train"] = evenly_spaced(starts["train"], 8_192)
    starts["validation"] = evenly_spaced(starts["validation"], 1_024)
    starts["test"] = evenly_spaced(starts["test"], 1_024)
    datasets = {part: WindowDataset(values, indices) for part, indices in starts.items()}
    metadata = {
        "rows": int(payload["raw_rows"]),
        "raw_channels": int(payload["raw_channels"]),
        "selected_channels": payload["selected_channels"].tolist(),
        **{f"n_{part}": len(dataset) for part, dataset in datasets.items()},
    }
    return datasets, metadata


def phase_fill(inputs: Tensor, phase: int) -> Tensor:
    if inputs.ndim != 3 or inputs.shape[1:] != (OBS_CONTEXT, CHANNELS):
        raise ValueError(inputs.shape)
    if not 0 <= phase < PATCH:
        raise ValueError(phase)
    output = inputs.new_zeros((inputs.shape[0], MODEL_CONTEXT, CHANNELS))
    output[:, phase : phase + OBS_CONTEXT] = inputs
    return output


class ForecastModel(nn.Module):
    def __init__(self, backbone: str, seed: int):
        super().__init__()
        self.backbone = backbone
        torch.manual_seed(seed)
        if backbone == "patchtst":
            config = PatchTSTConfig(
                num_input_channels=CHANNELS,
                context_length=MODEL_CONTEXT,
                prediction_length=HORIZON,
                patch_length=PATCH,
                patch_stride=PATCH,
                d_model=128,
                num_hidden_layers=3,
                num_attention_heads=4,
                ffn_dim=256,
                head_dropout=0.1,
                loss="mse",
                scaling="std",
            )
            self.model = PatchTSTForPrediction(config)
        elif backbone == "moment":
            self.model = MOMENTPipeline.from_pretrained(
                MOMENT_ID,
                revision=MOMENT_REVISION,
                model_kwargs={"task_name": "forecasting", "forecast_horizon": HORIZON},
            )
            self.model.init()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            for parameter in self.model.head.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(backbone)

    def train(self, mode: bool = True) -> "ForecastModel":
        super().train(mode)
        if self.backbone == "moment":
            self.model.normalizer.eval()
            self.model.tokenizer.eval()
            self.model.patch_embedding.eval()
            self.model.encoder.eval()
            self.model.head.train(mode)
        return self

    def forward(self, inputs: Tensor, phase: int) -> Tensor:
        padded = phase_fill(inputs, phase)
        if self.backbone == "patchtst":
            observed = torch.ones_like(padded, dtype=torch.bool)
            output = self.model(past_values=padded, past_observed_mask=observed)
            return output.prediction_outputs
        input_mask = torch.ones((len(inputs), MODEL_CONTEXT), device=inputs.device, dtype=torch.long)
        output = self.model(x_enc=padded.transpose(1, 2), input_mask=input_mask)
        return output.forecast.transpose(1, 2)

    def checkpoint_state(self) -> dict[str, Tensor]:
        module = self.model if self.backbone == "patchtst" else self.model.head
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def load_checkpoint_state(self, state: dict[str, Tensor]) -> None:
        module = self.model if self.backbone == "patchtst" else self.model.head
        module.load_state_dict(state)


def make_loader(dataset: Dataset, shuffle: bool, seed: int, batch_size: int = 128) -> DataLoader:
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
def loss_on_loader(model: ForecastModel, loader: DataLoader, device: torch.device, phase: int) -> float:
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


def train_model(
    dataset_name: str,
    backbone: str,
    train_mode: str,
    seed: int,
    device: torch.device,
) -> tuple[ForecastModel, dict[str, Any]]:
    datasets, _ = load_splits(dataset_name)
    checkpoint = CHECKPOINTS / f"{dataset_name}__{backbone}__{train_mode}__seed-{seed}.pt"
    model = ForecastModel(backbone, seed).to(device)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_checkpoint_state(saved["state_dict"])
        return model, saved["fit"]
    train_loader = make_loader(datasets["train"], True, seed)
    validation_loader = make_loader(datasets["validation"], False, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 8e-4 if backbone == "patchtst" else 1e-3
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    phase_rng = np.random.default_rng(seed + 701)
    best_loss = np.inf
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(12):
        model.train()
        train_total, train_count = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phase = 0 if train_mode == "canonical_train" else int(phase_rng.integers(0, PATCH))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(inputs, phase)
                loss = nn.functional.mse_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError((dataset_name, backbone, train_mode, seed, epoch))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            train_total += float(loss) * targets.numel()
            train_count += targets.numel()
        validation_loss = loss_on_loader(model, validation_loader, device, phase=0)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_total / train_count,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = model.checkpoint_state()
            stale = 0
        else:
            stale += 1
        if epoch >= 3 and stale >= 3:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_checkpoint_state(best_state)
    fit = {
        "epochs": len(history),
        "best_validation_loss": float(best_loss),
        "wall_seconds": time.perf_counter() - started,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in parameters),
        "history": history,
    }
    torch.save({"state_dict": best_state, "fit": fit}, checkpoint)
    return model, fit


@torch.no_grad()
def predict_phases(
    model: ForecastModel,
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
    return torch.stack([torch.cat(parts) for parts in phase_outputs]).numpy(), torch.cat(targets).numpy()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def evaluate(predictions: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    full = predictions.mean(axis=0)
    canonical = predictions[0]
    cover = predictions[list(PHASECOVER)].mean(axis=0)
    designs = []
    for phases in combinations(range(PATCH), 4):
        prediction = predictions[list(phases)].mean(axis=0)
        designs.append({
            "phases": list(phases),
            **metric(prediction, target),
            "quotient_mse": float(np.mean(np.square(prediction - full))),
        })
    iid_rmse = np.asarray([row["rmse"] for row in designs])
    iid_mae = np.asarray([row["mae"] for row in designs])
    iid_q = np.asarray([row["quotient_mse"] for row in designs])
    cover_metrics = metric(cover, target)
    cover_q = float(np.mean(np.square(cover - full)))
    canonical_metrics = metric(canonical, target)
    phase_spread = float(np.sqrt(np.mean(np.square(predictions - full[None]))))
    return {
        "canonical": {
            **canonical_metrics,
            "quotient_mse": float(np.mean(np.square(canonical - full))),
            "phases": [0],
        },
        "exact_iid4": {
            "rmse": float(iid_rmse.mean()),
            "rmse_sd": float(iid_rmse.std(ddof=0)),
            "mae": float(iid_mae.mean()),
            "quotient_mse": float(iid_q.mean()),
            "designs": designs,
        },
        "phasecover4": {
            **cover_metrics,
            "quotient_mse": cover_q,
            "phases": list(PHASECOVER),
            "better_quotient_fraction": float(np.mean(iid_q > cover_q)),
            "better_rmse_fraction": float(np.mean(iid_rmse > cover_metrics["rmse"])),
        },
        "full8": {**metric(full, target), "quotient_mse": 0.0, "phases": list(range(PATCH))},
        "phase_spread_rms": phase_spread,
        "phase_materiality": phase_spread / max(canonical_metrics["rmse"], 1e-12),
    }


def integrity_check() -> dict[str, Any]:
    rng = np.random.default_rng(20261213)
    inputs = torch.from_numpy(rng.normal(size=(3, OBS_CONTEXT, CHANNELS)).astype(np.float32))
    maximum_error = 0.0
    fill_errors = 0
    for phase in range(PATCH):
        padded = phase_fill(inputs, phase)
        reconstructed = padded[:, phase : phase + OBS_CONTEXT]
        maximum_error = max(maximum_error, float((reconstructed - inputs).abs().max()))
        fill_errors += int(float(padded[:, :phase].abs().sum()) != 0.0)
        fill_errors += int(float(padded[:, phase + OBS_CONTEXT :].abs().sum()) != 0.0)
    designs = list(combinations(range(PATCH), 4))
    designs_valid = len(designs) == 70 and all(len(set(item)) == 4 for item in designs)
    return {
        "maximum_reconstruction_error": maximum_error,
        "boundary_fill_errors": fill_errors,
        "designs": len(designs),
        "designs_valid": designs_valid,
        "phasecover_valid": len(set(PHASECOVER)) == 4 and min(PHASECOVER) >= 0 and max(PHASECOVER) < PATCH,
        "passed": maximum_error == 0.0 and fill_errors == 0 and designs_valid,
    }


def run_cell(dataset_name: str, backbone: str, train_mode: str, seed: int, device: torch.device) -> None:
    stem = f"{dataset_name}__{backbone}__{train_mode}__seed-{seed}"
    output = CELLS / f"{stem}.json"
    if output.exists() and json.loads(output.read_text()).get("status") == "complete":
        return
    print(f"run {stem} device={device}", flush=True)
    datasets, metadata = load_splits(dataset_name)
    model, fit = train_model(dataset_name, backbone, train_mode, seed, device)
    predictions, target = predict_phases(model, datasets["test"], device, seed)
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise FloatingPointError(stem)
    prediction_path = PREDICTIONS / f"{stem}.npz"
    np.savez_compressed(prediction_path, predictions=predictions, target=target)
    atomic_json(output, {
        "status": "complete",
        "dataset": dataset_name,
        "backbone": backbone,
        "train_mode": train_mode,
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "metadata": metadata,
        "fit": fit,
        "test": evaluate(predictions, target),
        "prediction_file": str(prediction_path.relative_to(HERE)),
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "momentfm": "0.1.4",
            "moment_model": MOMENT_ID,
            "moment_revision": MOMENT_REVISION,
        },
    })
    print(f"complete {stem} epochs={fit['epochs']} seconds={fit['wall_seconds']:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "integrity"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        for dataset_name in DATASETS:
            prepare(dataset_name)
        return
    if args.stage == "integrity":
        audit = integrity_check()
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["passed"]:
            raise SystemExit(1)
        return
    cells = [
        (dataset_name, backbone, train_mode, seed)
        for dataset_name in DATASETS
        for backbone in BACKBONES
        for train_mode in TRAIN_MODES
        for seed in MODEL_SEEDS
    ]
    device = torch.device(args.device)
    for index, cell in enumerate(cells):
        if index % args.n_shards == args.shard:
            run_cell(*cell, device)


if __name__ == "__main__":
    main()
