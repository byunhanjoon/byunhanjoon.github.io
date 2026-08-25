"""Parameter-efficient ensembles over exactly equivalent tabular bases.

The primary model keeps TabM's trainable parameter count fixed while giving
different ensemble members cumulative/Helmert or local/adjacent coordinates.
All basis maps are fit blockwise on the training partition only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import tabm
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .broad_data import paired_natural_representations
from .broad_extension_data import load_extension_dataset
from .broad_models import member_loss
from .core import PARTS, Prepared, loss_numpy, make_prepared, metric
from .broad_data import load_broad_dataset


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/orbit_ensemble_preregistered.json"
RESULTS = ROOT / "results/day3/orbit_ensemble"


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _identity_orbits(dimension: int, k: int) -> np.ndarray:
    return np.repeat(np.eye(dimension, dtype=np.float64)[None, :, :], k, axis=0)


def natural_orbits(transform: np.ndarray, assignments: list[str]) -> np.ndarray:
    """Build cumulative/local member maps in reference coordinates."""

    identity = np.eye(transform.shape[0], dtype=np.float64)
    if transform.shape[0] != transform.shape[1]:
        raise ValueError("Natural orbit transform must be square")
    if np.linalg.matrix_rank(transform, tol=1e-10) != transform.shape[0]:
        raise np.linalg.LinAlgError("Natural orbit transform must be invertible")
    lookup = {"cumulative": identity, "local": transform}
    return np.stack([lookup[name] for name in assignments])


def random_orthogonal_orbits(dimension: int, k: int, seed: int) -> np.ndarray:
    """Return an identity member followed by deterministic Haar rotations."""

    rng = np.random.default_rng(seed)
    transforms = [np.eye(dimension, dtype=np.float64)]
    for _ in range(1, k):
        q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        transforms.append(q * signs[None, :])
    return np.stack(transforms)


class OrbitDenseStemTabM(nn.Module):
    """Dense-stem TabM accepting ordinary or member-specific input charts."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        transforms: np.ndarray,
        *,
        member_views: bool,
        latent_size: int = 64,
        n_blocks: int = 2,
        d_block: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if transforms.ndim != 3 or transforms.shape[1:] != (input_size, input_size):
            raise ValueError("Orbit transforms must have shape (k, input_size, input_size)")
        self.member_views = member_views
        self.register_buffer("orbit_transforms", torch.from_numpy(transforms.astype(np.float32)))
        self.first = nn.Linear(input_size, latent_size)
        self.backbone = tabm.TabM.make(
            n_num_features=latent_size,
            cat_cardinalities=[],
            d_out=output_size,
            num_embeddings=None,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=len(transforms),
        )

    def member_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,kde->bke", x, self.orbit_transforms)

    def forward_members_ordinary(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.first(x), None)

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        if not self.member_views:
            return self.forward_members_ordinary(x)
        return self.backbone(self.first(self.member_inputs(x)), None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=1)


def ensemble_logits(members: torch.Tensor, task: str) -> torch.Tensor:
    if task == "binclass":
        probability = members.squeeze(-1).sigmoid().mean(dim=1)
        return torch.logit(probability.clamp(1e-7, 1 - 1e-7))[:, None]
    if task == "multiclass":
        probability = members.softmax(dim=-1).mean(dim=1)
        return probability.clamp_min(1e-12).log()
    return members.mean(dim=1)


def _predict_members(
    model: OrbitDenseStemTabM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start : start + batch_size]).to(device)
            output.append(model.forward_members(tensor).float().cpu().numpy())
    return np.concatenate(output)


def _ensemble_numpy(members: np.ndarray, task: str) -> np.ndarray:
    if task == "binclass":
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(members[..., 0], -40, 40)))
        mean = np.clip(probabilities.mean(axis=1), 1e-7, 1 - 1e-7)
        return np.log(mean / (1 - mean))[:, None]
    if task == "multiclass":
        shifted = members - members.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        mean = np.clip(probabilities.mean(axis=1), 1e-12, None)
        return np.log(mean)
    return members.mean(axis=1)


def _member_proper_losses(task: str, members: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray([loss_numpy(task, members[:, index], y) for index in range(members.shape[1])])


def _diversity_values(task: str, members: np.ndarray) -> tuple[float, float]:
    if task == "binclass":
        values = 1.0 / (1.0 + np.exp(-np.clip(members[..., 0], -40, 40)))
    elif task == "multiclass":
        shifted = members - members.max(axis=-1, keepdims=True)
        values = np.exp(shifted)
        values /= values.sum(axis=-1, keepdims=True)
        values = values.reshape(len(values), values.shape[1], -1)
        values = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    else:
        values = members[..., 0]
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[1])
    disagreement = float(np.sqrt(np.mean((values - values.mean(axis=1, keepdims=True)) ** 2)))
    correlation = np.corrcoef(values, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    finite = upper[np.isfinite(upper)]
    return disagreement, float(finite.mean()) if len(finite) else 1.0


def _evaluate(data: Prepared, members: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, float]:
    prediction = _ensemble_numpy(members, data.task)
    member_losses = _member_proper_losses(data.task, members, y)
    disagreement, correlation = _diversity_values(data.task, members)
    output = {
        f"{prefix}_proper_loss": loss_numpy(data.task, prediction, y),
        f"{prefix}_metric": metric(data, prediction, y),
        f"{prefix}_member_loss_mean": float(member_losses.mean()),
        f"{prefix}_member_loss_std": float(member_losses.std(ddof=1)),
        f"{prefix}_ensemble_gain_vs_mean_member": float(
            (member_losses.mean() - loss_numpy(data.task, prediction, y))
            / max(member_losses.mean(), 1e-30)
        ),
        f"{prefix}_member_disagreement": disagreement,
        f"{prefix}_member_prediction_correlation": correlation,
    }
    if data.task == "binclass":
        output[f"{prefix}_roc_auc"] = float(roc_auc_score(y, prediction.reshape(-1)))
    if data.task == "regression":
        output[f"{prefix}_rmse"] = output[f"{prefix}_metric"]
    else:
        output[f"{prefix}_accuracy"] = output[f"{prefix}_metric"]
    return output


def train_cell(
    data: Prepared,
    transforms: np.ndarray,
    *,
    member_views: bool,
    seed: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cfg = config()["training"]
    _seed(seed)
    resolved = torch.device(device)
    output_size = data.n_classes if data.task == "multiclass" else 1
    model = OrbitDenseStemTabM(
        data.x["train"].shape[1],
        output_size,
        transforms,
        member_views=member_views,
        latent_size=int(cfg["latent_size"]),
        n_blocks=int(cfg["n_blocks"]),
        d_block=int(cfg["d_block"]),
        dropout=float(cfg["dropout"]),
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    batch_size = int(cfg["batch_size"])
    if len(data.x["train"]) >= int(cfg["large_dataset_threshold"]):
        batch_size = int(cfg["large_batch_size"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 70000),
        pin_memory=resolved.type == "cuda",
    )
    memory_valid = True
    if resolved.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(resolved.index)
        except RuntimeError:
            memory_valid = False
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    curves: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        total = 0.0
        count = 0
        for features, target in loader:
            features = features.to(resolved, non_blocking=True)
            target = target.to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = member_loss(model, features, target, data.task)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(features)
            count += len(features)
        val_members = _predict_members(model, data.x["val"], resolved, batch_size * 2)
        val_prediction = _ensemble_numpy(val_members, data.task)
        val_loss = loss_numpy(data.task, val_prediction, data.y["val"])
        curves.append({"epoch": epoch, "train_member_loss": total / count, "val_proper_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale > int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    result: dict[str, object] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "train_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(resolved.index))
            if resolved.type == "cuda" and memory_valid
            else math.nan
        ),
        "peak_memory_observation_valid": memory_valid,
    }
    for part in ("val", "test"):
        members = _predict_members(model, data.x[part], resolved, batch_size * 2)
        result.update(_evaluate(data, members, data.y[part], part))
    return result, curves


def _load_dataset(name: str):
    if name.endswith("_extension"):
        return load_extension_dataset(name)
    return load_broad_dataset(name)


def _stage_datasets(stage: str, cfg: dict[str, object]) -> list[str]:
    if stage == "screen":
        return list(cfg["screen"]["datasets"])
    confirmation = cfg["confirmation"]
    return list(confirmation["broad_datasets"]) + list(confirmation["extension_datasets"])


def run(args: argparse.Namespace) -> None:
    cfg = config()
    datasets = args.datasets or _stage_datasets(args.stage, cfg)
    arms = args.arms or list(cfg[args.stage]["arms"])
    seeds = args.seeds or list(cfg[args.stage]["seeds"])
    rows = _read(args.output)
    curves_path = args.output.with_name(args.output.stem + "_curves.csv")
    curve_rows = _read(curves_path)
    completed = {
        (row["stage"], row["dataset"], row["arm"], int(row["seed"]))
        for row in rows
        if not row.get("failure", "").strip()
    }
    assignments = list(cfg["natural_member_assignment"])
    k = int(cfg["ensemble_size"])
    if len(assignments) != k:
        raise ValueError("Natural member assignment does not match ensemble size")
    selected_datasets = [
        name for index, name in enumerate(datasets) if index % args.num_shards == args.shard
    ]
    for dataset_name in selected_datasets:
        dataset = _load_dataset(dataset_name)
        reference, changed, transform = paired_natural_representations(dataset)
        relation_error = max(changed.metadata["basis_relation_errors"].values())
        threshold = float(cfg["integrity"]["maximum_basis_relation_error"])
        if relation_error > threshold:
            raise AssertionError(f"{dataset_name}: basis relation error {relation_error} > {threshold}")
        dimension = reference.parts["train"].shape[1]
        identity = _identity_orbits(dimension, k)
        natural = natural_orbits(transform, assignments)
        random_orbits = random_orthogonal_orbits(
            dimension, k, int(cfg["random_orbit_seed"]) + datasets.index(dataset_name)
        )
        specifications = {
            "cumulative": (reference.parts, identity, False),
            "local": (changed.parts, identity, False),
            "orbit_natural": (reference.parts, natural, True),
            "orbit_random": (reference.parts, random_orbits, True),
        }
        for arm in arms:
            parts, transforms, member_views = specifications[arm]
            prepared = make_prepared(dataset, parts, {})
            for seed in seeds:
                key = (args.stage, dataset_name, arm, int(seed))
                if key in completed:
                    continue
                failure = ""
                try:
                    fit, curves = train_cell(
                        prepared,
                        transforms,
                        member_views=member_views,
                        seed=int(seed),
                        device=args.device,
                    )
                except Exception as error:  # failures are recorded scientific outcomes
                    failure = f"{type(error).__name__}: {error}"
                    fit, curves = {}, []
                row = {
                    "experiment": "equivalent_basis_orbit_ensemble",
                    "stage": args.stage,
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "split_fingerprint": dataset.split_fingerprint,
                    "arm": arm,
                    "seed": seed,
                    "failure": failure,
                    "input_features": dimension,
                    "ensemble_size": k,
                    "member_specific_views": member_views,
                    "basis_relation_error": relation_error,
                    "basis_transform_condition": float(np.linalg.cond(transform)),
                    "all_orbit_transforms_full_rank": all(
                        np.linalg.matrix_rank(value, tol=1e-10) == dimension for value in transforms
                    ),
                    **fit,
                }
                rows.append(row)
                _write(args.output, rows)
                curve_rows.extend(
                    {
                        "stage": args.stage,
                        "dataset": dataset_name,
                        "arm": arm,
                        "seed": seed,
                        **curve,
                    }
                    for curve in curves
                )
                _write(curves_path, curve_rows)
                status = failure or f"loss={fit['test_proper_loss']:.6f}"
                print(f"{dataset_name:34s} {arm:18s} s{seed} {status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "confirmation"], required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    if args.output is None:
        args.output = RESULTS / f"{args.stage}_shard{args.shard}.csv"
    run(args)


if __name__ == "__main__":
    main()
