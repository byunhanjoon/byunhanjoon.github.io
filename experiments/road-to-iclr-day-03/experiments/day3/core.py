"""Core data, geometry, exact-basis, and neural-training utilities for Day 3.

All fitted transformations in this module are trained on the official training
partition only.  Geometry calculations use float64; networks receive float32.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "road-to-iclr-day-01" / "data"
Task = Literal["binclass", "multiclass", "regression"]
PARTS = ("train", "val", "test")
RANK_RTOL = 1e-10


@dataclass
class Dataset:
    name: str
    task: Task
    x_num: dict[str, np.ndarray] | None
    x_bin: dict[str, np.ndarray] | None
    x_cat: dict[str, np.ndarray] | None
    y: dict[str, np.ndarray]
    n_classes: int
    split_fingerprint: str


@dataclass
class Prepared:
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    task: Task
    n_classes: int
    y_mean: float
    y_scale: float
    metadata: dict[str, object]


def _limited(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(indices) <= limit:
        return indices
    local = np.sort(np.random.default_rng(seed).choice(len(indices), limit, replace=False))
    return indices[local]


def load_dataset(
    name: str,
    data_root: Path = DATA_ROOT,
    max_train_rows: int | None = 100_000,
    max_eval_rows: int | None = 25_000,
    sample_seed: int = 2026,
) -> Dataset:
    directory = data_root / name
    info = json.loads((directory / "info.json").read_text())
    task = info["task"]["type"]
    raw_indices = {
        p: np.load(directory / "splits" / "default" / f"{p}.npy") for p in PARTS
    }
    indices = {
        p: _limited(raw_indices[p], max_train_rows if p == "train" else max_eval_rows, sample_seed + i)
        for i, p in enumerate(PARTS)
    }

    def optional(stem: str) -> dict[str, np.ndarray] | None:
        path = directory / f"{stem}.npy"
        if not path.exists():
            return None
        array = np.load(path, mmap_mode="r")
        return {p: np.asarray(array[indices[p]]) for p in PARTS}

    y = optional("y")
    assert y is not None
    if task == "multiclass":
        classes = np.unique(y["train"])
        y = {p: np.searchsorted(classes, y[p]).astype(np.int64) for p in PARTS}
        n_classes = len(classes)
    else:
        y = {p: y[p].astype(np.float32) for p in PARTS}
        n_classes = 2 if task == "binclass" else 1
    import hashlib

    digest = hashlib.sha256()
    for p in PARTS:
        digest.update(np.asarray(indices[p], dtype=np.int64).tobytes())
    return Dataset(
        name=name,
        task=task,
        x_num=optional("x_num"),
        x_bin=optional("x_bin"),
        x_cat=optional("x_cat"),
        y=y,
        n_classes=n_classes,
        split_fingerprint=digest.hexdigest()[:16],
    )


def clean_numeric(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train = parts["train"].astype(np.float64, copy=True)
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    out = {}
    for p, source in parts.items():
        values = source.astype(np.float64, copy=True)
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = medians[np.where(bad)[1]]
        out[p] = values
    return out


def standardize(parts: dict[str, np.ndarray], eps: float = 1e-12) -> dict[str, np.ndarray]:
    mean = np.mean(parts["train"], axis=0)
    scale = np.std(parts["train"], axis=0)
    keep = scale > eps
    if not np.any(keep):
        return {p: np.empty((len(v), 0), dtype=np.float64) for p, v in parts.items()}
    return {p: (v[:, keep] - mean[keep]) / scale[keep] for p, v in parts.items()}


def target(parts: dict[str, np.ndarray], task: Task) -> tuple[dict[str, np.ndarray], float, float]:
    mean = float(parts["train"].mean()) if task == "regression" else 0.0
    scale = (float(parts["train"].std()) if task == "regression" else 1.0) or 1.0
    out = {
        p: ((v.astype(np.float32) - mean) / scale if task == "regression" else v)
        for p, v in parts.items()
    }
    return out, mean, scale


def quantile_numeric(parts: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    clean = clean_numeric(parts)
    n_quantiles = max(min(len(clean["train"]) // 30, 1000), 10)
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=seed,
    ).fit(clean["train"])
    return {p: qt.transform(v).astype(np.float64) for p, v in clean.items()}


def category_codes(parts: dict[str, np.ndarray], column: int, levels: list[str] | None = None) -> tuple[dict[str, np.ndarray], list[str]]:
    train = parts["train"][:, column].astype(str)
    learned = sorted(np.unique(train).tolist()) if levels is None else list(levels)
    lookup = {value: i for i, value in enumerate(learned)}
    codes = {
        p: np.asarray([lookup.get(str(v), -1) for v in values[:, column]], dtype=np.int64)
        for p, values in parts.items()
    }
    return codes, learned


def one_hot_codes(codes: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    out = {}
    for p, values in codes.items():
        matrix = np.zeros((len(values), k), dtype=np.float64)
        valid = values >= 0
        matrix[np.arange(len(values))[valid], values[valid]] = 1.0
        out[p] = matrix
    return out


def helmert(k: int) -> np.ndarray:
    """Euclidean-orthonormal K x (K-1) Helmert contrast matrix."""
    h = np.zeros((k, k - 1), dtype=np.float64)
    for j in range(k - 1):
        h[: j + 1, j] = 1.0 / math.sqrt((j + 1) * (j + 2))
        h[j + 1, j] = -(j + 1) / math.sqrt((j + 1) * (j + 2))
    return h


def contrast_block(codes: dict[str, np.ndarray], k: int, basis: np.ndarray | None = None) -> dict[str, np.ndarray]:
    basis = helmert(k) if basis is None else basis
    return {p: one_hot_codes(codes, k)[p] @ basis for p in PARTS}


def cumulative_ordinal(codes: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    thresholds = np.arange(k - 1)
    return {p: (v[:, None] > thresholds[None, :]).astype(np.float64) for p, v in codes.items()}


def path_spectral(k: int) -> np.ndarray:
    adjacency = np.zeros((k, k), dtype=np.float64)
    for i in range(k - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    _, vectors = np.linalg.eigh(laplacian)
    return vectors[:, 1:]


def ple_blocks(parts: dict[str, np.ndarray], bins: int = 32) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    clean = clean_numeric(parts)
    blocks: list[dict[str, np.ndarray]] = []
    knot_list: list[np.ndarray] = []
    for j in range(clean["train"].shape[1]):
        knots = np.unique(np.quantile(clean["train"][:, j], np.linspace(0, 1, bins + 1)))
        if len(knots) < 2:
            knots = np.array([knots[0], knots[0] + 1.0])
        left, right = knots[:-1], knots[1:]
        widths = np.maximum(right - left, 1e-12)
        block = {
            p: np.clip((v[:, j, None] - left) / widths, 0.0, 1.0)
            for p, v in clean.items()
        }
        blocks.append(block)
        knot_list.append(knots)
    return blocks, knot_list


def exact_state_ple_and_identity(parts: dict[str, np.ndarray], column: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    clean = clean_numeric(parts)
    levels = np.unique(clean["train"][:, column])
    if len(levels) < 2:
        raise ValueError("Exact-state block needs at least two levels")
    left, right = levels[:-1], levels[1:]
    widths = np.maximum(right - left, 1e-12)
    ple = {
        p: np.clip((v[:, column, None] - left) / widths, 0.0, 1.0)
        for p, v in clean.items()
    }
    # On the observed states this is exactly a centered one-hot/Helmert basis.
    # Between states we use its unique affine extension through the PLE basis,
    # preserving exact equivalence on validation/test instead of mapping unseen
    # values to an arbitrary all-zero fallback.
    train_codes = np.searchsorted(levels, clean["train"][:, column])
    train_identity = one_hot_codes({p: train_codes for p in PARTS}, len(levels))["train"] @ helmert(len(levels))
    design = np.column_stack((np.ones(len(ple["train"])), ple["train"]))
    affine = np.linalg.lstsq(design, train_identity, rcond=RANK_RTOL)[0]
    identity = {
        p: np.column_stack((np.ones(len(ple[p])), ple[p])) @ affine for p in PARTS
    }
    return ple, identity, levels


def geometry(matrix: np.ndarray, rtol: float = RANK_RTOL) -> dict[str, float | int]:
    z = np.asarray(matrix, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    if z.shape[1] == 0:
        return {k: 0 for k in ("rank", "effective_rank", "sigma_max", "sigma_min_nonzero", "condition_number", "log_condition_number", "trace_cov", "logdet_nonzero_cov", "max_variance", "min_nonzero_variance")}
    covariance = z.T @ z / max(len(z), 1)
    eigen = np.linalg.eigvalsh(covariance)
    eigen = np.maximum(eigen, 0.0)
    threshold = (eigen[-1] if len(eigen) else 0.0) * rtol
    kept = eigen[eigen > threshold]
    if not len(kept):
        kept = np.array([0.0])
    total = kept.sum()
    probabilities = kept / total if total > 0 else np.ones_like(kept) / len(kept)
    eff = float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))))
    variances = np.diag(covariance)
    nonzero_var = variances[variances > max(float(variances.max(initial=0.0)) * rtol, 0.0)]
    condition = math.sqrt(float(kept[-1] / kept[0])) if kept[0] > 0 else math.inf
    return {
        "rank": int(len(kept)) if kept[-1] > 0 else 0,
        "effective_rank": eff,
        "sigma_max": math.sqrt(float(kept[-1])),
        "sigma_min_nonzero": math.sqrt(float(kept[0])),
        "condition_number": condition,
        "log_condition_number": math.log10(condition) if condition > 0 else -math.inf,
        "trace_cov": float(total),
        "logdet_nonzero_cov": float(np.log(np.maximum(kept, 1e-300)).sum()),
        "max_variance": float(variances.max(initial=0.0)),
        "min_nonzero_variance": float(nonzero_var.min()) if len(nonzero_var) else 0.0,
    }


def fit_whitener(train: np.ndarray, rtol: float = RANK_RTOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(train, dtype=np.float64)
    mean = z.mean(axis=0)
    centered = z - mean
    covariance = centered.T @ centered / max(len(z), 1)
    eigen, vectors = np.linalg.eigh(covariance)
    threshold = max(float(eigen[-1]), 0.0) * rtol
    keep = eigen > threshold
    transform = vectors[:, keep] / np.sqrt(eigen[keep])[None, :]
    return mean, transform, eigen[keep]


def whiten(parts: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    mean, transform, eigen = fit_whitener(parts["train"])
    out = {p: (v - mean) @ transform for p, v in parts.items()}
    return out, {"retained_rank": int(len(eigen)), "eigenvalues": eigen.tolist(), "transform": transform}


def diagonal_standardize(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = parts["train"].mean(axis=0)
    std = parts["train"].std(axis=0)
    keep = std > 1e-12
    return {p: (v[:, keep] - mean[keep]) / std[keep] for p, v in parts.items()}


def condition_transform(dimension: int, kappa: float, seed: int, random_rotation: bool = True) -> np.ndarray:
    if dimension == 0:
        return np.empty((0, 0), dtype=np.float64)
    rng = np.random.default_rng(seed)
    if random_rotation:
        q1, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        q2, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    else:
        q1 = q2 = np.eye(dimension)
    logs = np.linspace(-0.5 * math.log(kappa), 0.5 * math.log(kappa), dimension)
    singular = np.exp(logs)  # geometric mean exactly one
    return q1 @ np.diag(singular) @ q2


def apply_transform(parts: dict[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    return {p: v @ transform for p, v in parts.items()}


def reconstruction(source: dict[str, np.ndarray], target_block: dict[str, np.ndarray]) -> dict[str, float]:
    design = np.column_stack((np.ones(len(source["train"])), source["train"]))
    coef = np.linalg.lstsq(design, target_block["train"], rcond=RANK_RTOL)[0]
    errors = {}
    for p in PARTS:
        pred = np.column_stack((np.ones(len(source[p])), source[p])) @ coef
        errors[p] = float(np.linalg.norm(pred - target_block[p]) / max(np.linalg.norm(target_block[p]), 1e-12))
    return errors


def equivalence_diagnostics(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, object]:
    ac = a["train"] - a["train"].mean(axis=0)
    bc = b["train"] - b["train"].mean(axis=0)
    qa = np.linalg.qr(ac, mode="reduced")[0]
    qb = np.linalg.qr(bc, mode="reduced")[0]
    angles = subspace_angles(qa, qb)
    return {
        "a_to_b": reconstruction(a, b),
        "b_to_a": reconstruction(b, a),
        "max_principal_angle_deg": float(np.degrees(angles).max(initial=0.0)),
        "rank_a": geometry(a["train"])["rank"],
        "rank_b": geometry(b["train"])["rank"],
    }


def procrustes_align(reference: dict[str, np.ndarray], moving: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cross = moving["train"].T @ reference["train"]
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    return {p: moving[p] @ rotation for p in PARTS}, rotation


def block_pair_geometry(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    norm = float(np.linalg.norm(ac.T @ bc) / max(np.linalg.norm(ac) * np.linalg.norm(bc), 1e-12))
    aw, _ = whiten({p: ac for p in PARTS})
    bw, _ = whiten({p: bc for p in PARTS})
    cross = aw["train"].T @ bw["train"] / len(ac)
    singular = np.linalg.svd(cross, compute_uv=False)
    return {
        "normalized_cross_gram": norm,
        "top_canonical_correlation": float(singular.max(initial=0.0)),
        "mean_canonical_correlation": float(singular.mean()) if len(singular) else 0.0,
    }


def residualize(numeric: dict[str, np.ndarray], categorical: dict[str, np.ndarray], ridge: float = 0.0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = numeric["train"]
    c = categorical["train"]
    if ridge:
        gram = n.T @ n + ridge * np.eye(n.shape[1])
        beta = np.linalg.solve(gram, n.T @ c)
    else:
        beta = np.linalg.lstsq(n, c, rcond=RANK_RTOL)[0]
    return {p: categorical[p] - numeric[p] @ beta for p in PARTS}, beta


def combine(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        p: np.ascontiguousarray(np.column_stack([b[p] for b in blocks]), dtype=np.float32)
        for p in PARTS
    }


def base_schema(dataset: Dataset, seed: int = 0, include_num: bool = True, include_cat: bool = True) -> dict[str, np.ndarray]:
    blocks: list[dict[str, np.ndarray]] = []
    if include_num and dataset.x_num is not None:
        blocks.append(quantile_numeric(dataset.x_num, seed))
    if dataset.x_bin is not None:
        blocks.append(standardize(clean_numeric(dataset.x_bin)))
    if include_cat and dataset.x_cat is not None:
        for j in range(dataset.x_cat["train"].shape[1]):
            codes, levels = category_codes(dataset.x_cat, j)
            blocks.append(contrast_block(codes, len(levels)))
    if not blocks:
        return {p: np.empty((len(dataset.y[p]), 0), dtype=np.float64) for p in PARTS}
    return combine(blocks)


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_size
        self.first = nn.Linear(current, width)
        layers.extend((self.first, nn.GELU(), nn.Dropout(dropout)))
        current = width
        for _ in range(depth - 1):
            layers.extend((nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(width, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.first = nn.Linear(input_size, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(depth)))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.first(x)))


def loss_numpy(task: Task, prediction: np.ndarray, y: np.ndarray) -> float:
    if task == "binclass":
        logits = prediction.reshape(-1).astype(np.float64)
        return float(np.mean(np.logaddexp(0, logits) - y * logits))
    if task == "multiclass":
        logits = prediction.astype(np.float64)
        maximum = logits.max(axis=1, keepdims=True)
        logz = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
        return float(np.mean(logz - logits[np.arange(len(y)), y]))
    return float(np.mean((prediction.reshape(-1) - y) ** 2))


def metric(prepared: Prepared, prediction: np.ndarray, y: np.ndarray) -> float:
    if prepared.task == "binclass":
        return float(((prediction.reshape(-1) >= 0) == y).mean())
    if prepared.task == "multiclass":
        return float((prediction.argmax(axis=1) == y).mean())
    return float(np.sqrt(np.mean((prediction.reshape(-1) - y) ** 2)) * prepared.y_scale)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(out)


def train_model(
    data: Prepared,
    seed: int,
    device: str = "cuda:0",
    model_name: str = "mlp",
    width: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 40,
    patience: int = 6,
    regularizer: str = "standard",
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    target_size = data.n_classes if data.task == "multiclass" else 1
    cls = MLP if model_name == "mlp" else ResNet
    torch_device = torch.device(device)
    model = cls(data.x["train"].shape[1], target_size, width, depth, dropout).to(torch_device)
    first = model.first.weight
    other = [p for p in model.parameters() if p is not first]
    first_wd = weight_decay if regularizer == "standard" else 0.0
    optimizer = torch.optim.AdamW(
        [{"params": [first], "weight_decay": first_wd}, {"params": other, "weight_decay": weight_decay}],
        lr=learning_rate,
    )
    if data.task == "binclass":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    elif data.task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    covariance_t = None
    if regularizer == "invariant":
        z = data.x["train"].astype(np.float64)
        z = z - z.mean(axis=0)
        covariance_t = torch.from_numpy((z.T @ z / len(z)).astype(np.float32)).to(torch_device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch_device.type == "cuda",
    )
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_state = None
    curves: list[dict[str, float | int]] = []
    first_gradient_norm = math.nan
    first_update_ratio = math.nan
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_index, (x, y) in enumerate(loader):
            x, y = x.to(torch_device, non_blocking=True), y.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if data.task != "multiclass":
                prediction = prediction.squeeze(-1)
            loss = criterion(prediction, y)
            if covariance_t is not None:
                activation_energy = torch.sum((first @ covariance_t) * first) / first.shape[0]
                loss = loss + 0.5 * weight_decay * activation_energy
            loss.backward()
            if epoch == 1 and batch_index == 0:
                first_gradient_norm = float(first.grad.norm().detach().cpu())
                before = first.detach().clone()
            optimizer.step()
            if epoch == 1 and batch_index == 0:
                first_update_ratio = float((first.detach() - before).norm().cpu() / before.norm().cpu().clamp_min(1e-12))
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        val_prediction = _predict(model, data.x["val"], torch_device, batch_size * 4)
        val_loss = loss_numpy(data.task, val_prediction, data.y["val"])
        curves.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale > patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_prediction = _predict(model, data.x["train"], torch_device, batch_size * 4)
    val_prediction = _predict(model, data.x["val"], torch_device, batch_size * 4)
    test_prediction = _predict(model, data.x["test"], torch_device, batch_size * 4)
    result = {
        "input_features": data.x["train"].shape[1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "train_loss": loss_numpy(data.task, train_prediction, data.y["train"]),
        "val_loss": loss_numpy(data.task, val_prediction, data.y["val"]),
        "test_loss": loss_numpy(data.task, test_prediction, data.y["test"]),
        "val_metric": metric(data, val_prediction, data.y["val"]),
        "test_metric": metric(data, test_prediction, data.y["test"]),
        "first_gradient_norm": first_gradient_norm,
        "first_weight_norm": float(first.detach().norm().cpu()),
        "first_update_ratio": first_update_ratio,
        "train_seconds": time.perf_counter() - started,
    }
    return result, curves


def make_prepared(dataset: Dataset, x: dict[str, np.ndarray], metadata: dict[str, object]) -> Prepared:
    y, mean, scale = target(dataset.y, dataset.task)
    return Prepared(
        x={p: np.ascontiguousarray(v, dtype=np.float32) for p, v in x.items()},
        y=y,
        task=dataset.task,
        n_classes=dataset.n_classes,
        y_mean=mean,
        y_scale=scale,
        metadata=metadata,
    )


def real_fourier_basis(k: int, phase: float = 0.0) -> np.ndarray:
    """Return a real orthonormal DFT basis excluding the constant vector."""
    states = np.arange(k, dtype=np.float64) + phase
    columns = []
    for frequency in range(1, (k - 1) // 2 + 1):
        theta = 2 * np.pi * frequency * states / k
        columns.extend((math.sqrt(2 / k) * np.cos(theta), math.sqrt(2 / k) * np.sin(theta)))
    if k % 2 == 0:
        columns.append(((-1.0) ** states) / math.sqrt(k))
    return np.column_stack(columns)


def invariant_penalty(weight: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.trace(weight @ covariance @ weight.T))


def transformed_invariant_penalty(weight: np.ndarray, covariance: np.ndarray, transform: np.ndarray) -> float:
    inverse = np.linalg.inv(transform)
    wp = weight @ inverse
    sp = transform @ covariance @ transform.T
    return invariant_penalty(wp, sp)
