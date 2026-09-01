"""Shared models and data utilities for the Day-8 direction search.

The implementations are deliberately compact screening implementations.  The
TabR model preserves the paper's key-only squared-L2 retrieval and label-plus-
key-difference value construction.  ModernNCA predicts directly from a
softmax-weighted neighborhood.  They are not claimed to reproduce published
leaderboard configurations.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_ROOT = REPO / "experiments/road-to-iclr-day-01/data"
SEEDS = (20260831, 20260832, 20260833)
PANEL = (
    "adult", "churn", "higgs-small", "otto",
    "california", "diamond", "house", "black-friday",
)


@dataclass
class ArrayData:
    name: str
    task: Literal["classification", "regression"]
    n_classes: int
    x_num: dict[str, np.ndarray]
    x_cat: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    y_mean: float
    y_scale: float

    @property
    def n_num(self) -> int:
        return self.x_num["train"].shape[1]

    @property
    def n_cat(self) -> int:
        return self.x_cat["train"].shape[1]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cap_indices(
    indices: np.ndarray,
    y: np.ndarray,
    cap: int,
    task: str,
    seed: int,
) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    stratify = y[indices] if task == "classification" else None
    chosen, _ = train_test_split(
        indices, train_size=cap, random_state=seed, stratify=stratify
    )
    return np.asarray(chosen)


def load_real_dataset(name: str, split_seed: int = 20260831) -> ArrayData:
    root = DATA_ROOT / name
    info = json.loads((root / "info.json").read_text())
    task = "regression" if info["task"]["type"] == "regression" else "classification"
    arrays: dict[str, np.ndarray] = {}
    for key in ("x_num", "x_cat", "x_bin"):
        path = root / f"{key}.npy"
        if path.exists():
            arrays[key] = np.asarray(np.load(path, mmap_mode="r"))
    raw_y = np.asarray(np.load(root / "y.npy", mmap_mode="r"))
    if task == "classification":
        _, raw_y = np.unique(raw_y.astype(str), return_inverse=True)
        raw_y = raw_y.astype(np.int64)
    else:
        raw_y = raw_y.astype(np.float64)
    rows = np.arange(len(raw_y))
    stratify = raw_y if task == "classification" else None
    train_val, test = train_test_split(
        rows, test_size=0.2, random_state=split_seed, stratify=stratify
    )
    stratify_tv = raw_y[train_val] if task == "classification" else None
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=split_seed + 1, stratify=stratify_tv
    )
    train = _cap_indices(train, raw_y, 4096, task, split_seed + 11)
    val = _cap_indices(val, raw_y, 1024, task, split_seed + 12)
    test = _cap_indices(test, raw_y, 1024, task, split_seed + 13)
    split = {"train": train, "validation": val, "test": test}

    raw_num = arrays.get("x_num", np.empty((len(raw_y), 0), dtype=np.float32))
    raw_cat_parts = [arrays[key] for key in ("x_cat", "x_bin") if key in arrays]
    raw_cat = (
        np.concatenate(raw_cat_parts, axis=1)
        if raw_cat_parts else np.empty((len(raw_y), 0), dtype=np.float32)
    )
    if raw_num.shape[1]:
        medians = np.nanmedian(raw_num[train].astype(np.float64), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        clean_num = np.asarray(raw_num, dtype=np.float64).copy()
        bad = ~np.isfinite(clean_num)
        if bad.any():
            clean_num[bad] = medians[np.where(bad)[1]]
        scaler = StandardScaler().fit(clean_num[train])
        x_num = {
            part: np.ascontiguousarray(scaler.transform(clean_num[idx]), dtype=np.float32)
            for part, idx in split.items()
        }
    else:
        x_num = {
            part: np.empty((len(idx), 0), dtype=np.float32)
            for part, idx in split.items()
        }
    if raw_cat.shape[1]:
        # Ordinal/string identities are fit on training only.  Dense one-hot is
        # affordable after the frozen 4,096-row cap.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        encoder.fit(raw_cat[train].astype(str))
        x_cat = {
            part: np.ascontiguousarray(encoder.transform(raw_cat[idx].astype(str)), dtype=np.float32)
            for part, idx in split.items()
        }
    else:
        x_cat = {
            part: np.empty((len(idx), 0), dtype=np.float32)
            for part, idx in split.items()
        }
    if task == "regression":
        y_mean = float(raw_y[train].mean())
        y_scale = float(raw_y[train].std()) or 1.0
        y = {
            part: np.ascontiguousarray((raw_y[idx] - y_mean) / y_scale, dtype=np.float32)
            for part, idx in split.items()
        }
        n_classes = 1
    else:
        y_mean, y_scale = 0.0, 1.0
        y = {part: np.ascontiguousarray(raw_y[idx], dtype=np.int64) for part, idx in split.items()}
        n_classes = int(np.max(raw_y) + 1)
    return ArrayData(name, task, n_classes, x_num, x_cat, y, y_mean, y_scale)


def make_synthetic(name: str, seed: int, n_train: int = 2048, n_val: int = 512, n_test: int = 512) -> tuple[ArrayData, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test
    x = rng.uniform(-1.0, 1.0, size=(total, 2)).astype(np.float32)

    def truth(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z0, z1 = z[:, 0], z[:, 1]
        if name == "S1_rotating":
            a = 1.0 / (1.0 + np.exp(-4.0 * z0))
            f0, f1 = np.sin(4.0 * z0), np.sin(4.0 * z1)
            m = (1.0 - a) * f0 + a * f1
            da = 4.0 * a * (1.0 - a)
            g0 = (1.0 - a) * 4.0 * np.cos(4.0 * z0) + da * (f1 - f0)
            g1 = a * 4.0 * np.cos(4.0 * z1)
            sigma = 0.08 + 0.35 / (1.0 + np.exp(-3.0 * z1))
        elif name == "S2_global":
            m = 3.0 * z0 + 0.3 * z1
            g0, g1 = np.full_like(z0, 3.0), np.full_like(z1, 0.3)
            sigma = np.full_like(z0, 0.22)
        elif name == "S3_noise":
            m = np.sin(3.0 * z0)
            g0, g1 = 3.0 * np.cos(3.0 * z0), np.zeros_like(z1)
            sigma = 0.05 + 0.80 / (1.0 + np.exp(-10.0 * z1))
        elif name == "S4_warp":
            tau = z0 + 0.75 * z0**3
            m = np.sin(3.0 * tau) + 0.25 * z1
            g0 = 3.0 * np.cos(3.0 * tau) * (1.0 + 2.25 * z0**2)
            g1 = np.full_like(z1, 0.25)
            sigma = np.full_like(z0, 0.18)
        else:
            raise ValueError(name)
        grad = np.column_stack((g0, g1)).astype(np.float32)
        return m.astype(np.float32), sigma.astype(np.float32), grad

    m, sigma, grad = truth(x)
    y = m + sigma * rng.normal(size=total).astype(np.float32)
    split_idx = {
        "train": np.arange(n_train),
        "validation": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, total),
    }
    data = ArrayData(
        name=name,
        task="regression",
        n_classes=1,
        x_num={p: np.ascontiguousarray(x[idx]) for p, idx in split_idx.items()},
        x_cat={p: np.empty((len(idx), 0), dtype=np.float32) for p, idx in split_idx.items()},
        y={p: np.ascontiguousarray(y[idx]) for p, idx in split_idx.items()},
        y_mean=0.0,
        y_scale=1.0,
    )
    meta = {
        "m": {p: m[idx] for p, idx in split_idx.items()},
        "sigma": {p: sigma[idx] for p, idx in split_idx.items()},
        "grad": {p: grad[idx] for p, idx in split_idx.items()},
        "truth": truth,
    }
    return data, meta


class BasisMap(nn.Module):
    """Per-feature numerical map; categoricals pass through unchanged."""

    def __init__(self, kind: str, train_num: np.ndarray, n_cat: int, bins: int = 8) -> None:
        super().__init__()
        self.kind = kind
        self.n_num = int(train_num.shape[1])
        self.n_cat = int(n_cat)
        self.bins = int(bins)
        if self.n_num:
            q = np.quantile(train_num.astype(np.float64), np.linspace(0, 1, bins + 1), axis=0).T
            for j in range(self.n_num):
                for b in range(1, bins + 1):
                    q[j, b] = max(q[j, b], q[j, b - 1] + 1e-4)
            self.register_buffer("edges", torch.tensor(q, dtype=torch.float32))
        else:
            self.register_buffer("edges", torch.empty((0, bins + 1)))
        if kind == "localwarp" and self.n_num:
            # Equal initial positive increments make the map approximately
            # quantile-linear; softplus/cumulative construction is monotone.
            self.raw_increments = nn.Parameter(torch.zeros(self.n_num, bins))
        else:
            self.register_parameter("raw_increments", None)

    @property
    def out_dim(self) -> int:
        if self.kind == "ple":
            numeric = self.n_num * self.bins
        elif self.kind == "plr":
            numeric = self.n_num * 9
        else:
            numeric = self.n_num
        return numeric + self.n_cat

    def _ple(self, x: Tensor) -> Tensor:
        left, right = self.edges[:, :-1], self.edges[:, 1:]
        value = (x[:, :, None] - left[None]) / (right - left)[None]
        return value.clamp(0.0, 1.0)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if self.kind == "raw":
            z = x_num
        elif self.kind == "ple":
            z = self._ple(x_num).flatten(1)
        elif self.kind == "plr":
            frequencies = x_num.new_tensor([0.5, 1.0, 2.0, 4.0])
            phase = math.pi * x_num[:, :, None] * frequencies
            z = torch.cat((x_num[:, :, None], torch.sin(phase), torch.cos(phase)), dim=2).flatten(1)
        elif self.kind == "localwarp":
            if not self.n_num:
                z = x_num
            else:
                basis = self._ple(x_num)
                increments = nn.functional.softplus(self.raw_increments) + 1e-4
                increments = increments / increments.mean(dim=1, keepdim=True)
                z = (basis * increments[None]).sum(dim=2)
                z = z - 0.5 * increments.sum(dim=1)[None]
        elif self.kind == "wrongwarp":
            z = 2.0 * torch.sinh(x_num.clamp(-2.0, 2.0)) / math.sinh(2.0)
        elif self.kind == "oraclewarp":
            z = x_num.clone()
            if self.n_num:
                z[:, 0] = x_num[:, 0] + 0.75 * x_num[:, 0].pow(3)
        elif self.kind == "inversewarp":
            z = torch.sign(x_num) * torch.abs(x_num).pow(1.0 / 3.0)
        else:
            raise ValueError(self.kind)
        return torch.cat((z, x_cat), dim=1)


class BranchEncoder(nn.Module):
    def __init__(
        self,
        kind: str,
        train_num: np.ndarray,
        n_cat: int,
        width: int,
        capacity: str = "shallow",
    ) -> None:
        super().__init__()
        self.basis = BasisMap(kind, train_num, n_cat)
        d = self.basis.out_dim
        if capacity == "identity":
            if d != width:
                self.net = nn.Linear(d, width, bias=False)
            else:
                self.net = nn.Identity()
        elif capacity == "linear":
            self.net = nn.Linear(d, width, bias=False)
        elif capacity == "shallow":
            self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width))
        elif capacity == "standard":
            self.net = nn.Sequential(
                nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
            )
        elif capacity == "deep":
            self.net = nn.Sequential(
                nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width),
            )
        else:
            raise ValueError(capacity)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.net(self.basis(x_num, x_cat))


class MLPModel(nn.Module):
    def __init__(self, data: ArrayData, pred_kind: str, width: int = 64) -> None:
        super().__init__()
        self.pred = BranchEncoder(pred_kind, data.x_num["train"], data.n_cat, width, "standard")
        out = 1 if data.task == "regression" else data.n_classes
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out))
        self.model_name = "MLP"

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.head(self.pred(x_num, x_cat))


class TabRModel(nn.Module):
    def __init__(
        self,
        data: ArrayData,
        pred_kind: str,
        retr_kind: str,
        key_capacity: str = "standard",
        width: int = 64,
    ) -> None:
        super().__init__()
        self.pred = BranchEncoder(pred_kind, data.x_num["train"], data.n_cat, width, "standard")
        self.retr = BranchEncoder(retr_kind, data.x_num["train"], data.n_cat, width, key_capacity)
        self.key_norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, width, bias=False)
        self.task, self.n_classes, self.width = data.task, data.n_classes, width
        if data.task == "regression":
            self.label = nn.Linear(1, width)
            out = 1
        else:
            self.label = nn.Embedding(data.n_classes, width)
            out = data.n_classes
        self.value = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Linear(width * 2, width, bias=False))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out))
        self.model_name = "TabR"

    def keys(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.key(self.key_norm(self.retr(x_num, x_cat)))

    def forward_context(
        self,
        q_num: Tensor,
        q_cat: Tensor,
        c_num: Tensor,
        c_cat: Tensor,
        c_y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        q_rep = self.pred(q_num, q_cat)
        q_key = self.keys(q_num, q_cat)
        b, k = c_num.shape[:2]
        c_key = self.keys(c_num.flatten(0, 1), c_cat.flatten(0, 1)).reshape(b, k, -1)
        distance = (q_key[:, None] - c_key).square().sum(dim=2)
        attention = torch.softmax(-distance / math.sqrt(self.width), dim=1)
        if self.task == "regression":
            label = self.label(c_y.float()[..., None])
        else:
            label = self.label(c_y.long())
        values = label + self.value(q_key[:, None] - c_key)
        mixed = q_rep + torch.bmm(attention[:, None], values).squeeze(1)
        return self.head(mixed), distance


class ModernNCAModel(nn.Module):
    def __init__(self, data: ArrayData, retr_kind: str, key_capacity: str = "deep", width: int = 64) -> None:
        super().__init__()
        self.retr = BranchEncoder(retr_kind, data.x_num["train"], data.n_cat, width, key_capacity)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.75)))
        self.task, self.n_classes = data.task, data.n_classes
        self.model_name = "ModernNCA"

    def keys(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.retr(x_num, x_cat)

    def forward_candidates(
        self,
        q_num: Tensor,
        q_cat: Tensor,
        c_num: Tensor,
        c_cat: Tensor,
        c_y: Tensor,
        same_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q = self.keys(q_num, q_cat)
        c = self.keys(c_num, c_cat)
        # Avoid cdist's undefined zero-distance backward at a query's own row.
        # The expanded squared-L2 form has a stable derivative at equality.
        qf, cf = q.float(), c.float()
        distance = (
            qf.square().sum(dim=1, keepdim=True)
            - 2.0 * qf @ cf.T
            + cf.square().sum(dim=1)[None]
        ).clamp_min(0.0)
        temperature = self.log_temperature.exp().clamp(0.05, 5.0)
        logits = -distance / temperature
        if same_mask is not None:
            # Mask after temperature scaling: differentiating inf/temperature
            # yields NaN even though the corresponding softmax weight is zero.
            logits = logits.masked_fill(same_mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        if self.task == "regression":
            pred = weights @ c_y.float()[:, None]
        else:
            onehot = nn.functional.one_hot(c_y.long(), self.n_classes).float()
            pred = weights @ onehot
        reported_distance = distance.masked_fill(same_mask, torch.inf) if same_mask is not None else distance
        return pred, reported_distance


def build_model(
    data: ArrayData,
    model_name: str,
    pred_kind: str,
    retr_kind: str,
    key_capacity: str = "standard",
) -> nn.Module:
    if model_name == "MLP":
        return MLPModel(data, pred_kind)
    if model_name == "TabR":
        return TabRModel(data, pred_kind, retr_kind, key_capacity)
    if model_name == "ModernNCA":
        # ModernNCA has no separate parametric prediction branch.
        return ModernNCAModel(data, retr_kind, key_capacity if key_capacity != "standard" else "deep")
    raise ValueError(model_name)


def _loss(model: nn.Module, output: Tensor, target: Tensor, data: ArrayData) -> Tensor:
    if data.task == "regression":
        return nn.functional.mse_loss(output[:, 0], target.float())
    if isinstance(model, ModernNCAModel):
        return nn.functional.nll_loss(torch.log(output.clamp_min(1e-8)), target.long())
    return nn.functional.cross_entropy(output, target.long())


def _select_context(
    model: TabRModel,
    q_num: Tensor,
    q_cat: Tensor,
    c_num: Tensor,
    c_cat: Tensor,
    q_indices: Tensor | None,
    c_indices: Tensor | None,
    k: int,
) -> Tensor:
    with torch.no_grad():
        distance = torch.cdist(model.keys(q_num, q_cat).float(), model.keys(c_num, c_cat).float()).square()
        if q_indices is not None and c_indices is not None:
            distance.masked_fill_(q_indices[:, None] == c_indices[None, :], torch.inf)
        return torch.topk(distance, k=min(k, c_num.shape[0] - int(q_indices is not None)), largest=False).indices


@torch.no_grad()
def predict_model(
    model: nn.Module,
    data: ArrayData,
    part: str,
    device: torch.device,
    return_neighbors: bool = False,
    query_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    model.eval()
    q_num = torch.as_tensor(data.x_num[part], device=device)
    q_cat = torch.as_tensor(data.x_cat[part], device=device)
    if query_limit is not None:
        q_num, q_cat = q_num[:query_limit], q_cat[:query_limit]
    c_num = torch.as_tensor(data.x_num["train"], device=device)
    c_cat = torch.as_tensor(data.x_cat["train"], device=device)
    c_y = torch.as_tensor(data.y["train"], device=device)
    outputs, all_idx, all_dist = [], [], []
    c_keys = model.keys(c_num, c_cat) if hasattr(model, "keys") else None
    for start in range(0, len(q_num), 256):
        qn, qc = q_num[start:start + 256], q_cat[start:start + 256]
        if isinstance(model, MLPModel):
            out = model(qn, qc)
            idx = dist = None
        elif isinstance(model, TabRModel):
            q_keys = model.keys(qn, qc)
            full_dist = torch.cdist(q_keys.float(), c_keys.float()).square()
            idx = torch.topk(full_dist, k=min(16, len(c_num)), largest=False).indices
            cn, cc, cy = c_num[idx], c_cat[idx], c_y[idx]
            out, _ = model.forward_context(qn, qc, cn, cc, cy)
            dist = full_dist
        else:
            out, dist = model.forward_candidates(qn, qc, c_num, c_cat, c_y)
            idx = torch.topk(dist, k=min(16, len(c_num)), largest=False).indices
        outputs.append(out.detach().cpu())
        if return_neighbors and idx is not None and dist is not None:
            all_idx.append(idx.cpu())
            all_dist.append(dist.cpu())
    raw = torch.cat(outputs).numpy()
    if data.task == "classification" and not isinstance(model, ModernNCAModel):
        raw = torch.softmax(torch.from_numpy(raw), dim=1).numpy()
    return (
        raw,
        torch.cat(all_idx).numpy() if all_idx else None,
        torch.cat(all_dist).numpy() if all_dist else None,
    )


def evaluate_predictions(pred: np.ndarray, y: np.ndarray, data: ArrayData) -> dict[str, float]:
    if data.task == "regression":
        rmse = float(np.sqrt(np.mean((pred[:, 0] - y.astype(float)) ** 2)))
        return {"loss": rmse, "metric": rmse, "score": -rmse, "metric_name": "standardized_rmse"}
    labels = pred.argmax(axis=1)
    accuracy = float(np.mean(labels == y))
    logloss = float(-np.log(pred[np.arange(len(y)), y].clip(1e-8)).mean())
    return {"loss": logloss, "metric": accuracy, "score": accuracy, "metric_name": "accuracy"}


def train_model(
    data: ArrayData,
    model_name: str,
    pred_kind: str,
    retr_kind: str,
    seed: int,
    device: torch.device,
    key_capacity: str = "standard",
    max_epochs: int = 24,
) -> tuple[nn.Module, dict[str, float]]:
    seed_all(seed)
    model = build_model(data, model_name, pred_kind, retr_kind, key_capacity).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    x_num = torch.as_tensor(data.x_num["train"], device=device)
    x_cat = torch.as_tensor(data.x_cat["train"], device=device)
    y = torch.as_tensor(data.y["train"], device=device)
    rng = np.random.default_rng(seed)
    best_state, best_loss, stale = None, math.inf, 0
    batch_size, candidate_size = 128, min(512, len(y))
    steps = min(20, max(8, math.ceil(len(y) / batch_size)))
    for epoch in range(max_epochs):
        model.train()
        for _ in range(steps):
            q_idx_np = rng.choice(len(y), size=min(batch_size, len(y)), replace=False)
            q_idx = torch.as_tensor(q_idx_np, device=device)
            qn, qc, qy = x_num[q_idx], x_cat[q_idx], y[q_idx]
            if isinstance(model, MLPModel):
                output = model(qn, qc)
            else:
                c_idx_np = rng.choice(len(y), size=candidate_size, replace=False)
                c_idx = torch.as_tensor(c_idx_np, device=device)
                cn, cc, cy = x_num[c_idx], x_cat[c_idx], y[c_idx]
                if isinstance(model, TabRModel):
                    pos = _select_context(model, qn, qc, cn, cc, q_idx, c_idx, 16)
                    output, _ = model.forward_context(qn, qc, cn[pos], cc[pos], cy[pos])
                else:
                    same = q_idx[:, None] == c_idx[None, :]
                    output, _ = model.forward_candidates(qn, qc, cn, cc, cy, same)
            loss = _loss(model, output, qy, data)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        pred, _, _ = predict_model(model, data, "validation", device)
        val = evaluate_predictions(pred, data.y["validation"], data)
        if val["loss"] < best_loss - 1e-5:
            best_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= 6 and stale >= 4:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, _, _ = predict_model(model, data, "test", device)
    metrics = evaluate_predictions(pred, data.y["test"], data)
    metrics.update(
        epochs=float(epoch + 1),
        parameters=float(sum(p.numel() for p in model.parameters())),
        best_validation_loss=float(best_loss),
    )
    return model, metrics


def cross_fitted_risk_proxy(data: ArrayData, seed: int = 20260831) -> dict[str, np.ndarray]:
    x_train = np.concatenate((data.x_num["train"], data.x_cat["train"]), axis=1)
    x_test = np.concatenate((data.x_num["test"], data.x_cat["test"]), axis=1)
    y = data.y["train"]
    if data.task == "regression":
        model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8, max_features=0.8, n_jobs=1, random_state=seed)
        cv = KFold(5, shuffle=True, random_state=seed)
        m_train = cross_val_predict(model, x_train, y, cv=cv, n_jobs=1, method="predict")
        model.fit(x_train, y)
        m_test = model.predict(x_test)
        residual2 = (y - m_train) ** 2
        noise_model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.8, n_jobs=1, random_state=seed + 1)
        noise_cv = KFold(5, shuffle=True, random_state=seed + 2)
        sigma_train = np.maximum(
            cross_val_predict(noise_model, x_train, residual2, cv=noise_cv, n_jobs=1, method="predict"),
            1e-6,
        )
    else:
        model = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=8, max_features=0.8, n_jobs=1, random_state=seed)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        m_train = cross_val_predict(model, x_train, y, cv=cv, n_jobs=1, method="predict_proba")
        model.fit(x_train, y)
        m_test = model.predict_proba(x_test)
        # Align absent-fold classes defensively; the frozen panel has enough of every class.
        if m_train.shape[1] != data.n_classes:
            raise AssertionError("OOF class-probability width mismatch")
        sigma_train = np.maximum(1.0 - np.square(m_train).sum(axis=1), 1e-6)
    return {
        "m_train": np.asarray(m_train),
        "m_test": np.asarray(m_test),
        "sigma_train": np.asarray(sigma_train),
    }


def retrieval_diagnostics(
    model: nn.Module,
    data: ArrayData,
    device: torch.device,
    proxy: dict[str, np.ndarray] | None = None,
    limit: int = 128,
) -> dict[str, float]:
    if isinstance(model, MLPModel):
        return {}
    _, indices, distances = predict_model(model, data, "test", device, True, limit)
    assert indices is not None and distances is not None
    qn = min(limit, len(data.y["test"]))
    if proxy is None:
        proxy = cross_fitted_risk_proxy(data)
    m_train, m_test, sigma = proxy["m_train"], proxy["m_test"][:qn], proxy["sigma_train"]
    correlations, top_risk, top_mismatch, top_noise, overlaps = [], [], [], [], []
    rng = np.random.default_rng(20260831)
    for q in range(qn):
        if m_train.ndim == 1:
            mismatch = (m_train - m_test[q]) ** 2
        else:
            mismatch = np.square(m_train - m_test[q]).sum(axis=1)
        risk = mismatch + sigma
        candidate = rng.choice(len(risk), size=min(512, len(risk)), replace=False)
        rho = spearmanr(distances[q, candidate], risk[candidate]).statistic
        correlations.append(float(rho) if np.isfinite(rho) else 0.0)
        idx = indices[q]
        top_risk.append(float(risk[idx].mean()))
        top_mismatch.append(float(mismatch[idx].mean()))
        top_noise.append(float(sigma[idx].mean()))
        oracle = np.argpartition(risk, min(len(idx), len(risk) - 1))[:len(idx)]
        overlaps.append(len(set(idx.tolist()) & set(oracle.tolist())) / len(idx))
    selected = indices.reshape(-1)
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    counts = np.bincount(selected, minlength=len(data.y["train"])).astype(float)
    probs = counts[counts > 0] / counts.sum()
    frequency_entropy = float(-(probs * np.log(probs)).sum() / math.log(len(counts)))
    y_neighbors = data.y["train"][indices]
    if data.task == "regression":
        within_var = float(np.var(y_neighbors, axis=1).mean())
        label_entropy = float("nan")
        target_consistency = float(np.square(y_neighbors - data.y["test"][:qn, None]).mean())
        residual_train = data.y["train"] - m_train
        residual_test = data.y["test"][:qn] - m_test
        residual_consistency = float(
            np.square(residual_train[indices] - residual_test[:, None]).mean()
        )
    else:
        within_var = float("nan")
        ent = []
        for row in y_neighbors:
            p = np.bincount(row, minlength=data.n_classes).astype(float) / len(row)
            p = p[p > 0]
            ent.append(-(p * np.log(p)).sum())
        label_entropy = float(np.mean(ent))
        target_consistency = float(np.mean(y_neighbors == data.y["test"][:qn, None]))
        train_onehot = np.eye(data.n_classes)[data.y["train"]]
        test_onehot = np.eye(data.n_classes)[data.y["test"][:qn]]
        residual_train = train_onehot - m_train
        residual_test = test_onehot - m_test
        residual_consistency = float(
            np.square(residual_train[indices] - residual_test[:, None, :]).sum(axis=2).mean()
        )
    return {
        "risk_spearman": float(np.mean(correlations)),
        "topk_proxy_risk": float(np.mean(top_risk)),
        "topk_target_mismatch": float(np.mean(top_mismatch)),
        "topk_candidate_noise": float(np.mean(top_noise)),
        "oracle_topk_overlap": float(np.mean(overlaps)),
        "candidate_frequency_entropy": frequency_entropy,
        "mean_selected_retrieval_distance": float(selected_distances.mean()),
        "neighbor_target_consistency": target_consistency,
        "neighbor_residual_consistency": residual_consistency,
        "neighbor_label_entropy": label_entropy,
        "within_neighborhood_target_variance": within_var,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
