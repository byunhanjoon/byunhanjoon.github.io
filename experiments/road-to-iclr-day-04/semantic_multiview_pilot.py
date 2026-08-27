"""Neural-only semantic multi-view pilot for the Day 4 continuation.

The experiment pairs two deterministic charts of every numerical field:

* quantile piecewise-linear encoding (PLE), the established baseline;
* a topology-aware chart: empirical-rank cosine modes for ordered fields and
  Fourier modes with a declared period for cyclic fields.

Both charts contain no target information and are fitted on the training
partition only.  In ``multiview_vicreg`` they use separate tokenizers and a
shared MLP, ResNet, or FT-Transformer backbone.  The supervised loss is
applied to both views and a VICReg-style latent loss aligns the two renderings
of the same row.  ``multiview_noalign`` isolates ordinary two-view ensembling;
``multiview_wrong`` replaces cyclic adjacency by a fixed bin permutation.

This is a screening runner, not a claim of benchmark superiority.  In
particular, the cyclic field declarations below come from the official TabReD
preprocessing scripts and are deliberately explicit rather than inferred from
the target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
TABRED_ROOT = Path(
    "/home/byunhanjoon/2027ICLR/projects/multifeature_ple_tabular/data/tabred_legacy"
)
PARTS = ("train", "val", "test")
View = Literal["ple", "topology", "wrong"]

# Negative indices are intentional: the official preprocessing scripts append
# these timestamp-derived columns in this order after the anonymous numerical
# fields.  Origins are learned from X_train; periods are schema declarations.
CYCLIC_SUFFIXES: dict[str, tuple[tuple[str, int, float], ...]] = {
    "weather": (
        ("day_of_week", -5, 7.0),
        ("day_of_month", -4, 31.0),
        ("minute_of_day", -3, 1440.0),
        ("hour_of_day", -2, 24.0),
        ("month", -1, 12.0),
    ),
    "cooking-time": (
        ("day_of_week", -3, 7.0),
        ("minute_of_day", -2, 1440.0),
        ("hour_of_day", -1, 24.0),
    ),
    "delivery-eta": (
        ("day_of_week", -3, 7.0),
        ("minute_of_day", -2, 1440.0),
        ("hour_of_day", -1, 24.0),
    ),
    "maps-routing": (
        ("day_of_week", -3, 7.0),
        ("minute_of_day", -2, 1440.0),
        ("hour_of_day", -1, 24.0),
    ),
}


@dataclass
class SplitData:
    x_num: dict[str, np.ndarray]
    x_bin: dict[str, np.ndarray] | None
    x_cat: dict[str, np.ndarray] | None
    y: dict[str, np.ndarray]
    y_mean: float
    y_scale: float
    category_cardinalities: list[int]
    cyclic_columns: list[int]
    cyclic_names: list[str]
    cyclic_periods: list[float]
    cyclic_origins: list[float]
    split_sizes_full: dict[str, int]


def _optional(directory: Path, stem: str, indices: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    if not (directory / f"{stem}_train.npy").exists():
        return None
    return {
        part: np.asarray(
            np.load(directory / f"{stem}_{part}.npy", mmap_mode="r")[indices[part]]
        )
        for part in PARTS
    }


def _impute_and_standardize(
    arrays: dict[str, np.ndarray], *, standardize: bool
) -> dict[str, np.ndarray]:
    train = np.asarray(arrays["train"], dtype=np.float64)
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled: dict[str, np.ndarray] = {}
    for part, source in arrays.items():
        values = np.asarray(source, dtype=np.float64).copy()
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = median[np.where(bad)[1]]
        filled[part] = values
    if standardize:
        mean = filled["train"].mean(axis=0)
        scale = filled["train"].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        filled = {part: (values - mean) / scale for part, values in filled.items()}
    return {
        part: np.ascontiguousarray(values, dtype=np.float32)
        for part, values in filled.items()
    }


def _encode_categories(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[int]]:
    encoded = {
        part: np.zeros(np.asarray(values).shape, dtype=np.int64)
        for part, values in arrays.items()
    }
    cardinalities: list[int] = []
    for column in range(arrays["train"].shape[1]):
        levels = np.unique(arrays["train"][:, column])
        cardinalities.append(len(levels) + 1)  # zero is unknown/missing
        for part, source in arrays.items():
            values = source[:, column]
            positions = np.searchsorted(levels, values)
            clipped = np.minimum(positions, max(len(levels) - 1, 0))
            valid = (positions < len(levels)) & (levels[clipped] == values)
            encoded[part][:, column] = np.where(valid, positions + 1, 0)
    return encoded, cardinalities


def load_tabred(
    name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    sample_seed: int,
    root: Path = TABRED_ROOT,
) -> SplitData:
    """Load deterministic subsets of the official temporal partitions."""
    directory = root / name
    if not directory.exists():
        raise FileNotFoundError(directory)
    limits = {"train": max_train_rows, "val": max_eval_rows, "test": max_eval_rows}
    split_sizes_full: dict[str, int] = {}
    indices: dict[str, np.ndarray] = {}
    for offset, part in enumerate(PARTS):
        size = len(np.load(directory / f"Y_{part}.npy", mmap_mode="r"))
        split_sizes_full[part] = size
        if size > limits[part]:
            indices[part] = np.sort(
                np.random.default_rng(sample_seed + offset).choice(
                    size, limits[part], replace=False
                )
            )
        else:
            indices[part] = np.arange(size)

    raw_num = _optional(directory, "X_num", indices)
    if raw_num is None:
        raise ValueError(f"{name} has no numerical fields")
    x_num = _impute_and_standardize(raw_num, standardize=False)
    raw_bin = _optional(directory, "X_bin", indices)
    x_bin = (
        _impute_and_standardize(raw_bin, standardize=True)
        if raw_bin is not None
        else None
    )
    raw_cat = _optional(directory, "X_cat", indices)
    if raw_cat is not None:
        x_cat, cardinalities = _encode_categories(raw_cat)
    else:
        x_cat, cardinalities = None, []

    y_raw = _optional(directory, "Y", indices)
    assert y_raw is not None
    y_mean = float(np.mean(y_raw["train"]))
    y_scale = float(np.std(y_raw["train"])) or 1.0
    y = {
        part: np.ascontiguousarray((values - y_mean) / y_scale, dtype=np.float32)
        for part, values in y_raw.items()
    }

    n_num = x_num["train"].shape[1]
    declarations = CYCLIC_SUFFIXES.get(name, ())
    cyclic_names, cyclic_columns, cyclic_periods, cyclic_origins = [], [], [], []
    for field_name, suffix, period in declarations:
        column = suffix if suffix >= 0 else n_num + suffix
        if not 0 <= column < n_num:
            raise ValueError(f"invalid cyclic suffix {suffix} for {name}/{n_num}")
        cyclic_names.append(field_name)
        cyclic_columns.append(column)
        cyclic_periods.append(period)
        cyclic_origins.append(float(np.min(x_num["train"][:, column])))
    return SplitData(
        x_num=x_num,
        x_bin=x_bin,
        x_cat=x_cat,
        y=y,
        y_mean=y_mean,
        y_scale=y_scale,
        category_cardinalities=cardinalities,
        cyclic_columns=cyclic_columns,
        cyclic_names=cyclic_names,
        cyclic_periods=cyclic_periods,
        cyclic_origins=cyclic_origins,
        split_sizes_full=split_sizes_full,
    )


def quantile_edges(train: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(
        train, np.linspace(0.0, 1.0, n_bins + 1), axis=0
    ).T.astype(np.float64)
    for field in range(len(edges)):
        scale = max(float(np.ptp(edges[field])), 1.0)
        # The model consumes float32.  A float64 machine-epsilon nudge can
        # disappear during the cast and recreate zero-width bins, so make the
        # separation explicitly resolvable in the downstream dtype.
        epsilon = np.finfo(np.float32).eps * scale * 8
        for boundary in range(1, n_bins + 1):
            if edges[field, boundary] <= edges[field, boundary - 1]:
                edges[field, boundary] = edges[field, boundary - 1] + epsilon
    output = edges.astype(np.float32)
    # At large offsets (for example a constant field at 110), the epsilon
    # above can still be smaller than one float32 ULP. Repair in the actual
    # downstream dtype with the next representable value.
    for field in range(len(output)):
        for boundary in range(1, n_bins + 1):
            if output[field, boundary] <= output[field, boundary - 1]:
                output[field, boundary] = np.nextafter(
                    output[field, boundary - 1], np.float32(np.inf)
                )
    if not np.all(np.diff(output, axis=1) > 0):
        raise RuntimeError("quantile-edge separation was lost in float32")
    return output


def ple_basis(x: Tensor, edges: Tensor) -> Tensor:
    left, right = edges[:, :-1], edges[:, 1:]
    ratio = (x[:, :, None] - left[None]) / (right - left)[None]
    n_bins = ratio.shape[-1]
    index = torch.arange(n_bins, device=x.device)
    ratio = torch.where(
        (x[:, :, None] < left[None]) & (index[None, None] > 0),
        torch.zeros((), device=x.device, dtype=x.dtype),
        ratio,
    )
    ratio = torch.where(
        (x[:, :, None] >= right[None]) & (index[None, None] < n_bins - 1),
        torch.ones((), device=x.device, dtype=x.dtype),
        ratio,
    )
    return ratio.clamp(0.0, 1.0)


def ordered_cosine_basis(rank: Tensor, n_bins: int) -> Tensor:
    frequencies = torch.arange(1, n_bins + 1, device=rank.device, dtype=rank.dtype)
    return math.sqrt(2.0) * torch.cos(math.pi * rank[:, :, None] * frequencies)


def cyclic_fourier_basis(phase: Tensor, n_bins: int) -> Tensor:
    if n_bins % 2:
        raise ValueError("the cyclic basis needs an even number of modes")
    harmonics = torch.arange(
        1, n_bins // 2 + 1, device=phase.device, dtype=phase.dtype
    )
    angles = 2.0 * math.pi * phase[:, :, None] * harmonics
    return math.sqrt(2.0) * torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def scrambled_phase(phase: Tensor, permutation: Tensor) -> Tensor:
    """Destroy ring adjacency while retaining equal-sized phase cells."""
    cells = len(permutation)
    scaled = phase * cells
    index = torch.floor(scaled).long().clamp(0, cells - 1)
    fraction = scaled - index.to(scaled.dtype)
    return (permutation[index] + fraction) / cells


class FieldTokenizer(nn.Module):
    def __init__(
        self,
        edges: np.ndarray,
        n_bin_fields: int,
        category_cardinalities: list[int],
        d_token: int,
        *,
        view: View,
        cyclic_columns: list[int],
        cyclic_periods: list[float],
        cyclic_origins: list[float],
        scramble_seed: int = 20260827,
    ) -> None:
        super().__init__()
        self.view = view
        self.n_bins = edges.shape[1] - 1
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float32))
        self.register_buffer(
            "cyclic_columns", torch.as_tensor(cyclic_columns, dtype=torch.long)
        )
        self.register_buffer(
            "cyclic_periods", torch.as_tensor(cyclic_periods, dtype=torch.float32)
        )
        self.register_buffer(
            "cyclic_origins", torch.as_tensor(cyclic_origins, dtype=torch.float32)
        )
        permutation = np.random.default_rng(scramble_seed).permutation(self.n_bins)
        self.register_buffer("phase_permutation", torch.as_tensor(permutation, dtype=torch.long))
        self.num_weight = nn.Parameter(
            torch.empty(edges.shape[0], self.n_bins, d_token)
        )
        self.num_bias = nn.Parameter(torch.zeros(edges.shape[0], d_token))
        nn.init.xavier_uniform_(self.num_weight)
        self.bin_weight = nn.Parameter(torch.empty(n_bin_fields, d_token))
        self.bin_bias = nn.Parameter(torch.zeros(n_bin_fields, d_token))
        if n_bin_fields:
            nn.init.normal_(self.bin_weight, std=1.0 / math.sqrt(d_token))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_token) for cardinality in category_cardinalities
        )
        for embedding in self.cat_embeddings:
            nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_token))

    def numerical_basis(self, x_num: Tensor) -> Tensor:
        p = ple_basis(x_num, self.edges)
        if self.view == "ple":
            return p
        # mean(PLE) is a continuous empirical-rank coordinate in [0, 1].
        basis = ordered_cosine_basis(p.mean(dim=-1), self.n_bins)
        if len(self.cyclic_columns):
            cyclic = x_num[:, self.cyclic_columns]
            phase = torch.remainder(
                (cyclic - self.cyclic_origins) / self.cyclic_periods, 1.0
            )
            if self.view == "wrong":
                phase = scrambled_phase(phase, self.phase_permutation)
            basis[:, self.cyclic_columns] = cyclic_fourier_basis(phase, self.n_bins)
        return basis

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor
    ) -> Tensor:
        basis = self.numerical_basis(x_num)
        output = [
            torch.einsum("nfb,fbd->nfd", basis, self.num_weight) + self.num_bias
        ]
        if self.bin_weight.shape[0]:
            output.append(x_bin[:, :, None] * self.bin_weight[None] + self.bin_bias[None])
        if len(self.cat_embeddings):
            output.append(
                torch.stack(
                    [embedding(x_cat[:, field]) for field, embedding in enumerate(self.cat_embeddings)],
                    dim=1,
                )
            )
        return torch.cat(output, dim=1)


class MLPBackbone(nn.Module):
    def __init__(self, n_fields: int, d_token: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_fields * d_token, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Dropout(0.1), nn.Linear(width, width), nn.GELU()))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.body(tokens.flatten(1))
        return self.head(latent).squeeze(-1), latent


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResNetBackbone(nn.Module):
    def __init__(self, n_fields: int, d_token: int, width: int, depth: int) -> None:
        super().__init__()
        self.input = nn.Linear(n_fields * d_token, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(depth)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.norm(self.blocks(self.input(tokens.flatten(1))))
        return self.head(torch.nn.functional.gelu(latent)).squeeze(-1), latent


class FTTransformerBackbone(nn.Module):
    def __init__(self, n_fields: int, d_token: int, width: int, depth: int) -> None:
        super().__init__()
        del n_fields, width
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        n_heads = 4 if d_token % 4 == 0 else 2
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        cls = self.cls.expand(len(tokens), -1, -1)
        latent = self.norm(self.encoder(torch.cat((cls, tokens), dim=1))[:, 0])
        return self.head(torch.nn.functional.gelu(latent)).squeeze(-1), latent


BACKBONES = {
    "mlp": MLPBackbone,
    "resnet": ResNetBackbone,
    "ft_transformer": FTTransformerBackbone,
}
DUAL_METHODS = {"multiview_noalign", "multiview_vicreg", "multiview_wrong"}


class SemanticMultiViewModel(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        backbone: str,
        edges: np.ndarray,
        n_bin_fields: int,
        category_cardinalities: list[int],
        cyclic_columns: list[int],
        cyclic_periods: list[float],
        cyclic_origins: list[float],
        d_token: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        n_fields = edges.shape[0] + n_bin_fields + len(category_cardinalities)
        # Construct first for paired backbone initialization across all methods.
        self.backbone = BACKBONES[backbone](n_fields, d_token, width, depth)
        common = dict(
            edges=edges,
            n_bin_fields=n_bin_fields,
            category_cardinalities=category_cardinalities,
            d_token=d_token,
            cyclic_columns=cyclic_columns,
            cyclic_periods=cyclic_periods,
            cyclic_origins=cyclic_origins,
        )
        self.method = method
        if method == "ple":
            self.primary = FieldTokenizer(view="ple", **common)
            self.secondary = None
        elif method in {"topology", "topology_wrong"}:
            self.primary = FieldTokenizer(
                view="topology" if method == "topology" else "wrong", **common
            )
            self.secondary = None
        elif method in DUAL_METHODS:
            self.primary = FieldTokenizer(view="ple", **common)
            secondary_view: View = "wrong" if method == "multiview_wrong" else "topology"
            self.secondary = FieldTokenizer(view=secondary_view, **common)
        else:
            raise KeyError(method)

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        first_prediction, first_latent = self.backbone(
            self.primary(x_num, x_bin, x_cat)
        )
        if self.secondary is None:
            return first_prediction, first_latent, None, None
        second_prediction, second_latent = self.backbone(
            self.secondary(x_num, x_bin, x_cat)
        )
        return first_prediction, first_latent, second_prediction, second_latent


def vicreg_loss(first: Tensor, second: Tensor) -> tuple[Tensor, dict[str, float]]:
    first = first.float()
    second = second.float()
    invariance = torch.nn.functional.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        torch.relu(1.0 - first_std).mean() + torch.relu(1.0 - second_std).mean()
    )

    def covariance_penalty(centered: Tensor) -> Tensor:
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        diagonal = torch.diagonal(covariance)
        return (covariance.square().sum() - diagonal.square().sum()) / covariance.shape[0]

    covariance = 0.5 * (
        covariance_penalty(first_centered) + covariance_penalty(second_centered)
    )
    total = 25.0 * invariance + 25.0 * variance + covariance
    return total, {
        "invariance": float(invariance.detach().cpu()),
        "variance": float(variance.detach().cpu()),
        "covariance": float(covariance.detach().cpu()),
    }


def _empty_fields(rows: int, columns: int, dtype: torch.dtype) -> Tensor:
    return torch.empty((rows, columns), dtype=dtype)


def make_loader(
    data: SplitData,
    part: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    rows = len(data.y[part])
    x_bin = (
        data.x_bin[part]
        if data.x_bin is not None
        else np.empty((rows, 0), dtype=np.float32)
    )
    x_cat = (
        data.x_cat[part]
        if data.x_cat is not None
        else np.empty((rows, 0), dtype=np.int64)
    )
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data.x_num[part]),
            torch.from_numpy(x_bin),
            torch.from_numpy(x_cat),
            torch.from_numpy(data.y[part]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        num_workers=0,
    )


@torch.inference_mode()
def evaluate(
    model: SemanticMultiViewModel,
    loader: DataLoader,
    device: torch.device,
    y_scale: float,
) -> dict[str, float]:
    model.eval()
    prediction, first_all, second_all, target = [], [], [], []
    for x_num, x_bin, x_cat, y in loader:
        x_num, x_bin, x_cat = x_num.to(device), x_bin.to(device), x_cat.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            first, _, second, _ = model(x_num, x_bin, x_cat)
        combined = first if second is None else 0.5 * (first + second)
        prediction.append(combined.float().cpu())
        first_all.append(first.float().cpu())
        if second is not None:
            second_all.append(second.float().cpu())
        target.append(y)
    pred = torch.cat(prediction).numpy()
    first = torch.cat(first_all).numpy()
    truth = torch.cat(target).numpy()
    mse = float(np.mean((pred - truth) ** 2))
    first_mse = float(np.mean((first - truth) ** 2))
    second_mse = (
        float(np.mean((torch.cat(second_all).numpy() - truth) ** 2))
        if second_all
        else first_mse
    )
    return {
        "loss": mse,
        "rmse": math.sqrt(mse) * y_scale,
        "primary_loss": first_mse,
        "secondary_loss": second_mse,
    }


def train_one(
    data: SplitData,
    *,
    method: str,
    backbone: str,
    seed: int,
    device: str,
    n_bins: int,
    d_token: int,
    width: int,
    depth: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    alignment_weight: float,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    edges = quantile_edges(data.x_num["train"], n_bins)
    n_bin_fields = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
    model = SemanticMultiViewModel(
        method=method,
        backbone=backbone,
        edges=edges,
        n_bin_fields=n_bin_fields,
        category_cardinalities=data.category_cardinalities,
        cyclic_columns=data.cyclic_columns,
        cyclic_periods=data.cyclic_periods,
        cyclic_origins=data.cyclic_origins,
        d_token=d_token,
        width=width,
        depth=depth,
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = make_loader(
        data, "train", batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = make_loader(
        data, "val", batch_size=batch_size * 2, shuffle=False, seed=seed
    )
    test_loader = make_loader(
        data, "test", batch_size=batch_size * 2, shuffle=False, seed=seed
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    last_alignment = {"invariance": 0.0, "variance": 0.0, "covariance": 0.0}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_bin, x_cat, y in train_loader:
            x_num, x_bin, x_cat, y = (
                x_num.to(resolved, non_blocking=True),
                x_bin.to(resolved, non_blocking=True),
                x_cat.to(resolved, non_blocking=True),
                y.to(resolved, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                first, first_latent, second, second_latent = model(x_num, x_bin, x_cat)
                if second is None:
                    supervised = torch.nn.functional.mse_loss(first, y)
                else:
                    supervised = 0.5 * (
                        torch.nn.functional.mse_loss(first, y)
                        + torch.nn.functional.mse_loss(second, y)
                    )
            if method in {"multiview_vicreg", "multiview_wrong"}:
                assert second_latent is not None
                alignment, last_alignment = vicreg_loss(first_latent, second_latent)
                loss = supervised.float() + alignment_weight * alignment
            else:
                loss = supervised.float()
            loss.backward()
            optimizer.step()
        validation = evaluate(model, val_loader, resolved, data.y_scale)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, stale = validation["loss"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation = evaluate(model, val_loader, resolved, data.y_scale)
    test = evaluate(model, test_loader, resolved, data.y_scale)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "val_loss": validation["loss"],
        "val_rmse": validation["rmse"],
        "val_primary_loss": validation["primary_loss"],
        "val_secondary_loss": validation["secondary_loss"],
        "test_loss": test["loss"],
        "test_rmse": test["rmse"],
        "test_primary_loss": test["primary_loss"],
        "test_secondary_loss": test["secondary_loss"],
        "alignment_invariance_last": last_alignment["invariance"],
        "alignment_variance_last": last_alignment["variance"],
        "alignment_covariance_last": last_alignment["covariance"],
        "train_seconds": time.perf_counter() - started,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["weather", "cooking-time", "delivery-eta"]
    )
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "resnet", "ft_transformer"]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "ple",
            "topology",
            "topology_wrong",
            "multiview_noalign",
            "multiview_vicreg",
            "multiview_wrong",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260827])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--max-eval-rows", type=int, default=15_000)
    parser.add_argument("--sample-seed", type=int, default=20260827)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--d-token", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results/semantic_multiview_pilot.csv"
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (row["dataset"], row["model"], int(row["seed"]), row["method"])
        for row in rows
    }
    metadata: dict[str, object] = {
        "protocol": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datasets": {},
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else None,
    }
    for dataset_name in args.datasets:
        data = load_tabred(
            dataset_name,
            max_train_rows=args.max_train_rows,
            max_eval_rows=args.max_eval_rows,
            sample_seed=args.sample_seed,
        )
        metadata["datasets"][dataset_name] = {
            "full_split_sizes": data.split_sizes_full,
            "used_split_sizes": {part: len(data.y[part]) for part in PARTS},
            "numerical_fields": data.x_num["train"].shape[1],
            "binary_fields": 0 if data.x_bin is None else data.x_bin["train"].shape[1],
            "categorical_cardinalities": data.category_cardinalities,
            "cyclic_fields": [
                {
                    "name": name,
                    "column": column,
                    "period": period,
                    "origin_from_train": origin,
                }
                for name, column, period, origin in zip(
                    data.cyclic_names,
                    data.cyclic_columns,
                    data.cyclic_periods,
                    data.cyclic_origins,
                )
            ],
        }
        for model_name in args.models:
            batch_size = min(args.batch_size, 256) if model_name == "ft_transformer" else args.batch_size
            for seed in args.seeds:
                for method in args.methods:
                    key = (dataset_name, model_name, seed, method)
                    if key in completed:
                        continue
                    result = train_one(
                        data,
                        method=method,
                        backbone=model_name,
                        seed=seed,
                        device=args.device,
                        n_bins=args.n_bins,
                        d_token=args.d_token,
                        width=args.width,
                        depth=args.depth,
                        batch_size=batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        alignment_weight=args.alignment_weight,
                    )
                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "method": method,
                        "n_train": len(data.y["train"]),
                        "n_val": len(data.y["val"]),
                        "n_test": len(data.y["test"]),
                        "n_num": data.x_num["train"].shape[1],
                        "n_bin": 0 if data.x_bin is None else data.x_bin["train"].shape[1],
                        "n_cat": len(data.category_cardinalities),
                        "n_cyclic": len(data.cyclic_columns),
                        "n_bins": args.n_bins,
                        "d_token": args.d_token,
                        "alignment_weight": args.alignment_weight,
                        **result,
                    }
                    rows.append(row)
                    completed.add(key)
                    write_rows(args.output, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
