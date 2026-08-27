#!/usr/bin/env python3
"""Exact-support residual-token transfer pilot for MLP, ResNet, and FT-Transformer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_multiview_pilot import (
    MLPBackbone,
    PARTS,
    ResNetBackbone,
    SplitData,
    load_tabred,
    ple_basis,
    quantile_edges,
)


HERE = Path(__file__).resolve().parent
METHODS = ("qple", "tple", "qple_bin_control", "qple_support", "tple_support")
SUPPORT_METHODS = {"qple_bin_control", "qple_support", "tple_support"}


@dataclass
class Encodings:
    qple_edges: np.ndarray
    tple_edges: np.ndarray
    selected_columns: list[int]
    cardinalities: list[int]
    exact_codes: dict[str, np.ndarray]
    bin_codes: dict[str, np.ndarray]


def target_aware_edges(
    train: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    min_samples_leaf: int,
    min_impurity_decrease: float,
) -> np.ndarray:
    """Fixed-width T-PLE edges; unused tail bins are outside train support."""

    output = np.empty((train.shape[1], n_bins + 1), dtype=np.float64)
    fallback = quantile_edges(train, n_bins).astype(np.float64)
    for field in range(train.shape[1]):
        values = train[:, field]
        if np.all(values == values[0]):
            output[field] = fallback[field]
            continue
        tree = DecisionTreeRegressor(
            max_leaf_nodes=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=0,
        ).fit(values[:, None], target).tree_
        thresholds = tree.threshold[tree.children_left != tree.children_right]
        knots = np.unique(np.r_[values.min(), thresholds, values.max()])
        scale = max(float(np.ptp(values)), 1.0)
        epsilon = np.finfo(np.float32).eps * scale * 8
        padded = list(knots)
        while len(padded) < n_bins + 1:
            padded.append(padded[-1] + epsilon)
        output[field] = padded[: n_bins + 1]
    result = output.astype(np.float32)
    if not np.all(np.diff(result, axis=1) > 0):
        raise RuntimeError("T-PLE edge padding was lost in float32")
    return result


def exact_support_codes(
    parts: dict[str, np.ndarray], max_cardinality: int
) -> tuple[list[int], list[int], dict[str, np.ndarray]]:
    selected, cardinalities, levels_by_field = [], [], []
    for field in range(parts["train"].shape[1]):
        levels = np.unique(parts["train"][:, field])
        if 2 <= len(levels) <= max_cardinality:
            selected.append(field)
            cardinalities.append(len(levels))
            levels_by_field.append(levels)
    codes = {
        part: np.zeros((len(values), len(selected)), dtype=np.int64)
        for part, values in parts.items()
    }
    for support_field, (field, levels) in enumerate(zip(selected, levels_by_field)):
        for part, values in parts.items():
            query = values[:, field]
            positions = np.searchsorted(levels, query)
            clipped = np.minimum(positions, len(levels) - 1)
            known = (positions < len(levels)) & (levels[clipped] == query)
            codes[part][:, support_field] = np.where(known, positions + 1, 0)
    return selected, cardinalities, codes


def quantile_bin_codes(
    parts: dict[str, np.ndarray],
    edges: np.ndarray,
    selected: list[int],
    cardinalities: list[int],
) -> dict[str, np.ndarray]:
    output = {
        part: np.zeros((len(values), len(selected)), dtype=np.int64)
        for part, values in parts.items()
    }
    n_bins = edges.shape[1] - 1
    for support_field, (field, cardinality) in enumerate(
        zip(selected, cardinalities)
    ):
        interior = edges[field, 1:-1]
        for part, values in parts.items():
            raw_bin = np.searchsorted(
                interior, values[:, field], side="right"
            )
            output[part][:, support_field] = (
                np.minimum(raw_bin * cardinality // n_bins, cardinality - 1) + 1
            )
    return output


def prepare_encodings(data: SplitData, config: dict) -> Encodings:
    qple = quantile_edges(data.x_num["train"], config["qple_bins"])
    tple = target_aware_edges(
        data.x_num["train"],
        data.y["train"],
        config["tple_bins"],
        config["tple_min_samples_leaf"],
        config["tple_min_impurity_decrease"],
    )
    selected, cardinalities, exact = exact_support_codes(
        data.x_num, config["support_cardinality_max"]
    )
    binned = quantile_bin_codes(data.x_num, qple, selected, cardinalities)
    return Encodings(qple, tple, selected, cardinalities, exact, binned)


class SupportTokenizer(nn.Module):
    def __init__(
        self,
        *,
        edges: np.ndarray,
        n_bin_fields: int,
        category_cardinalities: list[int],
        support_columns: list[int],
        support_cardinalities: list[int],
        d_token: int,
        use_support: bool,
        support_interface: str = "additive",
        support_gate_mode: str = "sigmoid",
    ) -> None:
        super().__init__()
        if support_interface not in {"additive", "separate"}:
            raise ValueError(f"unknown support interface: {support_interface}")
        self.support_interface = support_interface
        if support_gate_mode not in {"sigmoid", "zero_linear"}:
            raise ValueError(f"unknown support gate mode: {support_gate_mode}")
        self.support_gate_mode = support_gate_mode
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float32))
        self.register_buffer(
            "support_columns", torch.as_tensor(support_columns, dtype=torch.long)
        )
        self.num_weight = nn.Parameter(
            torch.empty(edges.shape[0], edges.shape[1] - 1, d_token)
        )
        self.num_bias = nn.Parameter(torch.zeros(edges.shape[0], d_token))
        nn.init.xavier_uniform_(self.num_weight)
        self.bin_weight = nn.Parameter(torch.empty(n_bin_fields, d_token))
        self.bin_bias = nn.Parameter(torch.zeros(n_bin_fields, d_token))
        if n_bin_fields:
            nn.init.normal_(self.bin_weight, std=1.0 / math.sqrt(d_token))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_token)
            for cardinality in category_cardinalities
        )
        for embedding in self.cat_embeddings:
            nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_token))
        self.support_embeddings = (
            nn.ModuleList(
                nn.Embedding(cardinality + 1, d_token, padding_idx=0)
                for cardinality in support_cardinalities
            )
            if use_support
            else nn.ModuleList()
        )
        if use_support:
            for embedding in self.support_embeddings:
                nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_token))
                with torch.no_grad():
                    embedding.weight[0].zero_()
            self.support_gate_logits = nn.Parameter(
                torch.full((len(support_cardinalities),), -1.0)
                if support_gate_mode == "sigmoid"
                else torch.zeros(len(support_cardinalities))
            )
        else:
            self.register_parameter("support_gate_logits", None)

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor, support_codes: Tensor
    ) -> Tensor:
        basis = ple_basis(x_num, self.edges)
        numerical = (
            torch.einsum("nfb,fbd->nfd", basis, self.num_weight)
            + self.num_bias
        )
        if len(self.support_embeddings):
            residual = torch.stack(
                [
                    embedding(support_codes[:, field])
                    for field, embedding in enumerate(self.support_embeddings)
                ],
                dim=1,
            )
            gate = (
                torch.sigmoid(self.support_gate_logits)
                if self.support_gate_mode == "sigmoid"
                else self.support_gate_logits
            )
            residual = residual * gate[None, :, None]
            if self.support_interface == "additive":
                numerical[:, self.support_columns] = (
                    numerical[:, self.support_columns] + residual
                )
        output = [numerical]
        if len(self.support_embeddings) and self.support_interface == "separate":
            output.append(residual)
        if self.bin_weight.shape[0]:
            output.append(
                x_bin[:, :, None] * self.bin_weight[None] + self.bin_bias[None]
            )
        if len(self.cat_embeddings):
            output.append(
                torch.stack(
                    [
                        embedding(x_cat[:, field])
                        for field, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
            )
        return torch.cat(output, dim=1)


class MatchedFTTransformer(nn.Module):
    def __init__(
        self, n_fields: int, d_token: int, depth: int, feedforward_width: int, dropout: float
    ) -> None:
        super().__init__()
        del n_fields
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=4,
            dim_feedforward=feedforward_width,
            dropout=dropout,
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


class HybridAttentionMLP(nn.Module):
    """MLP predictor plus a function-preserving zero-start attention residual."""

    def __init__(
        self,
        n_fields: int,
        d_token: int,
        width: int,
        depth: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = MLPBackbone(n_fields, d_token, width, depth)
        self.attention = MatchedFTTransformer(
            n_fields, d_token, depth, feedforward_width, dropout
        )
        self.attention_gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        mlp_prediction, mlp_latent = self.mlp(tokens)
        attention_prediction, _ = self.attention(tokens)
        return (
            mlp_prediction + self.attention_gate * attention_prediction,
            mlp_latent,
        )


class SupportModel(nn.Module):
    def __init__(
        self,
        *,
        data: SplitData,
        encoding: Encodings,
        method: str,
        architecture: str,
        d_token: int,
        width: int,
        depth: int,
        ft_feedforward_width: int,
        dropout: float,
        support_interface: str = "additive",
        support_gate_mode: str = "sigmoid",
    ) -> None:
        super().__init__()
        use_support = method in SUPPORT_METHODS
        edges = encoding.tple_edges if method.startswith("tple") else encoding.qple_edges
        n_bin = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
        self.tokenizer = SupportTokenizer(
            edges=edges,
            n_bin_fields=n_bin,
            category_cardinalities=data.category_cardinalities,
            support_columns=encoding.selected_columns,
            support_cardinalities=encoding.cardinalities,
            d_token=d_token,
            use_support=use_support,
            support_interface=support_interface,
            support_gate_mode=support_gate_mode,
        )
        n_fields = data.x_num["train"].shape[1] + n_bin + len(data.category_cardinalities)
        if use_support and support_interface == "separate":
            n_fields += len(encoding.selected_columns)
        if architecture == "mlp":
            self.backbone = MLPBackbone(n_fields, d_token, width, depth)
        elif architecture == "resnet":
            self.backbone = ResNetBackbone(n_fields, d_token, width, depth)
        elif architecture == "ft_transformer":
            self.backbone = MatchedFTTransformer(
                n_fields, d_token, depth, ft_feedforward_width, dropout
            )
        elif architecture == "hybrid":
            self.backbone = HybridAttentionMLP(
                n_fields,
                d_token,
                width,
                depth,
                ft_feedforward_width,
                dropout,
            )
        else:
            raise KeyError(architecture)

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor, support_codes: Tensor
    ) -> Tensor:
        prediction, _ = self.backbone(
            self.tokenizer(x_num, x_bin, x_cat, support_codes)
        )
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_matched_model(
    data: SplitData,
    encoding: Encodings,
    config: dict,
    method: str,
    architecture: str,
    target_parameters: int | None,
) -> tuple[SupportModel, int, int]:
    base_width = config["width"]
    base_ff = config["ft_feedforward_width"]

    def build(width: int, ff_width: int) -> SupportModel:
        return SupportModel(
            data=data,
            encoding=encoding,
            method=method,
            architecture=architecture,
            d_token=config["d_token"],
            width=width,
            depth=config["depth"],
            ft_feedforward_width=ff_width,
            dropout=config["dropout"],
            support_gate_mode=config.get("support_gate_mode", "sigmoid"),
        )

    if target_parameters is None:
        return build(base_width, base_ff), base_width, base_ff
    if architecture == "ft_transformer":
        candidates = range(4, 257)
        chosen = min(
            candidates,
            key=lambda value: abs(parameter_count(build(base_width, value)) - target_parameters),
        )
        return build(base_width, chosen), base_width, chosen
    candidates = range(8, 513)
    chosen = min(
        candidates,
        key=lambda value: abs(parameter_count(build(value, base_ff)) - target_parameters),
    )
    return build(chosen, base_ff), chosen, base_ff


def codes_for_method(encoding: Encodings, method: str) -> dict[str, np.ndarray]:
    if method == "qple_bin_control":
        return encoding.bin_codes
    if method in {"qple_support", "tple_support"}:
        return encoding.exact_codes
    return {
        part: np.empty((len(values), 0), dtype=np.int64)
        for part, values in encoding.exact_codes.items()
    }


def make_loader(
    data: SplitData,
    codes: dict[str, np.ndarray],
    part: str,
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
            torch.from_numpy(codes[part]),
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
    model: SupportModel, loader: DataLoader, device: torch.device, y_scale: float
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x_num, x_bin, x_cat, codes, target in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                x_num.to(device), x_bin.to(device), x_cat.to(device), codes.to(device)
            )
        predictions.append(prediction.float().cpu())
        targets.append(target)
    pred = torch.cat(predictions).numpy()
    truth = torch.cat(targets).numpy()
    mse = float(np.mean((pred - truth) ** 2))
    return {"loss": mse, "rmse": math.sqrt(mse) * y_scale}, pred


def train_one(
    data: SplitData,
    encoding: Encodings,
    config: dict,
    method: str,
    architecture: str,
    seed: int,
    device: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    reference, _, _ = build_matched_model(
        data, encoding, config, "qple_support", architecture, None
    )
    target_parameters = parameter_count(reference)
    del reference
    # Re-seed so constructing the parameter-budget reference cannot perturb init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, width, ff_width = build_matched_model(
        data, encoding, config, method, architecture, target_parameters
    )
    model = model.to(resolved)
    codes = codes_for_method(encoding, method)
    batch_size = (
        min(config["batch_size"], 256)
        if architecture in {"ft_transformer", "hybrid"}
        else config["batch_size"]
    )
    loaders = {
        part: make_loader(
            data, codes, part, batch_size if part == "train" else batch_size * 2,
            part == "train", seed,
        )
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for x_num, x_bin, x_cat, support_codes, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                prediction = model(
                    x_num.to(resolved, non_blocking=True),
                    x_bin.to(resolved, non_blocking=True),
                    x_cat.to(resolved, non_blocking=True),
                    support_codes.to(resolved, non_blocking=True),
                )
                loss = torch.nn.functional.mse_loss(
                    prediction, target.to(resolved, non_blocking=True)
                )
            loss.backward()
            optimizer.step()
        validation, _ = evaluate(model, loaders["val"], resolved, data.y_scale)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, stale = validation["loss"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > config["patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation, val_prediction = evaluate(model, loaders["val"], resolved, data.y_scale)
    test, test_prediction = evaluate(model, loaders["test"], resolved, data.y_scale)
    gates = model.tokenizer.support_gate_logits
    return (
        {
            "parameters": parameter_count(model),
            "target_parameters": target_parameters,
            "matched_width": width,
            "matched_ft_feedforward_width": ff_width,
            "best_epoch": best_epoch,
            "val_loss": validation["loss"],
            "val_rmse": validation["rmse"],
            "test_loss": test["loss"],
            "test_rmse": test["rmse"],
            "mean_support_gate": (
                float(
                    (
                        torch.sigmoid(gates)
                        if model.tokenizer.support_gate_mode == "sigmoid"
                        else gates
                    ).mean().detach().cpu()
                )
                if gates is not None and len(gates)
                else 0.0
            ),
            "train_seconds": time.perf_counter() - started,
        },
        val_prediction,
        test_prediction,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def record_path(output: Path, dataset: str, architecture: str, seed: int, method: str) -> Path:
    return output.parent / f"{output.stem}_predictions" / f"{dataset}__{architecture}__{seed}__{method}.npz"


def analyze(config: dict, output: Path) -> dict[str, object]:
    frame = pd.read_csv(output)
    development = frame[frame.dataset.isin(config["development_datasets"])]
    cells = []
    for (dataset, model, seed), group in development.groupby(["dataset", "model", "seed"]):
        indexed = group.set_index("method")
        if not set(METHODS).issubset(indexed.index):
            continue
        q_pass = bool(
            indexed.loc["qple_support", "val_rmse"] < indexed.loc["qple", "val_rmse"]
            and indexed.loc["qple_support", "val_rmse"]
            < indexed.loc["qple_bin_control", "val_rmse"]
        )
        t_pass = bool(
            indexed.loc["tple_support", "val_rmse"] < indexed.loc["tple", "val_rmse"]
        )
        cells.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "q_support_gain_val_pct": 100 * (
                    indexed.loc["qple", "val_rmse"]
                    - indexed.loc["qple_support", "val_rmse"]
                ) / indexed.loc["qple", "val_rmse"],
                "q_support_vs_bin_val_pct": 100 * (
                    indexed.loc["qple_bin_control", "val_rmse"]
                    - indexed.loc["qple_support", "val_rmse"]
                ) / indexed.loc["qple_bin_control", "val_rmse"],
                "t_support_gain_val_pct": 100 * (
                    indexed.loc["tple", "val_rmse"]
                    - indexed.loc["tple_support", "val_rmse"]
                ) / indexed.loc["tple", "val_rmse"],
                "q_support_gain_test_pct": 100 * (
                    indexed.loc["qple", "test_rmse"]
                    - indexed.loc["qple_support", "test_rmse"]
                ) / indexed.loc["qple", "test_rmse"],
                "t_support_gain_test_pct": 100 * (
                    indexed.loc["tple", "test_rmse"]
                    - indexed.loc["tple_support", "test_rmse"]
                ) / indexed.loc["tple", "test_rmse"],
                "cell_gate": q_pass and t_pass,
            }
        )
    cell_frame = pd.DataFrame(cells)
    cell_path = output.with_name(output.stem + "_cells.csv")
    cell_frame.to_csv(cell_path, index=False)
    dataset_gate = {
        dataset: int(group.cell_gate.sum()) >= 2
        for dataset, group in cell_frame.groupby("dataset")
    }
    architecture_gate = any(dataset_gate.values())
    decision = {
        "architecture_gate_passed": architecture_gate,
        "dataset_gates": dataset_gate,
        "passing_cells": int(cell_frame.cell_gate.sum()) if len(cell_frame) else 0,
        "development_cells": int(len(cell_frame)),
        "transfer_authorized": architecture_gate,
    }
    output.with_name(output.stem + "_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Exact-support residual-token transfer pilot",
        "",
        config["status"].capitalize() + ".",
        "",
        "The proposed token adds a gated learned embedding of an exact numerical level to the ordinary PLE field token. Fields are activated by the frozen target-free rule `2 <= train cardinality <= 128`; unseen values receive a zero residual. The bin control has identical tables, gates, backbone shape, and parameter count but indexes them with Q-PLE bins instead of exact levels.",
        "",
        "| Dataset | Model | Q-support val gain | Exact vs bin-control val gain | T-support val gain | Q-support test gain | T-support test gain | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in cell_frame.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['q_support_gain_val_pct']:+.3f}% | "
            f"{row['q_support_vs_bin_val_pct']:+.3f}% | {row['t_support_gain_val_pct']:+.3f}% | "
            f"{row['q_support_gain_test_pct']:+.3f}% | {row['t_support_gain_test_pct']:+.3f}% | "
            f"{'PASS' if row['cell_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"- Architecture gate: **{'PASS' if architecture_gate else 'FAIL'}**.",
            f"- Development dataset gates: `{json.dumps(dataset_gate, sort_keys=True)}`.",
            f"- Delivery ETA transfer: **{'authorized' if architecture_gate else 'not authorized'}** by the frozen rule.",
        ]
    )
    output.with_name(output.stem + "_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=HERE / "support_identity_transfer_config.json"
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results/support_identity_transfer.csv"
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.analyze_only:
        analyze(config, args.output)
        return
    datasets = args.datasets or config["development_datasets"]
    models = args.models or config["architectures"]
    methods = args.methods or config["methods"]
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (row["dataset"], row["model"], int(row["seed"]), row["method"])
        for row in rows
    }
    metadata: dict[str, object] = {
        "config": config,
        "datasets": {},
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda") else None,
    }
    for dataset_name in datasets:
        data = load_tabred(
            dataset_name,
            max_train_rows=config["max_train_rows"],
            max_eval_rows=config["max_eval_rows"],
            sample_seed=config["sample_seed"],
        )
        encoding = prepare_encodings(data, config)
        metadata["datasets"][dataset_name] = {
            "selected_support_columns": encoding.selected_columns,
            "support_cardinalities": encoding.cardinalities,
            "n_num": data.x_num["train"].shape[1],
            "n_selected": len(encoding.selected_columns),
            "full_split_sizes": data.split_sizes_full,
        }
        for model_name in models:
            for method in methods:
                key = (dataset_name, model_name, config["seed"], method)
                if key in completed:
                    print(f"skip {key}", flush=True)
                    continue
                result, val_prediction, test_prediction = train_one(
                    data, encoding, config, method, model_name, config["seed"], args.device
                )
                prediction_path = record_path(
                    args.output, dataset_name, model_name, config["seed"], method
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                with prediction_path.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        validation=val_prediction.astype(np.float32),
                        test=test_prediction.astype(np.float32),
                    )
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": config["seed"],
                    "method": method,
                    "n_train": len(data.y["train"]),
                    "n_val": len(data.y["val"]),
                    "n_test": len(data.y["test"]),
                    "n_num": data.x_num["train"].shape[1],
                    "n_support_fields": len(encoding.selected_columns),
                    "support_levels": sum(encoding.cardinalities),
                    **result,
                }
                rows.append(row)
                completed.add(key)
                write_rows(args.output, rows)
                print(json.dumps(row, sort_keys=True), flush=True)
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    required = {
        (dataset, model, config["seed"], method)
        for dataset in config["development_datasets"]
        for model in config["architectures"]
        for method in config["methods"]
    }
    analysis_methods = {
        "qple",
        "tple",
        "qple_bin_control",
        "qple_support",
        "tple_support",
    }
    if analysis_methods.issubset(config["methods"]) and required.issubset(completed):
        analyze(config, args.output)


if __name__ == "__main__":
    main()
