#!/usr/bin/env python3
"""Validation-selected neural benchmark on frozen state representations.

The runner is deliberately one-cell-at-a-time and resume-safe so the full
matrix can be distributed across both GPUs without duplicate fits.  It uses
the reference FT-Transformer and TabM packages already installed in the frozen
environment.  Test rows are never passed to the training/HPO routine.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tabm
import torch
from rtdl_revisiting_models import FTTransformerBackbone
from scipy import sparse
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mpe import state_balanced_mean, state_loss_table  # noqa: E402
from representations import (  # noqa: E402
    candidate_bandwidths,
    corrupted_mpe_table,
    load_task,
    representation_tables,
    split_row_indices,
)
from ridge_benchmark import DEFAULT_TASKS, ordinary_design  # noqa: E402


TRAINING_SEEDS = [20261101, 20261102, 20261103]
ALIASES = {
    "rbf_normalized": "similarity_same_metric",
    "hierarchy_shortest_path_similarity": "similarity_same_metric",
    "support_complete_categorical": "unknown_embedding",
    "tree_rbf": "similarity_unnormalized",
    "rbf_unnormalized": "similarity_unnormalized",
}


def frozen_trials() -> list[dict[str, Any]]:
    grid = list(
        itertools.product(
            [3e-4, 1e-3, 3e-3],
            [0.0, 1e-4, 1e-3],
            [128, 256],
            [2, 3],
            [0.0, 0.1],
            [512, 2048],
        )
    )
    order = np.random.default_rng(20261301).permutation(len(grid))[:8]
    return [
        {
            "learning_rate": grid[index][0],
            "weight_decay": grid[index][1],
            "width": grid[index][2],
            "depth": grid[index][3],
            "dropout": grid[index][4],
            "batch_size": grid[index][5],
        }
        for index in order
    ]


HPO_TRIALS = frozen_trials()


class MLP(nn.Module):
    def __init__(self, input_size: int, width: int, depth: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_size
        for _ in range(depth):
            blocks.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        blocks.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_size: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_size, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(depth)))
        self.output = nn.Sequential(nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


class DenseFTTransformer(nn.Module):
    def __init__(self, input_size: int, width: int, depth: int, dropout: float):
        super().__init__()
        d_token = 32 if width == 128 else 64
        n_tokens = 8
        self.stem = nn.Linear(input_size, n_tokens * d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token**-0.5)
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.backbone = FTTransformerBackbone(
            d_out=1,
            n_blocks=depth,
            d_block=d_token,
            attention_n_heads=4,
            attention_dropout=dropout,
            ffn_d_hidden=3 * d_token,
            ffn_d_hidden_multiplier=None,
            ffn_dropout=dropout,
            ffn_activation="ReGLU",
            residual_dropout=0.0,
            n_tokens=None,
            linformer_kv_compression_ratio=None,
            linformer_kv_compression_sharing=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).reshape(len(x), self.n_tokens, self.d_token)
        return self.backbone(torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1))


class DenseTabM(nn.Module):
    is_tabm = True

    def __init__(self, input_size: int, width: int, depth: int, dropout: float):
        super().__init__()
        latent = 64
        self.stem = nn.Linear(input_size, latent)
        self.backbone = tabm.TabM.make(
            n_num_features=latent,
            cat_cardinalities=[],
            d_out=1,
            num_embeddings=None,
            n_blocks=depth,
            d_block=width,
            dropout=dropout,
            k=8,
        )

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.stem(x), None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=1)


class MetricTokenizedModel(nn.Module):
    """Apply a frozen-width learned tokenizer to trailing field coordinates.

    For MPE the trailing coordinates are partition weights and this layer is
    exactly ``wV``.  For the categorical control they are training-state plus
    UNK one-hot coordinates, making the same layer a width-D lookup table.
    """

    def __init__(self, backbone: nn.Module, representation_size: int, token_dimension: int):
        super().__init__()
        self.backbone = backbone
        self.representation_size = representation_size
        self.token_dimension = token_dimension
        self.tokenizer = nn.Linear(representation_size, token_dimension, bias=False)
        self.is_tabm = bool(getattr(backbone, "is_tabm", False))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        ordinary = x[:, : x.shape[1] - self.representation_size]
        weights = x[:, x.shape[1] - self.representation_size :]
        token = self.tokenizer(weights)
        return torch.cat([ordinary, token], dim=1) if ordinary.shape[1] else token

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_tabm:
            raise TypeError("forward_members is only defined for a TabM backbone")
        return self.backbone.forward_members(self.transform(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.transform(x))


def make_model(
    backbone: str,
    input_size: int,
    trial: dict[str, Any],
    *,
    tokenized_representation_size: int = 0,
    token_dimension: int = 32,
) -> nn.Module:
    backbone_input_size = (
        input_size - tokenized_representation_size + token_dimension
        if tokenized_representation_size
        else input_size
    )
    kwargs = (backbone_input_size, int(trial["width"]), int(trial["depth"]), float(trial["dropout"]))
    if backbone == "mlp":
        model = MLP(*kwargs)
    elif backbone == "resnet":
        model = ResNet(*kwargs)
    elif backbone == "ft_transformer":
        model = DenseFTTransformer(*kwargs)
    elif backbone == "tabm":
        model = DenseTabM(*kwargs)
    else:
        raise KeyError(backbone)
    if tokenized_representation_size:
        model = MetricTokenizedModel(model, tokenized_representation_size, token_dimension)
    return model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dense_batch(
    design: sparse.csr_matrix | torch.Tensor,
    indices: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Select a dense batch without changing the frozen row ordering.

    Real benchmark designs are small enough in width to reside on the H100s.
    Keeping a cell's immutable design tensor on-device avoids repeating a
    SciPy CSR slice, densification, and host-to-device transfer every epoch.
    The sparse branch is retained for CPU runs and imported callers.
    """
    if isinstance(design, torch.Tensor):
        index = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, device=device)
        return design.index_select(0, index)
    values = design[indices].toarray().astype(np.float32, copy=False)
    return torch.from_numpy(values).to(device, non_blocking=True)


def predict(
    model: nn.Module,
    design: sparse.csr_matrix | torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            prediction = model(dense_batch(design, batch_indices, device)).reshape(-1)
            output.append(prediction.float().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def state_balanced_score(target: np.ndarray, prediction: np.ndarray, states: np.ndarray) -> float:
    return state_balanced_mean((prediction - target) ** 2, states)


def fit_validation(
    design: sparse.csr_matrix | torch.Tensor,
    target: np.ndarray,
    states: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    backbone: str,
    trial: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    max_epochs: int = 300,
    patience: int = 30,
    tokenized_representation_size: int = 0,
    token_dimension: int = 32,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_everything(seed)
    model = make_model(
        backbone, design.shape[1], trial,
        tokenized_representation_size=tokenized_representation_size,
        token_dimension=token_dimension,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(trial["learning_rate"]), weight_decay=float(trial["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(trial["learning_rate"]) * 0.05)
    batch_size = int(trial["batch_size"])
    unique, counts = np.unique(states[train_rows], return_counts=True)
    count_lookup = dict(zip(unique.tolist(), counts.tolist()))
    row_weight = np.asarray(
        [1.0 / count_lookup[state] if state in count_lookup else 0.0 for state in states],
        dtype=np.float32,
    )
    row_weight *= len(train_rows) / row_weight[train_rows].sum()
    target_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(device, non_blocking=True)
    row_weight_tensor = torch.from_numpy(row_weight).to(device, non_blocking=True)
    generator = np.random.default_rng(seed + 991)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    curve = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(max_epochs):
        model.train()
        order = generator.permutation(train_rows)
        training_loss_sum = 0.0
        training_weight_sum = 0.0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            chosen_tensor = torch.as_tensor(chosen, device=device)
            x = dense_batch(design, chosen_tensor, device)
            y = target_tensor.index_select(0, chosen_tensor)
            weight = row_weight_tensor.index_select(0, chosen_tensor)
            optimizer.zero_grad(set_to_none=True)
            if bool(getattr(model, "is_tabm", False)):
                raw = model.forward_members(x).squeeze(-1)
                loss_rows = ((raw - y[:, None]) ** 2).mean(dim=1)
            else:
                raw = model(x).reshape(-1)
                loss_rows = (raw - y) ** 2
            loss = (loss_rows * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            training_loss_sum += float((loss_rows.detach() * weight).sum().cpu())
            training_weight_sum += float(weight.sum().cpu())
        scheduler.step()
        validation_prediction = predict(model, design, validation_rows, device)
        validation_score = state_balanced_score(
            target[validation_rows], validation_prediction, states[validation_rows]
        )
        curve.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss_sum / training_weight_sum,
                "validation_state_balanced_mse": validation_score,
            }
        )
        if validation_score < best_score - 1e-8:
            best_score = validation_score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise AssertionError("no finite validation checkpoint")
    telemetry = {
        "validation_score": best_score,
        "best_epoch": best_epoch,
        "stop_epoch": len(curve),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "tokenizer_parameters": int(
            sum(parameter.numel() for parameter in model.tokenizer.parameters())
            if isinstance(model, MetricTokenizedModel)
            else 0
        ),
        "curve": curve,
    }
    del model
    return best_state, telemetry


def selected_bandwidth(task_name: str, split_index: int, setting: str) -> float:
    path = HERE / "raw" / "ridge_cells" / f"{task_name}__split{split_index}__{setting}.json"
    if path.exists():
        return float(json.loads(path.read_text())["selected_bandwidth"])
    task = load_task(task_name)
    values = candidate_bandwidths(task, split_index)
    return float(values[len(values) // 2])


def evaluate_test(
    state: dict[str, torch.Tensor],
    design: sparse.csr_matrix | torch.Tensor,
    target: np.ndarray,
    raw_target: np.ndarray,
    states: np.ndarray,
    test_rows: np.ndarray,
    backbone: str,
    trial: dict[str, Any],
    target_scale: float,
    device: torch.device,
    *,
    tokenized_representation_size: int = 0,
    token_dimension: int = 32,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model = make_model(
        backbone, design.shape[1], trial,
        tokenized_representation_size=tokenized_representation_size,
        token_dimension=token_dimension,
    ).to(device)
    model.load_state_dict(state)
    prediction = predict(model, design, test_rows, device)
    error = prediction - target[test_rows]
    losses = error**2
    state_table = state_loss_table(losses, states[test_rows])
    summary = {
        "state_balanced_standardized_mse": state_balanced_mean(losses, states[test_rows]),
        "row_weighted_standardized_mse": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean(losses)) * target_scale),
        "mae": float(np.mean(np.abs(error)) * target_scale),
    }
    rows = [
        {
            "state_id": str(state_id),
            "rows": int(np.sum(states[test_rows] == state_id)),
            "standardized_mse": value,
        }
        for state_id, value in state_table.items()
    ]
    del model
    return summary, rows


def run_cell(
    task_name: str,
    split_index: int,
    setting: str,
    backbone: str,
    representation: str,
    device_name: str,
    output: Path,
) -> dict[str, Any]:
    canonical = ALIASES.get(representation, representation)
    uses_metric_tokenizer = canonical == "mpe" or canonical == "mpe_equality" or canonical.startswith("mpe_corrupt_")
    uses_categorical_tokenizer = canonical == "unknown_embedding"
    uses_trainable_tokenizer = uses_metric_tokenizer or uses_categorical_tokenizer
    mpe_implementation_version = 2
    categorical_implementation_version = 2
    cell_id = f"{task_name}__split{split_index}__{setting}__{backbone}__{representation}"
    path = output / f"{cell_id}.json"
    state_path = output / f"{cell_id}__state_metrics.parquet"
    if path.exists() and state_path.exists():
        payload = json.loads(path.read_text())
        valid_version = (
            (not uses_metric_tokenizer or payload.get("mpe_implementation_version") == mpe_implementation_version)
            and (
                not uses_categorical_tokenizer
                or payload.get("categorical_implementation_version") == categorical_implementation_version
            )
        )
        if payload.get("status") == "complete" and valid_version:
            print(f"resume {cell_id}", flush=True)
            return payload
        if payload.get("status") == "complete" and not valid_version:
            print(f"recompute legacy tokenizer cell {cell_id}", flush=True)
    task = load_task(task_name)
    row_parts = split_row_indices(task, split_index)
    bandwidth = selected_bandwidth(task_name, split_index, setting)
    tables, metadata = representation_tables(task, split_index, bandwidth)
    if canonical.startswith("mpe_corrupt_"):
        corruption_index = int(canonical.rsplit("_", 1)[-1])
        table = corrupted_mpe_table(task, split_index, bandwidth, corruption_index)
    else:
        if canonical not in tables:
            raise KeyError(f"{canonical} unavailable for {task_name}; choices={sorted(tables)}")
        table = tables[canonical]
    state_indices = task.row_state_indices()
    representation_design = sparse.csr_matrix(table[state_indices], dtype=np.float32)
    if setting == "full_table":
        ordinary = ordinary_design(task, row_parts["train"])
        design = sparse.hstack([ordinary, representation_design], format="csr")
    else:
        design = representation_design
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target_mean = float(raw_target[row_parts["train"]].mean())
    target_scale = float(raw_target[row_parts["train"]].std()) or 1.0
    target = (raw_target - target_mean) / target_scale
    states = task.rows["field_state"].astype(str).to_numpy()
    device = torch.device(device_name)
    if device.type == "cuda":
        # All frozen real designs fit comfortably in accelerator memory.  This
        # is a transport optimization only: values and row order are identical
        # to the CSR design used by ridge and by the prior neural path.
        training_design: sparse.csr_matrix | torch.Tensor = torch.from_numpy(
            design.toarray().astype(np.float32, copy=False)
        ).to(device, non_blocking=True)
    else:
        training_design = design
    tokenized_representation_size = int(table.shape[1]) if uses_trainable_tokenizer else 0

    hpo = []
    hpo_states = []
    for trial_index, trial in enumerate(HPO_TRIALS):
        best_state, telemetry = fit_validation(
            training_design, target, states, row_parts["train"], row_parts["validation"],
            backbone, trial, TRAINING_SEEDS[0], device,
            tokenized_representation_size=tokenized_representation_size,
            token_dimension=32,
        )
        hpo_states.append(best_state)
        hpo.append({"trial": trial_index, "config": trial, **telemetry})
        print(
            f"{cell_id} hpo={trial_index} val={telemetry['validation_score']:.6f} "
            f"epoch={telemetry['best_epoch']}/{telemetry['stop_epoch']}",
            flush=True,
        )
    selected = min(hpo, key=lambda item: (item["validation_score"], item["trial"]))
    selected_index = int(selected["trial"])
    selected_trial = HPO_TRIALS[selected_index]
    final_states = [hpo_states[selected_index]]
    final_telemetry = [hpo[selected_index]]
    for seed in TRAINING_SEEDS[1:]:
        best_state, telemetry = fit_validation(
            training_design, target, states, row_parts["train"], row_parts["validation"],
            backbone, selected_trial, seed, device,
            tokenized_representation_size=tokenized_representation_size,
            token_dimension=32,
        )
        final_states.append(best_state)
        final_telemetry.append({"seed": seed, **telemetry})

    results, state_rows = [], []
    for seed, model_state, telemetry in zip(TRAINING_SEEDS, final_states, final_telemetry):
        summary, per_state = evaluate_test(
            model_state, training_design, target, raw_target, states, row_parts["test"],
            backbone, selected_trial, target_scale, device,
            tokenized_representation_size=tokenized_representation_size,
            token_dimension=32,
        )
        results.append({"seed": seed, **summary})
        for row in per_state:
            state_rows.append({"seed": seed, **row})
    pd.DataFrame(state_rows).assign(
        task=task_name, split=split_index, setting=setting, backbone=backbone,
        representation=representation,
    ).to_parquet(state_path, index=False, compression="zstd")
    payload = {
        "status": "complete",
        "cell_id": cell_id,
        "task": task_name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "setting": setting,
        "backbone": backbone,
        "representation": representation,
        "canonical_representation": canonical,
        "alias_contract": ALIASES.get(representation),
        "mpe_implementation_version": mpe_implementation_version if uses_metric_tokenizer else None,
        "categorical_implementation_version": (
            categorical_implementation_version if uses_categorical_tokenizer else None
        ),
        "uses_learned_landmark_tokens": uses_metric_tokenizer,
        "uses_learned_categorical_embedding": uses_categorical_tokenizer,
        "uses_trainable_tokenizer": uses_trainable_tokenizer,
        "token_dimension": 32 if uses_trainable_tokenizer else None,
        "tokenizer_parameters": int(table.shape[1] * 32) if uses_trainable_tokenizer else 0,
        "bandwidth": bandwidth,
        "feature_dimension": int(table.shape[1]),
        "design_dimension": int(design.shape[1]),
        "representation_metadata": metadata,
        "hpo_trials": hpo,
        "selected_trial": selected_index,
        "selected_config": selected_trial,
        "training_seeds": TRAINING_SEEDS,
        "test_evaluations": len(TRAINING_SEEDS),
        "results": results,
        "final_fit_telemetry": final_telemetry,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)
    return payload


def available_representations(task_name: str, split_index: int, setting: str) -> list[str]:
    task = load_task(task_name)
    tables, _ = representation_tables(task, split_index, selected_bandwidth(task_name, split_index, setting))
    names = [name for name in tables if name not in ALIASES]
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=DEFAULT_TASKS)
    parser.add_argument("--split", required=True, type=int, choices=range(5))
    parser.add_argument("--setting", required=True, choices=["isolated_field", "full_table"])
    parser.add_argument("--backbone", required=True, choices=["mlp", "resnet", "ft_transformer", "tabm"])
    parser.add_argument("--representation", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "neural_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.representation == "all":
        names = available_representations(args.task, args.split, args.setting)
    elif args.representation == "core":
        available = available_representations(args.task, args.split, args.setting)
        preferred = [
            "mpe", "similarity_same_metric", "similarity_unnormalized", "nystrom",
            "unknown_embedding", "q_ple", "uniform_ple", "mpe_equality",
            "ancestor_multihot", "path_to_root", "laplacian", "node2vec",
            "raw_coordinates", "coordinate_fourier", "spatial_rbf", "graph_laplacian",
            "character_3gram_hash",
        ]
        names = [name for name in preferred if name in available]
        # Corrupt metrics are generated lazily and therefore are not members of
        # the fixed representation table returned by ``available_representations``.
        # They nevertheless use the identical learned wV tokenizer and backbone.
        if "mpe" in available:
            names.extend(f"mpe_corrupt_{index}" for index in range(10))
    else:
        names = [args.representation]
    for name in names:
        run_cell(args.task, args.split, args.setting, args.backbone, name, args.device, args.output)


if __name__ == "__main__":
    main()
