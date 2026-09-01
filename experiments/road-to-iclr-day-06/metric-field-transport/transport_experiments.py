#!/usr/bin/env python3
"""Gated E2 whole-state transport experiments.

Execution is disabled until the complete E1 analysis satisfies every frozen
promotion criterion. The E1b selected-distance cells are reused as raw-base
cells instead of being trained a second time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from successor_experiments import (
    E1B_TASKS,
    HERE,
    NEURAL_SEEDS,
    PROTOCOL_PATH,
    atomic_json,
    balanced_mse,
    compose_design,
    dense_batch,
    load_task,
    maybe_device_design,
    neural_dev_design,
    ordinary_design_subset,
    representation_tables,
    seed_everything,
    sha256_path,
    split_row_indices,
    split_state_indices,
    stable_hash,
    state_balanced_training_weights,
    train_metric_scale,
)


E2_CONDITIONS = [
    "raw_base",
    "lookup_unknown",
    "transport_zero",
    "transport_first_order",
    "transport_shuffled_metric",
]
EXPERT_RANK = 16
CORRECTION_RANK = 4
NEIGHBORS = 8


@dataclass(frozen=True)
class TransportGraph:
    neighbor_train_indices: np.ndarray
    neighbor_weights: np.ndarray
    coordinate_differences: np.ndarray
    valid_query: np.ndarray
    association: np.ndarray


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def build_transport_graph(
    task: Any,
    training_states: np.ndarray,
    validation_states: np.ndarray,
    coordinates: np.ndarray,
    *,
    shuffled: bool,
    split_index: int,
) -> TransportGraph:
    training_states = np.asarray(training_states, dtype=np.int64)
    validation_states = np.asarray(validation_states, dtype=np.int64)
    if len(training_states) <= 1:
        raise ValueError("transport requires at least two development-training states")
    neighbor_count = min(NEIGHBORS, len(training_states) - 1)
    state_count = len(task.state_ids)
    coordinate_size = int(coordinates.shape[1])
    association = np.arange(state_count, dtype=np.int64)
    if shuffled:
        rng = np.random.default_rng(stable_hash("mft-e2-shuffle", task.name, split_index))
        association[training_states] = rng.permutation(training_states)
        association[validation_states] = rng.permutation(validation_states)

    neighbors = np.zeros((state_count, neighbor_count), dtype=np.int64)
    weights = np.zeros((state_count, neighbor_count), dtype=np.float32)
    differences = np.zeros((state_count, neighbor_count, coordinate_size), dtype=np.float32)
    valid_query = np.zeros(state_count, dtype=bool)
    training_lookup = {int(state): local for local, state in enumerate(training_states)}
    bandwidth = train_metric_scale(task.distance, training_states)
    candidate_metric_states = association[training_states]

    for query in np.concatenate([training_states, validation_states]):
        query = int(query)
        distances = np.asarray(task.distance[association[query], candidate_metric_states], dtype=np.float64)
        candidates = [local for local, state in enumerate(training_states) if int(state) != query]
        ordered = sorted(
            candidates,
            key=lambda local: (distances[local], str(task.state_ids[int(training_states[local])])),
        )[:neighbor_count]
        chosen = np.asarray(ordered, dtype=np.int64)
        chosen_distances = distances[chosen]
        logits = -0.5 * (chosen_distances / bandwidth) ** 2
        logits -= np.max(logits)
        chosen_weights = np.exp(logits)
        chosen_weights /= chosen_weights.sum()
        anchor_states = training_states[chosen]
        neighbors[query] = chosen
        weights[query] = chosen_weights.astype(np.float32)
        differences[query] = (
            coordinates[association[query]][None, :] - coordinates[association[anchor_states]]
        ).astype(np.float32)
        valid_query[query] = True

    if not np.all(valid_query[np.concatenate([training_states, validation_states])]):
        raise AssertionError("missing development transport query")
    for state in training_states:
        local = training_lookup[int(state)]
        if local in neighbors[int(state)]:
            raise AssertionError("self entered training-state transport neighborhood")
    if not np.allclose(weights[valid_query].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("transport weights do not sum to one")
    return TransportGraph(neighbors, weights, differences, valid_query, association)


class TransportMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_count: int,
        training_states: np.ndarray,
        condition: str,
        graph: TransportGraph | None,
        coordinate_size: int,
        width: int = 128,
    ):
        super().__init__()
        if condition not in E2_CONDITIONS:
            raise KeyError(condition)
        self.condition = condition
        self.width = int(width)
        self.backbone = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.base_head = nn.Linear(width, 1)

        training_states = np.asarray(training_states, dtype=np.int64)
        state_to_train = np.full(state_count, -1, dtype=np.int64)
        state_to_train[training_states] = np.arange(len(training_states), dtype=np.int64)
        self.register_buffer("training_states", torch.from_numpy(training_states))
        self.register_buffer("state_to_train", torch.from_numpy(state_to_train))
        self.experts: nn.Parameter | None = None
        self.projection: nn.Linear | None = None
        self.transport_scale: nn.Parameter | None = None
        self.correction_right: nn.Linear | None = None
        self.correction_left: nn.Parameter | None = None

        if condition != "raw_base":
            self.experts = nn.Parameter(torch.empty(len(training_states), EXPERT_RANK + 1))
            with torch.no_grad():
                self.experts.normal_(mean=0.0, std=0.02)
                self.experts[:, -1].zero_()
            self.projection = nn.Linear(width, EXPERT_RANK, bias=False)
            self.transport_scale = nn.Parameter(torch.zeros(()))

        uses_transport = condition in {
            "transport_zero",
            "transport_first_order",
            "transport_shuffled_metric",
        }
        if uses_transport:
            if graph is None:
                raise ValueError(f"{condition} requires a transport graph")
            self.register_buffer("neighbor_train_indices", torch.from_numpy(graph.neighbor_train_indices))
            self.register_buffer("neighbor_weights", torch.from_numpy(graph.neighbor_weights))
            self.register_buffer("coordinate_differences", torch.from_numpy(graph.coordinate_differences))
            self.register_buffer("valid_query", torch.from_numpy(graph.valid_query))
        else:
            self.neighbor_train_indices = None
            self.neighbor_weights = None
            self.coordinate_differences = None
            self.valid_query = None

        if condition in {"transport_first_order", "transport_shuffled_metric"}:
            self.correction_right = nn.Linear(coordinate_size, CORRECTION_RANK, bias=False)
            self.correction_left = nn.Parameter(
                torch.empty(len(training_states), CORRECTION_RANK, EXPERT_RANK + 1)
            )
            nn.init.normal_(self.correction_left, mean=0.0, std=0.02)

    @property
    def uses_transport(self) -> bool:
        return self.condition in {
            "transport_zero",
            "transport_first_order",
            "transport_shuffled_metric",
        }

    def transported_experts(self, global_states: torch.Tensor) -> torch.Tensor:
        if not self.uses_transport or self.experts is None:
            raise RuntimeError("transport is unavailable")
        if self.valid_query is None or not bool(self.valid_query.index_select(0, global_states).all()):
            raise AssertionError("non-development state entered transport")
        assert self.neighbor_train_indices is not None
        assert self.neighbor_weights is not None
        neighbors = self.neighbor_train_indices.index_select(0, global_states)
        weights = self.neighbor_weights.index_select(0, global_states)
        anchor_experts = self.experts[neighbors]
        if self.correction_right is not None and self.correction_left is not None:
            assert self.coordinate_differences is not None
            differences = self.coordinate_differences.index_select(0, global_states)
            reduced = self.correction_right(differences)
            left = self.correction_left[neighbors]
            correction = torch.einsum("bkr,bkrd->bkd", reduced, left)
            anchor_experts = anchor_experts + correction
        return torch.sum(weights.unsqueeze(-1) * anchor_experts, dim=1)

    def auxiliary_loss(self, masked_training_states: torch.Tensor) -> torch.Tensor:
        if not self.uses_transport or self.experts is None:
            return self.base_head.weight.sum() * 0.0
        selected = torch.nonzero(masked_training_states, as_tuple=False).flatten()
        if selected.numel() == 0:
            return self.base_head.weight.sum() * 0.0
        global_states = self.training_states.index_select(0, selected)
        reconstructed = self.transported_experts(global_states)
        target = self.experts.index_select(0, selected).detach()
        return F.mse_loss(reconstructed, target)

    def forward(
        self,
        x: torch.Tensor,
        global_states: torch.Tensor,
        *,
        masked_training_states: torch.Tensor | None = None,
        evaluation: bool = False,
    ) -> torch.Tensor:
        hidden = self.backbone(x)
        prediction = self.base_head(hidden).reshape(-1)
        if self.condition == "raw_base":
            return prediction
        assert self.experts is not None
        assert self.projection is not None
        assert self.transport_scale is not None
        if self.condition == "lookup_unknown":
            if evaluation:
                expert = torch.zeros(
                    (len(global_states), EXPERT_RANK + 1), dtype=hidden.dtype, device=hidden.device
                )
            else:
                local_states = self.state_to_train.index_select(0, global_states)
                if bool((local_states < 0).any()):
                    raise AssertionError("held-out state entered lookup training")
                expert = self.experts.index_select(0, local_states)
        else:
            transported = self.transported_experts(global_states)
            if evaluation:
                expert = transported
            else:
                if masked_training_states is None:
                    raise ValueError("training transport requires an epoch-level state mask")
                local_states = self.state_to_train.index_select(0, global_states)
                if bool((local_states < 0).any()):
                    raise AssertionError("held-out state entered transport training")
                warm = self.experts.index_select(0, local_states)
                row_mask = masked_training_states.index_select(0, local_states).unsqueeze(1)
                expert = torch.where(row_mask, transported, warm)
        interaction = torch.sum(self.projection(hidden) * expert[:, :EXPERT_RANK], dim=1) / math.sqrt(
            EXPERT_RANK
        )
        interaction = interaction + expert[:, EXPERT_RANK]
        return prediction + self.transport_scale * interaction


def predict_transport(
    model: TransportMLP,
    design: Any,
    global_states: torch.Tensor,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chosen = rows[start : start + batch_size]
            chosen_tensor = torch.as_tensor(chosen, device=device)
            x = dense_batch(design, chosen_tensor, device)
            states = global_states.index_select(0, chosen_tensor)
            predictions.append(model(x, states, evaluation=True).float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def fit_transport_validation(
    design: Any,
    target: np.ndarray,
    state_labels: np.ndarray,
    global_states: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    training_states: np.ndarray,
    condition: str,
    graph: TransportGraph | None,
    coordinate_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    model = TransportMLP(
        design.shape[1],
        int(np.max(global_states)) + 1 if graph is None else len(graph.valid_query),
        training_states,
        condition,
        graph,
        coordinate_size,
    ).to(device)
    device_design = maybe_device_design(design, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=5e-5)
    weights = np.zeros(len(state_labels), dtype=np.float32)
    weights[train_rows] = state_balanced_training_weights(state_labels[train_rows]).astype(np.float32)
    target_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(device)
    weight_tensor = torch.from_numpy(weights).to(device)
    global_state_tensor = torch.from_numpy(global_states.astype(np.int64, copy=False)).to(device)
    order_generator = np.random.default_rng(seed + 991)
    mask_generator = np.random.default_rng(seed + 1771)
    initial_prediction = predict_transport(
        model, device_design, global_state_tensor, validation_rows, device
    )
    initial_score = balanced_mse(
        target[validation_rows], initial_prediction, state_labels[validation_rows]
    )
    best_score = float("inf")
    best_epoch = -1
    stale = 0
    curve = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(160):
        model.train()
        if model.uses_transport:
            mask_numpy = mask_generator.random(len(training_states)) < 0.5
        else:
            mask_numpy = np.zeros(len(training_states), dtype=bool)
        masked_states = torch.from_numpy(mask_numpy).to(device)
        order = order_generator.permutation(train_rows)
        for start in range(0, len(order), 2048):
            chosen = order[start : start + 2048]
            chosen_tensor = torch.as_tensor(chosen, device=device)
            x = dense_batch(device_design, chosen_tensor, device)
            y = target_tensor.index_select(0, chosen_tensor)
            row_weight = weight_tensor.index_select(0, chosen_tensor)
            batch_states = global_state_tensor.index_select(0, chosen_tensor)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x,
                batch_states,
                masked_training_states=masked_states,
                evaluation=False,
            )
            task_rows = (prediction - y) ** 2
            task_loss = (task_rows * row_weight).sum() / row_weight.sum()
            auxiliary = model.auxiliary_loss(masked_states)
            loss = task_loss + 0.1 * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        validation_prediction = predict_transport(
            model, device_design, global_state_tensor, validation_rows, device
        )
        score = balanced_mse(
            target[validation_rows], validation_prediction, state_labels[validation_rows]
        )
        curve.append(
            {
                "epoch": epoch + 1,
                "validation_state_balanced_standardized_mse": score,
                "masked_training_state_fraction": float(np.mean(mask_numpy)),
            }
        )
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    result = {
        "initial_validation_state_balanced_standardized_mse": initial_score,
        "validation_state_balanced_standardized_mse": best_score,
        "best_epoch": best_epoch,
        "stop_epoch": len(curve),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "transport_scale_final": (
            float(model.transport_scale.detach().cpu()) if model.transport_scale is not None else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "curve": curve,
    }
    del model, device_design, target_tensor, weight_tensor, global_state_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_e1_gate(output_root: Path) -> tuple[dict[str, Any], str]:
    analysis_path = output_root / "analysis.json"
    if not analysis_path.exists():
        raise RuntimeError("E2 disabled: run analyze_successor.py after complete E1")
    analysis = json.loads(analysis_path.read_text())
    gate = analysis.get("e1b", {}).get("promotion_gate")
    if analysis.get("status") != "complete" or not gate or gate.get("promote_to_e2") is not True:
        raise RuntimeError("E2 disabled: the frozen E1 promotion gate has not passed")
    selected = analysis["e1a"]["selected_raw_representation"]
    return analysis, selected


def reuse_raw_base(
    task_name: str,
    split_index: int,
    seed: int,
    selected_representation: str,
    output_root: Path,
) -> dict[str, Any]:
    source_path = (
        output_root
        / "e1b_cells"
        / f"{task_name}__split{split_index}__{selected_representation}__seed{seed}.json"
    )
    cell_id = f"{task_name}__split{split_index}__raw_base__seed{seed}"
    path = output_root / "e2_cells" / f"{cell_id}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol_sha256") == sha256_path(
            PROTOCOL_PATH
        ):
            print(f"resume e2 {cell_id}", flush=True)
            return existing
    source = json.loads(source_path.read_text())
    payload = {
        **source,
        "stage": "e2",
        "cell_id": cell_id,
        "condition": "raw_base",
        "representation": selected_representation,
        "reused_without_retraining": True,
        "reused_from": str(source_path.relative_to(HERE)),
    }
    atomic_json(payload, path)
    print(f"e2 {cell_id} reused={selected_representation}", flush=True)
    return payload


def run_e2_cell(
    task_name: str,
    split_index: int,
    condition: str,
    seed: int,
    selected_representation: str,
    device_name: str,
    output_root: Path,
) -> dict[str, Any]:
    if condition == "raw_base":
        return reuse_raw_base(task_name, split_index, seed, selected_representation, output_root)
    cell_id = f"{task_name}__split{split_index}__{condition}__seed{seed}"
    path = output_root / "e2_cells" / f"{cell_id}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == sha256_path(
            PROTOCOL_PATH
        ):
            print(f"resume e2 {cell_id}", flush=True)
            return payload

    task = load_task(task_name)
    state_parts = split_state_indices(task, split_index)
    row_parts = split_row_indices(task, split_index)
    design, target, labels, train_rows, validation_rows, metadata = neural_dev_design(
        task, split_index, selected_representation
    )
    output_rows = np.concatenate([row_parts["train"], row_parts["validation"]])
    global_states = task.row_state_indices()[output_rows]
    tables, _ = representation_tables(task, state_parts["train"])
    coordinates = tables[selected_representation]
    uses_transport = condition in {
        "transport_zero",
        "transport_first_order",
        "transport_shuffled_metric",
    }
    graph = None
    if uses_transport:
        graph = build_transport_graph(
            task,
            state_parts["train"],
            state_parts["validation"],
            coordinates,
            shuffled=condition == "transport_shuffled_metric",
            split_index=split_index,
        )
    started = time.perf_counter()
    result = fit_transport_validation(
        design,
        target,
        labels,
        global_states,
        train_rows,
        validation_rows,
        state_parts["train"],
        condition,
        graph,
        int(coordinates.shape[1]),
        seed,
        torch.device(device_name),
    )
    graph_metadata = None
    if graph is not None:
        graph_metadata = {
            "neighbors": int(graph.neighbor_train_indices.shape[1]),
            "shuffled_metric_association": condition == "transport_shuffled_metric",
            "graph_sha256": array_sha256(
                graph.neighbor_train_indices,
                graph.neighbor_weights,
                graph.coordinate_differences,
                graph.association,
            ),
        }
    payload = {
        "status": "complete",
        "stage": "e2",
        "cell_id": cell_id,
        "task": task_name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "setting": "full_table",
        "condition": condition,
        "representation": selected_representation,
        "seed": seed,
        "device": device_name,
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "feature_dimension": int(design.shape[1]),
        "representation_dimension": int(coordinates.shape[1]),
        "representation_metadata": metadata,
        "transport_graph": graph_metadata,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "width": 128,
            "depth": 2,
            "dropout": 0.1,
            "batch_size": 2048,
            "max_epochs": 160,
            "patience": 20,
            "whole_state_mask_probability": 0.5 if uses_transport else 0.0,
            "auxiliary_reconstruction_weight": 0.1 if uses_transport else 0.0,
            "expert_rank": EXPERT_RANK,
            "correction_rank": CORRECTION_RANK if condition.endswith("first_order") or condition.endswith("metric") else 0,
        },
        "result": result,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, path)
    print(
        f"e2 {cell_id} dev={result['validation_state_balanced_standardized_mse']:.6f} "
        f"epoch={result['best_epoch']}",
        flush=True,
    )
    del design
    gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()

    _, selected_representation = load_e1_gate(args.output)
    tasks = E1B_TASKS if args.task == "all" else [args.task]
    splits = [0, 1] if args.split == "all" else [int(args.split)]
    conditions = E2_CONDITIONS if args.condition == "all" else [args.condition]
    seeds = NEURAL_SEEDS if args.seed == "all" else [int(args.seed)]
    for task_name in tasks:
        if task_name not in E1B_TASKS:
            raise ValueError(task_name)
        for split_index in splits:
            for condition in conditions:
                if condition not in E2_CONDITIONS:
                    raise ValueError(condition)
                for seed in seeds:
                    if seed not in NEURAL_SEEDS:
                        raise ValueError(seed)
                    run_e2_cell(
                        task_name,
                        split_index,
                        condition,
                        seed,
                        selected_representation,
                        args.device,
                        args.output,
                    )


if __name__ == "__main__":
    main()
