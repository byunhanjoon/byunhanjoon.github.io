"""Single-model dual Raw/Gram feature encoders with MLP, TabM-D, and ResNet backbones."""

from __future__ import annotations

import copy
import gc
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.linear1(self.norm(values))
        residual = self.dropout(torch.relu(residual))
        return values + self.linear2(residual)


class ResNetBackbone(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, 256)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(3)))
        self.norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input(values))
        return self.output(torch.relu(self.norm(self.blocks(hidden))))[:, None, :]


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(3):
            layers.extend((nn.Linear(current, 256), nn.GELU()))
            current = 256
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)[:, None, :]


class TabMDBackbone(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        from pytabkit.models.nn_models.tabm import Model

        self.network = Model(
            n_num_features=input_dim,
            cat_cardinalities=[],
            n_classes=output_dim,
            backbone={"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.1},
            bins=None,
            num_embeddings=None,
            arch_type="tabm",
            k=32,
            share_training_batches=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DualViewModel(nn.Module):
    def __init__(
        self,
        *,
        raw_feature_blocks: dict[str, list[int]],
        gram_feature_blocks: dict[str, list[int]],
        raw_width: int,
        output_dim: int,
        backbone: str,
        gate_kind: str,
        fixed_gates: dict[str, float],
        hidden_width_per_feature: int = 16,
    ) -> None:
        super().__init__()
        self.features = list(raw_feature_blocks)
        self.raw_indices = {name: list(raw_feature_blocks[name]) for name in self.features}
        self.gram_indices = {name: list(gram_feature_blocks[name]) for name in self.features}
        used = {index for indices in self.raw_indices.values() for index in indices}
        self.passthrough_indices = [index for index in range(raw_width) if index not in used]
        self.raw_layers = nn.ModuleDict(
            {name: nn.Linear(len(self.raw_indices[name]), hidden_width_per_feature) for name in self.features}
        )
        self.gram_layers = nn.ModuleDict(
            {name: nn.Linear(len(self.gram_indices[name]), hidden_width_per_feature) for name in self.features}
        )
        self.gate_kind = gate_kind
        if gate_kind == "learnable":
            self.gate_logits = nn.Parameter(torch.zeros(len(self.features)))
            self.register_buffer("fixed_gate_values", torch.empty(0))
        else:
            self.gate_logits = None
            self.register_buffer(
                "fixed_gate_values",
                torch.tensor([float(fixed_gates[name]) for name in self.features], dtype=torch.float32),
            )
        encoded_width = len(self.features) * hidden_width_per_feature + len(self.passthrough_indices)
        if backbone == "controlled_mlp":
            self.backbone = MLPBackbone(encoded_width, output_dim)
        elif backbone == "resnet_tabular":
            self.backbone = ResNetBackbone(encoded_width, output_dim)
        elif backbone == "tabm_d":
            self.backbone = TabMDBackbone(encoded_width, output_dim)
        else:
            raise ValueError(f"unsupported DualView backbone {backbone}")

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits) if self.gate_logits is not None else self.fixed_gate_values

    def encode(self, raw: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        gates = self.gates()
        parts: list[torch.Tensor] = []
        for index, feature in enumerate(self.features):
            raw_hidden = self.raw_layers[feature](raw[:, self.raw_indices[feature]])
            gram_hidden = self.gram_layers[feature](gram[:, self.gram_indices[feature]])
            alpha = gates[index]
            parts.append((1.0 - alpha) * raw_hidden + alpha * gram_hidden)
        if self.passthrough_indices:
            parts.append(raw[:, self.passthrough_indices])
        return torch.cat(parts, dim=1)

    def forward(self, raw: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.encode(raw, gram))


def _loss(
    problem_type: str,
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    ensemble = output.shape[1]
    if problem_type == "classification":
        return nn.functional.cross_entropy(
            output.flatten(0, 1), target.repeat_interleave(ensemble)
        )
    return nn.functional.mse_loss(output.squeeze(-1).flatten(), target.repeat_interleave(ensemble))


def fit_dualview_predictions(
    *,
    problem_type: str,
    raw_rep: Any,
    gram_rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    backbone: str,
    gate_kind: str,
    fixed_gates: dict[str, float] | None = None,
    gate_penalty: float = 0.0,
    inference_warmups: int = 3,
    inference_repeats: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one joint encoder/backbone and return validation/test predictions."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    features = list(raw_rep.feature_blocks)
    fixed_gates = ({name: 0.5 for name in features} if fixed_gates is None else fixed_gates)
    output_dim = 1 if problem_type == "regression" else int(np.max(y_train)) + 1
    model = DualViewModel(
        raw_feature_blocks=raw_rep.feature_blocks,
        gram_feature_blocks=gram_rep.feature_blocks,
        raw_width=raw_rep.X_train.shape[1],
        output_dim=output_dim,
        backbone=backbone,
        gate_kind=gate_kind,
        fixed_gates=fixed_gates,
    ).to(torch_device)
    learning_rate = 0.002 if backbone == "tabm_d" else 0.001
    max_epochs, patience = (30, 5) if backbone == "tabm_d" else (80, 12) if backbone == "resnet_tabular" else (100, 15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split, raw_values, gram_values in (
        ("train", raw_rep.X_train, gram_rep.X_train),
        ("validation", raw_rep.X_validation, gram_rep.X_validation),
        ("test", raw_rep.X_test, gram_rep.X_test),
    ):
        tensors[split] = (
            torch.as_tensor(np.asarray(raw_values, dtype=np.float32), device=torch_device),
            torch.as_tensor(np.asarray(gram_values, dtype=np.float32), device=torch_device),
        )
    if problem_type == "regression":
        y_mean = float(np.mean(y_train))
        y_scale = max(float(np.std(y_train)), 1e-8)
        train_target = torch.as_tensor(((y_train - y_mean) / y_scale).astype(np.float32), device=torch_device)
        validation_target = torch.as_tensor(
            ((y_validation - y_mean) / y_scale).astype(np.float32), device=torch_device
        )
    else:
        y_mean, y_scale = 0.0, 1.0
        train_target = torch.as_tensor(y_train.astype(np.int64), device=torch_device)
        validation_target = torch.as_tensor(y_validation.astype(np.int64), device=torch_device)

    generator = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order = generator.permutation(len(y_train))
        for start in range(0, len(order), 256):
            rows = torch.as_tensor(order[start : start + 256], device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            raw_batch, gram_batch = tensors["train"]
            output = model(raw_batch[rows], gram_batch[rows])
            loss = _loss(problem_type, output, train_target[rows])
            if gate_kind == "learnable":
                gates = model.gates()
                loss = loss + float(gate_penalty) * torch.sum(gates * (1.0 - gates))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_output = model(*tensors["validation"])
            validation_loss = float(_loss(problem_type, validation_output, validation_target).item())
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("DualView failed to produce a checkpoint")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started
    model.eval()

    def prediction(split: str) -> np.ndarray:
        with torch.no_grad():
            output = model(*tensors[split]).mean(dim=1)
            if problem_type == "regression":
                return output.squeeze(-1).cpu().numpy() * y_scale + y_mean
            return torch.softmax(output, dim=-1).cpu().numpy()

    prediction_started = time.perf_counter()
    validation_prediction = prediction("validation")
    test_prediction = prediction("test")
    predict_seconds = time.perf_counter() - prediction_started
    with torch.no_grad():
        for _ in range(int(inference_warmups)):
            model(*tensors["test"])
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch_device)
        inference_started = time.perf_counter()
        for _ in range(int(inference_repeats)):
            model(*tensors["test"])
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch_device)
    repeated_inference_seconds = time.perf_counter() - inference_started
    gates = model.gates().detach().cpu().numpy()
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "inference_repeats": int(inference_repeats),
        "mean_inference_seconds": repeated_inference_seconds / max(int(inference_repeats), 1),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(torch_device)) if str(device).startswith("cuda") else 0,
        "best_epoch": int(best_epoch),
        "best_validation_objective": best_loss,
        "architecture": f"DualViewGram-{backbone}-h16",
        "backbone": backbone,
        "gate_kind": gate_kind,
        "gate_penalty": float(gate_penalty),
        "gate_values": {name: float(value) for name, value in zip(features, gates)},
        "mean_gate": float(np.mean(gates)),
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": max_epochs,
        "patience": patience,
        "single_model": True,
        "inference_passes": 1,
    }
    del model, tensors
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"validation": validation_prediction, "test": test_prediction}, telemetry
