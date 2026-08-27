"""Validation-gated field-local chart distillation for late Day 4.

This runner is the predeclared follow-up to ``semantic_multiview_pilot.py``.
It keeps one deployed PLE token stream and adds a zero-initialized gated
residual only at schema-declared cyclic fields.  The semantic residual is
rendered from Fourier coordinates and can be locally distilled toward the
stop-gradient PLE token for the same field.  No full-row contrastive loss,
second prediction branch, tree model, or target-derived field declaration is
used.

The selection screen intentionally reports validation metrics only.  Its gate
is fixed in ``analyze_field_local_distillation.py``; test predictions must not
be computed unless that gate clears.
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
import torch
from torch import Tensor, nn

import semantic_multiview_pilot as base


HERE = Path(__file__).resolve().parent
ADAPTER_METHODS = {
    "ple_adapter",
    "semantic_noalign",
    "semantic_local",
    "semantic_wrong_local",
}
LOCAL_METHODS = {"semantic_local", "semantic_wrong_local"}


class FieldLocalTokenizer(nn.Module):
    """PLE tokenizer plus a residual at declared numerical fields only."""

    def __init__(
        self,
        *,
        method: str,
        edges: np.ndarray,
        n_bin_fields: int,
        category_cardinalities: list[int],
        d_token: int,
        cyclic_columns: list[int],
        cyclic_periods: list[float],
        cyclic_origins: list[float],
        scramble_seed: int = 20260827,
    ) -> None:
        super().__init__()
        if method not in ADAPTER_METHODS:
            raise KeyError(method)
        self.method = method
        self.n_bins = edges.shape[1] - 1
        self.base = base.FieldTokenizer(
            edges=edges,
            n_bin_fields=n_bin_fields,
            category_cardinalities=category_cardinalities,
            d_token=d_token,
            view="ple",
            cyclic_columns=cyclic_columns,
            cyclic_periods=cyclic_periods,
            cyclic_origins=cyclic_origins,
        )
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
        self.register_buffer(
            "phase_permutation", torch.as_tensor(permutation, dtype=torch.long)
        )
        self.residual_weight = nn.Parameter(
            torch.empty(len(cyclic_columns), self.n_bins, d_token)
        )
        nn.init.xavier_uniform_(self.residual_weight)
        # Exact PLE at initialization, regardless of residual initialization.
        self.gate_logits = nn.Parameter(torch.zeros(len(cyclic_columns)))

    def residual_basis(self, x_num: Tensor) -> Tensor:
        if self.method == "ple_adapter":
            p = base.ple_basis(x_num, self.base.edges)
            return p[:, self.cyclic_columns]
        cyclic = x_num[:, self.cyclic_columns]
        phase = torch.remainder(
            (cyclic - self.cyclic_origins) / self.cyclic_periods, 1.0
        )
        if self.method == "semantic_wrong_local":
            phase = base.scrambled_phase(phase, self.phase_permutation)
        return base.cyclic_fourier_basis(phase, self.n_bins)

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.base(x_num, x_bin, x_cat)
        teacher = tokens[:, self.cyclic_columns]
        residual = torch.einsum(
            "ncb,cbd->ncd", self.residual_basis(x_num), self.residual_weight
        )
        gate = torch.tanh(self.gate_logits)
        # Avoid an in-place update on a view needed by the distillation graph.
        update = torch.zeros_like(tokens)
        update[:, self.cyclic_columns] = gate[None, :, None] * residual
        return tokens + update, teacher, residual


class FieldLocalModel(nn.Module):
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
        # Keep paired backbone and base-PLE initialization invariant to method.
        self.backbone = base.BACKBONES[backbone](n_fields, d_token, width, depth)
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
            self.tokenizer: nn.Module = base.FieldTokenizer(view="ple", **common)
        elif method in ADAPTER_METHODS:
            self.tokenizer = FieldLocalTokenizer(method=method, **common)
        else:
            raise KeyError(method)

    def forward(
        self, x_num: Tensor, x_bin: Tensor, x_cat: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if self.method == "ple":
            prediction, _ = self.backbone(self.tokenizer(x_num, x_bin, x_cat))
            return prediction, None, None
        assert isinstance(self.tokenizer, FieldLocalTokenizer)
        tokens, teacher, residual = self.tokenizer(x_num, x_bin, x_cat)
        prediction, _ = self.backbone(tokens)
        return prediction, teacher, residual


def local_distillation_loss(teacher: Tensor, student: Tensor) -> Tensor:
    """Align field tokens without changing the base PLE teacher."""
    teacher_normalized = torch.nn.functional.layer_norm(
        teacher.detach().float(), (teacher.shape[-1],)
    )
    student_normalized = torch.nn.functional.layer_norm(
        student.float(), (student.shape[-1],)
    )
    return torch.nn.functional.mse_loss(student_normalized, teacher_normalized)


@torch.inference_mode()
def evaluate_validation(
    model: FieldLocalModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_scale: float,
) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for x_num, x_bin, x_cat, y in loader:
        x_num, x_bin, x_cat = x_num.to(device), x_bin.to(device), x_cat.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction, _, _ = model(x_num, x_bin, x_cat)
        predictions.append(prediction.float().cpu())
        targets.append(y)
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    mse = float(np.mean((prediction - target) ** 2))
    return {"val_loss": mse, "val_rmse": math.sqrt(mse) * y_scale}


def train_one(
    data: base.SplitData,
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
    local_weight: float,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    edges = base.quantile_edges(data.x_num["train"], n_bins)
    n_bin_fields = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
    model = FieldLocalModel(
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
    train_loader = base.make_loader(
        data, "train", batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = base.make_loader(
        data, "val", batch_size=batch_size * 2, shuffle=False, seed=seed
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    last_local_loss = 0.0
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
                prediction, teacher, residual = model(x_num, x_bin, x_cat)
                supervised = torch.nn.functional.mse_loss(prediction, y)
            if method in LOCAL_METHODS:
                assert teacher is not None and residual is not None
                local = local_distillation_loss(teacher, residual)
                last_local_loss = float(local.detach().cpu())
                loss = supervised.float() + local_weight * local
            else:
                loss = supervised.float()
            loss.backward()
            optimizer.step()
        validation = evaluate_validation(model, val_loader, resolved, data.y_scale)
        if validation["val_loss"] < best_loss:
            best_loss, best_epoch, stale = validation["val_loss"], epoch, 0
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
    validation = evaluate_validation(model, val_loader, resolved, data.y_scale)
    if isinstance(model.tokenizer, FieldLocalTokenizer):
        gates = torch.tanh(model.tokenizer.gate_logits).detach().cpu().numpy()
    else:
        gates = np.zeros(len(data.cyclic_columns), dtype=np.float32)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        **validation,
        "last_local_loss": last_local_loss,
        "mean_abs_gate": float(np.mean(np.abs(gates))) if len(gates) else 0.0,
        "max_abs_gate": float(np.max(np.abs(gates))) if len(gates) else 0.0,
        "gates": json.dumps([float(value) for value in gates]),
        "train_seconds": time.perf_counter() - started,
        "test_evaluated": False,
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
    parser.add_argument("--datasets", nargs="+", default=["weather", "cooking-time"])
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "resnet", "ft_transformer"]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "ple",
            "ple_adapter",
            "semantic_noalign",
            "semantic_local",
            "semantic_wrong_local",
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
    parser.add_argument("--local-weight", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/field_local_distillation_selection.csv",
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
        "selection_only": True,
        "test_metrics_computed": False,
        "datasets": {},
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else None,
    }
    for dataset_name in args.datasets:
        data = base.load_tabred(
            dataset_name,
            max_train_rows=args.max_train_rows,
            max_eval_rows=args.max_eval_rows,
            sample_seed=args.sample_seed,
        )
        metadata["datasets"][dataset_name] = {
            "used_train_rows": len(data.y["train"]),
            "used_validation_rows": len(data.y["val"]),
            "test_rows_loaded_but_never_evaluated": len(data.y["test"]),
            "cyclic_fields": [
                {"name": name, "column": column, "period": period}
                for name, column, period in zip(
                    data.cyclic_names, data.cyclic_columns, data.cyclic_periods
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
                        local_weight=args.local_weight,
                    )
                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "method": method,
                        "n_train": len(data.y["train"]),
                        "n_val": len(data.y["val"]),
                        "n_num": data.x_num["train"].shape[1],
                        "n_bin": 0 if data.x_bin is None else data.x_bin["train"].shape[1],
                        "n_cat": len(data.category_cardinalities),
                        "n_cyclic": len(data.cyclic_columns),
                        "local_weight": args.local_weight,
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
