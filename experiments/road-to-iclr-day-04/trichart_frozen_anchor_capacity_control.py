#!/usr/bin/env python3
"""Parameter-matched second-T-PLE control for Frozen-Anchor TriChart."""
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

from semantic_multiview_pilot import PARTS, MLPBackbone, ResNetBackbone, load_tabred
from support_identity_transfer_pilot import (
    MatchedFTTransformer,
    SupportTokenizer,
    parameter_count,
    prepare_encodings,
)
from trichart_frozen_anchor_classification import (
    _classification_metrics,
    fit_anchor as fit_classification_anchor,
    load_binary,
    prepare_classification_encodings,
)
from trichart_frozen_anchor_pilot import (
    FrozenAnchorResidual,
    fit_anchor as fit_regression_anchor,
)
from trichart_shared_pilot import make_loader, move
from universal_mass_identity_pilot import UniversalTokenizer, prepare
from openml_external_data import load_openml


HERE = Path(__file__).resolve().parent


class FrozenTResidualControl(nn.Module):
    """Frozen anchor plus an independently trainable T-PLE residual model."""

    def __init__(
        self,
        anchor,
        data,
        universal,
        encoding,
        config: dict,
        architecture: str,
        width: int,
        ff_width: int,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        n_bin = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
        self.residual_tokenizer = SupportTokenizer(
            edges=encoding.tple_edges,
            n_bin_fields=n_bin,
            category_cardinalities=data.category_cardinalities,
            support_columns=encoding.selected_columns,
            support_cardinalities=encoding.cardinalities,
            d_token=config["d_token"],
            use_support=False,
        )
        if architecture == "mlp":
            self.residual_backbone = MLPBackbone(
                universal.n_fields, config["d_token"], width, config["depth"]
            )
        elif architecture == "resnet":
            self.residual_backbone = ResNetBackbone(
                universal.n_fields, config["d_token"], width, config["depth"]
            )
        elif architecture == "ft_transformer":
            self.residual_backbone = MatchedFTTransformer(
                universal.n_fields,
                config["d_token"],
                config["depth"],
                ff_width,
                config["dropout"],
            )
        else:
            raise KeyError(architecture)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self,
        x_num: Tensor,
        x_bin: Tensor,
        x_cat: Tensor,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del rank, rank_lower, rank_upper, information
        empty_codes = code[:, :0]
        with torch.no_grad():
            anchor_tokens = self.anchor.tokenizer(
                x_num, x_bin, x_cat, empty_codes
            )
            anchor_prediction = self.anchor.backbone(anchor_tokens)[0]
        residual_tokens = self.residual_tokenizer(
            x_num, x_bin, x_cat, empty_codes
        )
        residual_prediction = self.residual_backbone(residual_tokens)[0]
        return (
            anchor_prediction + self.residual_gate * residual_prediction,
            anchor_prediction,
        )


class FrozenSupervisedMix(FrozenTResidualControl):
    """Zero-start learned mixture whose second member is supervised directly."""

    def forward_all(
        self,
        x_num: Tensor,
        x_bin: Tensor,
        x_cat: Tensor,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del rank, rank_lower, rank_upper, information
        empty_codes = code[:, :0]
        with torch.no_grad():
            anchor_tokens = self.anchor.tokenizer(
                x_num, x_bin, x_cat, empty_codes
            )
            anchor_prediction = self.anchor.backbone(anchor_tokens)[0]
        member_tokens = self.residual_tokenizer(
            x_num, x_bin, x_cat, empty_codes
        )
        member_prediction = self.residual_backbone(member_tokens)[0]
        prediction = anchor_prediction + self.residual_gate * (
            member_prediction - anchor_prediction
        )
        return prediction, anchor_prediction, member_prediction

    def forward(self, *features: Tensor) -> tuple[Tensor, Tensor]:
        prediction, anchor, _ = self.forward_all(*features)
        return prediction, anchor


class FrozenGatedChartResidual(nn.Module):
    """T residual tokens with selectively gated Q/rank chart corrections."""

    def __init__(
        self,
        anchor,
        data,
        universal,
        encoding,
        config: dict,
        architecture: str,
        width: int,
        ff_width: int,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        n_bin = 0 if data.x_bin is None else data.x_bin["train"].shape[1]
        common = {
            "n_bin_fields": n_bin,
            "category_cardinalities": data.category_cardinalities,
            "support_columns": encoding.selected_columns,
            "support_cardinalities": encoding.cardinalities,
            "d_token": config["d_token"],
            "use_support": False,
        }
        self.q_tokenizer = SupportTokenizer(edges=encoding.qple_edges, **common)
        self.rank_tokenizer = UniversalTokenizer(
            universal.n_fields, config, "rank_only"
        )
        for name, parameter in self.rank_tokenizer.named_parameters():
            if name not in {"rank_weight", "field_bias"}:
                parameter.requires_grad_(False)
        if architecture == "mlp":
            self.residual_backbone = MLPBackbone(
                universal.n_fields, config["d_token"], width, config["depth"]
            )
        elif architecture == "resnet":
            self.residual_backbone = ResNetBackbone(
                universal.n_fields, config["d_token"], width, config["depth"]
            )
        elif architecture == "ft_transformer":
            self.residual_backbone = MatchedFTTransformer(
                universal.n_fields,
                config["d_token"],
                config["depth"],
                ff_width,
                config["dropout"],
            )
        else:
            raise KeyError(architecture)
        self.q_gate = nn.Parameter(torch.zeros(universal.n_fields))
        self.rank_gate = nn.Parameter(torch.zeros(universal.n_fields))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self,
        x_num: Tensor,
        x_bin: Tensor,
        x_cat: Tensor,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> tuple[Tensor, Tensor]:
        empty_codes = code[:, :0]
        with torch.no_grad():
            t_tokens = self.anchor.tokenizer(x_num, x_bin, x_cat, empty_codes)
            anchor_prediction = self.anchor.backbone(t_tokens)[0]
        q_tokens = self.q_tokenizer(x_num, x_bin, x_cat, empty_codes)
        rank_tokens = self.rank_tokenizer(
            rank, rank_lower, rank_upper, code, information
        )
        base = t_tokens.detach()
        residual_tokens = (
            base
            + self.q_gate[None, :, None] * (q_tokens - base)
            + self.rank_gate[None, :, None] * (rank_tokens - base)
        )
        residual_prediction = self.residual_backbone(residual_tokens)[0]
        return (
            anchor_prediction + self.residual_gate * residual_prediction,
            anchor_prediction,
        )


def trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _closest(low: int, high: int, target: int, count) -> int:
    while low < high:
        middle = (low + high) // 2
        if count(middle) < target:
            low = middle + 1
        else:
            high = middle
    candidates = {max(4, low - 1), low}
    return min(candidates, key=lambda value: abs(count(value) - target))


def chart_capacity_target(
    anchor, data, universal, encoding, config: dict, architecture: str
) -> int:
    chart = FrozenAnchorResidual(
        anchor, data, universal, encoding, config, architecture
    )
    # UniversalTokenizer deliberately allocates identity/frequency modules for
    # every ablation, but rank_only never executes them. Exclude these
    # zero-gradient parameters from the effective-capacity target.
    for name, parameter in chart.rank_tokenizer.named_parameters():
        if name not in {"rank_weight", "field_bias"}:
            parameter.requires_grad_(False)
    return trainable_parameters(chart)


def build_matched_control(
    anchor, data, universal, encoding, config: dict, architecture: str
):
    target = chart_capacity_target(
        anchor, data, universal, encoding, config, architecture
    )

    def build(width: int, ff_width: int):
        return FrozenTResidualControl(
            anchor,
            data,
            universal,
            encoding,
            config,
            architecture,
            width,
            ff_width,
        )

    if architecture == "ft_transformer":
        chosen_ff = _closest(
            4,
            4096,
            target,
            lambda value: trainable_parameters(build(config["width"], value)),
        )
        model = build(config["width"], chosen_ff)
        return model, target, config["width"], chosen_ff
    chosen_width = _closest(
        8,
        1024,
        target,
        lambda value: trainable_parameters(
            build(value, config["ft_feedforward_width"])
        ),
    )
    model = build(chosen_width, config["ft_feedforward_width"])
    return model, target, chosen_width, config["ft_feedforward_width"]


def build_matched_gated(
    anchor, data, universal, encoding, config: dict, architecture: str
):
    target = chart_capacity_target(
        anchor, data, universal, encoding, config, architecture
    )

    def build(width: int, ff_width: int):
        return FrozenGatedChartResidual(
            anchor,
            data,
            universal,
            encoding,
            config,
            architecture,
            width,
            ff_width,
        )

    if architecture == "ft_transformer":
        chosen_ff = _closest(
            4,
            4096,
            target,
            lambda value: trainable_parameters(build(config["width"], value)),
        )
        model = build(config["width"], chosen_ff)
        return model, target, config["width"], chosen_ff
    chosen_width = _closest(
        8,
        1024,
        target,
        lambda value: trainable_parameters(
            build(value, config["ft_feedforward_width"])
        ),
    )
    model = build(chosen_width, config["ft_feedforward_width"])
    return model, target, chosen_width, config["ft_feedforward_width"]


def build_matched_supervised_mix(
    anchor, data, universal, encoding, config: dict, architecture: str
):
    target = chart_capacity_target(
        anchor, data, universal, encoding, config, architecture
    )

    def build(width: int, ff_width: int):
        return FrozenSupervisedMix(
            anchor,
            data,
            universal,
            encoding,
            config,
            architecture,
            width,
            ff_width,
        )

    if architecture == "ft_transformer":
        chosen_ff = _closest(
            4,
            4096,
            target,
            lambda value: trainable_parameters(build(config["width"], value)),
        )
        model = build(config["width"], chosen_ff)
        return model, target, config["width"], chosen_ff
    chosen_width = _closest(
        8,
        1024,
        target,
        lambda value: trainable_parameters(
            build(value, config["ft_feedforward_width"])
        ),
    )
    model = build(chosen_width, config["ft_feedforward_width"])
    return model, target, chosen_width, config["ft_feedforward_width"]


@torch.inference_mode()
def evaluate(model, stream, device: torch.device, task: str, scale: float):
    model.eval()
    predictions, anchors, targets = [], [], []
    for batch in stream:
        *features, target = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction, anchor = model(*features)
        predictions.append(prediction.float().cpu())
        anchors.append(anchor.float().cpu())
        targets.append(target.float().cpu())
    prediction = torch.cat(predictions).numpy()
    anchor = torch.cat(anchors).numpy()
    truth = torch.cat(targets).numpy()
    if task == "classification":
        return (
            _classification_metrics(prediction, truth),
            _classification_metrics(anchor, truth),
            prediction,
        )
    loss = float(np.mean((prediction - truth) ** 2))
    anchor_loss = float(np.mean((anchor - truth) ** 2))
    return (
        {"loss": loss, "rmse": math.sqrt(loss) * scale},
        {"loss": anchor_loss, "rmse": math.sqrt(anchor_loss) * scale},
        prediction,
    )


def train_control(
    data,
    universal,
    encoding,
    anchor,
    config: dict,
    architecture: str,
    device: str,
    task: str,
    residual_kind: str,
    evaluate_test: bool = True,
):
    seed = config["seed"]
    random.seed(seed + 101)
    np.random.seed(seed + 101)
    torch.manual_seed(seed + 101)
    torch.cuda.manual_seed_all(seed + 101)
    resolved = torch.device(device)
    builder = {
        "t_control": build_matched_control,
        "gated_chart": build_matched_gated,
        "supervised_mix": build_matched_supervised_mix,
    }[residual_kind]
    model, target_parameters, width, ff_width = builder(
        anchor, data, universal, encoding, config, architecture
    )
    model = model.to(resolved)
    batch = (
        min(config["batch_size"], 256)
        if architecture == "ft_transformer"
        else config["batch_size"]
    )
    loader_seed = seed + 101 if residual_kind == "supervised_mix" else seed
    streams = {
        part: make_loader(
            data,
            universal,
            part,
            {**config, "batch_size": batch, "seed": loader_seed},
            part == "train",
        )
        for part in PARTS
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    initial, _, _ = evaluate(model, streams["val"], resolved, task, data.y_scale)
    metric = "log_loss" if task == "classification" else "loss"
    best, best_epoch, stale = initial[metric], 0, 0
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for values in streams["train"]:
            *features, target = move(values, resolved)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                if residual_kind == "supervised_mix":
                    prediction, _, member = model.forward_all(*features)
                    loss = criterion(prediction, target) + criterion(member, target)
                else:
                    prediction, _ = model(*features)
                    loss = criterion(prediction, target)
                loss = loss + config["residual_gate_l1_weight"] * model.residual_gate.abs()
            loss.backward()
            optimizer.step()
        validation, _, _ = evaluate(
            model, streams["val"], resolved, task, data.y_scale
        )
        if validation[metric] < best:
            best, best_epoch, stale = validation[metric], epoch, 0
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > config["patience"]:
            break
    model.load_state_dict(state)
    validation, val_anchor, val_prediction = evaluate(
        model, streams["val"], resolved, task, data.y_scale
    )
    test = test_anchor = test_prediction = None
    if evaluate_test:
        test, test_anchor, test_prediction = evaluate(
            model, streams["test"], resolved, task, data.y_scale
        )
    result = {
        "residual_target_parameters": target_parameters,
        "residual_parameters": trainable_parameters(model),
        "residual_parameter_difference": trainable_parameters(model)
        - target_parameters,
        "matched_width": width,
        "matched_ft_feedforward_width": ff_width,
        "residual_best_epoch": best_epoch,
        **{f"val_{key}": value for key, value in validation.items()},
        **{f"val_anchor_{key}": value for key, value in val_anchor.items()},
        "residual_gate": float(model.residual_gate.detach().cpu()),
        "residual_train_seconds": time.perf_counter() - started,
    }
    if test is not None and test_anchor is not None:
        result.update(
            **{f"test_{key}": value for key, value in test.items()},
            **{f"test_anchor_{key}": value for key, value in test_anchor.items()},
        )
    if residual_kind == "gated_chart":
        result.update(
            q_gate_abs_mean=float(model.q_gate.detach().abs().mean().cpu()),
            q_gate_nonzero=int((model.q_gate.detach().abs() > 1e-6).sum().cpu()),
            rank_gate_abs_mean=float(
                model.rank_gate.detach().abs().mean().cpu()
            ),
            rank_gate_nonzero=int(
                (model.rank_gate.detach().abs() > 1e-6).sum().cpu()
            ),
        )
    return result, val_prediction, test_prediction


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("regression", "classification"), default="regression")
    parser.add_argument(
        "--residual-kind",
        choices=("t_control", "gated_chart", "supervised_mix"),
        default="t_control",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.config is None:
        args.config = HERE / (
            "trichart_frozen_anchor_config.json"
            if args.task == "regression"
            else "trichart_frozen_anchor_classification_parity_config.json"
        )
    if args.output is None:
        suffix = {
            "t_control": "t_control",
            "gated_chart": "gated_chart",
            "supervised_mix": "supervised_mix",
        }[args.residual_kind]
        args.output = HERE / "results" / f"trichart_frozen_anchor_{args.task}_{suffix}.csv"
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    rows: list[dict[str, object]] = list(read(args.output))
    done = {(row["dataset"], row["model"]) for row in rows}
    metadata = {"config": config, "task": args.task, "datasets": {}}
    for dataset_name in args.datasets or config["development_datasets"]:
        if dataset_name.startswith("openml-"):
            data = load_openml(dataset_name, config)
        elif args.task == "classification":
            data = load_binary(dataset_name, config)
        else:
            data = load_tabred(
                dataset_name,
                max_train_rows=config["max_train_rows"],
                max_eval_rows=config["max_eval_rows"],
                sample_seed=config["sample_seed"],
            )
        if args.task == "classification":
            encoding = prepare_classification_encodings(data, config)
            fit_anchor = fit_classification_anchor
        else:
            encoding = prepare_encodings(data, config)
            fit_anchor = fit_regression_anchor
        universal = prepare(data, config)
        metadata["datasets"][dataset_name] = {
            "n_fields": universal.n_fields,
            "full_split_sizes": data.split_sizes_full,
        }
        models = args.models or (
            config.get("maps_architectures", config["architectures"])
            if dataset_name == "maps-routing"
            else config["architectures"]
        )
        for model_name in models:
            key = (dataset_name, model_name)
            if key in done:
                continue
            started = time.perf_counter()
            anchor, anchor_result = fit_anchor(
                data,
                encoding,
                config,
                model_name,
                args.device,
                evaluate_test=not args.validation_only,
            )
            result, val, test = train_control(
                data,
                universal,
                encoding,
                anchor,
                config,
                model_name,
                args.device,
                args.task,
                args.residual_kind,
                evaluate_test=not args.validation_only,
            )
            directory = args.output.parent / f"{args.output.stem}_predictions"
            directory.mkdir(parents=True, exist_ok=True)
            arrays = {"validation": val}
            if test is not None:
                arrays["test"] = test
            with (
                directory / f"{dataset_name}__{model_name}__{config['seed']}.npz"
            ).open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "method": (
                    "frozen_t_anchor_t_residual_control"
                    if args.residual_kind == "t_control"
                    else (
                        "frozen_t_anchor_field_gated_chart_residual"
                        if args.residual_kind == "gated_chart"
                        else "safeple_zero_start_supervised_mix"
                    )
                ),
                "seed": config["seed"],
                "n_fields": universal.n_fields,
                **anchor_result,
                **result,
                "total_train_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            done.add(key)
            write(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
