"""Causal closure test for the Day-3 ordinal chart orbit.

Each ordinal block is sample-whitened within its own declared field.  The five
equivalent bases then differ by block-orthogonal maps.  SGD, isotropic weight
decay, and an initial first-layer weight transported by the same map are
equivariant, so corresponding training trajectories should predict the same
function up to numerical error.

This is a mechanism/control experiment, not a claim that SGD is the best
tabular optimizer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DAY3_ROOT = REPOSITORY / "experiments" / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3_ROOT))

from experiments.day3.core import (  # noqa: E402
    MLP,
    combine,
    load_dataset,
    make_prepared,
    whiten,
)
from experiments.day3.run_suite import _nonordinal_schema, _ordinal_blocks  # noqa: E402
from orbit_anova import risk_summary  # noqa: E402


CHARTS = (
    "local",
    "cumulative",
    "cumulative_standardized",
    "path_spectral",
    "whitened",
)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 2048):
            outputs.append(model(torch.from_numpy(x[start : start + 2048]).to(device)).cpu().numpy())
    logits = np.concatenate(outputs).reshape(-1).astype(np.float64)
    positive = 1.0 / (1.0 + np.exp(-logits))
    return np.column_stack((1.0 - positive, positive))


def train_covariant(
    prepared,
    first_weight: np.ndarray,
    schema_width: int,
    block_widths: list[int],
    seed: int,
    device_name: str,
    epochs: int,
    optimizer_name: str,
) -> tuple[np.ndarray, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = MLP(prepared.x["train"].shape[1], 1, 256, 3, 0.1).to(device)
    with torch.no_grad():
        model.first.weight.copy_(torch.from_numpy(first_weight).to(device))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4
        )
        other_optimizer = None
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        other_optimizer = None
    elif optimizer_name == "field_vector_adam":
        optimizer = None
        other_parameters = [
            parameter for parameter in model.parameters()
            if parameter is not model.first.weight
        ]
        other_optimizer = torch.optim.AdamW(
            other_parameters, lr=1e-3, weight_decay=1e-4
        )
        first_moment = torch.zeros_like(model.first.weight)
        # Unchanged schema coordinates may retain ordinary Adam adaptivity.
        schema_second = torch.zeros_like(model.first.weight[:, :schema_width])
        block_seconds = [
            torch.zeros((model.first.weight.shape[0], 1), device=device)
            for _ in block_widths
        ]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    else:
        raise ValueError(optimizer_name)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(prepared.x["train"]),
            torch.from_numpy(prepared.y["train"]),
        ),
        batch_size=512,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    curve = []
    step = 0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch).squeeze(-1), y_batch)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            else:
                assert other_optimizer is not None
                other_optimizer.step()
                step += 1
                with torch.no_grad():
                    gradient = model.first.weight.grad
                    first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    first_moment_hat = first_moment / (1.0 - beta1**step)
                    update = torch.empty_like(first_moment)
                    schema_gradient = gradient[:, :schema_width]
                    schema_second.mul_(beta2).addcmul_(
                        schema_gradient, schema_gradient, value=1.0 - beta2
                    )
                    update[:, :schema_width] = first_moment_hat[:, :schema_width] / (
                        torch.sqrt(schema_second / (1.0 - beta2**step)) + epsilon
                    )
                    offset = schema_width
                    for width, second in zip(block_widths, block_seconds):
                        block_gradient = gradient[:, offset : offset + width]
                        second.mul_(beta2).add_(
                            torch.mean(block_gradient**2, dim=1, keepdim=True),
                            alpha=1.0 - beta2,
                        )
                        update[:, offset : offset + width] = (
                            first_moment_hat[:, offset : offset + width]
                            / (torch.sqrt(second / (1.0 - beta2**step)) + epsilon)
                        )
                        offset += width
                    model.first.weight.mul_(1.0 - 1e-3 * 1e-4)
                    model.first.weight.add_(update, alpha=-1e-3)
            total += float(loss.detach().cpu()) * len(x_batch)
        curve.append(total / len(prepared.y["train"]))
    return _predict(model, prepared.x["test"], device), curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw", "field_vector_adam"),
        default="sgd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "chart_covariant_training_adult.json",
    )
    args = parser.parse_args()

    dataset = load_dataset("adult")
    schema = _nonordinal_schema(dataset)
    x_by_chart = {}
    white_blocks_by_chart = {}
    for chart in CHARTS:
        blocks, _ = _ordinal_blocks(dataset, chart)
        white_blocks = [whiten(block)[0] for block in blocks]
        white_blocks_by_chart[chart] = white_blocks
        x_by_chart[chart] = combine([schema, *white_blocks])

    reference = CHARTS[0]
    schema_width = schema["train"].shape[1]
    block_widths = [
        block["train"].shape[1]
        for block in white_blocks_by_chart[reference]
    ]
    dimension = x_by_chart[reference]["train"].shape[1]
    transforms = {}
    diagnostics = {}
    for chart in CHARTS:
        transform = np.eye(dimension, dtype=np.float64)
        offset = schema_width
        for reference_block, chart_block in zip(
            white_blocks_by_chart[reference], white_blocks_by_chart[chart]
        ):
            width = reference_block["train"].shape[1]
            cross = (
                reference_block["train"].T @ chart_block["train"]
                / len(reference_block["train"])
            )
            transform[offset : offset + width, offset : offset + width] = cross
            offset += width
        transforms[chart] = transform
        diagnostics[chart] = {
            "orthogonality_error": float(
                np.max(np.abs(transform.T @ transform - np.eye(dimension)))
            ),
            "train_coordinate_residual": float(
                np.max(
                    np.abs(
                        x_by_chart[reference]["train"] @ transform
                        - x_by_chart[chart]["train"]
                    )
                )
            ),
            "test_coordinate_residual": float(
                np.max(
                    np.abs(
                        x_by_chart[reference]["test"] @ transform
                        - x_by_chart[chart]["test"]
                    )
                )
            ),
        }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_model = MLP(dimension, 1, 256, 3, 0.1)
    base_first = base_model.first.weight.detach().numpy().astype(np.float32)

    predictions = []
    curves = []
    for chart in CHARTS:
        prepared = make_prepared(dataset, x_by_chart[chart], {})
        transported_first = base_first @ transforms[chart]
        prediction, curve = train_covariant(
            prepared,
            transported_first.astype(np.float32),
            schema_width,
            block_widths,
            args.seed,
            args.device,
            args.epochs,
            args.optimizer,
        )
        predictions.append(prediction)
        curves.append(curve)
        print(f"{chart:26s} final_train_loss={curve[-1]:.8f}", flush=True)

    prediction_array = np.asarray(predictions)
    summary = risk_summary(
        prediction_array, dataset.y["test"].astype(np.int64), ("chart",)
    )
    output = {
        "design": {
            "dataset": "adult",
            "charts": list(CHARTS),
            "seed": args.seed,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "intervention": "within-field sample whitening and transported first-layer initialization",
        },
        "coordinate_diagnostics": diagnostics,
        "max_training_curve_range": float(np.max(np.ptp(np.asarray(curves), axis=0))),
        "risk": summary,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
