"""Regression transfer gate for raw and covariant Day-3 chart training."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DAY3_ROOT = REPOSITORY / "experiments" / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3_ROOT))

from experiments.day3.core import MLP, ResNet, combine, load_dataset, make_prepared, whiten  # noqa: E402
from experiments.day3.run_suite import _nonordinal_schema, _ordinal_blocks  # noqa: E402
from orbit_anova import decompose  # noqa: E402


CHARTS = (
    "local",
    "cumulative",
    "cumulative_standardized",
    "path_spectral",
    "whitened",
)


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 2048):
            outputs.append(model(torch.from_numpy(x[start : start + 2048]).to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def loader_for(prepared, seed: int, device: torch.device) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(prepared.x["train"]),
            torch.from_numpy(prepared.y["train"]),
        ),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )


def seeded_model(
    prepared, seed: int, device: torch.device, model_name: str = "mlp"
) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_class = MLP if model_name == "mlp" else ResNet
    return model_class(prepared.x["train"].shape[1], 1, 256, 3, 0.1).to(device)


def fit_adamw(
    prepared, seed: int, device_name: str, model_name: str = "mlp"
) -> tuple[np.ndarray, dict[str, object]]:
    device = torch.device(device_name)
    model = seeded_model(prepared, seed, device, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    loader = loader_for(prepared, seed, device)
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, 41):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch).squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()
        validation = predict(model, prepared.x["val"], device).reshape(-1)
        validation_loss = float(np.mean((validation - prepared.y["val"]) ** 2))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > 6:
            break
    if best_state is None:
        raise RuntimeError("No AdamW checkpoint")
    model.load_state_dict(best_state)
    return predict(model, prepared.x["test"], device), {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "seconds": time.perf_counter() - started,
    }


def fit_covariant_sgd(
    prepared,
    transported_first: np.ndarray,
    seed: int,
    device_name: str,
    epochs: int,
    learning_rate: float = 0.03,
) -> tuple[np.ndarray, list[float]]:
    device = torch.device(device_name)
    model = seeded_model(prepared, seed, device)
    with torch.no_grad():
        model.first.weight.copy_(torch.from_numpy(transported_first).to(device))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    criterion = nn.MSELoss()
    loader = loader_for(prepared, seed, device)
    curve = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch).squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x_batch)
        curve.append(total / len(prepared.y["train"]))
    return predict(model, prepared.x["test"], device), curve


def fit_field_vector_adam(
    prepared,
    transported_first: np.ndarray,
    schema_width: int,
    block_widths: list[int],
    seed: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """AdamW with rotation-equivariant second moments on field blocks."""
    device = torch.device(device_name)
    model = seeded_model(prepared, seed, device)
    with torch.no_grad():
        model.first.weight.copy_(torch.from_numpy(transported_first).to(device))
    other_parameters = [
        parameter for parameter in model.parameters()
        if parameter is not model.first.weight
    ]
    other_optimizer = torch.optim.AdamW(
        other_parameters, lr=1e-3, weight_decay=1e-4
    )
    first_moment = torch.zeros_like(model.first.weight)
    schema_second = torch.zeros_like(model.first.weight[:, :schema_width])
    block_seconds = [
        torch.zeros((model.first.weight.shape[0], 1), device=device)
        for _ in block_widths
    ]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    criterion = nn.MSELoss()
    loader = loader_for(prepared, seed, device)
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    step = 0
    curve = []
    for epoch in range(1, 41):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch).squeeze(-1), y_batch)
            loss.backward()
            other_optimizer.step()
            step += 1
            with torch.no_grad():
                gradient = model.first.weight.grad
                first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                first_hat = first_moment / (1.0 - beta1**step)
                update = torch.empty_like(first_hat)
                schema_gradient = gradient[:, :schema_width]
                schema_second.mul_(beta2).addcmul_(
                    schema_gradient, schema_gradient, value=1.0 - beta2
                )
                update[:, :schema_width] = first_hat[:, :schema_width] / (
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
                        first_hat[:, offset : offset + width]
                        / (torch.sqrt(second / (1.0 - beta2**step)) + epsilon)
                    )
                    offset += width
                model.first.weight.mul_(1.0 - 1e-3 * 1e-4)
                model.first.weight.add_(update, alpha=-1e-3)
            total += float(loss.detach().cpu()) * len(x_batch)
        curve.append(total / len(prepared.y["train"]))
        validation = predict(model, prepared.x["val"], device).reshape(-1)
        validation_loss = float(np.mean((validation - prepared.y["val"]) ** 2))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > 6:
            break
    if best_state is None:
        raise RuntimeError("No field-VectorAdam checkpoint")
    model.load_state_dict(best_state)
    return predict(model, prepared.x["test"], device), {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "training_curve": curve,
    }


def full_rank_white(prepared) -> tuple[np.ndarray, np.ndarray, int]:
    train = prepared.x["train"].astype(np.float64)
    test = prepared.x["test"].astype(np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    keep = singular > singular[0] * 1e-10
    scale = np.sqrt(len(train)) / singular[keep]
    return (
        (centered @ right[keep].T) * scale,
        ((test - mean) @ right[keep].T) * scale,
        int(keep.sum()),
    )


def squared_summary(
    predictions: np.ndarray,
    y: np.ndarray,
    factor_names: tuple[str, ...],
    target_scale: float,
) -> dict[str, object]:
    factor_axes = tuple(range(len(factor_names)))
    mean_prediction = predictions.mean(axis=factor_axes)
    flat = predictions.reshape((-1,) + predictions.shape[-2:])
    member_mse = np.mean((flat[..., 0] - y[None]) ** 2, axis=1)
    mean_mse = float(np.mean((mean_prediction[:, 0] - y) ** 2))
    decomposition = decompose(predictions, factor_names)
    row_ranges = np.ptp(flat[..., 0], axis=0)
    return {
        "reference_mse_standardized": float(
            np.mean((predictions[(0,) * len(factor_names)][..., 0] - y) ** 2)
        ),
        "mean_member_mse_standardized": float(member_mse.mean()),
        "best_member_mse_standardized": float(member_mse.min()),
        "worst_member_mse_standardized": float(member_mse.max()),
        "orbit_mean_mse_standardized": mean_mse,
        "orbit_mean_rmse_original_units": math.sqrt(mean_mse) * target_scale,
        "mse_reduction_by_averaging": float(member_mse.mean() - mean_mse),
        "risk_identity_absolute_error": abs(
            float(member_mse.mean() - mean_mse) - decomposition["total"]
        ),
        "prediction_range_original_units": {
            "mean": float(row_ranges.mean() * target_scale),
            "p95": float(np.quantile(row_ranges, 0.95) * target_scale),
            "max": float(row_ranges.max() * target_scale),
        },
        "anova": decomposition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("diamond", "black-friday"), default="diamond")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--covariant-epochs", type=int, default=100)
    parser.add_argument("--covariant-learning-rate", type=float, default=0.03)
    parser.add_argument("--covariant-seed", type=int, default=17)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--reuse-adam-from", type=Path,
        help="Read a frozen AdamW/ridge tensor from this NPZ while writing a separate output.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = HERE / f"chart_regression_{args.dataset.replace('-', '_')}.npz"

    dataset = load_dataset(args.dataset)
    if dataset.task != "regression":
        raise ValueError("Regression dataset required")
    schema = _nonordinal_schema(dataset)
    raw_prepared = {}
    raw_blocks = {}
    for chart in CHARTS:
        blocks, metadata = _ordinal_blocks(dataset, chart)
        raw_blocks[chart] = blocks
        raw_prepared[chart] = make_prepared(
            dataset, combine([schema, *blocks]), metadata
        )

    reuse_path = args.reuse_adam_from
    if reuse_path is None and args.reuse_existing and args.output.exists():
        reuse_path = args.output
    if reuse_path is not None and reuse_path.exists():
        existing = np.load(reuse_path)
        adam_predictions = existing["adamw_predictions"]
        adam_fits = [{"reused_from": str(reuse_path)}]
        if "ridge_predictions" in existing:
            ridge_predictions = list(existing["ridge_predictions"])
            ridge_ranks = [
                raw_prepared[chart].x["train"].shape[1] for chart in CHARTS
            ]
        else:
            ridge_predictions = []
            ridge_ranks = []
            for chart in CHARTS:
                train_white, test_white, rank = full_rank_white(raw_prepared[chart])
                ridge = Ridge(alpha=1.0).fit(
                    train_white, raw_prepared[chart].y["train"]
                )
                ridge_predictions.append(ridge.predict(test_white)[:, None])
                ridge_ranks.append(rank)
    else:
        adam_predictions = np.empty(
            (len(CHARTS), len(args.seeds), len(dataset.y["test"]), 1),
            dtype=np.float64,
        )
        adam_fits = []
        for chart_index, chart in enumerate(CHARTS):
            for seed_index, seed in enumerate(args.seeds):
                prediction, fit = fit_adamw(raw_prepared[chart], seed, args.device)
                adam_predictions[chart_index, seed_index] = prediction
                adam_fits.append({"chart": chart, "seed": seed, **fit})
                print(
                    f"adamw {chart:26s} seed={seed:2d} epoch={fit['best_epoch']:2d} "
                    f"seconds={fit['seconds']:.2f}",
                    flush=True,
                )

        ridge_predictions = []
        ridge_ranks = []
        for chart in CHARTS:
            train_white, test_white, rank = full_rank_white(raw_prepared[chart])
            ridge = Ridge(alpha=1.0).fit(train_white, raw_prepared[chart].y["train"])
            ridge_predictions.append(ridge.predict(test_white)[:, None])
            ridge_ranks.append(rank)

    white_blocks = {
        chart: [whiten(block)[0] for block in raw_blocks[chart]]
        for chart in CHARTS
    }
    x_covariant = {
        chart: combine([schema, *white_blocks[chart]]) for chart in CHARTS
    }
    reference = CHARTS[0]
    dimension = x_covariant[reference]["train"].shape[1]
    schema_width = schema["train"].shape[1]
    transforms = {}
    coordinate_diagnostics = {}
    for chart in CHARTS:
        transform = np.eye(dimension)
        offset = schema_width
        for reference_block, chart_block in zip(
            white_blocks[reference], white_blocks[chart]
        ):
            width = reference_block["train"].shape[1]
            transform[offset : offset + width, offset : offset + width] = (
                reference_block["train"].T @ chart_block["train"]
                / len(reference_block["train"])
            )
            offset += width
        transforms[chart] = transform
        coordinate_diagnostics[chart] = {
            "orthogonality_error": float(
                np.max(np.abs(transform.T @ transform - np.eye(dimension)))
            ),
            "test_coordinate_residual": float(
                np.max(
                    np.abs(
                        x_covariant[reference]["test"] @ transform
                        - x_covariant[chart]["test"]
                    )
                )
            ),
        }

    reference_prepared = make_prepared(dataset, x_covariant[reference], {})
    base_model = seeded_model(
        reference_prepared, args.covariant_seed, torch.device("cpu")
    )
    base_first = base_model.first.weight.detach().numpy()
    block_widths = [
        block["train"].shape[1] for block in white_blocks[reference]
    ]
    covariant_predictions = []
    curves = []
    for chart in CHARTS:
        prepared = make_prepared(dataset, x_covariant[chart], {})
        prediction, curve = fit_covariant_sgd(
            prepared,
            (base_first @ transforms[chart]).astype(np.float32),
            args.covariant_seed,
            args.device,
            args.covariant_epochs,
            args.covariant_learning_rate,
        )
        covariant_predictions.append(prediction)
        curves.append(curve)
        print(f"covariant {chart:22s} train_mse={curve[-1]:.8f}", flush=True)

    vector_predictions = []
    vector_fits = []
    for chart in CHARTS:
        prepared = make_prepared(dataset, x_covariant[chart], {})
        prediction, fit = fit_field_vector_adam(
            prepared,
            (base_first @ transforms[chart]).astype(np.float32),
            schema_width,
            block_widths,
            args.covariant_seed,
            args.device,
        )
        vector_predictions.append(prediction)
        vector_fits.append({"chart": chart, **fit})
        print(
            f"field-vector-adam {chart:14s} epoch={fit['best_epoch']:2d} "
            f"val={fit['best_validation_mse']:.8f}",
            flush=True,
        )

    y_test = raw_prepared[reference].y["test"].astype(np.float64)
    ridge_array = np.asarray(ridge_predictions)
    covariant_array = np.asarray(covariant_predictions)
    vector_array = np.asarray(vector_predictions)
    common_vector_epochs = min(
        len(fit["training_curve"]) for fit in vector_fits
    )
    vector_curve_range = float(
        np.max(
            np.ptp(
                np.asarray(
                    [
                        fit["training_curve"][:common_vector_epochs]
                        for fit in vector_fits
                    ]
                ),
                axis=0,
            )
        )
    )
    output = {
        "design": {
            "dataset": args.dataset,
            "charts": list(CHARTS),
            "seeds": args.seeds,
            "target_scale": raw_prepared[reference].y_scale,
            "split_fingerprint": dataset.split_fingerprint,
        },
        "raw_adamw": {
            "fits": adam_fits,
            "summary": squared_summary(
                adam_predictions,
                y_test,
                ("chart", "seed"),
                raw_prepared[reference].y_scale,
            ),
        },
        "ridge_control": {
            "retained_ranks": ridge_ranks,
            "summary": squared_summary(
                ridge_array,
                y_test,
                ("chart",),
                raw_prepared[reference].y_scale,
            ),
        },
        "covariant_sgd": {
            "epochs": args.covariant_epochs,
            "learning_rate": args.covariant_learning_rate,
            "seed": args.covariant_seed,
            "coordinate_diagnostics": coordinate_diagnostics,
            "max_training_curve_range": float(
                np.max(np.ptp(np.asarray(curves), axis=0))
            ),
            "summary": squared_summary(
                covariant_array,
                y_test,
                ("chart",),
                raw_prepared[reference].y_scale,
            ),
        },
        "field_vector_adam": {
            "seed": args.covariant_seed,
            "fits": vector_fits,
            "max_training_curve_range": vector_curve_range,
            "summary": squared_summary(
                vector_array,
                y_test,
                ("chart",),
                raw_prepared[reference].y_scale,
            ),
        },
    }
    np.savez_compressed(
        args.output,
        charts=np.asarray(CHARTS),
        seeds=np.asarray(args.seeds),
        y_test=y_test,
        adamw_predictions=adam_predictions,
        ridge_predictions=ridge_array,
        covariant_predictions=covariant_array,
        field_vector_adam_predictions=vector_array,
    )
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
