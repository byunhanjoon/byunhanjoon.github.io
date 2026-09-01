"""Prediction-orbit audit over Day-3 equivalent ordinal charts.

The Day-3 suite established that the five ordinal encodings below span the
same within-field function spaces, but stored only scalar metrics.  This
companion reruns the registered Adult MLP protocol and retains aligned test
probabilities so chart and seed can be crossed in OrbitANOVA.

The chart levels form a finite, declared audit distribution.  They are not
claimed to be a group.
"""

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
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DAY3_ROOT = REPOSITORY / "experiments" / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3_ROOT))

from experiments.day3.core import MLP, Prepared, ResNet, combine, load_dataset, make_prepared  # noqa: E402
from experiments.day3.run_suite import _nonordinal_schema, _ordinal_blocks  # noqa: E402
from orbit_anova import log_orbit_summary, risk_summary, symmetrization_frontier  # noqa: E402


CHARTS = (
    "local",
    "cumulative",
    "cumulative_standardized",
    "path_spectral",
    "whitened",
)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def fit_mlp_probabilities(
    data: Prepared,
    seed: int,
    device_name: str,
    width: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 40,
    patience: int = 6,
    model_name: str = "mlp",
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Match the Day-3 binary MLP training loop and return test probabilities."""
    if data.task != "binclass":
        raise ValueError("This focused pilot currently supports binary classification")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model_class = MLP if model_name == "mlp" else ResNet
    model = model_class(
        data.x["train"].shape[1], 1, width, depth, dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.x["train"]),
            torch.from_numpy(data.y["train"]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch).squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()
        validation_logits = _predict(
            model, data.x["val"], device, batch_size * 4
        ).reshape(-1)
        validation_loss = float(
            np.mean(np.logaddexp(0.0, validation_logits) - data.y["val"] * validation_logits)
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    logits = _predict(model, data.x["test"], device, batch_size * 4).reshape(-1)
    positive = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    probabilities = np.column_stack((1.0 - positive, positive))
    return probabilities, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_seconds": time.perf_counter() - started,
    }


def fit_linear_probabilities(data: Prepared) -> tuple[np.ndarray, dict[str, object]]:
    """Chart-covariantly whitened ridge-logistic negative control.

    Full-rank sample whitening maps equivalent design-matrix column spaces to
    coordinates that differ only by an orthogonal transform.  The L2 penalty
    and convex objective are therefore invariant to the original chart.
    """
    train = data.x["train"].astype(np.float64)
    test = data.x["test"].astype(np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    keep = singular_values > singular_values[0] * 1e-10
    scale = np.sqrt(len(train)) / singular_values[keep]
    train_white = (centered @ right[keep].T) * scale
    test_white = ((test - mean) @ right[keep].T) * scale
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-11,
    ).fit(train_white, data.y["train"])
    return model.predict_proba(test_white), {
        "iterations": int(model.n_iter_[0]),
        "converged": bool(model.n_iter_[0] < model.max_iter),
        "retained_rank": int(keep.sum()),
        "control": "full-rank sample whitening plus L2 logistic regression",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "chart_orbit_adult.npz",
    )
    args = parser.parse_args()

    dataset = load_dataset("adult")
    schema = _nonordinal_schema(dataset)
    prepared_by_chart: dict[str, Prepared] = {}
    for chart in CHARTS:
        blocks, metadata = _ordinal_blocks(dataset, chart)
        prepared_by_chart[chart] = make_prepared(
            dataset, combine([schema, *blocks]), metadata
        )

    linear_predictions = []
    linear_fits = []
    for chart in CHARTS:
        predictions, fit = fit_linear_probabilities(prepared_by_chart[chart])
        linear_predictions.append(predictions)
        linear_fits.append({"chart": chart, **fit})
        print(f"linear {chart:26s} iterations={fit['iterations']}", flush=True)

    mlp_predictions = np.empty(
        (len(CHARTS), len(args.seeds), len(dataset.y["test"]), 2),
        dtype=np.float64,
    )
    mlp_fits = []
    for chart_index, chart in enumerate(CHARTS):
        for seed_index, seed in enumerate(args.seeds):
            predictions, fit = fit_mlp_probabilities(
                prepared_by_chart[chart], seed, args.device
            )
            mlp_predictions[chart_index, seed_index] = predictions
            mlp_fits.append({"chart": chart, "seed": seed, **fit})
            print(
                f"mlp {chart:26s} seed={seed} epoch={fit['best_epoch']} "
                f"seconds={fit['train_seconds']:.2f}",
                flush=True,
            )

    linear_predictions_array = np.asarray(linear_predictions)
    y_test = dataset.y["test"].astype(np.int64)
    linear_summary = risk_summary(linear_predictions_array, y_test, ("chart",))
    mlp_summary = risk_summary(mlp_predictions, y_test, ("chart", "seed"))
    output = {
        "design": {
            "dataset": "adult",
            "charts": list(CHARTS),
            "seeds": args.seeds,
            "chart_semantics": "finite declared distribution over exactly span-equivalent ordinal bases",
            "split_fingerprint": dataset.split_fingerprint,
        },
        "linear_control": {
            "fits": linear_fits,
            "brier": linear_summary,
            "log": log_orbit_summary(linear_predictions_array, 1, y_test),
        },
        "mlp": {
            "fits": mlp_fits,
            "brier": mlp_summary,
            "log": log_orbit_summary(mlp_predictions, 2, y_test),
            "orbit_cover": symmetrization_frontier(
                mlp_predictions, ("chart", "seed")
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        charts=np.asarray(CHARTS),
        seeds=np.asarray(args.seeds),
        y_test=y_test,
        linear_predictions=linear_predictions_array,
        mlp_predictions=mlp_predictions,
    )
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
