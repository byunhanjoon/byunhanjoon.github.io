"""Geometry of information-equivalent PLE and identity bases on finite support."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
DAY2 = HERE.parent / "road-to-iclr-day-02"
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DAY2))

import cross_dataset_models as experiment  # noqa: E402
from hierarchical_residual import DiscoveryConfig, discover  # noqa: E402


def centered_geometry(
    basis: np.ndarray, counts: np.ndarray, effect: np.ndarray
) -> tuple[int, float, float, float]:
    weights = counts / counts.sum()
    centered_basis = basis - weights @ basis
    centered_effect = effect - weights @ effect
    weighted = np.sqrt(weights[:, None]) * centered_basis
    singular = np.linalg.svd(weighted, compute_uv=False)
    nonzero = singular[singular > singular.max(initial=0.0) * 1e-10]
    rank = len(nonzero)
    condition = float(nonzero.max() / nonzero.min()) if rank else float("nan")
    coefficient, *_ = np.linalg.lstsq(centered_basis, centered_effect, rcond=1e-10)
    reconstruction = centered_basis @ coefficient
    weighted_error = float(np.sum(weights * (reconstruction - centered_effect) ** 2))
    return rank, condition, float(np.linalg.norm(coefficient)), weighted_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=experiment.DATASETS,
        default=["adult", "black-friday"],
    )
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "basis_geometry.csv"
    )
    args = parser.parse_args()
    config = DiscoveryConfig()
    rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        cache: dict[str, object] = {}
        encoded = experiment.benchmark.encode_dataset(
            dataset, "schema_ple", 0, args.bins, 128, 20.0, 1e-3, cache
        )
        assert dataset.x_num is not None
        numeric = experiment.benchmark._clean_numeric(dataset.x_num)["train"]
        target = encoded.y["train"].astype(np.float64)
        selection, scores, smooth_prediction = discover(
            encoded.x["train"].astype(np.float64),
            numeric,
            target,
            dataset.task,
            0,
            config,
        )
        score_by_column = {
            score.columns[0]: score
            for score in scores
            if score.kind == "singleton"
        }
        residual = target - smooth_prediction
        for column, score in score_by_column.items():
            states, inverse, counts = np.unique(
                numeric[:, column], return_inverse=True, return_counts=True
            )
            state_effect = np.bincount(inverse, weights=residual) / counts
            parts = {"train": states[:, None], "val": states[:, None], "test": states[:, None]}
            ple = experiment.benchmark._piecewise_linear(parts, args.bins)["train"]
            identity = np.eye(len(states), dtype=np.float64)
            support = counts / (counts + config.smoothing)
            gated_identity = identity * support[None, :]
            geometries = {
                "ple": centered_geometry(ple, counts, state_effect),
                "identity": centered_geometry(identity, counts, state_effect),
                "support_gated_identity": centered_geometry(
                    gated_identity, counts, state_effect
                ),
            }
            for basis_name, (rank, condition, coefficient_norm, error) in geometries.items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "column": column,
                        "states": len(states),
                        "selected": int(column in selection.singletons),
                        "diagnostic_gain": score.relative_gain,
                        "fold_wins": score.fold_wins,
                        "basis": basis_name,
                        "centered_rank": rank,
                        "condition_number": condition,
                        "minimum_coefficient_norm": coefficient_norm,
                        "weighted_reconstruction_mse": error,
                    }
                )
        print(
            f"{dataset_name}: selected singletons={selection.singletons or '-'}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
