"""Label-permutation false-discovery control for the frozen selector."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent
DAY1 = EXPERIMENTS / "road-to-iclr-day-01"
DAY2 = HERE.parent
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DAY2))

import cross_dataset_models as experiment  # noqa: E402
from hierarchical_residual import DiscoveryConfig, discover  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=experiment.DATASETS, default=["adult", "black-friday"]
    )
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "permutation_null.csv"
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
        design = encoded.x["train"].astype(np.float64)
        for permutation in range(args.permutations):
            rng = np.random.default_rng(100_000 + permutation)
            shuffled = rng.permutation(target)
            selection, scores, _ = discover(
                design,
                numeric,
                shuffled,
                dataset.task,
                permutation,
                config,
            )
            singleton_scores = [
                score.relative_gain for score in scores if score.kind == "singleton"
            ]
            pair_scores = [
                score.incremental_gain for score in scores if score.kind == "pair"
            ]
            rows.append(
                {
                    "dataset": dataset_name,
                    "permutation": permutation,
                    "selected_singletons": ";".join(map(str, selection.singletons)),
                    "selected_pairs": ";".join(
                        f"{a}+{b}" for a, b in selection.pairs
                    ),
                    "selection_nonempty": int(
                        bool(selection.singletons or selection.pairs)
                    ),
                    "maximum_singleton_gain": max(singleton_scores, default=float("nan")),
                    "maximum_pair_incremental_gain": max(pair_scores, default=float("nan")),
                }
            )
        false_discoveries = sum(
            int(row["selection_nonempty"])
            for row in rows
            if row["dataset"] == dataset_name
        )
        print(
            f"{dataset_name}: false discoveries {false_discoveries}/{args.permutations}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
