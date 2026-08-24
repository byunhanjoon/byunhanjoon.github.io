"""Run hierarchical residual discovery and its nested audit on TabPack data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
DAY2 = HERE.parent / "road-to-iclr-day-02"
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DAY2))

import cross_dataset_models as experiment  # noqa: E402
import residual_map_benchmark  # noqa: E402
from hierarchical_residual import (  # noqa: E402
    CandidateScore,
    DiscoveryConfig,
    discover,
    nested_audit,
    nonnested_selected_gain,
)


DATASETS = experiment.DATASETS


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_row(
    dataset: str, seed: int, score: CandidateScore
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "kind": score.kind,
        "columns": ";".join(map(str, score.columns)),
        "cardinality": score.cardinality,
        "relative_gain": score.relative_gain,
        "incremental_gain": score.incremental_gain,
        "fold_wins": score.fold_wins,
        "incremental_fold_wins": score.incremental_fold_wins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=["adult", "black-friday"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--max-cardinality", type=int, default=128)
    parser.add_argument("--max-pair-cardinality", type=int, default=512)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=5)
    parser.add_argument("--minimum-pair-fold-wins", type=int, default=5)
    parser.add_argument("--maximum-singletons", type=int, default=4)
    parser.add_argument("--maximum-pairs", type=int, default=4)
    parser.add_argument("--representation-budget", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results" / "nested_discovery"
    )
    args = parser.parse_args()

    config = DiscoveryConfig(
        folds=args.folds,
        smoothing=args.smoothing,
        max_cardinality=args.max_cardinality,
        max_pair_cardinality=args.max_pair_cardinality,
        minimum_relative_gain=args.minimum_relative_gain,
        minimum_fold_wins=args.minimum_fold_wins,
        minimum_pair_fold_wins=args.minimum_pair_fold_wins,
        maximum_singletons=args.maximum_singletons,
        maximum_pairs=args.maximum_pairs,
        representation_budget=args.representation_budget,
    )
    score_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        dataset = residual_map_benchmark.subsample_dataset(
            dataset, args.max_train_rows, None, args.sample_seed
        )
        for seed in args.seeds:
            cache: dict[str, object] = {}
            encoded = experiment.benchmark.encode_dataset(
                dataset,
                "schema_ple",
                seed,
                args.bins,
                args.max_cardinality,
                args.smoothing,
                1e-3,
                cache,
            )
            assert dataset.x_num is not None
            numeric = experiment.benchmark._clean_numeric(dataset.x_num)["train"]
            target = encoded.y["train"].astype(np.float64)
            design = encoded.x["train"].astype(np.float64)
            selection, scores, _ = discover(
                design, numeric, target, dataset.task, seed, config
            )
            score_rows.extend(score_row(dataset_name, seed, score) for score in scores)
            selections.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "train_rows": len(target),
                    "design_features": design.shape[1],
                    "singletons": ";".join(map(str, selection.singletons)),
                    "pairs": ";".join(
                        f"{left}+{right}" for left, right in selection.pairs
                    ),
                    "nonnested_relative_gain": nonnested_selected_gain(
                        design,
                        numeric,
                        target,
                        dataset.task,
                        seed,
                        selection,
                        config,
                    ),
                    "config": json.dumps(asdict(config), sort_keys=True),
                }
            )
            print(
                f"{dataset_name} seed={seed}: singletons={selection.singletons or '-'} "
                f"pairs={selection.pairs or '-'}",
                flush=True,
            )
            for row in nested_audit(
                design,
                numeric,
                target,
                dataset.task,
                seed,
                config,
                args.outer_folds,
                args.inner_folds,
            ):
                audit_rows.append({"dataset": dataset_name, "seed": seed, **row})
                print(
                    f"  outer={row['outer_fold']} selected="
                    f"{row['singletons'] or '-'}|{row['pairs'] or '-'} "
                    f"gain={100 * float(row['relative_gain']):+.4f}%",
                    flush=True,
                )

    write_csv(args.output_dir / "candidate_scores.csv", score_rows)
    write_csv(args.output_dir / "selections.csv", selections)
    write_csv(args.output_dir / "nested_audit.csv", audit_rows)


if __name__ == "__main__":
    main()
