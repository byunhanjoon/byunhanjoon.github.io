"""Prospective nested-discovery screen on the frozen untouched TabArena block."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
DEFAULT_JPLE = Path("/home/byunhanjoon/2027ICLR/projects/multi_ple/jple_tabarena")
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DEFAULT_JPLE))

import real_data_benchmark as benchmark  # noqa: E402
from hierarchical_residual import (  # noqa: E402
    DiscoveryConfig,
    discover,
    nested_audit,
    nonnested_selected_gain,
)
from src.data import load_official_split  # type: ignore[import-not-found]  # noqa: E402
from tabarena_bridge import to_dataset  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jple-root", type=Path, default=DEFAULT_JPLE)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "wine_quality",
            "miami_housing",
            "Food_Delivery_Time",
            "seismic-bumps",
            "heloc",
            "credit_card_clients_default",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results" / "tabarena_nested_screen"
    )
    args = parser.parse_args()
    specs = json.loads(
        (args.jple_root / "configs" / "stage1_datasets.json").read_text()
    )["datasets"]
    by_name = {spec["dataset"]: spec for spec in specs}
    missing = sorted(set(args.datasets) - set(by_name))
    if missing:
        raise ValueError(f"Datasets absent from frozen TabArena config: {missing}")

    config = DiscoveryConfig()
    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        spec = by_name[dataset_name]
        if spec["problem_type"] not in ("regression", "binary"):
            continue
        for fold in args.folds:
            split = load_official_split(
                spec,
                repeat=0,
                fold=fold,
                n_bins=args.bins,
                validation_fraction=args.validation_fraction,
                seed=args.split_seed,
                cache_dir=args.jple_root / "data_cache" / "openml",
            )
            dataset = to_dataset(split)
            cache: dict[str, object] = {}
            encoded = benchmark.encode_dataset(
                dataset,
                "schema_ple",
                fold,
                args.bins,
                config.max_cardinality,
                config.smoothing,
                1e-3,
                cache,
            )
            assert dataset.x_num is not None
            task = dataset.task
            design = encoded.x["train"].astype(np.float64)
            numeric = benchmark._clean_numeric(dataset.x_num)["train"]
            target = encoded.y["train"].astype(np.float64)
            selection, scores, _ = discover(
                design, numeric, target, task, fold, config
            )
            selection_rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "problem_type": split.problem_type,
                    "train_rows": len(target),
                    "numerical_features": numeric.shape[1],
                    "design_features": design.shape[1],
                    "singletons": ";".join(map(str, selection.singletons)),
                    "pairs": ";".join(f"{a}+{b}" for a, b in selection.pairs),
                    "nonempty": int(bool(selection.singletons or selection.pairs)),
                    "nonnested_relative_gain": nonnested_selected_gain(
                        design,
                        numeric,
                        target,
                        task,
                        fold,
                        selection,
                        config,
                    ),
                    "split_hash": split.split_hash,
                }
            )
            for score in scores:
                score_rows.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold,
                        "kind": score.kind,
                        "columns": ";".join(map(str, score.columns)),
                        "cardinality": score.cardinality,
                        "relative_gain": score.relative_gain,
                        "incremental_gain": score.incremental_gain,
                        "fold_wins": score.fold_wins,
                        "incremental_fold_wins": score.incremental_fold_wins,
                    }
                )
            audit = nested_audit(
                design,
                numeric,
                target,
                task,
                fold,
                config,
                outer_folds=5,
                inner_folds=5,
            )
            audit_rows.extend(
                {"dataset": dataset_name, "fold": fold, **row} for row in audit
            )
            print(
                f"{dataset_name} fold={fold}: singletons={selection.singletons or '-'} "
                f"pairs={selection.pairs or '-'} nested_gain="
                f"{100 * np.mean([float(row['relative_gain']) for row in audit]):+.4f}% "
                f"wins={sum(int(row['win']) for row in audit)}/5",
                flush=True,
            )
    write_csv(args.output_dir / "selections.csv", selection_rows)
    write_csv(args.output_dir / "candidate_scores.csv", score_rows)
    write_csv(args.output_dir / "nested_audit.csv", audit_rows)


if __name__ == "__main__":
    main()
