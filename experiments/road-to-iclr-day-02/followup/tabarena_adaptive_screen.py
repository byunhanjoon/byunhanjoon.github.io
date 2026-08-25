"""Frozen Stage-1 downstream screen on untouched TabArena datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent
DAY1 = EXPERIMENTS / "road-to-iclr-day-01"
DAY2 = HERE.parent
DEFAULT_JPLE = Path("/home/byunhanjoon/2027ICLR/projects/multi_ple/jple_tabarena")
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DAY2))
sys.path.insert(0, str(DEFAULT_JPLE))

import cross_dataset_models as experiment  # noqa: E402
from adaptive_atomic_benchmark import (  # noqa: E402
    encode_selection,
    matched_random_selection,
    selection_text,
    serialize_scores,
)
from hierarchical_residual import (  # noqa: E402
    DiscoveryConfig,
    Selection,
    candidate_columns,
    discover,
)
from src.data import load_official_split  # type: ignore[import-not-found]  # noqa: E402
from tabarena_bridge import to_dataset  # noqa: E402


DEFAULT_DATASETS = [
    "wine_quality",
    "miami_housing",
    "Food_Delivery_Time",
    "seismic-bumps",
    "heloc",
    "credit_card_clients_default",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jple-root", type=Path, default=DEFAULT_JPLE)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=["mlp", "resnet"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "tabarena_adaptive_screen.csv"
    )
    args = parser.parse_args()

    specs = json.loads(
        (args.jple_root / "configs" / "stage1_datasets.json").read_text()
    )["datasets"]
    by_name = {spec["dataset"]: spec for spec in specs}
    config = DiscoveryConfig()
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (
            row["dataset"],
            int(row["fold"]),
            row["model"],
            int(row["seed"]),
            row["representation"],
        )
        for row in rows
    }
    device = torch.device(args.device)
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
            for model_seed in args.model_seeds:
                cache: dict[str, object] = {}
                base = experiment.benchmark.encode_dataset(
                    dataset,
                    "schema_ple",
                    model_seed,
                    args.bins,
                    config.max_cardinality,
                    config.smoothing,
                    1e-3,
                    cache,
                )
                assert dataset.x_num is not None
                clean = experiment.benchmark._clean_numeric(dataset.x_num)
                selection, scores, _ = discover(
                    base.x["train"].astype(np.float64),
                    clean["train"],
                    base.y["train"].astype(np.float64),
                    dataset.task,
                    fold,
                    config,
                )
                variants: dict[
                    str,
                    tuple[
                        experiment.benchmark.EncodedDataset,
                        Selection,
                        list[dict[str, object]],
                    ],
                ] = {"baseline_ple": (base, Selection((), ()), [])}
                adaptive, metadata = encode_selection(
                    base, clean, selection, config.smoothing, True, "adaptive_atomic"
                )
                variants["adaptive_atomic"] = (adaptive, selection, metadata)
                ungated, metadata = encode_selection(
                    base,
                    clean,
                    selection,
                    config.smoothing,
                    False,
                    "adaptive_atomic_ungated",
                )
                variants["adaptive_atomic_ungated"] = (
                    ungated,
                    selection,
                    metadata,
                )
                all_singletons = Selection(
                    candidate_columns(clean["train"], config.max_cardinality), ()
                )
                all_identity, metadata = encode_selection(
                    base,
                    clean,
                    all_singletons,
                    config.smoothing,
                    True,
                    "all_eligible_singletons",
                )
                variants["all_eligible_singletons"] = (
                    all_identity,
                    all_singletons,
                    metadata,
                )
                seen_controls: set[Selection] = set()
                for control in range(args.random_controls):
                    random_selection = matched_random_selection(
                        clean["train"],
                        selection,
                        config.max_cardinality,
                        config.max_pair_cardinality,
                        fold + 10_000 * (control + 1),
                    )
                    if random_selection in seen_controls:
                        continue
                    seen_controls.add(random_selection)
                    name = f"matched_random_{control}"
                    encoded, metadata = encode_selection(
                        base,
                        clean,
                        random_selection,
                        config.smoothing,
                        True,
                        name,
                    )
                    variants[name] = (encoded, random_selection, metadata)
                print(
                    f"{dataset_name} fold={fold}: {selection_text(selection)}",
                    flush=True,
                )
                diagnostics = serialize_scores(scores)
                for model_name in args.models:
                    model_config = experiment.MODEL_CONFIGS[model_name]
                    budget = experiment.baseline_parameter_count(
                        base, model_name, model_config, args.ensemble_size
                    )
                    baseline_output = None
                    for representation, (encoded, used_selection, metadata) in variants.items():
                        key = (
                            dataset_name,
                            fold,
                            model_name,
                            model_seed,
                            representation,
                        )
                        if key in completed:
                            continue
                        reused = (
                            not used_selection.singletons
                            and not used_selection.pairs
                            and representation != "baseline_ple"
                        )
                        if reused:
                            assert baseline_output is not None
                            output = baseline_output
                        else:
                            output = experiment.train_model(
                                encoded,
                                model_name,
                                model_seed + 1009 * fold,
                                512,
                                device,
                                model_config,
                                args.ensemble_size,
                                args.max_epochs,
                                args.patience,
                                budget,
                            )
                        if representation == "baseline_ple":
                            baseline_output = output
                        extra = experiment._extra_metrics(dataset, encoded, output)
                        test_error = (
                            1.0 - extra["test_auc"]
                            if dataset.task == "binclass"
                            else float(output.result["test_score"])
                        )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "task": dataset.task,
                                "fold": fold,
                                "split_hash": split.split_hash,
                                "model": model_name,
                                "seed": model_seed,
                                "representation": representation,
                                "selection": selection_text(selection),
                                "encoded_selection": selection_text(used_selection),
                                "state_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                                "diagnostics": diagnostics,
                                "reused_baseline": int(reused),
                                "bins": args.bins,
                                "train_rows": len(dataset.y["train"]),
                                "test_error": test_error,
                                **output.result,
                                **extra,
                            }
                        )
                        completed.add(key)
                        write_rows(args.output, rows)
                        print(
                            f"  {model_name:<6} {representation:<25} "
                            f"error={test_error:.6f}" + (" reused" if reused else ""),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
