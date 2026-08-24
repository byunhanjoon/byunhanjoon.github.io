"""Downstream benchmark for the fixed support-gated atomic encoder.

The discovery rule is fixed by ``DiscoveryConfig``. Selected singleton states
and pure-interaction pairs are represented with train-vocabulary one-hot views,
scaled by empirical support. Random controls are matched by state cardinality.
All neural variants are width-adjusted to the PLE baseline parameter budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


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
    Selection,
    candidate_columns,
    discover,
    state_keys,
)


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


def encode_group(
    clean: dict[str, np.ndarray],
    columns: tuple[int, ...],
    smoothing: float,
    gated: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_keys = state_keys(clean["train"][:, columns])
    keys, inverse, counts = np.unique(
        train_keys, return_inverse=True, return_counts=True
    )
    weights = counts / (counts + smoothing) if gated else np.ones_like(counts)
    output: dict[str, np.ndarray] = {}
    train = np.zeros((len(train_keys), len(keys)), dtype=np.float32)
    train[np.arange(len(train_keys)), inverse] = weights[inverse]
    output["train"] = train
    unseen: dict[str, float] = {}
    for part in ("val", "test"):
        query = state_keys(clean[part][:, columns])
        positions = np.searchsorted(keys, query)
        valid = positions < len(keys)
        matched = np.zeros(len(query), dtype=bool)
        matched[valid] = keys[positions[valid]] == query[valid]
        encoded = np.zeros((len(query), len(keys)), dtype=np.float32)
        encoded[np.flatnonzero(matched), positions[matched]] = weights[
            positions[matched]
        ]
        output[part] = encoded
        unseen[part] = float(1.0 - matched.mean())
    return output, {
        "columns": list(columns),
        "cardinality": len(keys),
        "minimum_count": int(counts.min()),
        "median_count": float(np.median(counts)),
        "maximum_count": int(counts.max()),
        "gated": gated,
        "val_unseen_rate": unseen["val"],
        "test_unseen_rate": unseen["test"],
    }


def encode_selection(
    base: experiment.benchmark.EncodedDataset,
    clean: dict[str, np.ndarray],
    selection: Selection,
    smoothing: float,
    gated: bool,
    name: str,
) -> tuple[experiment.benchmark.EncodedDataset, list[dict[str, object]]]:
    groups = [(column,) for column in selection.singletons] + list(selection.pairs)
    if not groups:
        return base, []
    encoded_groups = [
        encode_group(clean, tuple(group), smoothing, gated) for group in groups
    ]
    parts = {
        part: np.column_stack([values[part] for values, _ in encoded_groups])
        for part in ("train", "val", "test")
    }
    columns = tuple(sorted({column for group in groups for column in group}))
    return (
        experiment._append_view(base, parts, name, columns),
        [metadata for _, metadata in encoded_groups],
    )


def _distance(left: int, right: int) -> float:
    return abs(math.log1p(left) - math.log1p(right))


def matched_random_selection(
    numeric: np.ndarray,
    selected: Selection,
    max_cardinality: int,
    max_pair_cardinality: int,
    seed: int,
) -> Selection:
    """Match selected terms by cardinality without consulting targets or scores."""

    rng = np.random.default_rng(seed)
    columns = candidate_columns(numeric, max_cardinality)
    column_cardinality = {
        column: len(np.unique(numeric[:, column])) for column in columns
    }
    pair_cardinality = {
        pair: len(np.unique(state_keys(numeric[:, pair])))
        for pair in combinations(columns, 2)
        if len(np.unique(state_keys(numeric[:, pair]))) <= max_pair_cardinality
    }
    available_columns = [
        column for column in columns if column not in selected.singletons
    ]
    random_singletons: list[int] = []
    for selected_column in selected.singletons:
        if not available_columns:
            break
        target = column_cardinality[selected_column]
        ranked = sorted(
            available_columns,
            key=lambda column: (
                _distance(column_cardinality[column], target),
                rng.random(),
            ),
        )
        chosen = ranked[int(rng.integers(0, min(3, len(ranked))))]
        random_singletons.append(chosen)
        available_columns.remove(chosen)

    selected_pairs = set(selected.pairs)
    available_pairs = [pair for pair in pair_cardinality if pair not in selected_pairs]
    random_pairs: list[tuple[int, int]] = []
    for selected_pair in selected.pairs:
        if not available_pairs:
            break
        target = pair_cardinality[selected_pair]
        ranked = sorted(
            available_pairs,
            key=lambda pair: (
                _distance(pair_cardinality[pair], target),
                rng.random(),
            ),
        )
        chosen = ranked[int(rng.integers(0, min(3, len(ranked))))]
        random_pairs.append(chosen)
        available_pairs.remove(chosen)
    return Selection(tuple(random_singletons), tuple(random_pairs))


def serialize_scores(scores: list[CandidateScore]) -> str:
    compact = [
        {
            "kind": score.kind,
            "columns": list(score.columns),
            "cardinality": score.cardinality,
            "relative_gain": score.relative_gain,
            "incremental_gain": score.incremental_gain,
            "fold_wins": score.fold_wins,
            "incremental_fold_wins": score.incremental_fold_wins,
        }
        for score in scores
    ]
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def selection_text(selection: Selection) -> str:
    singleton = ";".join(map(str, selection.singletons))
    pairs = ";".join(f"{a}+{b}" for a, b in selection.pairs)
    return f"singletons={singleton}|pairs={pairs}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=experiment.DATASETS, default=["adult", "black-friday"]
    )
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=experiment.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--max-cardinality", type=int, default=128)
    parser.add_argument("--max-pair-cardinality", type=int, default=512)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=5)
    parser.add_argument("--minimum-pair-fold-wins", type=int, default=5)
    parser.add_argument("--maximum-singletons", type=int, default=4)
    parser.add_argument("--maximum-pairs", type=int, default=4)
    parser.add_argument("--representation-budget", type=int, default=512)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=[
            "baseline_ple",
            "adaptive_atomic",
            "adaptive_atomic_ungated",
            "all_eligible_singletons",
            "matched_random_0",
            "matched_random_1",
            "matched_random_2",
        ],
    )
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "adaptive_atomic.csv"
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
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (row["dataset"], row["model"], int(row["seed"]), row["representation"])
        for row in rows
    }
    device = torch.device(args.device)
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        dataset = residual_map_benchmark.subsample_dataset(
            dataset,
            args.max_train_rows,
            args.max_eval_rows,
            args.sample_seed,
        )
        for seed in args.seeds:
            cache: dict[str, object] = {}
            base = experiment.benchmark.encode_dataset(
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
            clean = experiment.benchmark._clean_numeric(dataset.x_num)
            selection, scores, _ = discover(
                base.x["train"].astype(np.float64),
                clean["train"],
                base.y["train"].astype(np.float64),
                dataset.task,
                seed,
                config,
            )
            variants: dict[
                str, tuple[experiment.benchmark.EncodedDataset, Selection, list[dict[str, object]]]
            ] = {"baseline_ple": (base, Selection((), ()), [])}
            gated, metadata = encode_selection(
                base, clean, selection, args.smoothing, True, "adaptive_atomic"
            )
            variants["adaptive_atomic"] = (gated, selection, metadata)
            ungated, metadata = encode_selection(
                base, clean, selection, args.smoothing, False, "adaptive_atomic_ungated"
            )
            variants["adaptive_atomic_ungated"] = (ungated, selection, metadata)
            all_singletons = Selection(
                candidate_columns(clean["train"], args.max_cardinality), ()
            )
            all_identity, metadata = encode_selection(
                base,
                clean,
                all_singletons,
                args.smoothing,
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
                    args.max_cardinality,
                    args.max_pair_cardinality,
                    seed + 10_000 * (control + 1),
                )
                if random_selection in seen_controls:
                    continue
                seen_controls.add(random_selection)
                name = f"matched_random_{control}"
                encoded, metadata = encode_selection(
                    base,
                    clean,
                    random_selection,
                    args.smoothing,
                    True,
                    name,
                )
                variants[name] = (encoded, random_selection, metadata)
            print(
                f"{dataset_name} seed={seed}: {selection_text(selection)}",
                flush=True,
            )
            if args.representations is not None:
                variants = {
                    name: value
                    for name, value in variants.items()
                    if name in args.representations
                }
            diagnostics = serialize_scores(scores)
            for model_name in args.models:
                config_model = experiment.MODEL_CONFIGS[model_name]
                budget = experiment.baseline_parameter_count(
                    base, model_name, config_model, args.ensemble_size
                )
                baseline_output = None
                for representation, (encoded, used_selection, metadata) in variants.items():
                    key = (dataset_name, model_name, seed, representation)
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
                            seed,
                            experiment.batch_size(dataset_name),
                            device,
                            config_model,
                            args.ensemble_size,
                            args.max_epochs,
                            args.patience,
                            budget,
                        )
                    if representation == "baseline_ple":
                        baseline_output = output
                    row = {
                        "dataset": dataset_name,
                        "task": dataset.task,
                        "model": model_name,
                        "seed": seed,
                        "representation": representation,
                        "selection": selection_text(selection),
                        "encoded_selection": selection_text(used_selection),
                        "state_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        "diagnostics": diagnostics,
                        "reused_baseline": int(reused),
                        "bins": args.bins,
                        "train_rows": len(dataset.y["train"]),
                        **output.result,
                        **experiment._extra_metrics(dataset, encoded, output),
                    }
                    rows.append(row)
                    completed.add(key)
                    write_rows(args.output, rows)
                    print(
                        f"  {model_name:<6} {representation:<25} "
                        f"test={float(output.result['test_score']):.6f}" +
                        (" reused" if reused else ""),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
