"""Test stable exact-value pair interactions in downstream neural models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

import cross_dataset_models as experiment
import residual_map_benchmark as residual_benchmark
from interaction_utility_probe import mapped_correction, probe, row_keys


HERE = Path(__file__).resolve().parent
REPRESENTATIONS = (
    "baseline_ple",
    "additive_identity_pair",
    "crossed_identity_pair",
    "additive_residual_pair",
    "crossed_residual_pair",
)


def crossed_identity(
    clean: dict[str, np.ndarray], pair: tuple[int, int]
) -> dict[str, np.ndarray]:
    train_keys = row_keys(clean["train"][:, pair])
    keys, inverse = np.unique(train_keys, return_inverse=True)
    output: dict[str, np.ndarray] = {}
    train = np.zeros((len(train_keys), len(keys)), dtype=np.float32)
    train[np.arange(len(train_keys)), inverse] = 1.0
    output["train"] = train
    for part in ("val", "test"):
        query = row_keys(clean[part][:, pair])
        positions = np.searchsorted(keys, query)
        valid = positions < len(keys)
        matched = np.zeros(len(query), dtype=bool)
        matched[valid] = keys[positions[valid]] == query[valid]
        encoded = np.zeros((len(query), len(keys)), dtype=np.float32)
        encoded[np.flatnonzero(matched), positions[matched]] = 1.0
        output[part] = encoded
    return output


def crossed_residual_map(
    dataset: experiment.benchmark.Dataset,
    seed: int,
    pair: tuple[int, int],
    alpha: float,
    cache: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assert dataset.x_num is not None
    clean = experiment.benchmark._clean_numeric(dataset.x_num)
    residual = np.asarray(cache["numeric_residual"], dtype=np.float64)
    target = dataset.y["train"]
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
        if dataset.task == "binclass"
        else KFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
    )
    folds = list(
        splitter.split(
            clean["train"], target if dataset.task == "binclass" else None
        )
    )
    output = {
        part: np.zeros((len(values), 1), dtype=np.float64)
        for part, values in clean.items()
    }
    for fit_index, holdout_index in folds:
        correction, _ = mapped_correction(
            clean["train"][fit_index][:, pair],
            residual[fit_index],
            clean["train"][holdout_index][:, pair],
            alpha,
        )
        output["train"][holdout_index, 0] = correction
    full_cardinality = 0
    for part in ("val", "test"):
        correction, full_cardinality = mapped_correction(
            clean["train"][:, pair], residual, clean[part][:, pair], alpha
        )
        output[part][:, 0] = correction
    scale = max(float(np.sqrt(np.mean(output["train"] ** 2))), 1e-6)
    return (
        {
            part: (values / scale).astype(np.float32)
            for part, values in output.items()
        },
        {
            "alpha": alpha,
            "pair": list(pair),
            "pair_cardinality": full_cardinality,
            "scale": scale,
        },
    )


def encode_variants(
    dataset: experiment.benchmark.Dataset,
    seed: int,
    bins: int,
    alpha: float,
    minimum_relative_gain: float,
    minimum_fold_wins: int,
) -> tuple[
    dict[str, experiment.benchmark.EncodedDataset],
    tuple[int, int] | None,
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    cache: dict[str, object] = {}
    base = experiment.benchmark.encode_dataset(
        dataset, "schema_ple", seed, bins, 128, 20.0, 1e-3, cache
    )
    statistics = probe(dataset, seed, bins, 128, 20.0, cache)
    eligible = [
        row
        for row in statistics
        if float(row["relative_gain"]) >= minimum_relative_gain
        and int(row["fold_wins"]) >= minimum_fold_wins
    ]
    pair = None
    if eligible:
        best = max(eligible, key=lambda row: float(row["relative_gain"]))
        pair = (int(best["left_column"]), int(best["right_column"]))
    variants = {"baseline_ple": base}
    metadata: dict[str, dict[str, object]] = {"baseline_ple": {}}
    if pair is None:
        for name in REPRESENTATIONS[1:]:
            variants[name] = base
            metadata[name] = {}
        return variants, pair, statistics, metadata

    assert dataset.x_num is not None
    clean = experiment.benchmark._clean_numeric(dataset.x_num)
    variants["additive_identity_pair"] = experiment._append_view(
        base, experiment._full_identity(clean, pair), "additive_identity_pair", pair
    )
    variants["crossed_identity_pair"] = experiment._append_view(
        base, crossed_identity(clean, pair), "crossed_identity_pair", pair
    )
    additive_parts, additive_metadata = residual_benchmark.residual_map(
        dataset, seed, pair, alpha, cache
    )
    variants["additive_residual_pair"] = experiment._append_view(
        base, additive_parts, "additive_residual_pair", pair
    )
    crossed_parts, crossed_metadata = crossed_residual_map(
        dataset, seed, pair, alpha, cache
    )
    variants["crossed_residual_pair"] = experiment._append_view(
        base, crossed_parts, "crossed_residual_pair", pair
    )
    metadata.update(
        {
            "additive_identity_pair": {},
            "crossed_identity_pair": {},
            "additive_residual_pair": additive_metadata,
            "crossed_residual_pair": crossed_metadata,
        }
    )
    return variants, pair, statistics, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DAY1 / "data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=experiment.DATASETS,
        default=["adult", "black-friday", "churn"],
    )
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=experiment.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "interaction_view.csv"
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(residual_benchmark.read_rows(args.output))
    completed = {
        (row["dataset"], row["model"], int(row["seed"]), row["representation"])
        for row in rows
    }
    device = torch.device(args.device)
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        dataset = residual_benchmark.subsample_dataset(
            dataset,
            args.max_train_rows,
            args.max_eval_rows,
            args.sample_seed,
        )
        for seed in args.seeds:
            variants, pair, pair_statistics, metadata = encode_variants(
                dataset,
                seed,
                args.bins,
                args.alpha,
                args.minimum_relative_gain,
                args.minimum_fold_wins,
            )
            print(f"{dataset_name} seed={seed} pair={pair or '-'}", flush=True)
            serialized_statistics = {
                f"{row['left_column']};{row['right_column']}": {
                    "pair_cardinality": row["pair_cardinality"],
                    "relative_gain": row["relative_gain"],
                    "fold_wins": row["fold_wins"],
                }
                for row in pair_statistics
            }
            selected = () if pair is None else pair
            for model_name in args.models:
                config = experiment.MODEL_CONFIGS[model_name]
                budget = experiment.baseline_parameter_count(
                    variants["baseline_ple"], model_name, config, args.ensemble_size
                )
                baseline_output = None
                for representation, encoded in variants.items():
                    key = (dataset_name, model_name, seed, representation)
                    if key in completed:
                        continue
                    reused = pair is None and representation != "baseline_ple"
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
                            config,
                            args.ensemble_size,
                            args.max_epochs,
                            args.patience,
                            budget,
                        )
                    if representation == "baseline_ple":
                        baseline_output = output
                    row = residual_benchmark.result_row(
                        dataset,
                        model_name,
                        seed,
                        args.bins,
                        representation,
                        selected,
                        serialized_statistics,  # type: ignore[arg-type]
                        metadata[representation],
                        output,
                        encoded,
                        reused,
                    )
                    row["pair_statistics"] = json.dumps(
                        pair_statistics, sort_keys=True, separators=(",", ":")
                    )
                    rows.append(row)
                    completed.add(key)
                    residual_benchmark.write_rows(args.output, rows)
                    print(
                        f"  {model_name:<6} {representation:<24} "
                        f"test={float(output.result['test_score']):.6f}" +
                        (" reused" if reused else ""),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
