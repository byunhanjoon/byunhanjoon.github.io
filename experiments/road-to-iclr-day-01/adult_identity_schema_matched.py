#!/usr/bin/env python3
"""Matched Adult comparison of selected identity, seed, and schema diversity."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import real_data_benchmark as benchmark


ROOT = Path(__file__).resolve().parent


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def schema_transform(values: np.ndarray, variant: str) -> np.ndarray:
    if variant == "PAD":
        paired = np.stack([values, np.zeros_like(values)], axis=-1)
    elif variant == "DUPLICATE":
        paired = np.stack([values, values], axis=-1)
    elif variant == "POSNEG":
        paired = np.stack([np.maximum(values, 0), np.maximum(-values, 0)], axis=-1)
    elif variant == "SIGNMAG":
        paired = np.stack([np.abs(values), np.sign(values)], axis=-1)
    else:
        raise ValueError(variant)
    return paired.reshape(len(values), -1).astype(np.float32)


def equivalent_schema_encoded(
    baseline: benchmark.EncodedDataset,
    cache: dict[str, object],
    variant: str,
) -> benchmark.EncodedDataset:
    normalized = cast(dict[str, np.ndarray], cache["normalized_num"])
    transformed = {part: schema_transform(values, variant) for part, values in normalized.items()}
    mean = transformed["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = transformed["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    transformed = {
        part: ((values - mean) / std).astype(np.float32)
        for part, values in transformed.items()
    }

    schema_tail: dict[str, list[np.ndarray]] = {part: [] for part in baseline.x}
    if "clean_bin" in cache:
        clean_bin = cast(dict[str, np.ndarray], cache["clean_bin"])
        for part in schema_tail:
            schema_tail[part].append(clean_bin[part].astype(np.float32))
    if "cat_identity" in cache:
        cat_identity = cast(dict[str, np.ndarray], cache["cat_identity"])
        for part in schema_tail:
            schema_tail[part].append(cat_identity[part].astype(np.float32))
    ple = cast(dict[str, np.ndarray], cache["numeric_ple"])

    x = {}
    schema_size = transformed["train"].shape[1] + sum(
        component.shape[1] for component in schema_tail["train"]
    )
    for part in baseline.x:
        components = [transformed[part], *schema_tail[part], ple[part]]
        x[part] = np.ascontiguousarray(np.column_stack(components), dtype=np.float32)
    return benchmark.EncodedDataset(
        x=x,
        y=baseline.y,
        task=baseline.task,
        y_mean=baseline.y_mean,
        y_scale=baseline.y_scale,
        view_names=(f"schema_{variant.lower()}", "numeric_ple"),
        view_sizes=(schema_size, ple["train"].shape[1]),
        selected_numeric=(),
    )


def fixed_identity_encoded(
    baseline: benchmark.EncodedDataset,
    cache: dict[str, object],
    selected: tuple[int, ...],
) -> benchmark.EncodedDataset:
    clean_num = cast(dict[str, np.ndarray], cache["clean_num"])
    identity = benchmark._one_hot(
        {part: values[:, selected] for part, values in clean_num.items()}
    )
    x = {
        part: np.ascontiguousarray(
            np.column_stack([baseline.x[part], identity[part]]), dtype=np.float32
        )
        for part in baseline.x
    }
    return benchmark.EncodedDataset(
        x=x,
        y=baseline.y,
        task=baseline.task,
        y_mean=baseline.y_mean,
        y_scale=baseline.y_scale,
        view_names=(*baseline.view_names, "numeric_identity"),
        view_sizes=(*baseline.view_sizes, identity["train"].shape[1]),
        selected_numeric=selected,
    )


def fit_path(output: Path, family: str, seed: int, variant: str) -> Path:
    return output / "run_records" / f"{family}__seed{seed}__{variant}.json"


def save_fit(path: Path, result: benchmark.TrainOutput, selected: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = path.with_suffix(".npz")
    temporary_prediction = prediction_path.with_suffix(".npz.tmp")
    with temporary_prediction.open("wb") as handle:
        np.savez_compressed(
            handle,
            validation_logit=result.val_prediction.astype(np.float32),
            test_logit=result.test_prediction.astype(np.float32),
        )
    os.replace(temporary_prediction, prediction_path)
    record = {**result.result, "selected_numeric": list(selected)}
    temporary_record = path.with_suffix(".tmp")
    temporary_record.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(temporary_record, path)


def load_fit(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = np.load(path.with_suffix(".npz"))
    return (
        payload["validation_logit"].astype(np.float64),
        payload["test_logit"].astype(np.float64),
        json.loads(path.read_text()),
    )


def metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
        "auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "brier": float(np.mean((probability - target) ** 2)),
    }


def mean_pairwise_distance(predictions: list[np.ndarray]) -> float:
    return float(
        np.mean(
            [
                np.sqrt(np.mean((left - right) ** 2))
                for left, right in itertools.combinations(predictions, 2)
            ]
        )
    )


def mean_pairwise_disagreement(predictions: list[np.ndarray]) -> float:
    return float(
        np.mean(
            [
                np.mean((left >= 0.5) != (right >= 0.5))
                for left, right in itertools.combinations(predictions, 2)
            ]
        )
    )


def paired_test_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    repeats: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(2701)
    by_class = [np.flatnonzero(target == label) for label in (0, 1)]
    accuracy_differences = []
    auc_differences = []
    for _ in range(repeats):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
        )
        sampled_target = target[sample]
        accuracy_differences.append(
            accuracy_score(sampled_target, candidate[sample] >= 0.5)
            - accuracy_score(sampled_target, reference[sample] >= 0.5)
        )
        auc_differences.append(
            roc_auc_score(sampled_target, candidate[sample])
            - roc_auc_score(sampled_target, reference[sample])
        )
    accuracy_interval = np.quantile(accuracy_differences, [0.025, 0.975])
    auc_interval = np.quantile(auc_differences, [0.025, 0.975])
    return {
        "accuracy_difference_pp": 100
        * (
            accuracy_score(target, candidate >= 0.5)
            - accuracy_score(target, reference >= 0.5)
        ),
        "accuracy_ci_low_pp": 100 * accuracy_interval[0],
        "accuracy_ci_high_pp": 100 * accuracy_interval[1],
        "auc_difference_pp": 100
        * (roc_auc_score(target, candidate) - roc_auc_score(target, reference)),
        "auc_ci_low_pp": 100 * auc_interval[0],
        "auc_ci_high_pp": 100 * auc_interval[1],
    }


def train_required(config: dict, output: Path, device: torch.device) -> None:
    dataset = benchmark.load_dataset(ROOT / "data", config["dataset"])
    expected_selected = tuple(config["selected_numeric_expected"])
    for seed in config["seeds"]:
        preprocessing_cache: dict[str, object] = {}
        ple = benchmark.encode_dataset(
            dataset,
            "schema_ple",
            seed,
            config["bins"],
            config["low_cardinality"],
            config["smoothing"],
            config["identity_effect_threshold"],
            preprocessing_cache,
        )
        target_parameters = benchmark._parameter_count(
            benchmark._make_model(
                ple,
                config["width"],
                config["depth"],
                config["dropout"],
                False,
                config["model"],
                config["activation"],
            )
        )
        identity = fixed_identity_encoded(ple, preprocessing_cache, expected_selected)

        jobs: list[tuple[str, str, benchmark.EncodedDataset]] = [
            ("ple", "RAW", ple),
            ("identity", "SELECTED", identity),
        ]
        if seed == 0:
            jobs.extend(
                (
                    "schema",
                    variant,
                    equivalent_schema_encoded(ple, preprocessing_cache, variant),
                )
                for variant in config["schema_variants"]
            )
        for family, variant, encoded in jobs:
            path = fit_path(output, family, seed, variant)
            if path.exists() and path.with_suffix(".npz").exists():
                print(f"skip {path.name}", flush=True)
                continue
            result = benchmark.train_one(
                encoded,
                seed,
                benchmark.BATCH_SIZES[config["dataset"]],
                device,
                config["width"],
                config["depth"],
                config["dropout"],
                config["learning_rate"],
                config["weight_decay"],
                config["max_epochs"],
                config["patience"],
                gated=False,
                gate_entropy_weight=0.0,
                target_parameters=target_parameters if config["match_parameters"] else None,
                model_type=config["model"],
                activation=config["activation"],
            )
            save_fit(path, result, encoded.selected_numeric)
            print(
                f"{path.name}: accuracy={result.result['test_score']:.4f} "
                f"parameters={result.result['parameters']}",
                flush=True,
            )


def analyze(config: dict, output: Path) -> None:
    dataset = benchmark.load_dataset(ROOT / "data", config["dataset"])
    target = dataset.y["test"].astype(np.int64)
    seeds = config["seeds"]
    schemas = config["schema_variants"]

    ple_predictions = [
        sigmoid(load_fit(fit_path(output, "ple", seed, "RAW"))[1]) for seed in seeds
    ]
    identity_predictions = [
        sigmoid(load_fit(fit_path(output, "identity", seed, "SELECTED"))[1])
        for seed in seeds
    ]
    schema_predictions = [
        sigmoid(load_fit(fit_path(output, "schema", 0, variant))[1])
        for variant in schemas
    ]
    systems = {
        "Single PLE": ple_predictions[0],
        "Single PLE + identity": identity_predictions[0],
        "4 PLE seeds": np.mean(ple_predictions, axis=0),
        "4 PLE schemas": np.mean(schema_predictions, axis=0),
        "4 identity seeds": np.mean(identity_predictions, axis=0),
    }
    summary = pd.DataFrame(
        [{"system": name, **metrics(prediction, target)} for name, prediction in systems.items()]
    )
    baseline = summary.set_index("system").loc["Single PLE"]
    summary["accuracy_gain_pp"] = 100 * (summary["accuracy"] - baseline["accuracy"])
    summary["auc_gain_pp"] = 100 * (summary["auc"] - baseline["auc"])
    summary.to_csv(output / "summary.csv", index=False)

    individual_rows = []
    for family, labels, predictions in (
        ("PLE seed", [str(seed) for seed in seeds], ple_predictions),
        ("schema", schemas, schema_predictions),
        ("identity seed", [str(seed) for seed in seeds], identity_predictions),
    ):
        for label, prediction in zip(labels, predictions):
            individual_rows.append(
                {"family": family, "member": label, **metrics(prediction, target)}
            )
    pd.DataFrame(individual_rows).to_csv(output / "members.csv", index=False)

    diversity = pd.DataFrame(
        [
            {
                "family": family,
                "prediction_distance": mean_pairwise_distance(predictions),
                "decision_disagreement": mean_pairwise_disagreement(predictions),
            }
            for family, predictions in (
                ("4 PLE seeds", ple_predictions),
                ("4 PLE schemas", schema_predictions),
                ("4 identity seeds", identity_predictions),
            )
        ]
    )
    diversity.to_csv(output / "diversity.csv", index=False)

    comparisons = pd.DataFrame(
        [
            {
                "comparison": label,
                **paired_test_bootstrap(systems[candidate], systems[reference], target),
            }
            for label, candidate, reference in (
                ("Single identity vs single PLE", "Single PLE + identity", "Single PLE"),
                ("Schema ensemble vs PLE-seed ensemble", "4 PLE schemas", "4 PLE seeds"),
                ("Identity-seed ensemble vs PLE-seed ensemble", "4 identity seeds", "4 PLE seeds"),
            )
        ]
    )
    comparisons.to_csv(output / "comparisons.csv", index=False)

    colors = ["#a7a7a7", "#57a773", "#4c78a8", "#e07b39", "#6f5aa8"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    x = np.arange(len(summary))
    for axis, column, title in (
        (axes[0], "accuracy_gain_pp", "Accuracy gain over one PLE model"),
        (axes[1], "auc_gain_pp", "AUC gain over one PLE model"),
    ):
        axis.bar(x, summary[column], color=colors)
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_xticks(x, summary["system"], rotation=18, ha="right")
        axis.set_ylabel("Percentage points")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "adult_identity_schema_matched.png", dpi=180, facecolor="white")
    plt.close(fig)

    lines = [
        "# Adult matched identity-versus-diversity experiment",
        "",
        "All systems use the Day 1 TabPack split, 128-bin PLE, the tuned two-block GELU ResNet, and parameter matching. Four-member ensembles cost four fits.",
        "",
        "| System | Accuracy | AUC | Log loss | Accuracy gain vs single PLE | AUC gain vs single PLE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['system']} | {row['accuracy']:.4f} | {row['auc']:.4f} | "
            f"{row['log_loss']:.4f} | {row['accuracy_gain_pp']:+.4f} pp | "
            f"{row['auc_gain_pp']:+.4f} pp |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A single selected-identity model beats both four-fit PLE ensembles. Four schema views recover a small accuracy gain over one PLE model, but do not improve AUC as much as four ordinary seeds and remain far behind selected identity. The four-identity-seed ensemble is best, showing that representation improvement and ordinary ensemble improvement are additive rather than alternative explanations.",
            "",
            "## Paired test-row bootstrap",
            "",
            "Intervals below quantify uncertainty from the fixed test rows; they do not include training-protocol or dataset uncertainty.",
            "",
            "| Comparison | Accuracy difference (95% CI) | AUC difference (95% CI) |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['accuracy_difference_pp']:+.4f} "
            f"[{row['accuracy_ci_low_pp']:+.4f}, {row['accuracy_ci_high_pp']:+.4f}] pp | "
            f"{row['auc_difference_pp']:+.4f} [{row['auc_ci_low_pp']:+.4f}, "
            f"{row['auc_ci_high_pp']:+.4f}] pp |"
        )
    lines.extend(
        [
            "",
            "## Ensemble diversity",
            "",
            "| Family | Prediction distance | Decision disagreement |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in diversity.iterrows():
        lines.append(
            f"| {row['family']} | {row['prediction_distance']:.4f} | "
            f"{100 * row['decision_disagreement']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The schema ensemble keeps PLE fixed and replaces only the normalized numerical coordinates with PAD, DUPLICATE, POSNEG, or SIGNMAG. The identity model adds exact-value one-hot coordinates for the frozen Day 1 columns 3, 4, and 5.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "adult_identity_schema_matched.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "adult_identity_schema_matched"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        train_required(config, output, torch.device(args.device))
    analyze(config, output)


if __name__ == "__main__":
    main()
