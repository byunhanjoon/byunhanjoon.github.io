"""Label-free subgroup stratification of the Adult chart orbit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DAY3_ROOT = REPOSITORY / "experiments" / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3_ROOT))

from experiments.day3.core import load_dataset  # noqa: E402


def summarize(values: np.ndarray, groups: np.ndarray, minimum: int = 100) -> list[dict[str, object]]:
    rows = []
    for level in np.unique(groups.astype(str)):
        selected = groups.astype(str) == level
        if selected.sum() < minimum:
            continue
        rows.append(
            {
                "level": level,
                "rows": int(selected.sum()),
                "mean": float(values[selected].mean()),
                "p95": float(np.quantile(values[selected], 0.95)),
            }
        )
    return rows


def bootstrap_binary_difference(
    values: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> dict[str, object]:
    levels = np.unique(groups)
    if len(levels) != 2:
        raise ValueError("Binary grouping required")
    samples = [values[groups == level] for level in levels]
    differences = []
    for _ in range(repetitions):
        means = [
            rng.choice(sample, len(sample), replace=True).mean()
            for sample in samples
        ]
        differences.append(means[0] - means[1])
    point = float(samples[0].mean() - samples[1].mean())
    return {
        "levels": levels.astype(str).tolist(),
        "means": [float(sample.mean()) for sample in samples],
        "level_0_minus_level_1": point,
        "bootstrap_95_interval": np.quantile(
            differences, (0.025, 0.975)
        ).tolist(),
        "mean_ratio_level_0_over_level_1": float(
            samples[0].mean() / samples[1].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "chart_orbit_adult_s32.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "chart_subgroup_adult.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5_000)
    args = parser.parse_args()

    predictions = np.load(args.input)["mlp_predictions"].astype(np.float64)
    dataset = load_dataset("adult")
    seed_means = predictions.mean(axis=1)
    persistent_row_risk = np.mean(
        np.sum((seed_means - seed_means.mean(axis=0)) ** 2, axis=-1), axis=0
    )
    same_seed_conditional_row_risk = np.mean(
        np.sum(
            (predictions - predictions.mean(axis=0, keepdims=True)) ** 2,
            axis=-1,
        ),
        axis=(0, 1),
    )
    flat_hard = np.argmax(
        predictions.reshape((-1,) + predictions.shape[-2:]), axis=-1
    )
    hard_flip = (np.ptp(flat_hard, axis=0) > 0).astype(np.float64)

    binary = dataset.x_bin["test"][:, 0]
    race = dataset.x_cat["test"][:, 5].astype(str)
    education = dataset.x_cat["test"][:, 1].astype(str)
    rng = np.random.default_rng(260_826)
    metrics = {
        "persistent_mean_predictor_chart_risk": persistent_row_risk,
        "same_seed_conditional_chart_risk": same_seed_conditional_row_risk,
        "joint_hard_label_flip": hard_flip,
    }
    output = {
        "design": {
            "dataset": "adult",
            "warning": "the archive anonymizes the binary level mapping; instability is not an error or fairness-harm estimate",
        },
        "binary_feature": {
            name: {
                "groups": summarize(values, binary),
                "difference": bootstrap_binary_difference(
                    values, binary, rng, args.bootstrap
                ),
            }
            for name, values in metrics.items()
        },
        "race": {
            name: summarize(values, race) for name, values in metrics.items()
        },
        "education": {
            name: summarize(values, education) for name, values in metrics.items()
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
