"""Ground-truth recovery, permutation, and sample-efficiency experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY1))

import real_data_benchmark as benchmark  # noqa: E402
from hierarchical_residual import (  # noqa: E402
    DiscoveryConfig,
    Selection,
    discover,
    state_keys,
)


SCENARIOS = ("smooth", "additive", "interaction", "mixed")


def generate(
    rows: int, scenario: str, seed: int
) -> tuple[np.ndarray, np.ndarray, Selection]:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, 16, size=rows)
    second = rng.integers(0, 12, size=rows)
    nuisance_a = rng.integers(0, 10, size=rows)
    nuisance_b = rng.integers(0, 8, size=rows)
    continuous = rng.normal(size=rows)
    numeric = np.column_stack(
        (first, second, nuisance_a, nuisance_b, continuous)
    ).astype(np.float64)
    target = (
        0.07 * first
        - 0.04 * second
        + 0.35 * np.sin(first / 3.0)
        + 0.5 * continuous
        + 0.15 * continuous**2
    )
    singletons: tuple[int, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()
    if scenario in ("additive", "mixed"):
        target += 1.1 * (first == 3) - 0.9 * (first == 11)
        target += 0.8 * (second == 7)
        singletons = (0, 1)
    if scenario in ("interaction", "mixed"):
        # Product of centered indicators has zero marginal effect in expectation.
        left = (first == 5).astype(np.float64) - 1.0 / 16.0
        right = (second == 9).astype(np.float64) - 1.0 / 12.0
        target += 7.0 * left * right
        pairs = ((0, 1),)
    target += rng.normal(scale=0.35, size=rows)
    return numeric, target, Selection(singletons, pairs)


def ple_design(
    train: np.ndarray, test: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    parts = {"train": train, "val": test, "test": test}
    ple = benchmark._piecewise_linear(parts, bins)
    scaler = StandardScaler().fit(train)
    return (
        np.column_stack((scaler.transform(train), ple["train"])),
        np.column_stack((scaler.transform(test), ple["test"])),
    )


def encode_atomic(
    train: np.ndarray,
    test: np.ndarray,
    selection: Selection,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_groups: list[np.ndarray] = []
    test_groups: list[np.ndarray] = []
    groups = [(column,) for column in selection.singletons] + list(selection.pairs)
    for group in groups:
        train_key = state_keys(train[:, group])
        test_key = state_keys(test[:, group])
        keys, inverse, counts = np.unique(
            train_key, return_inverse=True, return_counts=True
        )
        weights = counts / (counts + smoothing)
        train_encoded = np.zeros((len(train), len(keys)), dtype=np.float64)
        train_encoded[np.arange(len(train)), inverse] = weights[inverse]
        positions = np.searchsorted(keys, test_key)
        valid = positions < len(keys)
        matched = np.zeros(len(test), dtype=bool)
        matched[valid] = keys[positions[valid]] == test_key[valid]
        test_encoded = np.zeros((len(test), len(keys)), dtype=np.float64)
        test_encoded[np.flatnonzero(matched), positions[matched]] = weights[
            positions[matched]
        ]
        train_groups.append(train_encoded)
        test_groups.append(test_encoded)
    if not groups:
        return np.empty((len(train), 0)), np.empty((len(test), 0))
    return np.column_stack(train_groups), np.column_stack(test_groups)


def fit_mse(
    train_design: np.ndarray,
    train_target: np.ndarray,
    test_design: np.ndarray,
    test_target: np.ndarray,
) -> float:
    model = Ridge(alpha=10.0).fit(train_design, train_target)
    prediction = model.predict(test_design)
    return float(np.mean((prediction - test_target) ** 2))


def term_metrics(selected: Selection, truth: Selection) -> dict[str, float]:
    selected_singletons = set(selected.singletons)
    true_singletons = set(truth.singletons)
    selected_pairs = set(selected.pairs)
    true_pairs = set(truth.pairs)

    def precision(selected_set: set, true_set: set) -> float:
        return (
            len(selected_set & true_set) / len(selected_set)
            if selected_set
            else float(not true_set)
        )

    def recall(selected_set: set, true_set: set) -> float:
        return (
            len(selected_set & true_set) / len(true_set)
            if true_set
            else float(not selected_set)
        )

    return {
        "singleton_precision": precision(selected_singletons, true_singletons),
        "singleton_recall": recall(selected_singletons, true_singletons),
        "pair_precision": precision(selected_pairs, true_pairs),
        "pair_recall": recall(selected_pairs, true_pairs),
    }


def run_once(
    train_rows: int,
    scenario: str,
    seed: int,
    bins: int,
    config: DiscoveryConfig,
    permute_codes: bool,
) -> dict[str, object]:
    numeric, target, truth = generate(train_rows + 10_000, scenario, seed)
    train, test = numeric[:train_rows].copy(), numeric[train_rows:].copy()
    train_target, test_target = target[:train_rows], target[train_rows:]
    if permute_codes:
        rng = np.random.default_rng(seed + 100_000)
        for column in (0, 1):
            values = np.unique(train[:, column])
            permutation = rng.permutation(values)
            mapping = dict(zip(values.tolist(), permutation.tolist()))
            train[:, column] = np.array([mapping[value] for value in train[:, column]])
            test[:, column] = np.array([mapping[value] for value in test[:, column]])
    train_design, test_design = ple_design(train, test, bins)
    selection, _, _ = discover(
        train_design,
        train,
        train_target,
        "regression",
        seed,
        config,
    )
    selected_train, selected_test = encode_atomic(
        train, test, selection, config.smoothing
    )
    oracle_train, oracle_test = encode_atomic(
        train, test, truth, config.smoothing
    )
    baseline_mse = fit_mse(
        train_design, train_target, test_design, test_target
    )
    selected_mse = fit_mse(
        np.column_stack((train_design, selected_train)),
        train_target,
        np.column_stack((test_design, selected_test)),
        test_target,
    )
    oracle_mse = fit_mse(
        np.column_stack((train_design, oracle_train)),
        train_target,
        np.column_stack((test_design, oracle_test)),
        test_target,
    )
    return {
        "scenario": scenario,
        "train_rows": train_rows,
        "seed": seed,
        "permuted_codes": int(permute_codes),
        "truth_singletons": ";".join(map(str, truth.singletons)),
        "truth_pairs": ";".join(f"{a}+{b}" for a, b in truth.pairs),
        "selected_singletons": ";".join(map(str, selection.singletons)),
        "selected_pairs": ";".join(f"{a}+{b}" for a, b in selection.pairs),
        **term_metrics(selection, truth),
        "baseline_mse": baseline_mse,
        "selected_mse": selected_mse,
        "oracle_mse": oracle_mse,
        "selected_gain": (baseline_mse - selected_mse) / baseline_mse,
        "oracle_gain": (baseline_mse - oracle_mse) / baseline_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[500, 1_000, 3_000, 10_000])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "synthetic_recovery.csv"
    )
    args = parser.parse_args()
    config = DiscoveryConfig(
        folds=5,
        smoothing=20.0,
        max_cardinality=128,
        max_pair_cardinality=512,
        minimum_relative_gain=5e-4,
        minimum_fold_wins=5,
        minimum_pair_fold_wins=5,
        maximum_singletons=4,
        maximum_pairs=4,
        representation_budget=512,
    )
    rows: list[dict[str, object]] = []
    for train_rows in args.sample_sizes:
        for scenario in args.scenarios:
            for seed in range(args.repetitions):
                for permuted in (False, True):
                    row = run_once(
                        train_rows,
                        scenario,
                        seed,
                        args.bins,
                        config,
                        permuted,
                    )
                    rows.append(row)
            subset = [
                row
                for row in rows
                if row["train_rows"] == train_rows
                and row["scenario"] == scenario
                and not row["permuted_codes"]
            ]
            print(
                f"n={train_rows:<6} {scenario:<11} "
                f"gain={100 * np.mean([float(row['selected_gain']) for row in subset]):+.2f}% "
                f"singleton_recall={np.mean([float(row['singleton_recall']) for row in subset]):.2f} "
                f"pair_recall={np.mean([float(row['pair_recall']) for row in subset]):.2f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
