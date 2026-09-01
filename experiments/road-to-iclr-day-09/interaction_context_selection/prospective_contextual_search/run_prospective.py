from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.core import DATASETS, TabICLEvaluator, indices_string, load_dataset, sample_context  # noqa: E402
from src.selectors import topk  # noqa: E402


K = 32
N_ROUNDS = 4
EXPLORE_PER_ROUND = 32
GUIDED_PER_ROUND = 32
SEEDS = (4101, 4102, 4103)
METHODS = ("random_search", "row_active", "contextual_active", "additive_static", "fm_static", "vip_style")
DEVELOPMENT = {"credit-g", "diamonds"}
CONFIRMATION = set(DATASETS) - DEVELOPMENT
RAW = HERE / "raw"
PROCESSED = HERE / "processed"


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def safe_cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return numerator / denominator


def summarize_distances(values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            values.min(axis=1),
            np.quantile(values, 0.25, axis=1),
            np.median(values, axis=1),
            values.mean(axis=1),
            values.max(axis=1),
        ]
    )


def entropy_rows(counts: np.ndarray) -> np.ndarray:
    probabilities = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return -np.sum(np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-12)), 0), axis=1)


def action_features(bundle: Any, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized features for every valid one-swap action from `current`."""
    z = np.asarray(bundle.feature_z, dtype=float)
    query_z = np.asarray(bundle.selector_feature_z, dtype=float)
    bins = np.asarray(bundle.target_bins, dtype=int)
    selected = set(map(int, current))
    available = np.asarray(sorted(set(range(len(z))) - selected), dtype=int)

    valid_old: list[int] = []
    for old in current:
        if bundle.task == "classification" and np.sum(bundle.y_candidate[current] == bundle.y_candidate[old]) <= 1:
            continue
        valid_old.append(int(old))
    old = np.repeat(np.asarray(valid_old, dtype=int), len(available))
    new = np.tile(available, len(valid_old))
    actions = np.column_stack([old, new])

    out_z, in_z = z[old], z[new]
    row = np.column_stack(
        [
            out_z,
            in_z,
            in_z - out_z,
            np.abs(in_z - out_z),
            in_z * out_z,
            np.linalg.norm(in_z - out_z, axis=1),
            safe_cosine_rows(in_z, out_z),
            (bins[old] == bins[new]).astype(float),
            (bins[new] - bins[old]).astype(float),
        ]
    )

    set_z = z[current]
    set_mean = set_z.mean(axis=0)
    set_std = set_z.std(axis=0)
    query_mean = query_z.mean(axis=0)
    query_std = query_z.std(axis=0)
    n_actions = len(actions)
    context = np.tile(np.concatenate([set_mean, set_std, query_mean, query_std]), (n_actions, 1))
    context = np.column_stack(
        [
            context,
            np.linalg.norm(out_z - set_mean, axis=1),
            np.linalg.norm(in_z - set_mean, axis=1),
            safe_cosine_rows(out_z, np.tile(set_mean, (n_actions, 1))),
            safe_cosine_rows(in_z, np.tile(set_mean, (n_actions, 1))),
            np.linalg.norm(out_z - query_mean, axis=1),
            np.linalg.norm(in_z - query_mean, axis=1),
            safe_cosine_rows(out_z, np.tile(query_mean, (n_actions, 1))),
            safe_cosine_rows(in_z, np.tile(query_mean, (n_actions, 1))),
        ]
    )

    distance_to_selected = cdist(z, z[current])
    old_position = {int(value): position for position, value in enumerate(current)}
    out_summaries = np.empty((n_actions, 5), dtype=float)
    in_summaries = np.empty((n_actions, 5), dtype=float)
    for old_value in valid_old:
        mask = old == old_value
        keep = np.ones(len(current), dtype=bool)
        keep[old_position[old_value]] = False
        out_distances = distance_to_selected[[old_value]][:, keep]
        out_summaries[mask] = np.repeat(summarize_distances(out_distances), mask.sum(), axis=0)
        in_summaries[mask] = summarize_distances(distance_to_selected[new[mask]][:, keep])
    context = np.column_stack([context, out_summaries, in_summaries])

    n_bins = int(bins.max()) + 1
    counts_before = np.bincount(bins[current], minlength=n_bins).astype(float)
    counts_matrix = np.tile(counts_before, (n_actions, 1))
    counts_after = counts_matrix.copy()
    counts_after[np.arange(n_actions), bins[old]] -= 1
    counts_after[np.arange(n_actions), bins[new]] += 1
    entropy_before = entropy_rows(counts_matrix)
    entropy_after = entropy_rows(counts_after)
    context = np.column_stack(
        [
            context,
            counts_before[bins[old]],
            counts_before[bins[new]],
            entropy_before,
            entropy_after,
            entropy_after - entropy_before,
        ]
    )
    return actions, row.astype(np.float32), context.astype(np.float32)


def proposal(current: np.ndarray, action: np.ndarray) -> np.ndarray:
    old, new = map(int, action)
    return np.sort(np.asarray(list((set(map(int, current)) - {old}) | {new}), dtype=int))


class Experiment:
    def __init__(self, dataset: str, device: str):
        self.dataset = dataset
        self.bundle = load_dataset(dataset)
        self.evaluator = TabICLEvaluator(self.bundle, device, seed=0, n_estimators=1)
        self.model_data = np.load(ROOT / "results" / "processed" / "selector_models" / f"{dataset}_k32.npz")
        self.result_path = PROCESSED / f"{dataset}.csv"
        self.call_path = RAW / f"{dataset}_calls.csv"
        self.results = pd.read_csv(self.result_path).to_dict("records") if self.result_path.exists() else []
        self.calls = pd.read_csv(self.call_path).to_dict("records") if self.call_path.exists() else []
        self.selector_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.test_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        for row in self.calls:
            key = tuple(int(value) for value in str(row["indices"]).split(";") if value)
            record = {
                metric: float(row[metric])
                for metric in ("utility", "logloss", "accuracy", "roc_auc", "rmse", "mae", "runtime_seconds")
                if metric in row and pd.notna(row[metric])
            }
            (self.selector_cache if row["split"] == "selector" else self.test_cache)[key] = record

    def persist(self) -> None:
        atomic_csv(pd.DataFrame(self.results), self.result_path)
        atomic_csv(pd.DataFrame(self.calls), self.call_path)

    def evaluate(
        self,
        indices: np.ndarray,
        split: str,
        method: str,
        start_seed: int,
        round_id: int,
        phase: str,
    ) -> float:
        key = tuple(sorted(map(int, indices)))
        cache = self.selector_cache if split == "selector" else self.test_cache
        hit = key in cache
        if not hit:
            cache[key] = self.evaluator.evaluate(np.asarray(key), split, return_prediction=False)
        record = cache[key]
        self.calls.append(
            {
                "dataset": self.dataset,
                "data_role": "development" if self.dataset in DEVELOPMENT else "confirmation",
                "method": method,
                "start_seed": start_seed,
                "round": round_id,
                "phase": phase,
                "split": split,
                "indices": indices_string(key),
                "context_size": len(key),
                "cache_hit": hit,
                **record,
            }
        )
        return float(record["utility"])

    def start(self, start_seed: int) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.dataset, "prospective-start", start_seed))
        return sample_context(self.bundle.y_candidate, K, rng, self.bundle.task)

    def finish(
        self,
        method: str,
        start_seed: int,
        initial: np.ndarray,
        final: np.ndarray,
        start_selector: float,
        final_selector: float,
        selector_calls: int,
        trace: list[dict[str, Any]],
    ) -> None:
        start_test = self.evaluate(initial, "test", method, start_seed, -1, "post_selection_start_test")
        final_test = self.evaluate(final, "test", method, start_seed, -1, "post_selection_final_test")
        self.results.append(
            {
                "dataset": self.dataset,
                "data_role": "development" if self.dataset in DEVELOPMENT else "confirmation",
                "method": method,
                "start_seed": start_seed,
                "selector_calls": selector_calls,
                "start_selector_utility": start_selector,
                "final_selector_utility": final_selector,
                "selector_improvement": final_selector - start_selector,
                "start_test_utility": start_test,
                "final_test_utility": final_test,
                "test_improvement": final_test - start_test,
                "initial_indices": indices_string(initial),
                "final_indices": indices_string(final),
                "trace_json": json.dumps(trace, separators=(",", ":")),
            }
        )
        self.persist()

    def run_random(self, start_seed: int) -> None:
        method = "random_search"
        initial = self.start(start_seed)
        current = initial.copy()
        current_utility = self.evaluate(current, "selector", method, start_seed, 0, "start")
        start_utility = current_utility
        selector_calls = 1
        rng = np.random.default_rng(stable_seed(self.dataset, method, start_seed))
        seen_by_state: dict[tuple[int, ...], set[tuple[int, int]]] = {}
        gains_by_state: dict[tuple[int, ...], dict[tuple[int, int], float]] = {}
        trace: list[dict[str, Any]] = []
        for round_id in range(1, N_ROUNDS + 1):
            actions, _, _ = action_features(self.bundle, current)
            state = tuple(map(int, current))
            seen = seen_by_state.setdefault(state, set())
            gains = gains_by_state.setdefault(state, {})
            candidates = np.asarray([i for i, action in enumerate(actions) if tuple(map(int, action)) not in seen])
            chosen = rng.choice(candidates, size=min(EXPLORE_PER_ROUND + GUIDED_PER_ROUND, len(candidates)), replace=False)
            for index in chosen:
                action = tuple(map(int, actions[index]))
                value = self.evaluate(proposal(current, actions[index]), "selector", method, start_seed, round_id, "random")
                selector_calls += 1
                seen.add(action)
                gains[action] = value - current_utility
            best_action, best_gain = max(gains.items(), key=lambda item: item[1])
            moved = best_gain > 1e-12
            if moved:
                current = proposal(current, np.asarray(best_action))
                current_utility += best_gain
            trace.append({"round": round_id, "best_gain": best_gain, "moved": moved, "indices": indices_string(current)})
        self.finish(method, start_seed, initial, current, start_utility, current_utility, selector_calls, trace)

    def run_active(self, start_seed: int, contextual: bool) -> None:
        method = "contextual_active" if contextual else "row_active"
        initial = self.start(start_seed)
        current = initial.copy()
        current_utility = self.evaluate(current, "selector", method, start_seed, 0, "start")
        start_utility = current_utility
        selector_calls = 1
        rng = np.random.default_rng(stable_seed(self.dataset, method, start_seed))
        observed_x: list[np.ndarray] = []
        observed_y: list[float] = []
        seen_by_state: dict[tuple[int, ...], set[tuple[int, int]]] = {}
        gains_by_state: dict[tuple[int, ...], dict[tuple[int, int], float]] = {}
        trace: list[dict[str, Any]] = []
        for round_id in range(1, N_ROUNDS + 1):
            actions, row_x, context_x = action_features(self.bundle, current)
            features = np.column_stack([row_x, context_x]) if contextual else row_x
            state = tuple(map(int, current))
            seen = seen_by_state.setdefault(state, set())
            gains = gains_by_state.setdefault(state, {})
            available = np.asarray([i for i, action in enumerate(actions) if tuple(map(int, action)) not in seen])
            explore = rng.choice(available, size=min(EXPLORE_PER_ROUND, len(available)), replace=False)
            for index in explore:
                action = tuple(map(int, actions[index]))
                value = self.evaluate(proposal(current, actions[index]), "selector", method, start_seed, round_id, "explore")
                selector_calls += 1
                gain = value - current_utility
                seen.add(action)
                gains[action] = gain
                observed_x.append(features[index].copy())
                observed_y.append(gain)

            model = ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=0.7,
                n_jobs=8,
                random_state=stable_seed(self.dataset, method, start_seed, round_id),
            )
            train_x = np.asarray(observed_x)
            train_y = np.asarray(observed_y)
            model.fit(train_x, train_y)
            remaining = np.asarray([i for i, action in enumerate(actions) if tuple(map(int, action)) not in seen])
            tree_predictions = np.asarray([tree.predict(features[remaining]) for tree in model.estimators_])
            acquisition = tree_predictions.mean(axis=0) + 0.5 * tree_predictions.std(axis=0)
            order = remaining[np.argsort(-acquisition)]
            guided: list[int] = []
            per_outgoing: dict[int, int] = {}
            for index in order:
                outgoing = int(actions[index, 0])
                if per_outgoing.get(outgoing, 0) >= 2:
                    continue
                guided.append(int(index))
                per_outgoing[outgoing] = per_outgoing.get(outgoing, 0) + 1
                if len(guided) >= GUIDED_PER_ROUND:
                    break
            for index in guided:
                action = tuple(map(int, actions[index]))
                value = self.evaluate(proposal(current, actions[index]), "selector", method, start_seed, round_id, "guided")
                selector_calls += 1
                gain = value - current_utility
                seen.add(action)
                gains[action] = gain
                observed_x.append(features[index].copy())
                observed_y.append(gain)

            best_action, best_gain = max(gains.items(), key=lambda item: item[1])
            moved = best_gain > 1e-12
            if moved:
                current = proposal(current, np.asarray(best_action))
                current_utility += best_gain
            trace.append(
                {
                    "round": round_id,
                    "best_gain": best_gain,
                    "moved": moved,
                    "train_rows": len(observed_y),
                    "indices": indices_string(current),
                }
            )
        self.finish(method, start_seed, initial, current, start_utility, current_utility, selector_calls, trace)

    def static_delta(self, current: np.ndarray, actions: np.ndarray, method: str) -> np.ndarray:
        old, new = actions[:, 0], actions[:, 1]
        if method == "additive_static":
            additive = np.asarray(self.model_data["additive"])
            return additive[new] - additive[old]
        additive = np.asarray(self.model_data["fm_additive"])
        pair = np.asarray(self.model_data["fm_pair"])
        values = additive[new] - additive[old]
        for position, (outgoing, incoming) in enumerate(actions):
            rest = current[current != outgoing]
            values[position] += pair[incoming, rest].sum() - pair[outgoing, rest].sum()
        return values

    def run_static(self, start_seed: int, method: str) -> None:
        initial = self.start(start_seed)
        current = initial.copy()
        current_utility = self.evaluate(current, "selector", method, start_seed, 0, "start")
        start_utility = current_utility
        selector_calls = 1
        trace: list[dict[str, Any]] = []
        for round_id in range(1, N_ROUNDS + 1):
            actions, _, _ = action_features(self.bundle, current)
            selected_action = actions[int(np.argmax(self.static_delta(current, actions, method)))]
            value = self.evaluate(proposal(current, selected_action), "selector", method, start_seed, round_id, "predicted_best")
            selector_calls += 1
            gain = value - current_utility
            moved = gain > 1e-12
            if moved:
                current = proposal(current, selected_action)
                current_utility = value
            trace.append({"round": round_id, "gain": gain, "moved": moved, "indices": indices_string(current)})
        self.finish(method, start_seed, initial, current, start_utility, current_utility, selector_calls, trace)

    def weighted_context(self, probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.bundle.task != "classification":
            return np.sort(rng.choice(len(probabilities), K, replace=False, p=probabilities))
        chosen: list[int] = []
        for cls in np.unique(self.bundle.y_candidate):
            candidates = np.flatnonzero(self.bundle.y_candidate == cls)
            local = probabilities[candidates] / probabilities[candidates].sum()
            chosen.append(int(rng.choice(candidates, p=local)))
        remaining = np.setdiff1d(np.arange(len(probabilities)), np.asarray(chosen))
        local = probabilities[remaining] / probabilities[remaining].sum()
        chosen.extend(map(int, rng.choice(remaining, K - len(chosen), replace=False, p=local)))
        return np.sort(np.asarray(chosen, dtype=int))

    def run_vip_style(self, start_seed: int) -> None:
        method = "vip_style"
        initial = self.start(start_seed)
        start_utility = self.evaluate(initial, "selector", method, start_seed, 0, "start")
        selector_calls = 1
        rng = np.random.default_rng(stable_seed(self.dataset, method, start_seed))
        memberships: list[np.ndarray] = []
        utilities: list[float] = []
        coefficients = np.zeros(256, dtype=float)
        seen = {tuple(map(int, initial))}
        trace: list[dict[str, Any]] = []
        for round_id, temperature in enumerate((2.0, 1.0, 0.5, 0.25), start=1):
            standardized = (coefficients - coefficients.mean()) / max(coefficients.std(), 1e-8)
            logits = np.clip(standardized / temperature, -10, 10)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            generated = 0
            while generated < 63:
                context = self.weighted_context(probabilities, rng)
                key = tuple(map(int, context))
                if key in seen:
                    continue
                seen.add(key)
                value = self.evaluate(context, "selector", method, start_seed, round_id, "value_guided_subset")
                selector_calls += 1
                membership = np.zeros(256, dtype=np.float32)
                membership[context] = 1
                memberships.append(membership)
                utilities.append(value)
                generated += 1
            regression = Ridge(alpha=10.0, fit_intercept=True)
            regression.fit(np.asarray(memberships), np.asarray(utilities))
            coefficients = np.asarray(regression.coef_)
            trace.append({"round": round_id, "contexts": len(utilities), "temperature": temperature})
        final = topk(coefficients, self.bundle.y_candidate, K, self.bundle.task == "classification")
        final_utility = self.evaluate(final, "selector", method, start_seed, N_ROUNDS + 1, "final_topk")
        selector_calls += 1
        self.finish(method, start_seed, initial, final, start_utility, final_utility, selector_calls, trace)

    def run(self) -> None:
        completed = {(str(row["method"]), int(row["start_seed"])) for row in self.results}
        for start_seed in SEEDS:
            for method in METHODS:
                if (method, start_seed) in completed:
                    print(f"{self.dataset} seed={start_seed} {method}: already complete", flush=True)
                    continue
                started = time.perf_counter()
                if method == "random_search":
                    self.run_random(start_seed)
                elif method == "row_active":
                    self.run_active(start_seed, contextual=False)
                elif method == "contextual_active":
                    self.run_active(start_seed, contextual=True)
                elif method in {"additive_static", "fm_static"}:
                    self.run_static(start_seed, method)
                elif method == "vip_style":
                    self.run_vip_style(start_seed)
                else:
                    raise KeyError(method)
                print(f"{self.dataset} seed={start_seed} {method}: {time.perf_counter() - started:.1f}s", flush=True)


def bootstrap_interval(values: np.ndarray, seed: int = 0, repetitions: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(repetitions)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report() -> None:
    paths = [PROCESSED / f"{dataset}.csv" for dataset in sorted(DATASETS) if (PROCESSED / f"{dataset}.csv").exists()]
    if not paths:
        raise FileNotFoundError("No prospective result files")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    atomic_csv(results, HERE / "results.csv")
    summary = (
        results.groupby(["data_role", "method"])[
            ["selector_calls", "selector_improvement", "test_improvement"]
        ]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    atomic_csv(summary, HERE / "summary.csv")

    confirmation = results[results.data_role == "confirmation"]
    pivot = confirmation.pivot(index=["dataset", "start_seed"], columns="method", values="test_improvement")
    contextual_positive = float((pivot.contextual_active > 0).mean())
    contextual_beats_row = float((pivot.contextual_active > pivot.row_active).mean())
    contextual_minus_random = (pivot.contextual_active - pivot.random_search).to_numpy()
    random_lower, random_upper = bootstrap_interval(contextual_minus_random, stable_seed("prospective-bootstrap-random"))
    contextual_minus_row = (pivot.contextual_active - pivot.row_active).to_numpy()
    row_lower, row_upper = bootstrap_interval(contextual_minus_row, stable_seed("prospective-bootstrap-row"))
    selector_pivot = confirmation.pivot(
        index=["dataset", "start_seed"], columns="method", values="selector_improvement"
    )
    selector_minus_row = (selector_pivot.contextual_active - selector_pivot.row_active).to_numpy()
    selector_lower, selector_upper = bootstrap_interval(
        selector_minus_row, stable_seed("prospective-bootstrap-selector-row")
    )
    audit = {
        "datasets_complete": sorted(results.dataset.unique()),
        "expected_datasets": sorted(DATASETS),
        "rows": len(results),
        "expected_rows": len(DATASETS) * len(SEEDS) * len(METHODS),
        "confirmation_trajectories": len(pivot),
        "contextual_positive_test_fraction": contextual_positive,
        "contextual_beats_row_test_fraction": contextual_beats_row,
        "contextual_minus_random_mean_test_improvement": float(contextual_minus_random.mean()),
        "contextual_minus_random_bootstrap_95": [random_lower, random_upper],
        "contextual_minus_row_mean_test_improvement": float(contextual_minus_row.mean()),
        "contextual_minus_row_bootstrap_95": [row_lower, row_upper],
        "contextual_minus_row_mean_selector_improvement": float(selector_minus_row.mean()),
        "contextual_minus_row_selector_bootstrap_95": [selector_lower, selector_upper],
        "gate_1_oracle_regret": "not_tested_in_end_to_end_run",
        "gate_2_positive_test_at_least_70pct": contextual_positive >= 0.70,
        "gate_3_beats_row_at_least_70pct": contextual_beats_row >= 0.70,
        "gate_4_beats_random_positive_bootstrap_interval": random_lower > 0,
        "test_labels_used_for_selection": False,
        "vip_style_is_official_vip_cop": False,
    }
    atomic_json(audit, HERE / "audit.json")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    run_parser.add_argument("--device", default="cuda:0")
    subparsers.add_parser("report")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        Experiment(args.dataset, args.device).run()
    else:
        report()


if __name__ == "__main__":
    main()
