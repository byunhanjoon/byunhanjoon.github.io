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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT / "prospective_contextual_search"))

from src.core import (  # noqa: E402
    DatasetBundle,
    TabICLEvaluator,
    _make_row_representations,
    _regression_bins,
    _split_indices,
    indices_string,
    prediction_metrics,
    sample_context,
)
from run_prospective import action_features, proposal  # noqa: E402


DATASETS: dict[str, dict[str, Any]] = {
    "breast-w": {"openml_id": 15, "task": "classification"},
    "credit-approval": {"openml_id": 29, "task": "classification"},
    "blood-transfusion": {"openml_id": 1464, "task": "classification"},
    "sick": {"openml_id": 38, "task": "classification"},
    "kin8nm": {"openml_id": 189, "task": "regression"},
    "puma32h": {"openml_id": 308, "task": "regression"},
    "cpu-act": {"openml_id": 573, "task": "regression"},
    "elevators": {"openml_id": 216, "task": "regression"},
}
METHODS = (
    "random_mean",
    "row_mean",
    "contextual_mean",
    "random_rotating",
    "row_rotating",
    "contextual_rotating",
)
SEEDS = (5201, 5202, 5203)
K = 32
N_ROUNDS = 4
EXPLORE = 32
GUIDED = 32
SPLIT_SEED = 1729
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


def load_new_dataset(name: str) -> DatasetBundle:
    from sklearn.datasets import fetch_openml

    spec = DATASETS[name]
    bunch = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
    X = bunch.data.copy()
    raw_target = pd.Series(np.asarray(bunch.target))
    if spec["task"] == "regression":
        numeric_target = pd.to_numeric(raw_target, errors="coerce")
        keep = numeric_target.notna().to_numpy()
        X = X.loc[keep].reset_index(drop=True)
        y = numeric_target.loc[keep].to_numpy(dtype=np.float64)
        classes = None
    else:
        keep = raw_target.notna().to_numpy()
        X = X.loc[keep].reset_index(drop=True)
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_target.loc[keep].astype(str))
        classes = encoder.classes_
    if len(y) < 640:
        raise ValueError(f"{name} has only {len(y)} usable rows")
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype("object").where(pd.notna(X[column]), "__MISSING__")
    candidate, selector, test = _split_indices(y, spec["task"], SPLIT_SEED)
    z, feature_z, selector_z, test_z, bins = _make_row_representations(
        X, y, spec["task"], candidate, selector, test
    )
    return DatasetBundle(
        name=name,
        task=spec["task"],
        source_id=f"OpenML:{spec['openml_id']} ({bunch.details.get('name', name)})",
        X=X,
        y=y,
        classes=classes,
        candidate_idx=candidate,
        selector_idx=selector,
        test_idx=test,
        z=z,
        feature_z=feature_z,
        selector_feature_z=selector_z,
        test_feature_z=test_z,
        target_bins=bins,
    )


def selector_folds(bundle: DatasetBundle) -> list[np.ndarray]:
    if bundle.task == "classification":
        strata = bundle.y_selector
    else:
        strata = _regression_bins(bundle.y_selector, 8)
    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=stable_seed(bundle.name, "selector-folds", SPLIT_SEED),
    )
    return [test for _, test in splitter.split(np.zeros(len(strata)), strata)]


class Experiment:
    def __init__(self, dataset: str, device: str):
        self.dataset = dataset
        self.bundle = load_new_dataset(dataset)
        self.folds = selector_folds(self.bundle)
        self.evaluator = TabICLEvaluator(self.bundle, device, seed=0, n_estimators=1)
        self.result_path = PROCESSED / f"{dataset}.csv"
        self.call_path = RAW / f"{dataset}_calls.csv"
        self.results = pd.read_csv(self.result_path).to_dict("records") if self.result_path.exists() else []
        self.calls = pd.read_csv(self.call_path).to_dict("records") if self.call_path.exists() else []
        self.selector_cache: dict[tuple[int, ...], dict[str, float]] = {}
        self.test_cache: dict[tuple[int, ...], dict[str, float]] = {}
        for row in self.calls:
            key = tuple(int(value) for value in str(row["indices"]).split(";") if value)
            fields = ["utility", "logloss", "accuracy", "roc_auc", "rmse", "mae", "runtime_seconds"]
            fields.extend(f"fold_{index}_utility" for index in range(4))
            record = {field: float(row[field]) for field in fields if field in row and pd.notna(row[field])}
            (self.selector_cache if row["split"] == "selector" else self.test_cache)[key] = record

    def persist(self) -> None:
        atomic_csv(pd.DataFrame(self.results), self.result_path)
        atomic_csv(pd.DataFrame(self.calls), self.call_path)

    def evaluate_selector(
        self, indices: np.ndarray, method: str, start_seed: int, round_id: int, phase: str
    ) -> dict[str, float]:
        key = tuple(sorted(map(int, indices)))
        hit = key in self.selector_cache
        if not hit:
            output = self.evaluator.evaluate(np.asarray(key), "selector", return_prediction=True)
            prediction = np.asarray(output.pop("prediction"))
            for fold_id, fold in enumerate(self.folds):
                metrics = prediction_metrics(
                    self.bundle.y_selector[fold],
                    prediction[fold],
                    self.bundle.task,
                    float(self.bundle.y_candidate.std()),
                )
                output[f"fold_{fold_id}_utility"] = float(metrics["utility"])
            self.selector_cache[key] = output
        record = self.selector_cache[key]
        self.calls.append(
            {
                "dataset": self.dataset,
                "method": method,
                "start_seed": start_seed,
                "round": round_id,
                "phase": phase,
                "split": "selector",
                "indices": indices_string(key),
                "context_size": len(key),
                "cache_hit": hit,
                **record,
            }
        )
        return record

    def evaluate_test(
        self, indices: np.ndarray, method: str, start_seed: int, phase: str
    ) -> dict[str, float]:
        key = tuple(sorted(map(int, indices)))
        hit = key in self.test_cache
        if not hit:
            self.test_cache[key] = self.evaluator.evaluate(np.asarray(key), "test", return_prediction=False)
        record = self.test_cache[key]
        self.calls.append(
            {
                "dataset": self.dataset,
                "method": method,
                "start_seed": start_seed,
                "round": -1,
                "phase": phase,
                "split": "test",
                "indices": indices_string(key),
                "context_size": len(key),
                "cache_hit": hit,
                **record,
            }
        )
        return record

    @staticmethod
    def fold_vector(record: dict[str, float]) -> np.ndarray:
        return np.asarray([record[f"fold_{index}_utility"] for index in range(4)], dtype=float)

    def initial_context(self, start_seed: int) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.dataset, "successor-start", start_seed))
        return sample_context(self.bundle.y_candidate, K, rng, self.bundle.task)

    def select_score(
        self,
        candidate: dict[str, Any],
        mode: str,
        judge: int,
        scouts: np.ndarray,
    ) -> float:
        if mode == "mean":
            return float(candidate["full_gain"])
        fold_gain = np.asarray(candidate["fold_gain"])
        scout_gain = float(fold_gain[scouts].mean())
        judge_gain = float(fold_gain[judge])
        return min(scout_gain, judge_gain)

    def run_method(self, method: str, start_seed: int) -> None:
        mode = "rotating" if method.endswith("rotating") else "mean"
        family = method.rsplit("_", 1)[0]
        contextual = family == "contextual"
        active = family in {"contextual", "row"}
        initial = self.initial_context(start_seed)
        current = initial.copy()
        current_record = self.evaluate_selector(current, method, start_seed, 0, "start")
        start_record = dict(current_record)
        selector_calls = 1
        rng = np.random.default_rng(stable_seed(self.dataset, method, start_seed))
        observed_x: list[np.ndarray] = []
        observed_full_gain: list[float] = []
        observed_fold_gain: list[np.ndarray] = []
        seen_by_state: dict[tuple[int, ...], set[tuple[int, int]]] = {}
        candidates_by_state: dict[tuple[int, ...], dict[tuple[int, int], dict[str, Any]]] = {}
        trace: list[dict[str, Any]] = []

        for round_id in range(1, N_ROUNDS + 1):
            judge = (round_id - 1) % 4
            scouts = np.asarray([index for index in range(4) if index != judge], dtype=int)
            actions, row_features, context_features = action_features(self.bundle, current)
            features = np.column_stack([row_features, context_features]) if contextual else row_features
            state = tuple(map(int, current))
            seen = seen_by_state.setdefault(state, set())
            state_candidates = candidates_by_state.setdefault(state, {})
            available = np.asarray(
                [index for index, action in enumerate(actions) if tuple(map(int, action)) not in seen], dtype=int
            )
            exploration_count = EXPLORE if active else EXPLORE + GUIDED
            explore = rng.choice(available, size=min(exploration_count, len(available)), replace=False)

            def observe(index: int, phase: str) -> None:
                nonlocal selector_calls
                action = tuple(map(int, actions[index]))
                record = self.evaluate_selector(
                    proposal(current, actions[index]), method, start_seed, round_id, phase
                )
                selector_calls += 1
                full_gain = float(record["utility"] - current_record["utility"])
                fold_gain = self.fold_vector(record) - self.fold_vector(current_record)
                seen.add(action)
                state_candidates[action] = {
                    "full_gain": full_gain,
                    "fold_gain": fold_gain.tolist(),
                    "record": record,
                }
                if active:
                    observed_x.append(features[index].copy())
                    observed_full_gain.append(full_gain)
                    observed_fold_gain.append(fold_gain.copy())

            for index in explore:
                observe(int(index), "explore" if active else "random")

            if active:
                train_x = np.asarray(observed_x)
                if mode == "mean":
                    train_y = np.asarray(observed_full_gain)
                else:
                    train_y = np.asarray(observed_fold_gain)[:, scouts].mean(axis=1)
                model = ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=0.7,
                    n_jobs=8,
                    random_state=stable_seed(self.dataset, method, start_seed, round_id),
                )
                model.fit(train_x, train_y)
                remaining = np.asarray(
                    [index for index, action in enumerate(actions) if tuple(map(int, action)) not in seen], dtype=int
                )
                tree_predictions = np.asarray([tree.predict(features[remaining]) for tree in model.estimators_])
                acquisition = tree_predictions.mean(axis=0) + 0.5 * tree_predictions.std(axis=0)
                order = remaining[np.argsort(-acquisition)]
                guided: list[int] = []
                outgoing_counts: dict[int, int] = {}
                for index in order:
                    outgoing = int(actions[index, 0])
                    if outgoing_counts.get(outgoing, 0) >= 2:
                        continue
                    guided.append(int(index))
                    outgoing_counts[outgoing] = outgoing_counts.get(outgoing, 0) + 1
                    if len(guided) >= GUIDED:
                        break
                for index in guided:
                    observe(index, "guided")

            best_action, best_candidate = max(
                state_candidates.items(),
                key=lambda item: self.select_score(item[1], mode, judge, scouts),
            )
            best_score = self.select_score(best_candidate, mode, judge, scouts)
            moved = best_score > 1e-12
            if moved:
                current = proposal(current, np.asarray(best_action))
                current_record = dict(best_candidate["record"])
            trace.append(
                {
                    "round": round_id,
                    "judge_fold": judge if mode == "rotating" else None,
                    "selection_score": best_score,
                    "selected_full_gain": best_candidate["full_gain"],
                    "selected_fold_gain": best_candidate["fold_gain"],
                    "moved": moved,
                    "indices": indices_string(current),
                }
            )

        start_test = self.evaluate_test(initial, method, start_seed, "post_selection_start_test")
        final_test = self.evaluate_test(current, method, start_seed, "post_selection_final_test")
        self.results.append(
            {
                "dataset": self.dataset,
                "task": self.bundle.task,
                "method": method,
                "start_seed": start_seed,
                "selector_calls": selector_calls,
                "start_selector_utility": start_record["utility"],
                "final_selector_utility": current_record["utility"],
                "selector_improvement": current_record["utility"] - start_record["utility"],
                "start_test_utility": start_test["utility"],
                "final_test_utility": final_test["utility"],
                "test_improvement": final_test["utility"] - start_test["utility"],
                "initial_indices": indices_string(initial),
                "final_indices": indices_string(current),
                "trace_json": json.dumps(trace, separators=(",", ":")),
            }
        )
        self.persist()

    def run(self) -> None:
        completed = {(str(row["method"]), int(row["start_seed"])) for row in self.results}
        for start_seed in SEEDS:
            for method in METHODS:
                if (method, start_seed) in completed:
                    print(f"{self.dataset} seed={start_seed} {method}: already complete", flush=True)
                    continue
                started = time.perf_counter()
                self.run_method(method, start_seed)
                print(f"{self.dataset} seed={start_seed} {method}: {time.perf_counter() - started:.1f}s", flush=True)


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report() -> None:
    paths = [PROCESSED / f"{dataset}.csv" for dataset in sorted(DATASETS) if (PROCESSED / f"{dataset}.csv").exists()]
    if not paths:
        raise FileNotFoundError("No successor results")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    atomic_csv(results, HERE / "results.csv")
    summary = (
        results.groupby(["task", "method"])[["selector_improvement", "test_improvement"]]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    atomic_csv(summary, HERE / "summary.csv")
    pivot = results.pivot(index=["dataset", "start_seed"], columns="method", values="test_improvement")
    comparisons: dict[str, Any] = {}
    for baseline in ("contextual_mean", "row_rotating", "random_mean"):
        difference = (pivot.contextual_rotating - pivot[baseline]).to_numpy()
        lower, upper = bootstrap_interval(difference, stable_seed("successor-bootstrap", baseline))
        comparisons[baseline] = {
            "mean_difference": float(difference.mean()),
            "wins": int((difference > 0).sum()),
            "ties": int((difference == 0).sum()),
            "total": len(difference),
            "win_fraction": float((difference > 0).mean()),
            "bootstrap_95": [lower, upper],
        }
    positive_fraction = float((pivot.contextual_rotating > 0).mean())
    task_means = results[results.method == "contextual_rotating"].groupby("task").test_improvement.mean().to_dict()
    audit = {
        "rows": len(results),
        "expected_rows": len(DATASETS) * len(SEEDS) * len(METHODS),
        "datasets_complete": sorted(results.dataset.unique()),
        "expected_datasets": sorted(DATASETS),
        "contextual_rotating_positive_fraction": positive_fraction,
        "comparisons": comparisons,
        "contextual_rotating_task_mean_test_improvement": task_means,
        "gate_1_positive_at_least_70pct": positive_fraction >= 0.70,
        "gate_2_beats_contextual_mean": comparisons["contextual_mean"]["win_fraction"] >= 0.60
        and comparisons["contextual_mean"]["mean_difference"] > 0,
        "gate_3_beats_row_rotating": comparisons["row_rotating"]["win_fraction"] >= 0.60
        and comparisons["row_rotating"]["mean_difference"] > 0,
        "gate_4_random_mean_noninferiority": comparisons["random_mean"]["mean_difference"] > 0
        and comparisons["random_mean"]["bootstrap_95"][0] >= -0.005,
        "gate_5_both_task_types_positive": all(value > 0 for value in task_means.values()),
        "test_labels_used_for_selection": False,
        "split_seed": SPLIT_SEED,
        "selector_folds": 4,
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
