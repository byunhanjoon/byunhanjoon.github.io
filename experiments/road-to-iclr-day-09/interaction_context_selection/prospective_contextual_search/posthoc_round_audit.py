from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.core import DATASETS, TabICLEvaluator, load_dataset, parse_indices  # noqa: E402


METHODS = {"random_search", "row_active", "contextual_active", "additive_static", "fm_static"}


def context_key(value: str) -> tuple[int, ...]:
    return tuple(map(int, parse_indices(value)))


def run_dataset(dataset: str, device: str) -> None:
    result = pd.read_csv(HERE / "processed" / f"{dataset}.csv")
    calls = pd.read_csv(HERE / "raw" / f"{dataset}_calls.csv")
    selector = {
        context_key(row.indices): float(row.utility)
        for row in calls[calls.split == "selector"].itertuples()
    }
    test = {
        context_key(row.indices): float(row.utility)
        for row in calls[calls.split == "test"].itertuples()
    }
    evaluator = None
    rows: list[dict[str, object]] = []
    for record in result.itertuples():
        if record.method not in METHODS:
            continue
        trace = json.loads(record.trace_json)
        contexts = [record.initial_indices] + [step["indices"] for step in trace]
        start_test = None
        for round_id, encoded in enumerate(contexts):
            key = context_key(encoded)
            if key not in test:
                if evaluator is None:
                    evaluator = TabICLEvaluator(load_dataset(dataset), device, seed=0, n_estimators=1)
                test[key] = float(evaluator.evaluate(np.asarray(key), "test", return_prediction=False)["utility"])
            if start_test is None:
                start_test = test[key]
            rows.append(
                {
                    "dataset": dataset,
                    "data_role": record.data_role,
                    "method": record.method,
                    "start_seed": int(record.start_seed),
                    "round": round_id,
                    "indices": encoded,
                    "selector_utility": selector[key],
                    "test_utility": test[key],
                    "selector_improvement": selector[key] - float(record.start_selector_utility),
                    "test_improvement": test[key] - start_test,
                    "posthoc_test_diagnostic": True,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(HERE / "processed" / f"{dataset}_posthoc_rounds.csv", index=False)
    print(f"{dataset}: {len(output)} round rows, {len(test)} unique test contexts", flush=True)


def report() -> None:
    paths = sorted((HERE / "processed").glob("*_posthoc_rounds.csv"))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(HERE / "posthoc_round_results.csv", index=False)
    confirmation = frame[frame.data_role == "confirmation"]
    summary = (
        confirmation.groupby(["method", "round"])[["selector_improvement", "test_improvement"]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary.to_csv(HERE / "posthoc_round_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    run_parser.add_argument("--device", default="cuda:0")
    subparsers.add_parser("report")
    args = parser.parse_args()
    if args.command == "run":
        run_dataset(args.dataset, args.device)
    else:
        report()


if __name__ == "__main__":
    main()
