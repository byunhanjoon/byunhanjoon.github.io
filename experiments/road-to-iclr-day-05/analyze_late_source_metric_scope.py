"""Metric-scope repeat on the four late OpenML sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_disjoint_pair32 import paired_ids
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_packing_metric_scope import METRICS, action_metrics, metric_values


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = HERE / "openml_late_source_cover_config.json"
TENSORS = RESULTS / "openml_late_source_cover"
COMPARISONS = {
    "pair32": ("disjoint_pair32", "independent_pair32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--tensors", type=Path, default=TENSORS)
    parser.add_argument("--output-prefix", default="late_source")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            archive = np.load(args.tensors / f"{dataset}__{model}.npz")
            predictions = archive["validation_predictions"]
            shape = tuple(int(value) for value in predictions.shape[:4])
            flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
            y = archive["validation_y"]
            disjoint, independent = paired_ids(shape, "late-source-metric", dataset)
            pack, pairs, _ = sample_pack_and_pairs(shape, "late-source-metric", dataset)
            actions = {
                "disjoint_pair32": np.stack(disjoint, axis=1),
                "independent_pair32": np.stack(independent, axis=1),
                "mutually_disjoint_pack64": pack,
                "two_disjoint_pairs64": pairs,
            }
            exact = {key: float(value[0]) for key, value in metric_values(y, flat.mean(axis=0)).items()}
            for method, blocks in actions.items():
                values = action_metrics(y, flat, blocks)
                for metric in METRICS:
                    error = values[metric] - exact[metric]
                    rows.append({
                        "dataset": dataset, "model": model, "method": method,
                        "metric": metric, "exact_metric": exact[metric],
                        "bias": float(error.mean()),
                        "rmse": float(np.sqrt(np.mean(error ** 2))),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / f"{args.output_prefix}_metric_scope.csv", index=False)
    summaries, all_pass = {}, True
    for name, (action, control) in COMPARISONS.items():
        metric_results = {}
        for metric in METRICS:
            current = frame[(frame.metric == metric) & frame.method.isin((action, control))]
            candidate = current.pivot(index=["dataset", "model"], columns="method", values="rmse")
            source = current.groupby(["dataset", "method"]).rmse.mean().unstack()
            wins = int((candidate[action] < candidate[control] - 1e-15).sum())
            source_wins = int((source[action] <= source[control] + 1e-15).sum())
            metric_results[metric] = {
                "candidate_strict_wins": wins, "candidates": int(len(candidate)),
                "source_mean_nohigher": source_wins, "sources": int(len(source)),
                "mean_rmse_ratio": float(current[current.method == action].rmse.mean() /
                                         current[current.method == control].rmse.mean()),
            }
        passed = bool(
            metric_results["brier"]["candidate_strict_wins"] >= 10
            and metric_results["log_loss"]["candidate_strict_wins"] >= 10
            and metric_results["roc_auc"]["candidate_strict_wins"] >= 9
            and all(metric_results[metric]["source_mean_nohigher"] == 4
                    for metric in ("brier", "log_loss", "roc_auc"))
        )
        all_pass &= passed
        summaries[name] = {"metrics": metric_results, "probabilistic_ranking_scope_passed": passed}
    summary = {
        "status": "complete", "sources": len(config["datasets"]),
        "candidates": len(config["datasets"]) * len(config["models"]),
        "comparisons": summaries, "all_probabilistic_ranking_gates_passed": bool(all_pass),
    }
    (RESULTS / f"{args.output_prefix}_metric_scope_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
