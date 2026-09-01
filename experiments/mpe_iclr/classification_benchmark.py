#!/usr/bin/env python3
"""Secondary BTS ArrDel15 cold-airport classification benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from mpe import state_balanced_mean, state_loss_table
from representations import corrupted_mpe_table, load_task, representation_tables, split_row_indices
from ridge_benchmark import ordinary_design, row_representation, state_balanced_training_weights


HERE = Path(__file__).resolve().parent
TASKS = ["airline_origin_airport", "airline_destination_airport"]
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def fit_predict(
    design: sparse.csr_matrix,
    target: np.ndarray,
    state_index: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha, max_iter=2000, tol=1e-5,
        random_state=20261101, average=True, n_jobs=1,
    )
    model.fit(
        design[train_rows], target[train_rows],
        sample_weight=state_balanced_training_weights(state_index[train_rows]),
    )
    return model.predict_proba(design[evaluation_rows])[:, 1]


def classification_metrics(target: np.ndarray, probability: np.ndarray, states: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    brier = (probability - target) ** 2
    state_brier = state_loss_table(brier, states)
    summary = {
        "state_balanced_brier": state_balanced_mean(brier, states),
        "row_weighted_brier": float(np.mean(brier)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "auroc": float(roc_auc_score(target, probability)),
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
    }
    per_state = [
        {"state_id": str(state), "rows": int(np.sum(states == state)), "brier": value}
        for state, value in state_brier.items()
    ]
    return summary, per_state


def run_cell(task_name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{task_name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume classification {cell}", flush=True)
        return
    task = load_task(task_name)
    parts = split_row_indices(task, split)
    ridge_path = HERE / "raw" / "ridge_cells" / f"{cell}.json"
    if not ridge_path.exists():
        raise FileNotFoundError(ridge_path)
    bandwidth = float(json.loads(ridge_path.read_text())["selected_bandwidth"])
    tables, metadata = representation_tables(task, split, bandwidth)
    for corruption in range(10):
        tables[f"mpe_corrupt_{corruption}"] = corrupted_mpe_table(task, split, bandwidth, corruption)
    target = pd.to_numeric(task.rows["arr_del15"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(target)).issubset({0, 1}):
        raise AssertionError("ArrDel15 is not binary")
    state_index = task.row_state_indices()
    state_ids = task.rows["field_state"].astype(str).to_numpy()
    ordinary = ordinary_design(task, parts["train"]) if setting == "full_table" else None
    results = []
    state_rows = []
    cache: dict[str, tuple[dict[str, float], list[dict[str, Any]], float]] = {}
    for name, table in tables.items():
        digest = hashlib.sha256(np.ascontiguousarray(table).view(np.uint8)).hexdigest()
        if digest in cache:
            summary, per_state, selected_alpha = cache[digest]
            trials = []
            alias_of = next(row["representation"] for row in results if row["feature_digest"] == digest)
        else:
            representation = row_representation(task, table)
            design = representation if ordinary is None else sparse.hstack([ordinary, representation], format="csr")
            trials = []
            for alpha in ALPHAS:
                probability = fit_predict(
                    design, target, state_index, parts["train"], parts["validation"], alpha
                )
                score = state_balanced_mean(
                    (probability - target[parts["validation"]]) ** 2,
                    state_ids[parts["validation"]],
                )
                trials.append({"alpha": alpha, "state_balanced_brier": float(score)})
            selected_alpha = float(min(trials, key=lambda row: (row["state_balanced_brier"], row["alpha"]))["alpha"])
            probability = fit_predict(
                design, target, state_index, parts["train"], parts["test"], selected_alpha
            )
            summary, per_state = classification_metrics(
                target[parts["test"]], probability, state_ids[parts["test"]]
            )
            cache[digest] = (summary, per_state, selected_alpha)
            alias_of = None
        results.append(
            {
                "representation": name, "feature_dimension": int(table.shape[1]),
                "feature_digest": digest, "alias_of": alias_of,
                "selected_alpha": selected_alpha, "validation_trials": trials, **summary,
            }
        )
        state_rows.extend({"representation": name, **row} for row in per_state)
    pd.DataFrame(state_rows).assign(task=task_name, split=split, setting=setting).to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "task": task_name, "source_unit": "BTS", "target": "ArrDel15",
        "split": split, "setting": setting, "selected_regression_bandwidth": bandwidth,
        "alpha_grid": ALPHAS, "representation_metadata": metadata, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete classification {cell}", flush=True)


def consolidate(output: Path) -> None:
    rows = []
    states = []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        rows.extend(
            {
                "task": payload["task"], "source_unit": payload["source_unit"],
                "target": payload["target"], "split": payload["split"], "setting": payload["setting"],
                **{key: value for key, value in result.items() if key != "validation_trials"},
            }
            for result in payload["results"]
        )
        state_path = path.with_name(path.stem + "__state_metrics.parquet")
        if state_path.exists():
            states.append(pd.read_parquet(state_path))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / "classification_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / "classification_results.csv", index=False)
    if states:
        pd.concat(states, ignore_index=True).to_parquet(
            HERE / "raw" / "classification_state_results.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=["isolated_field", "full_table", "all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "classification_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = TASKS if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        settings = ("isolated_field", "full_table") if args.setting == "all" else (args.setting,)
        for task in tasks:
            for split in splits:
                for setting in settings:
                    run_cell(task, split, setting, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
