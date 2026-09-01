#!/usr/bin/env python3
"""Join frozen router decisions to already-observed outer development results."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
MFT_ROOT = HERE.parent / "metric-field-transport"
if str(MFT_ROOT) not in sys.path:
    sys.path.insert(0, str(MFT_ROOT))

from successor_experiments import ALL_TASKS, E1B_TASKS, NEURAL_SEEDS  # noqa: E402
from router_experiment import PROTOCOL_PATH, sha256_path  # noqa: E402


RESULTS = HERE / "results"
MFT_RESULTS = MFT_ROOT / "results"
SCORE = "validation_state_balanced_standardized_mse"


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def relative(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    sources = {}
    for source, values in sorted(grouped.items()):
        baseline = mean(row["weights_m32"] for row in values)
        always_raw = mean(row["distance_m32"] for row in values)
        routed = mean(row["routed"] for row in values)
        sources[source] = {
            "weights_m32": baseline,
            "distance_m32": always_raw,
            "routed": routed,
            "always_raw_relative_improvement": relative(baseline, always_raw),
            "routed_relative_improvement": relative(baseline, routed),
            "router_raw_selections": sum(row["decision"] == "distance_m32" for row in values),
            "cells": len(values),
        }
    baseline = mean(row["weights_m32"] for row in sources.values())
    always_raw = mean(row["distance_m32"] for row in sources.values())
    routed = mean(row["routed"] for row in sources.values())
    return {
        "sources": sources,
        "source_balanced_weights_m32": baseline,
        "source_balanced_distance_m32": always_raw,
        "source_balanced_routed": routed,
        "source_balanced_always_raw_relative_improvement": relative(baseline, always_raw),
        "source_balanced_routed_relative_improvement": relative(baseline, routed),
        "raw_selections": sum(row["decision"] == "distance_m32" for row in rows),
        "cells": len(rows),
        "routed_wins": sum(row["routed"] < row["weights_m32"] for row in rows),
        "routed_ties": sum(row["routed"] == row["weights_m32"] for row in rows),
        "routed_losses": sum(row["routed"] > row["weights_m32"] for row in rows),
    }


def main() -> None:
    cell_paths = sorted((RESULTS / "router_cells").glob("*.json"))
    cells = [json.loads(path.read_text()) for path in cell_paths]
    protocol_hash = sha256_path(PROTOCOL_PATH)
    identifiers = set()
    for cell in cells:
        if cell["cell_id"] in identifiers:
            raise AssertionError(f"duplicate {cell['cell_id']}")
        identifiers.add(cell["cell_id"])
        if cell.get("protocol_sha256") != protocol_hash:
            raise AssertionError(f"protocol mismatch: {cell['cell_id']}")
        if cell.get("sealed_original_test") is not True or cell.get("test_target_evaluations") != 0:
            raise AssertionError(f"test seal violation: {cell['cell_id']}")
    expected = {(task, split) for task in ALL_TASKS for split in range(5)}
    observed = {(cell["task"], cell["split"]) for cell in cells}
    complete = observed == expected

    decisions = {(cell["task"], cell["split"]): cell["decision"] for cell in cells}
    ridge_rows = []
    if complete:
        for cell in cells:
            outer_path = (
                MFT_RESULTS
                / "e1a_cells"
                / f"{cell['task']}__split{cell['split']}__full_table.json"
            )
            outer = json.loads(outer_path.read_text())
            scores = {row["representation"]: row[SCORE] for row in outer["results"]}
            decision = cell["decision"]
            ridge_rows.append(
                {
                    "task": cell["task"],
                    "source": cell["source_unit"],
                    "split": cell["split"],
                    "decision": decision,
                    "weights_m32": scores["weights_m32"],
                    "distance_m32": scores["distance_m32"],
                    "routed": scores[decision],
                }
            )
    ridge = source_summary(ridge_rows) if ridge_rows else None

    neural_rows = []
    if complete:
        for task in E1B_TASKS:
            for split in (0, 1):
                decision = decisions[(task, split)]
                for seed in NEURAL_SEEDS:
                    scores = {}
                    source = None
                    for representation in ("weights_m32", "distance_m32"):
                        path = (
                            MFT_RESULTS
                            / "e1b_cells"
                            / f"{task}__split{split}__{representation}__seed{seed}.json"
                        )
                        payload = json.loads(path.read_text())
                        source = payload["source_unit"]
                        scores[representation] = payload["result"][SCORE]
                    neural_rows.append(
                        {
                            "task": task,
                            "source": source,
                            "split": split,
                            "seed": seed,
                            "decision": decision,
                            "weights_m32": scores["weights_m32"],
                            "distance_m32": scores["distance_m32"],
                            "routed": scores[decision],
                        }
                    )
    neural = source_summary(neural_rows) if neural_rows else None

    gate = None
    if complete and ridge is not None and neural is not None:
        medical_rejections = [
            decisions[("medical_charges", split)] == "weights_m32" for split in (0, 1)
        ]
        checks = {
            "neural_source_balanced_improvement_at_least_5_percent": neural[
                "source_balanced_routed_relative_improvement"
            ]
            >= 0.05,
            "no_neural_source_degradation_above_1_percent": min(
                row["routed_relative_improvement"] for row in neural["sources"].values()
            )
            >= -0.01,
            "rejects_raw_in_both_medical_neural_partitions": all(medical_rejections),
            "broad_ridge_source_balanced_no_worse": ridge[
                "source_balanced_routed_relative_improvement"
            ]
            >= 0.0,
            "no_broad_ridge_source_degradation_above_2_percent": min(
                row["routed_relative_improvement"] for row in ridge["sources"].values()
            )
            >= -0.02,
        }
        gate = {"recommend_new_data_confirmation": all(checks.values()), "checks": checks}

    summary = {
        "status": "complete" if complete else "running",
        "scientific_status": "post_outcome_exploratory_only",
        "protocol_sha256": protocol_hash,
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "completed_router_cells": len(cells),
        "expected_router_cells": 45,
        "ridge": ridge,
        "neural": neural,
        "feasibility_gate": gate,
    }
    atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", RESULTS / "analysis.json")

    lines = [
        "# Metric Trust Router — exploratory results",
        "",
        f"Status: **{summary['status'].upper()}** ({len(cells)}/45 router cells)",
        "",
        "These results are post-outcome feasibility evidence only. Original test targets remain sealed.",
        "",
    ]
    if complete and ridge is not None and neural is not None and gate is not None:
        lines.extend(
            [
                "## Broad nine-task Ridge join",
                "",
                "| Source | Weight MSE | Always raw MSE | Routed MSE | Routed improvement | Raw selections |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for source, row in ridge["sources"].items():
            lines.append(
                f"| {source} | {row['weights_m32']:.6f} | {row['distance_m32']:.6f} | "
                f"{row['routed']:.6f} | {100 * row['routed_relative_improvement']:+.2f}% | "
                f"{row['router_raw_selections']}/{row['cells']} |"
            )
        lines.extend(
            [
                "",
                f"Source-balanced routed improvement: "
                f"{100 * ridge['source_balanced_routed_relative_improvement']:+.2f}%. "
                f"Wins/ties/losses: {ridge['routed_wins']}/{ridge['routed_ties']}/{ridge['routed_losses']}.",
                "",
                "## Four-source neural join",
                "",
                "| Source | Weight MSE | Always raw MSE | Routed MSE | Routed improvement | Raw selections |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for source, row in neural["sources"].items():
            lines.append(
                f"| {source} | {row['weights_m32']:.6f} | {row['distance_m32']:.6f} | "
                f"{row['routed']:.6f} | {100 * row['routed_relative_improvement']:+.2f}% | "
                f"{row['router_raw_selections']}/{row['cells']} |"
            )
        lines.extend(
            [
                "",
                f"Source-balanced routed improvement: "
                f"{100 * neural['source_balanced_routed_relative_improvement']:+.2f}%. "
                f"Wins/ties/losses: {neural['routed_wins']}/{neural['routed_ties']}/{neural['routed_losses']}.",
                "",
                "## Frozen feasibility gate",
                "",
            ]
        )
        for name, passed in gate["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
        lines.extend(
            [
                "",
                f"Decision: **{'RECOMMEND NEW-DATA CONFIRMATION' if gate['recommend_new_data_confirmation'] else 'REJECT ROUTER AS LEAD'}**.",
            ]
        )
    atomic_text("\n".join(lines) + "\n", HERE / "RESULTS.md")
    print(f"status={summary['status']} cells={len(cells)}/45", flush=True)


if __name__ == "__main__":
    main()
