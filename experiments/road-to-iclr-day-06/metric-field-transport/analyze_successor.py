#!/usr/bin/env python3
"""Regenerate the post-MPE successor development summary and promotion gate."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from successor_experiments import (
    ALL_TASKS,
    E0_CONDITIONS,
    E0_TASKS,
    E1_REPRESENTATIONS,
    E1B_REPRESENTATIONS,
    E1B_TASKS,
    NEURAL_SEEDS,
    PROTOCOL_PATH,
    sha256_path,
)
from transport_experiments import E2_CONDITIONS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW_DISTANCE_CANDIDATES = ["distance_m32", "distance_m64", "distance_m128", "distance_all"]
SCORE_KEY = "validation_state_balanced_standardized_mse"


def load_json_cells(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def check_cells(cells: list[dict[str, Any]], stage: str) -> None:
    protocol_hash = sha256_path(PROTOCOL_PATH)
    identifiers: set[str] = set()
    for cell in cells:
        identifier = cell["cell_id"]
        if identifier in identifiers:
            raise AssertionError(f"duplicate {stage} cell: {identifier}")
        identifiers.add(identifier)
        if cell.get("stage") != stage or cell.get("status") != "complete":
            raise AssertionError(f"invalid {stage} status: {identifier}")
        if cell.get("protocol_sha256") != protocol_hash:
            raise AssertionError(f"protocol mismatch: {identifier}")
        if cell.get("sealed_original_test") is not True or cell.get("test_target_evaluations") != 0:
            raise AssertionError(f"test-target seal violation: {identifier}")


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def paired_summary(values: Iterable[tuple[float, float]]) -> dict[str, Any]:
    pairs = list(values)
    improvements = [relative_improvement(baseline, candidate) for baseline, candidate in pairs]
    return {
        "cells": len(pairs),
        "wins": sum(candidate < baseline for baseline, candidate in pairs),
        "win_fraction": sum(candidate < baseline for baseline, candidate in pairs) / len(pairs),
        "mean_relative_improvement": mean(improvements),
        "median_relative_improvement": median(improvements),
        "mean_baseline_mse": mean(baseline for baseline, _ in pairs),
        "mean_candidate_mse": mean(candidate for _, candidate in pairs),
    }


def analyze_e0(cells: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"]): cell["result"][SCORE_KEY]
        for cell in cells
    }
    expected_keys = [
        (task, split, seed, condition)
        for task in E0_TASKS
        for split in (0, 1)
        for seed in NEURAL_SEEDS
        for condition in E0_CONDITIONS
    ]
    missing = [key for key in expected_keys if key not in lookup]
    condition_summaries: dict[str, Any] = {}
    if not missing:
        for condition in E0_CONDITIONS:
            if condition == "weights_direct":
                continue
            pairs = [
                (
                    lookup[(task, split, seed, "weights_direct")],
                    lookup[(task, split, seed, condition)],
                )
                for task in E0_TASKS
                for split in (0, 1)
                for seed in NEURAL_SEEDS
            ]
            condition_summaries[condition] = paired_summary(pairs)

    initial_exact = True
    initial_comparisons = 0
    initial_lookup = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"]): cell["result"][
            "initial_validation_state_balanced_standardized_mse"
        ]
        for cell in cells
    }
    for condition in ("factor_identity_learned", "factor_rezero"):
        for task in E0_TASKS:
            for split in (0, 1):
                for seed in NEURAL_SEEDS:
                    direct_key = (task, split, seed, "weights_direct")
                    candidate_key = (task, split, seed, condition)
                    if direct_key in initial_lookup and candidate_key in initial_lookup:
                        initial_comparisons += 1
                        initial_exact &= initial_lookup[direct_key] == initial_lookup[candidate_key]
    return {
        "expected_cells": len(expected_keys),
        "completed_cells": len(lookup),
        "complete": not missing,
        "missing": [list(key) for key in missing],
        "identity_rezero_initial_score_exact": initial_exact,
        "identity_rezero_initial_comparisons": initial_comparisons,
        "vs_weights_direct": condition_summaries,
    }


def e1a_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        for result in cell["results"]:
            rows.append(
                {
                    "task": cell["task"],
                    "source": cell["source_unit"],
                    "split": cell["split"],
                    "setting": cell["setting"],
                    "representation": result["representation"],
                    "dimension": result["feature_dimension"],
                    "score": result[SCORE_KEY],
                }
            )
    return rows


def source_balanced_scores(
    rows: list[dict[str, Any]], representations: list[str], setting: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for representation in representations:
        chosen = [
            row
            for row in rows
            if row["representation"] == representation and (setting is None or row["setting"] == setting)
        ]
        by_source: dict[str, list[float]] = defaultdict(list)
        for row in chosen:
            by_source[row["source"]].append(row["score"])
        source_means = {source: mean(scores) for source, scores in sorted(by_source.items())}
        result[representation] = {
            "source_means": source_means,
            "source_balanced_mean_mse": mean(source_means.values()) if source_means else math.nan,
            "mean_dimension": mean(row["dimension"] for row in chosen) if chosen else math.nan,
            "cells": len(chosen),
        }
    return result


def analyze_e1a(cells: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(cell["task"], cell["split"], cell["setting"]): cell for cell in cells}
    expected_keys = [
        (task, split, setting)
        for task in ALL_TASKS
        for split in range(5)
        for setting in ("isolated_field", "full_table")
    ]
    missing = [key for key in expected_keys if key not in lookup]
    rows = e1a_rows(cells)
    full_scores = source_balanced_scores(rows, E1_REPRESENTATIONS, "full_table")
    isolated_scores = source_balanced_scores(rows, E1_REPRESENTATIONS, "isolated_field")
    selected_representation = None
    if not missing:
        selected_representation = min(
            RAW_DISTANCE_CANDIDATES,
            key=lambda name: (
                full_scores[name]["source_balanced_mean_mse"],
                full_scores[name]["mean_dimension"],
                RAW_DISTANCE_CANDIDATES.index(name),
            ),
        )
    return {
        "expected_cells": len(expected_keys),
        "completed_cells": len(lookup),
        "complete": not missing,
        "missing": [list(key) for key in missing],
        "full_table": full_scores,
        "isolated_field": isolated_scores,
        "raw_distance_candidates": RAW_DISTANCE_CANDIDATES,
        "selected_raw_representation": selected_representation,
    }


def analyze_e1b(cells: list[dict[str, Any]], selected_representation: str | None) -> dict[str, Any]:
    lookup = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"]): cell for cell in cells
    }
    expected_keys = [
        (task, split, seed, condition)
        for task in E1B_TASKS
        for split in (0, 1)
        for seed in NEURAL_SEEDS
        for condition in E1B_REPRESENTATIONS
    ]
    missing = [key for key in expected_keys if key not in lookup]
    result: dict[str, Any] = {
        "expected_cells": len(expected_keys),
        "completed_cells": len(lookup),
        "complete": not missing,
        "missing": [list(key) for key in missing],
        "selected_raw_representation": selected_representation,
        "promotion_gate": None,
    }
    if missing or selected_representation is None:
        return result

    pairs = []
    source_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task in E1B_TASKS:
        for split in (0, 1):
            for seed in NEURAL_SEEDS:
                baseline_cell = lookup[(task, split, seed, "weights_m32")]
                candidate_cell = lookup[(task, split, seed, selected_representation)]
                baseline = baseline_cell["result"][SCORE_KEY]
                candidate = candidate_cell["result"][SCORE_KEY]
                pairs.append((baseline, candidate))
                source = baseline_cell["source_unit"]
                source_values[source]["baseline"].append(baseline)
                source_values[source]["candidate"].append(candidate)
    paired = paired_summary(pairs)
    source_rows = {}
    for source, values in sorted(source_values.items()):
        baseline = mean(values["baseline"])
        candidate = mean(values["candidate"])
        source_rows[source] = {
            "baseline_mse": baseline,
            "candidate_mse": candidate,
            "relative_improvement": relative_improvement(baseline, candidate),
        }
    source_balanced_baseline = mean(row["baseline_mse"] for row in source_rows.values())
    source_balanced_candidate = mean(row["candidate_mse"] for row in source_rows.values())
    source_balanced_improvement = relative_improvement(source_balanced_baseline, source_balanced_candidate)
    checks = {
        "beats_at_least_3_of_4_sources": sum(
            row["relative_improvement"] > 0 for row in source_rows.values()
        )
        >= 3,
        "wins_at_least_60_percent_cells": paired["win_fraction"] >= 0.60,
        "source_balanced_improvement_at_least_1_percent": source_balanced_improvement >= 0.01,
        "no_source_degradation_above_5_percent": min(
            row["relative_improvement"] for row in source_rows.values()
        )
        >= -0.05,
    }
    result["promotion_gate"] = {
        "promote_to_e2": all(checks.values()),
        "checks": checks,
        "paired": paired,
        "sources": source_rows,
        "source_balanced_baseline_mse": source_balanced_baseline,
        "source_balanced_candidate_mse": source_balanced_candidate,
        "source_balanced_relative_improvement": source_balanced_improvement,
    }
    return result


def analyze_e2(cells: list[dict[str, Any]], e1b: dict[str, Any]) -> dict[str, Any]:
    e1_gate = e1b.get("promotion_gate")
    authorized = bool(e1_gate and e1_gate.get("promote_to_e2"))
    if not authorized:
        return {
            "authorized_by_e1": False,
            "expected_cells": 0,
            "completed_cells": len(cells),
            "complete": bool(e1_gate),
            "missing": [],
            "success_gate": None,
        }

    lookup = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"]): cell for cell in cells
    }
    expected_keys = [
        (task, split, seed, condition)
        for task in E1B_TASKS
        for split in (0, 1)
        for seed in NEURAL_SEEDS
        for condition in E2_CONDITIONS
    ]
    missing = [key for key in expected_keys if key not in lookup]
    result: dict[str, Any] = {
        "authorized_by_e1": True,
        "expected_cells": len(expected_keys),
        "completed_cells": len(lookup),
        "complete": not missing,
        "missing": [list(key) for key in missing],
        "candidate": "transport_first_order",
        "success_gate": None,
    }
    if missing:
        return result

    conditions = ["raw_base", "transport_zero", "transport_first_order", "transport_shuffled_metric"]
    source_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    paired_by_baseline: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for task in E1B_TASKS:
        for split in (0, 1):
            for seed in NEURAL_SEEDS:
                cells_here = {condition: lookup[(task, split, seed, condition)] for condition in conditions}
                scores = {condition: cells_here[condition]["result"][SCORE_KEY] for condition in conditions}
                source = cells_here["raw_base"]["source_unit"]
                for condition, score in scores.items():
                    source_values[source][condition].append(score)
                for baseline in ("raw_base", "transport_zero", "transport_shuffled_metric"):
                    paired_by_baseline[baseline].append((scores[baseline], scores["transport_first_order"]))

    source_rows: dict[str, Any] = {}
    for source, values in sorted(source_values.items()):
        means = {condition: mean(scores) for condition, scores in values.items()}
        source_rows[source] = {
            **{f"{condition}_mse": score for condition, score in means.items()},
            "improvement_vs_raw_base": relative_improvement(
                means["raw_base"], means["transport_first_order"]
            ),
            "improvement_vs_transport_zero": relative_improvement(
                means["transport_zero"], means["transport_first_order"]
            ),
        }
    source_balanced = {
        condition: mean(row[f"{condition}_mse"] for row in source_rows.values())
        for condition in conditions
    }
    improvement_vs_raw = relative_improvement(
        source_balanced["raw_base"], source_balanced["transport_first_order"]
    )
    paired = {baseline: paired_summary(pairs) for baseline, pairs in paired_by_baseline.items()}
    checks = {
        "beats_raw_and_zero_on_at_least_3_of_4_sources": sum(
            row["improvement_vs_raw_base"] > 0 and row["improvement_vs_transport_zero"] > 0
            for row in source_rows.values()
        )
        >= 3,
        "wins_at_least_60_percent_cells_vs_raw": paired["raw_base"]["win_fraction"] >= 0.60,
        "source_balanced_improvement_at_least_1_percent_vs_raw": improvement_vs_raw >= 0.01,
        "no_source_degradation_above_5_percent_vs_raw": min(
            row["improvement_vs_raw_base"] for row in source_rows.values()
        )
        >= -0.05,
        "beats_shuffled_metric_on_at_least_80_percent_cells": paired["transport_shuffled_metric"][
            "win_fraction"
        ]
        >= 0.80,
    }
    result["success_gate"] = {
        "success": all(checks.values()),
        "checks": checks,
        "sources": source_rows,
        "source_balanced_mse": source_balanced,
        "source_balanced_improvement_vs_raw": improvement_vs_raw,
        "paired": paired,
    }
    return result


def percentage(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def markdown_report(summary: dict[str, Any]) -> str:
    e0, e1a, e1b, e2 = summary["e0"], summary["e1a"], summary["e1b"], summary["e2"]
    lines = [
        "# Metric-Field Transport — development results",
        "",
        f"Status: **{summary['status'].upper()}**",
        "",
        "These are post-outcome development results. The original MPE test targets remain sealed; "
        "no result here is confirmatory evidence.",
        "",
        "## Integrity and completeness",
        "",
        "| Stage | Complete cells | Expected | Complete |",
        "|---|---:|---:|:---:|",
        f"| E0 | {e0['completed_cells']} | {e0['expected_cells']} | {e0['complete']} |",
        f"| E1a | {e1a['completed_cells']} | {e1a['expected_cells']} | {e1a['complete']} |",
        f"| E1b | {e1b['completed_cells']} | {e1b['expected_cells']} | {e1b['complete']} |",
        f"| E2 | {e2['completed_cells']} | {e2['expected_cells']} | {e2['complete']} |",
        "",
        f"Protocol SHA-256: `{summary['protocol_sha256']}`. Every loaded artifact records "
        "`sealed_original_test=true` and `test_target_evaluations=0`.",
        "",
        "## E0 — factorization control",
        "",
    ]
    if e0["complete"]:
        lines.extend(
            [
                "| Condition vs direct weights | Wins | Mean relative change | Median relative change |",
                "|---|---:|---:|---:|",
            ]
        )
        for condition, values in e0["vs_weights_direct"].items():
            lines.append(
                f"| {condition} | {values['wins']}/{values['cells']} | "
                f"{percentage(values['mean_relative_improvement'])} | "
                f"{percentage(values['median_relative_improvement'])} |"
            )
        lines.extend(
            [
                "",
                f"Identity/ReZero exact paired initial-score check: "
                f"**{e0['identity_rezero_initial_score_exact']}** "
                f"({e0['identity_rezero_initial_comparisons']} comparisons).",
            ]
        )
    else:
        lines.append("The paired E0 table will be emitted after all declared cells finish.")

    lines.extend(["", "## E1a — metric-coordinate Ridge screen", ""])
    if e1a["complete"]:
        lines.extend(
            [
                "| Representation | Full-table source-balanced MSE | Isolated-field source-balanced MSE |",
                "|---|---:|---:|",
            ]
        )
        for representation in E1_REPRESENTATIONS:
            lines.append(
                f"| {representation} | {e1a['full_table'][representation]['source_balanced_mean_mse']:.6f} | "
                f"{e1a['isolated_field'][representation]['source_balanced_mean_mse']:.6f} |"
            )
        lines.extend(
            [
                "",
                f"Predeclared raw-distance selection: **{e1a['selected_raw_representation']}**.",
            ]
        )
    else:
        lines.append("The representation selection remains unavailable until all E1a cells finish.")

    lines.extend(["", "## E1b — neural promotion gate", ""])
    gate = e1b["promotion_gate"]
    if gate is not None:
        lines.extend(
            [
                f"Selected candidate: **{e1b['selected_raw_representation']}**.",
                "",
                "| Source | Weight MSE | Candidate MSE | Relative improvement |",
                "|---|---:|---:|---:|",
            ]
        )
        for source, values in gate["sources"].items():
            lines.append(
                f"| {source} | {values['baseline_mse']:.6f} | {values['candidate_mse']:.6f} | "
                f"{percentage(values['relative_improvement'])} |"
            )
        lines.extend(
            [
                "",
                f"Paired wins: {gate['paired']['wins']}/{gate['paired']['cells']} "
                f"({100 * gate['paired']['win_fraction']:.1f}%). Source-balanced improvement: "
                f"{percentage(gate['source_balanced_relative_improvement'])}.",
                "",
            ]
        )
        for check, passed in gate["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{check}`")
        lines.extend(
            [
                "",
                f"Decision: **{'PROMOTE TO E2' if gate['promote_to_e2'] else 'REJECT E2 PROMOTION'}**.",
            ]
        )
    else:
        lines.append("The E1 neural promotion decision is pending complete E1a and E1b results.")
    lines.extend(["", "## E2 — whole-state task transport", ""])
    if not e2["authorized_by_e1"]:
        if e1b["promotion_gate"] is None:
            lines.append("E2 remains locked until E1 is complete.")
        else:
            lines.append("E2 was not authorized because E1 did not pass its frozen promotion gate.")
    elif not e2["complete"]:
        lines.append(
            f"E2 is authorized and running ({e2['completed_cells']}/{e2['expected_cells']} cells)."
        )
    else:
        e2_gate = e2["success_gate"]
        assert e2_gate is not None
        lines.extend(
            [
                "| Source | Raw MSE | Zero-order MSE | First-order MSE | First-order vs raw |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for source, values in e2_gate["sources"].items():
            lines.append(
                f"| {source} | {values['raw_base_mse']:.6f} | {values['transport_zero_mse']:.6f} | "
                f"{values['transport_first_order_mse']:.6f} | "
                f"{percentage(values['improvement_vs_raw_base'])} |"
            )
        lines.extend(
            [
                "",
                f"Paired wins vs raw: {e2_gate['paired']['raw_base']['wins']}/"
                f"{e2_gate['paired']['raw_base']['cells']}; vs shuffled metric: "
                f"{e2_gate['paired']['transport_shuffled_metric']['wins']}/"
                f"{e2_gate['paired']['transport_shuffled_metric']['cells']}.",
                "",
            ]
        )
        for check, passed in e2_gate["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{check}`")
        lines.extend(
            [
                "",
                f"Decision: **{'TRANSPORT SURVIVES DEVELOPMENT' if e2_gate['success'] else 'REJECT TRANSPORT AS LEAD'}**.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    e0_cells = load_json_cells(RESULTS / "e0_cells")
    e1a_cells = load_json_cells(RESULTS / "e1a_cells")
    e1b_cells = load_json_cells(RESULTS / "e1b_cells")
    e2_cells = load_json_cells(RESULTS / "e2_cells")
    check_cells(e0_cells, "e0")
    check_cells(e1a_cells, "e1a")
    check_cells(e1b_cells, "e1b")
    check_cells(e2_cells, "e2")

    e0 = analyze_e0(e0_cells)
    e1a = analyze_e1a(e1a_cells)
    e1b = analyze_e1b(e1b_cells, e1a["selected_raw_representation"])
    e2 = analyze_e2(e2_cells, e1b)
    prior_complete = e0["complete"] and e1a["complete"] and e1b["complete"]
    workflow_complete = prior_complete and e2["complete"]
    summary = {
        "status": "complete" if workflow_complete else "running",
        "scientific_status": "post_outcome_development_only",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "e0": e0,
        "e1a": e1a,
        "e1b": e1b,
        "e2": e2,
    }
    atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", RESULTS / "analysis.json")
    atomic_text(markdown_report(summary), HERE / "RESULTS.md")
    print(
        f"status={summary['status']} e0={e0['completed_cells']}/{e0['expected_cells']} "
        f"e1a={e1a['completed_cells']}/{e1a['expected_cells']} "
        f"e1b={e1b['completed_cells']}/{e1b['expected_cells']} "
        f"e2={e2['completed_cells']}/{e2['expected_cells']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
