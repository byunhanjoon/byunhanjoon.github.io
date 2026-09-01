#!/usr/bin/env python3
"""Final integrity audit for the test-sealed successor development ladder."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analyze_successor import check_cells, load_json_cells
from successor_experiments import (
    E0_CONDITIONS,
    E0_TASKS,
    E1_REPRESENTATIONS,
    E1B_REPRESENTATIONS,
    E1B_TASKS,
    NEURAL_SEEDS,
    PROTOCOL_PATH,
    RIDGE_ALPHAS,
    inner_state_split,
    load_task,
    sha256_path,
    split_state_indices,
)
from transport_experiments import E2_CONDITIONS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def scientific_result(result: dict[str, Any]) -> dict[str, Any]:
    ignored = {"wall_seconds", "peak_gpu_bytes"}
    return {key: value for key, value in result.items() if key not in ignored}


def main() -> None:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    expected_hash = (HERE / "PROTOCOL_SHA256.txt").read_text().split()[0]
    observed_hash = sha256_path(PROTOCOL_PATH)
    add("protocol hash", observed_hash == expected_hash, observed_hash)

    analysis_run = subprocess.run(
        [sys.executable, "analyze_successor.py"], cwd=HERE, capture_output=True, text=True
    )
    add("analysis regeneration", analysis_run.returncode == 0, analysis_run.stdout.strip())
    analysis = json.loads((RESULTS / "analysis.json").read_text())

    cells_by_stage = {
        stage: load_json_cells(RESULTS / f"{stage}_cells") for stage in ("e0", "e1a", "e1b", "e2")
    }
    for stage, cells in cells_by_stage.items():
        try:
            check_cells(cells, stage)
        except Exception as error:  # pragma: no cover - audit reporting path
            add(f"{stage} artifact seal", False, repr(error))
        else:
            add(f"{stage} artifact seal", True, f"{len(cells)} cells")

    expected_counts = {"e0": 90, "e1a": 90, "e1b": 144}
    e1_promoted = bool(
        analysis.get("e1b", {}).get("promotion_gate", {}).get("promote_to_e2", False)
    )
    expected_counts["e2"] = 120 if e1_promoted else 0
    for stage, expected in expected_counts.items():
        observed = len(cells_by_stage[stage])
        add(f"{stage} completeness", observed == expected, f"{observed}/{expected}")

    temporary_files = list(HERE.rglob("*.tmp"))
    add("no partial writes", not temporary_files, ", ".join(map(str, temporary_files)) or "none")

    forbidden_test_metrics = []

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                lowered = key.lower()
                if "test" in lowered and any(word in lowered for word in ("mse", "rmse", "mae", "loss", "score")):
                    forbidden_test_metrics.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for stage, cells in cells_by_stage.items():
        for cell in cells:
            visit(cell, f"{stage}.{cell['cell_id']}")
    add(
        "no emitted test metrics",
        not forbidden_test_metrics,
        ", ".join(forbidden_test_metrics) or "none",
    )

    e0_keys = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"])
        for cell in cells_by_stage["e0"]
    }
    expected_e0_keys = {
        (task, split, seed, condition)
        for task in E0_TASKS
        for split in (0, 1)
        for seed in NEURAL_SEEDS
        for condition in E0_CONDITIONS
    }
    add("E0 declared menu", e0_keys == expected_e0_keys, f"{len(e0_keys)} unique cells")

    e1b_keys = {
        (cell["task"], cell["split"], cell["seed"], cell["condition"])
        for cell in cells_by_stage["e1b"]
    }
    expected_e1b_keys = {
        (task, split, seed, condition)
        for task in E1B_TASKS
        for split in (0, 1)
        for seed in NEURAL_SEEDS
        for condition in E1B_REPRESENTATIONS
    }
    add("E1b declared menu", e1b_keys == expected_e1b_keys, f"{len(e1b_keys)} unique cells")

    ridge_ok = True
    landmark_ok = True
    for cell in cells_by_stage["e1a"]:
        ridge_ok &= {row["representation"] for row in cell["results"]} == set(E1_REPRESENTATIONS)
        ridge_ok &= set(cell["inner_trials"]) == set(E1_REPRESENTATIONS)
        for trials in cell["inner_trials"].values():
            ridge_ok &= [trial["alpha"] for trial in trials] == RIDGE_ALPHAS
        task = load_task(cell["task"])
        outer = split_state_indices(task, cell["split"])["train"]
        inner, _ = inner_state_split(task, cell["split"])
        for landmarks in cell["outer_representation_metadata"]["landmark_indices"].values():
            landmark_ok &= set(landmarks).issubset(set(outer.tolist()))
        for landmarks in cell["inner_representation_metadata"]["landmark_indices"].values():
            landmark_ok &= set(landmarks).issubset(set(inner.tolist()))
    add("E1a declared representations and alpha grid", ridge_ok, f"{len(cells_by_stage['e1a'])} cells")
    add("train-only landmark selection", landmark_ok, f"{len(cells_by_stage['e1a'])} cells")

    curves_ok = True
    for stage in ("e0", "e1b", "e2"):
        for cell in cells_by_stage[stage]:
            result = cell["result"]
            curve = result.get("curve", [])
            if not curve:
                curves_ok = False
                continue
            values = [row["validation_state_balanced_standardized_mse"] for row in curve]
            curves_ok &= all(math.isfinite(value) for value in values)
            curves_ok &= result["validation_state_balanced_standardized_mse"] == min(values)
    add("finite neural curves and exact best score", curves_ok, "E0/E1b/E2")

    e0_lookup = {
        (cell["task"], cell["split"], cell["seed"]): cell for cell in cells_by_stage["e0"]
        if cell["condition"] == "weights_direct"
    }
    overlap_ok = True
    overlap_count = 0
    for cell in cells_by_stage["e1b"]:
        if cell["condition"] == "weights_m32" and cell["task"] in E0_TASKS:
            source = e0_lookup[(cell["task"], cell["split"], cell["seed"])]
            overlap_ok &= scientific_result(cell["result"]) == scientific_result(source["result"])
            overlap_count += 1
    add("deduplicated baseline equivalence", overlap_ok and overlap_count == 18, f"{overlap_count}/18")

    e2_menu_ok = True
    if e1_promoted:
        e2_keys = {
            (cell["task"], cell["split"], cell["seed"], cell["condition"])
            for cell in cells_by_stage["e2"]
        }
        expected_e2_keys = {
            (task, split, seed, condition)
            for task in E1B_TASKS
            for split in (0, 1)
            for seed in NEURAL_SEEDS
            for condition in E2_CONDITIONS
        }
        e2_menu_ok = e2_keys == expected_e2_keys
    else:
        e2_menu_ok = not cells_by_stage["e2"]
    add("E2 gate and declared menu", e2_menu_ok, f"promoted={e1_promoted}")

    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_successor_experiments.py", "test_analyze_successor.py", "test_transport_experiments.py"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    add("unit tests", pytest_run.returncode == 0, pytest_run.stdout.strip().splitlines()[-1])

    passed = sum(check.passed for check in checks)
    payload = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "protocol_sha256": observed_hash,
        "checks": [asdict(check) for check in checks],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Metric-Field Transport — final integrity audit",
        "",
        f"Status: **{payload['status']}** ({passed}/{len(checks)} checks passed)",
        "",
        "| Check | Status | Detail |",
        "|---|:---:|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {detail} |")
    lines.append("")
    (HERE / "FINAL_AUDIT.md").write_text("\n".join(lines))
    print(f"{payload['status']}: {passed}/{len(checks)} checks", flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
