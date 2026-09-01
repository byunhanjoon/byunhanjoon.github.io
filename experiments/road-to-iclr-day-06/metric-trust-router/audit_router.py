#!/usr/bin/env python3
"""Integrity audit for the exploratory five-fold metric trust router."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from router_experiment import (
    ALL_TASKS,
    PROTOCOL_PATH,
    REPRESENTATIONS,
    RIDGE_ALPHAS,
    load_task,
    sha256_path,
    split_state_indices,
    state_folds,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def main() -> None:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    observed_hash = sha256_path(PROTOCOL_PATH)
    expected_hash = (HERE / "PROTOCOL_SHA256.txt").read_text().split()[0]
    add("protocol hash", observed_hash == expected_hash, observed_hash)

    paths = sorted((RESULTS / "router_cells").glob("*.json"))
    cells = [json.loads(path.read_text()) for path in paths]
    add("router completeness", len(cells) == 45, f"{len(cells)}/45")
    identifiers = {cell["cell_id"] for cell in cells}
    add("unique router cells", len(identifiers) == len(cells), f"{len(identifiers)} unique")
    seals_ok = all(
        cell.get("protocol_sha256") == observed_hash
        and cell.get("sealed_original_test") is True
        and cell.get("test_target_evaluations") == 0
        and cell.get("scientific_status") == "post_outcome_exploratory_only"
        for cell in cells
    )
    add("artifact seals", seals_ok, "protocol/status/test seal")

    expected_menu = {(task, split) for task in ALL_TASKS for split in range(5)}
    observed_menu = {(cell["task"], cell["split"]) for cell in cells}
    add("declared task menu", observed_menu == expected_menu, f"{len(observed_menu)} cells")

    folds_ok = True
    landmarks_ok = True
    trials_ok = True
    decisions_ok = True
    task_cache: dict[str, Any] = {}
    for cell in cells:
        if cell["task"] not in task_cache:
            task_cache[cell["task"]] = load_task(cell["task"])
        task = task_cache[cell["task"]]
        outer_train = set(split_state_indices(task, cell["split"])["train"].tolist())
        expected_folds = state_folds(task, cell["split"])
        stored_folds = cell["fold_results"]
        folds_ok &= len(stored_folds) == 5
        for expected_fold, stored in zip(expected_folds, stored_folds):
            expected_ids = [task.state_ids[state] for state in expected_fold]
            folds_ok &= stored["validation_state_ids"] == expected_ids
            fold_train = outer_train - set(expected_fold.tolist())
            for landmarks in stored["representation_metadata"]["landmark_indices"].values():
                landmarks_ok &= set(landmarks).issubset(fold_train)
            representation_rows = {
                row["representation"]: row["trials"] for row in stored["representations"]
            }
            trials_ok &= set(representation_rows) == set(REPRESENTATIONS)
            for trials in representation_rows.values():
                trials_ok &= [row["alpha"] for row in trials] == RIDGE_ALPHAS
        expected_decision = (
            "distance_m32"
            if cell["selected_scores"]["distance_m32"]
            < cell["selected_scores"]["weights_m32"]
            else "weights_m32"
        )
        decisions_ok &= cell["decision"] == expected_decision
    add("five-fold state partition", folds_ok, "all stored folds reproduce")
    add("fold-train-only landmarks", landmarks_ok, "all landmark sets checked")
    add("fixed representations and alpha grid", trials_ok, "45 x 5 folds")
    add("zero-threshold decisions", decisions_ok, "all 45 decisions recomputed")

    temporary = list(HERE.rglob("*.tmp"))
    add("no partial writes", not temporary, ", ".join(map(str, temporary)) or "none")

    runner_source = (HERE / "router_experiment.py").read_text()
    join_isolation = "e1a_cells" not in runner_source and "e1b_cells" not in runner_source
    add("router/outer-join code isolation", join_isolation, "runner cannot load outer result folders")

    analysis_run = subprocess.run(
        [sys.executable, "analyze_router.py"], cwd=HERE, capture_output=True, text=True
    )
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    add(
        "analysis regeneration",
        analysis_run.returncode == 0 and analysis.get("status") == "complete",
        analysis_run.stdout.strip(),
    )
    add(
        "frozen feasibility gate",
        analysis.get("feasibility_gate", {}).get("recommend_new_data_confirmation") is True,
        str(analysis.get("feasibility_gate", {}).get("checks")),
    )

    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_router_experiment.py"],
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
    (RESULTS / "AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Metric Trust Router — integrity audit",
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
    (HERE / "AUDIT.md").write_text("\n".join(lines))
    print(f"{payload['status']}: {passed}/{len(checks)} checks", flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
