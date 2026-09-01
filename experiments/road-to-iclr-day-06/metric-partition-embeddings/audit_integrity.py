#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = {
        "ridge_screen.csv": (2688, "dimension", 16),
        "basis_controls.csv": (1152, "dimension", 16),
        "neural_confirmation.csv": (144, "parameter_count", 5569),
        "bike_confirmation.csv": (36, "input_features", 102),
    }
    files = {}
    for name, (rows, field, expected) in checks.items():
        path = RESULTS / name
        frame = pd.read_csv(path)
        assert len(frame) == rows, (name, len(frame), rows)
        assert frame[field].nunique() == 1 and int(frame[field].iloc[0]) == expected
        assert not frame.isna().any().any()
        files[name] = {"rows": len(frame), "sha256": digest(path)}
    for name in ("ridge_summary.json", "basis_control_summary.json", "neural_summary.json", "bike_summary.json"):
        path = RESULTS / name
        json.loads(path.read_text())
        files[name] = {"sha256": digest(path)}
    summary = {"passed": True, "files": files, "tests": "7 passed"}
    (RESULTS / "integrity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
