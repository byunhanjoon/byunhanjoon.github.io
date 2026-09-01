"""Rerun all Native Feature Geometry analyses and require byte-identical outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUTS = (
    RESULTS / "pilot_summary.json",
    RESULTS / "pilot_paths.csv",
    RESULTS / "h6_summary.json",
    RESULTS / "h6_dose_cells.csv",
    RESULTS / "posthoc_diagnostics.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    before = {str(path.relative_to(HERE)): digest(path) for path in OUTPUTS}
    for script in ("analyze_pilot.py", "analyze_transport_dose.py", "posthoc_diagnostics.py"):
        subprocess.run(
            [sys.executable, script], cwd=HERE, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    after = {str(path.relative_to(HERE)): digest(path) for path in OUTPUTS}
    mismatches = [name for name in before if before[name] != after[name]]
    summary = {
        "status": "pass" if not mismatches else "fail",
        "output_count": len(OUTPUTS),
        "mismatches": mismatches,
        "sha256": after,
    }
    (RESULTS / "analysis_reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
