"""Run final syntax, test, and coverage audits and save machine-readable proof."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout[-12000:],
    }


def main() -> None:
    python_files = sorted(
        str(path.relative_to(ROOT))
        for directory in (ROOT / "experiments/day3", ROOT / "tests")
        for path in directory.glob("*.py")
    )
    syntax = run([sys.executable, "-m", "py_compile", *python_files])
    tests = run([sys.executable, "-m", "pytest", "-q"])
    audit_results = {}
    audit_details = {}
    for label, module in (
        ("broad", "experiments.day3.audit_broad_completion"),
        ("extension", "experiments.day3.audit_extension_completion"),
        ("distribution_shift", "experiments.day3.audit_distribution_shift"),
    ):
        result = run([sys.executable, "-m", module])
        audit_details[label] = result
        audit_results[label] = result["exit_code"] == 0
    payload = {
        "verified_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python": sys.version,
        "py_compile": syntax,
        "pytest": tests,
        "audits": audit_results,
        "audit_details": audit_details,
        "complete": bool(
            syntax["exit_code"] == 0
            and tests["exit_code"] == 0
            and all(audit_results.values())
        ),
    }
    output = RESULTS / "final_verification.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["complete"] else 1)


if __name__ == "__main__":
    main()
