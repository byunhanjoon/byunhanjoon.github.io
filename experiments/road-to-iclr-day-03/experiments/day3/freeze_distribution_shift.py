"""Freeze the distribution-shift protocol and source arrays pre-outcome."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .distribution_shift_data import CONFIG_PATH, shift_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def main() -> None:
    cfg = shift_config()
    protected = [
        CONFIG_PATH,
        ROOT / "experiments/day3/distribution_shift_data.py",
        ROOT / "experiments/day3/run_distribution_shift.py",
        RESULTS / "selected_hyperparameters.json",
    ]
    for spec in cfg["datasets"].values():
        directory = Path(spec["path"])
        protected.append(directory / "info.json")
        for part in ("train", "val", "test"):
            for stem in ("N", "C", "y"):
                path = directory / f"{stem}_{part}.npy"
                if path.exists():
                    protected.append(path)
    protected = sorted(set(protected), key=str)
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    aggregate = hashlib.sha256(
        "".join(f"{path}:{value}\n" for path, value in hashes.items()).encode()
    ).hexdigest()
    payload = {
        "frozen_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "timing": "before any random-split model outcome",
        "protected_files": len(hashes),
        "aggregate_sha256": aggregate,
        "sha256": hashes,
    }
    output = RESULTS / "distribution_shift_freeze.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
