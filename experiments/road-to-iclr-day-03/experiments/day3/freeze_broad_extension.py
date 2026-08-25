"""Freeze extension code, protocol, hyperparameters, and source arrays pre-outcome."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .broad_extension_data import CONFIG_PATH, extension_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = extension_config()
    protected = [
        CONFIG_PATH,
        ROOT / "experiments/day3/broad_extension_data.py",
        ROOT / "experiments/day3/run_broad_extension.py",
        RESULTS / "selected_hyperparameters.json",
    ]
    for spec in cfg["datasets"].values():
        directory = Path(spec["path"])
        protected.append(directory / "info.json")
        for part in ("train", "val", "test"):
            for stem in (spec["numeric_stem"], spec["categorical_stem"], "y"):
                if stem is not None:
                    protected.append(directory / f"{stem}_{part}.npy")
    protected = sorted(set(protected), key=str)
    missing = [str(path) for path in protected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing extension files: {missing}")
    hashes = {str(path.resolve()): digest(path) for path in protected}
    aggregate = hashlib.sha256(
        "".join(f"{path}:{value}\n" for path, value in hashes.items()).encode()
    ).hexdigest()
    payload = {
        "frozen_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "timing": "before any extension model outcome",
        "protected_files": len(hashes),
        "aggregate_sha256": aggregate,
        "sha256": hashes,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "broad_extension_freeze.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
