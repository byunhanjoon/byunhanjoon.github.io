"""Capture code, protocol, and data hashes before broad benchmark execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .broad_data import CONFIG_PATH, DATA_ROOT, FINANCE_ROOT, config


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/day3/broad_benchmark/broad_freeze.json"
PROTECTED = [
    ROOT / "BROAD_BENCHMARK_PROTOCOL.md",
    CONFIG_PATH,
    ROOT / "experiments/day3/broad_data.py",
    ROOT / "experiments/day3/broad_models.py",
    ROOT / "experiments/day3/run_broad_benchmark.py",
    ROOT / "experiments/day3/run_broad_calibration.sh",
    ROOT / "experiments/day3/run_broad_matrix_calibration.sh",
    ROOT / "experiments/day3/analyze_broad_calibration.py",
    ROOT / "experiments/day3/run_broad_phase1.py",
    ROOT / "experiments/day3/analyze_broad_phase1.py",
    ROOT / "experiments/day3/run_broad_confirmation.py",
    ROOT / "experiments/day3/analyze_broad_final.py",
    ROOT / "experiments/day3/run_broad_robustness.py",
    ROOT / "experiments/day3/freeze_broad_benchmark.py",
    ROOT / "THEORY_DAY3.md",
    ROOT / "tests/test_broad_benchmark.py",
    ROOT / "tests/test_optimizer_remedies.py",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def data_files(name: str) -> list[Path]:
    if (DATA_ROOT / name).is_dir():
        directory = DATA_ROOT / name
        paths = [directory / "info.json", directory / "y.npy"]
        paths.extend(directory / f"{stem}.npy" for stem in ("x_num", "x_bin", "x_cat") if (directory / f"{stem}.npy").exists())
        paths.extend(directory / "splits/default" / f"{part}.npy" for part in ("train", "val", "test"))
        return paths
    directory = FINANCE_ROOT / name
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix in (".json", ".npy"))


def main() -> None:
    files = PROTECTED.copy()
    for name in config()["datasets"]:
        files.extend(data_files(name))
    hashes = {str(path.resolve()): digest(path) for path in files}
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "captured_after_validation_calibration_code_freeze_and_before_broad_tiers",
        "files": len(hashes),
        "sha256": hashes,
        "aggregate_sha256": aggregate,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in ("captured_at", "files", "aggregate_sha256")}, indent=2))


if __name__ == "__main__":
    main()
