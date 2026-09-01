#!/usr/bin/env python3
"""Read-only integrity audit for the Day-09 artifact manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "MANIFEST.jsonl"
PATH_FIELDS = ("raw_bundle", "processed_summary")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    missing: list[str] = []
    config_mismatches: list[str] = []
    invalid_npz: list[str] = []
    episode_shape_mismatches: list[str] = []
    checked_paths: set[Path] = set()

    for record in records:
        run_key = str(record.get("run_key", "<unknown>"))
        for field in PATH_FIELDS:
            value = record.get(field)
            if not value:
                continue
            path = ROOT / str(value)
            checked_paths.add(path)
            if not path.is_file():
                missing.append(f"{run_key}:{field}:{value}")

        config = record.get("config")
        expected = record.get("config_sha256")
        if config and expected:
            path = ROOT / str(config)
            if not path.is_file():
                missing.append(f"{run_key}:config:{config}")
            elif sha256(path) != expected:
                config_mismatches.append(run_key)

        raw = record.get("raw_bundle")
        if raw and (ROOT / str(raw)).is_file() and str(raw).endswith(".npz"):
            path = ROOT / str(raw)
            try:
                with np.load(path, allow_pickle=False) as bundle:
                    for key in bundle.files:
                        array = bundle[key]
                        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                            invalid_npz.append(f"{run_key}:{key}:nonfinite")
                        episodes = record.get("episodes")
                        if episodes is not None and array.ndim > 0 and array.shape[0] != int(episodes):
                            episode_shape_mismatches.append(
                                f"{run_key}:{key}:{array.shape[0]}!={int(episodes)}"
                            )
            except Exception as exc:  # pragma: no cover - audit diagnostic
                invalid_npz.append(f"{run_key}:unreadable:{type(exc).__name__}")

    run_key_counts = Counter(str(record["run_key"]) for record in records if record.get("run_key"))
    duplicate_run_keys = sorted(key for key, count in run_key_counts.items() if count > 1)
    print(f"manifest_records={len(records)}")
    print(f"unique_referenced_artifacts={len(checked_paths)}")
    print(f"missing_artifacts={len(missing)}")
    print(f"config_hash_mismatches={len(config_mismatches)}")
    print(f"invalid_npz_arrays={len(invalid_npz)}")
    print(f"duplicate_run_keys={len(duplicate_run_keys)}")
    print(f"episode_shape_mismatches={len(episode_shape_mismatches)}")
    if missing:
        print("missing_detail=" + json.dumps(missing, sort_keys=True))
    if config_mismatches:
        print("config_mismatch_runs=" + json.dumps(config_mismatches, sort_keys=True))
    if invalid_npz:
        print("invalid_npz_detail=" + json.dumps(invalid_npz, sort_keys=True))
    if duplicate_run_keys:
        print("duplicate_run_key_detail=" + json.dumps(duplicate_run_keys, sort_keys=True))
    if episode_shape_mismatches:
        print("episode_shape_detail=" + json.dumps(episode_shape_mismatches, sort_keys=True))

    if missing or config_mismatches or invalid_npz or duplicate_run_keys or episode_shape_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
