#!/usr/bin/env python3
"""Verify immutable bundle hashes, frozen inputs, and final stage coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_bundle(metadata_path: Path, config_hash: str, panel_hash: str) -> dict:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise RuntimeError(f"incomplete bundle: {metadata_path}")
    if metadata.get("config_sha256") != config_hash or metadata.get("dataset_panel_sha256") != panel_hash:
        raise RuntimeError(f"frozen input drift: {metadata_path}")
    verified = []
    for filename, record in metadata.get("files", {}).items():
        path = metadata_path.parent / filename
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash/size failure: {path}")
        verified.append(filename)
    if not verified:
        raise RuntimeError(f"bundle has no hashed artifacts: {metadata_path}")
    return {
        "metadata": str(metadata_path.relative_to(ROOT)), "stage": metadata.get("stage"),
        "verified_files": len(verified), "model": metadata.get("model"),
        "dataset": metadata.get("dataset_spec", {}).get("key"), "model_seed": metadata.get("model_seed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-development-only", action="store_true")
    args = parser.parse_args()
    config_hash = sha256(ROOT / "configs" / "development_protocol.yaml")
    panel_hash = sha256(ROOT / "configs" / "dataset_panel.json")
    stage_patterns = {
        "replication": "raw/development/replication/*/*/seed_*/metadata.json",
        "natural": "raw/development/natural/*/*/seed_*/metadata.json",
        "repairs": "raw/development/repairs/*/*/seed_*/metadata.json",
        "consistency": "raw/development/consistency/*/*/seed_*/metadata.json",
        "mechanism": "raw/development/mechanism/*/seed_*/member_*/*/metadata.json",
        "equal_hpo": "raw/development/equal_hpo/*/*/seed_*/metadata.json",
        "prospective": "raw/prospective/evaluation/*/*/seed_*/metadata.json",
    }
    expected = {
        "replication": 165, "natural": 165, "repairs": 165, "consistency": 33,
        "mechanism": 240, "equal_hpo": 18, "prospective": 84,
    }
    records = []
    counts = {}
    for stage, pattern in stage_patterns.items():
        paths = sorted(RESULTS.glob(pattern))
        counts[stage] = len(paths)
        records.extend(audit_bundle(path, config_hash, panel_hash) for path in paths)
    required = dict(expected)
    if args.allow_development_only:
        required.pop("prospective")
    failures = {stage: {"observed": counts[stage], "expected": count}
                for stage, count in required.items() if counts[stage] != count}
    method_hash = None
    method_path = ROOT / "configs" / "FROZEN_METHOD_CONFIG.json"
    if not args.allow_development_only:
        if not method_path.exists():
            failures["frozen_method"] = "missing"
        else:
            method_hash = sha256(method_path)
            lock = json.loads((RESULTS / "PROSPECTIVE_LOCK.json").read_text())
            if lock.get("frozen_method_config_sha256") != method_hash:
                failures["frozen_method"] = "hash mismatch"
            for record in records:
                if record["stage"] == "prospective_evaluation":
                    metadata = json.loads((ROOT / record["metadata"]).read_text())
                    if metadata.get("frozen_method_config_sha256") != method_hash:
                        failures.setdefault("prospective_method_hash", []).append(record["metadata"])
    report = {
        "status": "pass" if not failures else "fail", "counts": counts,
        "expected": expected, "audited_bundles": len(records), "failures": failures,
        "development_protocol_sha256": config_hash, "dataset_panel_sha256": panel_hash,
        "frozen_method_config_sha256": method_hash, "bundles": records,
    }
    output = RESULTS / "integrity_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "bundles"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
