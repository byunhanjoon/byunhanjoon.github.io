"""Record the exact raw-manifest and derived-artifact state just regenerated."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import closure_core as core


def digest_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(core.HERE)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    manifests = list(core.RAW.glob("*/*/manifest.json"))
    artifacts = []
    for relative in ("summaries", "figures", "tables"):
        artifacts.extend(path for path in (core.HERE / relative).glob("*") if path.is_file())
    payload = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_manifest_count": len(manifests),
        "raw_manifest_digest": digest_paths(manifests),
        "derived_artifact_count": len(artifacts),
        "derived_artifact_digest": digest_paths(artifacts),
        "pipeline": [
            "analyze_experiment_a.py", "analyze_experiment_a_classical.py",
            "analyze_experiment_b.py", "analyze_experiment_c.py",
            "analyze_experiment_d.py", "make_final_claims.py",
        ],
    }
    core.save_json_atomic(core.HERE / "regeneration_record.json", payload)


if __name__ == "__main__":
    main()
