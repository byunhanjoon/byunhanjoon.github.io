"""Hash, shape, finiteness, menu, and summary audit for this Day-6 track."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pilot_config.json"
RESULTS = HERE / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    errors = []
    pilot_count = h6_count = pilot_paths = h6_paths = h6_interventions = 0
    elapsed = 0.0
    for domain in config["domains"]:
        for regime in config["regimes"]:
            for seed in config["seeds"]:
                stem = f"{domain}__{regime}__seed{seed}"
                artifact = RESULTS / "pilot" / f"{stem}.npz"
                manifest_path = RESULTS / "pilot" / f"{stem}.json"
                if not artifact.exists() or not manifest_path.exists():
                    errors.append(f"missing pilot {stem}")
                    continue
                manifest = json.loads(manifest_path.read_text())
                bundle = np.load(artifact, allow_pickle=False)
                if manifest.get("artifact_sha256") != sha256(artifact):
                    errors.append(f"hash pilot {stem}")
                if manifest.get("config_sha256") != sha256(CONFIG_PATH):
                    errors.append(f"config pilot {stem}")
                if bundle["predictions"].shape != (6, 5, 1024):
                    errors.append(f"shape pilot {stem}")
                if bundle["patch_predictions"].shape != (2, 5, 5, 1024):
                    errors.append(f"patch shape {stem}")
                for name in ("predictions", "initial_tables", "final_tables", "test_target"):
                    if not np.isfinite(bundle[name]).all():
                        errors.append(f"nonfinite {name} {stem}")
                pilot_count += 1
                pilot_paths += int(manifest["paths"])
                elapsed += float(manifest["elapsed_seconds"])
    for domain in config["domains"]:
        for seed in config["seeds"]:
            stem = f"{domain}__seed{seed}"
            artifact = RESULTS / "h6" / f"{stem}.npz"
            manifest_path = RESULTS / "h6" / f"{stem}.json"
            if not artifact.exists() or not manifest_path.exists():
                errors.append(f"missing h6 {stem}")
                continue
            manifest = json.loads(manifest_path.read_text())
            bundle = np.load(artifact, allow_pickle=False)
            if manifest.get("artifact_sha256") != sha256(artifact):
                errors.append(f"hash h6 {stem}")
            if manifest.get("config_sha256") != sha256(CONFIG_PATH):
                errors.append(f"config h6 {stem}")
            if bundle["predictions"].shape != (2, 5, 5, 1024):
                errors.append(f"shape h6 {stem}")
            if not np.isfinite(bundle["predictions"]).all():
                errors.append(f"nonfinite h6 {stem}")
            h6_count += 1
            h6_paths += int(manifest["trained_paths"])
            h6_interventions += int(manifest["interventions"])
            elapsed += float(manifest["elapsed_seconds"])
    pilot_summary = json.loads((RESULTS / "pilot_summary.json").read_text())
    h6_summary = json.loads((RESULTS / "h6_summary.json").read_text())
    if pilot_summary.get("path_count") != 720:
        errors.append("pilot summary paths")
    if h6_summary.get("trained_path_count") != 120:
        errors.append("h6 summary paths")
    if pilot_count != 24 or pilot_paths != 720:
        errors.append("pilot completeness")
    if h6_count != 12 or h6_paths != 120 or h6_interventions != 600:
        errors.append("h6 completeness")
    summary = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "pilot_bundles": pilot_count,
        "pilot_paths": pilot_paths,
        "h6_bundles": h6_count,
        "h6_replayed_paths": h6_paths,
        "h6_interventions": h6_interventions,
        "summed_fit_seconds": elapsed,
    }
    (RESULTS / "integrity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
