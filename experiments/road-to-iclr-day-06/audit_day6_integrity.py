"""Audit Day-6 bundle hashes, menus, tensor shapes, and finite predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPECS = {
    "h1": {
        "config": "hypothesis_01_config.json", "protocol": "HYPOTHESIS_01_PROTOCOL.md",
        "precisions": ["fp32", "iea64"],
    },
    "h2": {
        "config": "hypothesis_02_config.json", "protocol": "HYPOTHESIS_02_PROTOCOL.md",
        "precisions": ["bfloat16", "float16", "float32", "float64"],
    },
    "h3": {
        "config": "hypothesis_03_config.json", "protocol": "HYPOTHESIS_03_PROTOCOL.md",
        "precisions": ["fp32", "iea64"],
    },
    "h4": {
        "config": "hypothesis_04_config.json", "protocol": "HYPOTHESIS_04_PROTOCOL.md",
        "precisions": None,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed_seeds(config: dict) -> set[int]:
    values = config.get("seeds", [])
    values = values + config.get("pilot_seeds", []) + config.get("confirmation_seeds", [])
    return {int(value) for value in values}


def audit_family(name: str, result_dir: Path) -> tuple[list[str], dict]:
    spec = SPECS[name]
    config_path = HERE / spec["config"]
    protocol_path = HERE / spec["protocol"]
    config = json.loads(config_path.read_text())
    expected_config_hash = digest(config_path)
    expected_protocol_hash = digest(protocol_path)
    artifacts = {path.stem: path for path in result_dir.glob("*.npz")}
    manifests = {path.stem: path for path in result_dir.glob("*.json")}
    errors = []
    if artifacts.keys() != manifests.keys():
        errors.append(
            f"{name}: orphan stems npz={sorted(artifacts.keys() - manifests.keys())} "
            f"json={sorted(manifests.keys() - artifacts.keys())}"
        )
    expected_paths = (int(config["nonidentity_views"]) + 1) * (
        len(spec["precisions"]) if spec["precisions"] is not None else 1
    )
    expected_checkpoints = np.asarray(config["checkpoints"], dtype=int)
    total_paths = 0
    for stem in sorted(artifacts.keys() & manifests.keys()):
        try:
            manifest = json.loads(manifests[stem].read_text())
            bundle = np.load(artifacts[stem])
            if manifest.get("status") != "complete":
                errors.append(f"{name}/{stem}: manifest is not complete")
            if manifest.get("dataset") not in config["datasets"]:
                errors.append(f"{name}/{stem}: dataset outside frozen menu")
            if manifest.get("model") not in config["models"]:
                errors.append(f"{name}/{stem}: model outside frozen menu")
            if int(manifest.get("seed", -1)) not in allowed_seeds(config):
                errors.append(f"{name}/{stem}: seed outside frozen menu")
            if manifest.get("config_sha256") != expected_config_hash:
                errors.append(f"{name}/{stem}: config hash mismatch")
            if manifest.get("protocol_sha256") != expected_protocol_hash:
                errors.append(f"{name}/{stem}: protocol hash mismatch")
            if int(manifest.get("paths", -1)) != expected_paths:
                errors.append(f"{name}/{stem}: manifest path count mismatch")
            checkpoints = bundle["checkpoints"].astype(int)
            if not np.array_equal(checkpoints, expected_checkpoints):
                errors.append(f"{name}/{stem}: checkpoint menu mismatch")
            validation = bundle["validation_predictions"]
            test = bundle["test_predictions"]
            labels = bundle["labels"]
            if validation.ndim != 4 or test.ndim != 4:
                errors.append(f"{name}/{stem}: prediction tensor is not rank four")
            else:
                if validation.shape[:2] != (expected_paths, len(expected_checkpoints)):
                    errors.append(f"{name}/{stem}: validation leading shape mismatch")
                if test.shape[:2] != (expected_paths, len(expected_checkpoints)):
                    errors.append(f"{name}/{stem}: test leading shape mismatch")
                if validation.shape[2] != len(bundle["validation_y"]):
                    errors.append(f"{name}/{stem}: validation row mismatch")
                if test.shape[2] != len(bundle["test_y"]):
                    errors.append(f"{name}/{stem}: test row mismatch")
            if labels.shape[0] != expected_paths:
                errors.append(f"{name}/{stem}: label path count mismatch")
            if not np.isfinite(validation).all() or not np.isfinite(test).all():
                errors.append(f"{name}/{stem}: nonfinite prediction")
            if spec["precisions"] is not None:
                observed = labels[:, 0].tolist()
                wanted = [
                    precision
                    for precision in spec["precisions"]
                    for _ in range(int(config["nonidentity_views"]) + 1)
                ]
                if observed != wanted:
                    errors.append(f"{name}/{stem}: precision/path ordering mismatch")
            total_paths += expected_paths
        except Exception as error:  # preserve all audit failures in one report
            errors.append(f"{name}/{stem}: unreadable bundle: {error}")
    return errors, {
        "bundles": len(artifacts.keys() & manifests.keys()),
        "paths": total_paths,
        "errors": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "integrity_audit_summary.json"
    )
    args = parser.parse_args()
    all_errors = []
    families = {}
    for name in SPECS:
        errors, summary = audit_family(name, args.results_root / name)
        all_errors.extend(errors)
        families[name] = summary
    result = {
        "status": "pass" if not all_errors else "fail",
        "families": families,
        "errors": all_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
