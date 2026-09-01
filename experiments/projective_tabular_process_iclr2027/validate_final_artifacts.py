#!/usr/bin/env python3
"""Fail closed unless the complete experiment and paper artifact set is valid."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import CACHE, CONFIG, HERE, atomic_json


RESULTS = CACHE / "results"
PAPER = HERE / "paper" / "iclr2027"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_count(stage: str) -> int:
    if stage == "eval":
        return (
            len(CONFIG["evaluation_tasks"])
            * len(CONFIG["evaluation_folds"])
            * int(CONFIG["context_replicates"])
            * len(CONFIG["context_sizes"])
        )
    datasets = (
        CONFIG["development_datasets"]
        if stage == "dev"
        else CONFIG["application_datasets"]
    )
    return (
        len(datasets)
        * len(CONFIG["development_splits"])
        * int(CONFIG["development_context_replicates"])
        * len(CONFIG["context_sizes"])
    )


def file_set(root: Path, stage: str) -> set[str]:
    return {path.name for path in (root / stage).glob("*.npz")}


def metadata(data: Any) -> dict[str, Any]:
    return json.loads(str(data["metadata"].item()))


def validate_cache(
    label: str,
    root: Path,
    stage: str,
    reference: set[str],
    required: set[str],
) -> dict[str, Any]:
    paths = sorted((root / stage).glob("*.npz"))
    actual = {path.name for path in paths}
    if actual != reference:
        missing = sorted(reference - actual)[:10]
        extra = sorted(actual - reference)[:10]
        raise AssertionError(f"{label}/{stage} filename mismatch: missing={missing}, extra={extra}")
    elapsed = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise AssertionError(f"{path} lacks {required - set(data.files)}")
            meta = metadata(data)
            for key in required - {"metadata"}:
                if not np.isfinite(data[key]).all():
                    raise AssertionError(f"non-finite {key} in {path}")
            if "elapsed_seconds" in meta:
                value = meta["elapsed_seconds"]
                if isinstance(value, list):
                    elapsed.extend(float(item) for item in value)
                else:
                    elapsed.append(float(value))
    return {
        "label": label,
        "stage": stage,
        "files": len(paths),
        "median_elapsed_seconds": float(np.median(elapsed)) if elapsed else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def validate_failure_logs(prefix: str, stage: str) -> list[str]:
    paths = sorted((CACHE / "logs").glob(f"{prefix}_{stage}_shard*.json"))
    if not paths:
        raise AssertionError(f"no logs for {prefix}/{stage}")
    for path in paths:
        payload = load_json(path)
        if payload.get("failures"):
            raise AssertionError(f"recorded failures in {path}: {len(payload['failures'])}")
    return [str(path) for path in paths]


def validate_shared_marginal_control() -> dict[str, float]:
    cells = pd.read_parquet(RESULTS / "projective_singleton" / "cells.parquet")
    keys = ["dataset", "split", "replicate", "context_size", "family", "group"]
    subset = cells[cells["method"].isin(["projtabicl", "tabiclv2_diagonal"])]
    wide = subset.pivot(index=keys, columns="method", values=["target", "mean", "squared_error"])
    maxima = {}
    for quantity in ("target", "mean", "squared_error"):
        delta = wide[(quantity, "projtabicl")] - wide[(quantity, "tabiclv2_diagonal")]
        maxima[quantity] = float(delta.abs().max())
        if maxima[quantity] != 0.0:
            raise AssertionError(f"fixed-marginal control mismatch in {quantity}: {maxima[quantity]}")
    return maxima


def validate_manifest() -> dict[str, Any]:
    path = RESULTS / "paper_manifest.json"
    payload = load_json(path)
    for record in payload["generated_files"]:
        artifact = Path(record["path"])
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        observed = sha256_file(artifact)
        if observed != record["sha256"]:
            raise AssertionError(f"manifest hash mismatch: {artifact}")
    return {"path": str(path), "sha256": sha256_file(path), "files": len(payload["generated_files"])}


def validate_paper() -> dict[str, Any]:
    pdf = PAPER / "projective_tfm.pdf"
    aux = PAPER / "projective_tfm.aux"
    if not pdf.exists() or not aux.exists():
        raise FileNotFoundError("compiled paper PDF/AUX missing")
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if not match:
        raise AssertionError("could not read PDF page count")
    aux_text = aux.read_text(errors="replace")
    main_match = re.search(
        r"\\newlabel\{page:main-end\}\{\{[^}]*\}\{(\d+)\}", aux_text
    )
    if not main_match:
        raise AssertionError("main-page label absent from AUX")
    main_pages = int(main_match.group(1))
    if main_pages > 9:
        raise AssertionError(f"ICLR main text exceeds nine pages: {main_pages}")
    return {
        "path": str(pdf),
        "sha256": sha256_file(pdf),
        "bytes": pdf.stat().st_size,
        "total_pages": int(match.group(1)),
        "main_text_last_page": main_pages,
    }


def main() -> None:
    source_root = CACHE / "tabicl_singleton_episodes"
    cache_records: list[dict[str, Any]] = []
    stage_specs = {
        "dev": [
            ("tabpfn25", CACHE / "tabpfn_episodes", {"mean", "variance", "metadata"}),
            ("tabpfn3", CACHE / "tabpfn3_episodes", {"mean", "variance", "metadata"}),
        ],
        "eval": [
            ("tabpfn25", CACHE / "tabpfn_episodes", {"mean", "variance", "metadata"}),
            ("tabpfn3", CACHE / "tabpfn3_episodes", {"mean", "variance", "metadata"}),
            ("tabdpt", CACHE / "tabdpt_episodes", {"mean", "metadata"}),
            ("classical", CACHE / "classical_episodes", {"means", "covariances", "metadata"}),
        ],
        "app": [
            ("tabpfn25", CACHE / "tabpfn_episodes", {"mean", "variance", "metadata"}),
            ("tabpfn3", CACHE / "tabpfn3_episodes", {"mean", "variance", "metadata"}),
            ("tabdpt", CACHE / "tabdpt_episodes", {"mean", "metadata"}),
            ("classical", CACHE / "classical_episodes", {"means", "covariances", "metadata"}),
        ],
    }
    for stage, specs in stage_specs.items():
        reference = file_set(source_root, stage)
        if len(reference) != expected_count(stage):
            raise AssertionError(f"singleton source count for {stage}: {len(reference)}")
        cache_records.append(
            validate_cache(
                "tabiclv2-singleton",
                source_root,
                stage,
                reference,
                {"mean", "variance", "hidden", "target", "metadata"},
            )
        )
        for label, root, required in specs:
            cache_records.append(validate_cache(label, root, stage, reference, required))

    logs = []
    for prefix in ("extract_tabpfn", "extract_tabpfn3"):
        for stage in ("dev", "eval", "app"):
            logs.extend(validate_failure_logs(prefix, stage))
    for prefix in ("extract_tabdpt",):
        for stage in ("eval", "app"):
            logs.extend(validate_failure_logs(prefix, stage))

    projective = load_json(RESULTS / "projective_singleton" / "summary.json")
    baselines = load_json(RESULTS / "baselines_singleton" / "summary.json")
    applications = load_json(RESULTS / "applications_singleton" / "summary.json")
    audit = load_json(RESULTS / "projectivity_audit_singleton" / "summary.json")
    if projective["query_mode"] != "singleton" or projective["episode_count"] != expected_count("eval"):
        raise AssertionError("projective summary has the wrong mode/count")
    if baselines["query_mode"] != "singleton" or baselines["evaluation_episodes"] != expected_count("eval"):
        raise AssertionError("baseline summary has the wrong mode/count")
    if applications["episodes"] != expected_count("app"):
        raise AssertionError("application summary has the wrong count")
    if not audit["passes"] or audit["episodes"] != 3 * len(CONFIG["evaluation_tasks"]):
        raise AssertionError("singleton projectivity audit did not pass")
    if projective["integrity"]["max_diagonal_abs"] > 1e-10:
        raise AssertionError("marginal preservation failed")
    if projective["integrity"]["minimum_eigenvalue"] < -1e-7:
        raise AssertionError("projective covariance PSD check failed")

    payload = {
        "status": "PASS",
        "protocol_sha256": sha256_file(HERE / "PROTOCOL.md"),
        "config_sha256": sha256_file(HERE / "config.json"),
        "cache_records": cache_records,
        "failure_logs": logs,
        "shared_marginal_control_maxima": validate_shared_marginal_control(),
        "singleton_audit_maximum": max(map(float, audit["maxima"].values())),
        "paper_manifest": validate_manifest(),
        "paper": validate_paper(),
    }
    atomic_json(RESULTS / "final_validation.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
