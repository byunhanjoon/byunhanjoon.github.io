#!/usr/bin/env python3
"""Build the frozen experiment registry from resume-safe raw cell payloads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATABASE = HERE / "registry.sqlite"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_rows(suite: str, path: Path, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    common = {
        "suite": suite,
        "task": payload.get("task"),
        "source_unit": payload.get("source_unit"),
        "state_split": payload.get("split"),
        "row_split": payload.get("control", "unseen_state"),
        "setting": payload.get("setting"),
        "metric": payload.get("metric", payload.get("selected_regression_bandwidth", "frozen_primary")),
        "landmarks": payload.get("representation_metadata", {}).get("landmark_state_ids"),
        "bandwidth": payload.get("bandwidth", payload.get("selected_bandwidth", payload.get("primary_bandwidth"))),
        "status": payload.get("status", "unknown"),
        "payload_path": str(path.relative_to(HERE)),
        "payload_sha256": digest(path),
    }
    if payload.get("representation") is not None:
        representation = str(payload["representation"])
        corruption = representation.rsplit("_", 1)[-1] if representation.startswith("mpe_corrupt_") else None
        results = payload.get("results") or [{}]
        for result in results:
            yield {
                **common,
                "metric_corruption": corruption,
                "representation": representation,
                "backbone": payload.get("backbone"),
                "hyperparameters": payload.get("selected_config", payload.get("selected_trial")),
                "seed": result.get("seed"),
            }
        return
    if isinstance(payload.get("results"), list):
        for result in payload["results"]:
            representation = str(result.get("representation", suite))
            corruption = representation.rsplit("_", 1)[-1] if representation.startswith("mpe_corrupt_") else None
            yield {
                **common,
                "metric_corruption": corruption,
                "representation": representation,
                "backbone": payload.get("backbone", payload.get("model", "ridge")),
                "hyperparameters": result.get("selected_alpha", result.get("selected_k")),
                "seed": result.get("seed"),
            }
        return
    representation = "mpe_features" if payload.get("with_mpe") else payload.get("representation", "native_or_diagnostic")
    yield {
        **common,
        "metric_corruption": None,
        "representation": representation,
        "backbone": payload.get("backbone", payload.get("model", suite)),
        "hyperparameters": payload.get("selected_trial"),
        "seed": payload.get("seed"),
    }


def main() -> None:
    protocol_digest = digest(HERE / "PROTOCOL_HASHES.txt")
    source_digest = digest(RAW / "source_checksums.json")
    rows = []
    for folder in sorted(RAW.glob("*_cells")):
        suite = folder.name.removesuffix("_cells")
        for path in sorted(folder.glob("*.json")):
            payload = json.loads(path.read_text())
            for row in expanded_rows(suite, path, payload):
                key_material = json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "suite", "task", "state_split", "row_split", "setting", "metric",
                            "metric_corruption", "representation", "landmarks", "bandwidth",
                            "backbone", "hyperparameters", "seed",
                        )
                    },
                    sort_keys=True,
                    default=str,
                )
                row["cell_key"] = hashlib.sha256(key_material.encode()).hexdigest()
                row["protocol_digest"] = protocol_digest
                row["source_manifest_digest"] = source_digest
                rows.append(row)

    connection = sqlite3.connect(DATABASE)
    with connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS registry;
            CREATE TABLE registry (
                cell_key TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                task TEXT,
                source_unit TEXT,
                state_split TEXT,
                row_split TEXT,
                setting TEXT,
                metric TEXT,
                metric_corruption TEXT,
                representation TEXT,
                landmarks_json TEXT,
                bandwidth TEXT,
                backbone TEXT,
                hyperparameters_json TEXT,
                seed TEXT,
                status TEXT NOT NULL,
                payload_path TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                protocol_digest TEXT NOT NULL,
                source_manifest_digest TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO registry VALUES (
                :cell_key, :suite, :task, :source_unit, :state_split, :row_split,
                :setting, :metric, :metric_corruption, :representation,
                :landmarks_json, :bandwidth, :backbone, :hyperparameters_json,
                :seed, :status, :payload_path, :payload_sha256,
                :protocol_digest, :source_manifest_digest
            )
            """,
            [
                {
                    **row,
                    "state_split": None if row.get("state_split") is None else str(row["state_split"]),
                    "metric": None if row.get("metric") is None else str(row["metric"]),
                    "landmarks_json": json.dumps(row.get("landmarks"), sort_keys=True, default=str),
                    "bandwidth": None if row.get("bandwidth") is None else str(row["bandwidth"]),
                    "hyperparameters_json": json.dumps(row.get("hyperparameters"), sort_keys=True, default=str),
                    "seed": None if row.get("seed") is None else str(row["seed"]),
                }
                for row in rows
            ],
        )
        connection.execute("CREATE INDEX registry_task_idx ON registry(task, state_split, setting)")
        connection.execute("CREATE INDEX registry_method_idx ON registry(backbone, representation)")
    connection.close()
    print(f"registry rows={len(rows)} path={DATABASE}", flush=True)


if __name__ == "__main__":
    main()
