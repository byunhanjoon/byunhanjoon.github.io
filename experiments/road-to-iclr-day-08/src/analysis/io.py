"""Immutable run storage and manifest helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_RUN_FIELDS = {
    "run_id",
    "status",
    "git_commit",
    "tracked_diff_sha256",
    "worktree_status_sha256",
    "code_sha256",
    "python_version",
    "package_versions",
    "model",
    "model_checkpoint",
    "dataset",
    "split_id",
    "transformation",
    "seed",
    "device",
    "wall_clock_seconds",
    "peak_gpu_memory_bytes",
    "command",
    "config",
    "result_path",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for relative in ("src", "scripts", "configs"):
        paths.extend(
            path
            for path in (root / relative).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_provenance(repository: Path) -> dict[str, str]:
    def output(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args], cwd=repository, check=True, stdout=subprocess.PIPE
        ).stdout

    commit = output("rev-parse", "HEAD").decode().strip()
    diff = output("diff", "--binary", "HEAD")
    status = output("status", "--porcelain=v1", "-z")
    return {
        "git_commit": commit,
        "tracked_diff_sha256": sha256_bytes(diff),
        "worktree_status_sha256": sha256_bytes(status),
    }


def make_run_id(job_key: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    entropy = os.urandom(5).hex()
    return f"{timestamp}__{job_key[:12]}__{entropy}"


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    missing = REQUIRED_RUN_FIELDS - set(record)
    if missing:
        raise ValueError(f"manifest record is missing fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def iter_records(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def load_completed_run(manifest: Path, job_key: str, code_sha256: str) -> dict[str, Any] | None:
    matches = [
        record
        for record in (iter_records(manifest) or [])
        if record.get("job_key") == job_key
        and record.get("code_sha256") == code_sha256
        and record.get("status") == "complete"
    ]
    if not matches:
        return None
    record = matches[-1]
    result_path = Path(record["result_path"])
    metadata_path = Path(record["metadata_path"])
    if not result_path.exists() or not metadata_path.exists():
        return None
    if sha256_file(result_path) != record["result_sha256"]:
        raise ValueError(f"cached result checksum mismatch: {result_path}")
    return record


def load_run_predictions(record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["result_path"])
    if sha256_file(path) != record["result_sha256"]:
        raise ValueError(f"result checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}
