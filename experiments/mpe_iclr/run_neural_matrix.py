#!/usr/bin/env python3
"""Resume-safe multi-GPU scheduler for the frozen neural benchmark matrix.

The scheduling unit is one task/split/setting/backbone bundle.  A bundle runs
the frozen ``core`` representation list sequentially so task metadata stays
warm, while independent bundles are balanced across the visible GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from neural_benchmark import available_representations
from ridge_benchmark import DEFAULT_TASKS


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "neural_cells"
LOGS = HERE / "logs" / "neural_matrix"
BACKBONES = ["mlp", "resnet", "ft_transformer", "tabm"]
SETTINGS = ["isolated_field", "full_table"]
PREFERRED = [
    "mpe",
    "similarity_same_metric",
    "similarity_unnormalized",
    "nystrom",
    "unknown_embedding",
    "q_ple",
    "uniform_ple",
    "mpe_equality",
    "ancestor_multihot",
    "path_to_root",
    "laplacian",
    "node2vec",
    "raw_coordinates",
    "coordinate_fourier",
    "spatial_rbf",
    "graph_laplacian",
    "character_3gram_hash",
]


@lru_cache(maxsize=None)
def task_representations(task: str) -> tuple[str, ...]:
    """Representation families depend on task schema, not the frozen split."""
    return tuple(available_representations(task, 0, "isolated_field"))


@lru_cache(maxsize=None)
def required_representations(task: str, split: int, setting: str) -> tuple[str, ...]:
    del split, setting
    available = task_representations(task)
    names = [name for name in PREFERRED if name in available]
    if "mpe" in available:
        names.extend(f"mpe_corrupt_{index}" for index in range(10))
    return tuple(names)


def cell_is_complete(task: str, split: int, setting: str, backbone: str, representation: str) -> bool:
    cell = f"{task}__split{split}__{setting}__{backbone}__{representation}"
    result_path = RAW / f"{cell}.json"
    state_path = RAW / f"{cell}__state_metrics.parquet"
    if not result_path.exists() or not state_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "complete":
        return False
    if representation == "mpe" or representation == "mpe_equality" or representation.startswith("mpe_corrupt_"):
        return payload.get("mpe_implementation_version") == 2
    if representation == "unknown_embedding":
        return payload.get("categorical_implementation_version") == 2
    return True


def bundle_is_complete(bundle: tuple[str, int, str, str]) -> bool:
    task, split, setting, backbone = bundle
    return all(
        cell_is_complete(task, split, setting, backbone, representation)
        for representation in required_representations(task, split, setting)
    )


def build_queue(tasks: list[str]) -> list[tuple[str, int, str, str]]:
    bundles = [
        (task, split, setting, backbone)
        for task in tasks
        for split in range(5)
        for setting in SETTINGS
        for backbone in BACKBONES
    ]
    # A fixed shuffle spreads large and small sources/backbones across workers
    # while keeping the full execution order reproducible.
    random.Random(20261829).shuffle(bundles)
    return [bundle for bundle in bundles if not bundle_is_complete(bundle)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--task", action="append", choices=DEFAULT_TASKS)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Consume the fixed pending-bundle order from the far end.",
    )
    parser.add_argument(
        "--max-bundles",
        type=int,
        help="Run only this many pending bundles (for bounded auxiliary workers).",
    )
    parser.add_argument(
        "--skip-bundles",
        type=int,
        default=0,
        help="Skip this many entries after applying the requested queue direction.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("at least one GPU is required")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be positive")
    if args.max_bundles is not None and args.max_bundles < 1:
        raise SystemExit("--max-bundles must be positive")
    if args.skip_bundles < 0:
        raise SystemExit("--skip-bundles must be nonnegative")
    tasks = args.task or list(DEFAULT_TASKS)
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    queue = build_queue(tasks)
    if args.reverse:
        queue.reverse()
    if args.skip_bundles:
        queue = queue[args.skip_bundles :]
    if args.max_bundles is not None:
        queue = queue[: args.max_bundles]
    total = len(queue)
    required_cells = sum(len(required_representations(*bundle[:3])) for bundle in queue)
    print(f"pending bundles={total} pending representation cells<={required_cells}", flush=True)
    if args.dry_run or not queue:
        return

    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()

    def run_bundle(worker_index: int, bundle: tuple[str, int, str, str]) -> tuple[tuple[str, int, str, str], int]:
        task, split, setting, backbone = bundle
        gpu = gpus[worker_index % len(gpus)]
        identifier = f"{task}__split{split}__{setting}__{backbone}"
        log_path = LOGS / f"{identifier}.log"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            sys.executable,
            str(HERE / "neural_benchmark.py"),
            "--task",
            task,
            "--split",
            str(split),
            "--setting",
            setting,
            "--backbone",
            backbone,
            "--representation",
            "core",
            "--device",
            "cuda:0",
        ]
        with log_path.open("a") as stream:
            stream.write(f"\nSCHEDULER START gpu={gpu} command={' '.join(command)}\n")
            stream.flush()
            result = subprocess.run(command, cwd=HERE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            stream.write(f"SCHEDULER EXIT code={result.returncode}\n")
        return bundle, result.returncode

    # Assign a stable worker index to each long-lived queue consumer instead of
    # letting task submission order determine the physical GPU.
    work_lock = threading.Lock()
    next_index = 0

    def consume(worker_index: int) -> list[tuple[tuple[str, int, str, str], int]]:
        nonlocal next_index, completed
        outcomes = []
        while True:
            with work_lock:
                if next_index >= len(queue):
                    break
                bundle = queue[next_index]
                next_index += 1
            outcome = run_bundle(worker_index, bundle)
            outcomes.append(outcome)
            with lock:
                completed += 1
                elapsed = time.perf_counter() - started
                task, split, setting, backbone = bundle
                print(
                    f"[{completed}/{total}] rc={outcome[1]} gpu={gpus[worker_index % len(gpus)]} "
                    f"{task} split={split} {setting} {backbone} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
        return outcomes

    workers = len(gpus) * args.workers_per_gpu
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(consume, worker_index) for worker_index in range(workers)]
        for future in as_completed(futures):
            failures.extend(bundle for bundle, code in future.result() if code != 0)
    if failures:
        print(f"failed bundles={len(failures)}; rerun is resume-safe", flush=True)
        raise SystemExit(1)
    remaining = build_queue(tasks)
    if remaining and args.max_bundles is None:
        raise SystemExit(f"scheduler exited with {len(remaining)} incomplete bundles")
    suffix = f" remaining_global_bundles={len(remaining)}" if args.max_bundles is not None else ""
    print(
        f"complete bundles={total} elapsed={(time.perf_counter() - started) / 60:.1f}m{suffix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
