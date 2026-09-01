#!/usr/bin/env python3
"""Run resume-safe prospective units with bounded local parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guarded_basis.common import load_protocol, prospective_specs  # noqa: E402


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "COMPLETE"
    except (json.JSONDecodeError, OSError):
        return False


def run(job: dict[str, Any]) -> tuple[dict[str, Any], str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / job["script"]),
        "--dataset", job["dataset"],
        "--model", job["model"],
        "--seed", str(job["seed"]),
        "--device", job["device"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(
            f"prospective unit failed ({job['dataset']}/{job['model']}/seed{job['seed']}):\n"
            + result.stdout[-8000:]
        )
    return job, result.stdout[-1000:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["general", "embedding"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    specs, finalists, digest = prospective_specs()
    protocol = load_protocol()
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    if args.kind == "general":
        models = list(protocol["general_models"])
        script = "run_guarded_prospective.py"
        unit_root = ROOT / "results" / "raw" / "prospective" / "units"
    else:
        models = ["controlled_mlp", "tabm_d", "resnet_tabular"]
        script = "run_embedding_prospective.py"
        unit_root = ROOT / "results" / "raw" / "prospective" / "embedding_units"
    if args.model is not None:
        if args.model not in models:
            raise RuntimeError(f"model {args.model} is not part of {args.kind} matrix")
        models = [args.model]
    seeds = [int(value) for value in protocol["prospective_seeds"]]
    jobs: list[dict[str, Any]] = []
    # Model/seed-major ordering puts all distinct datasets first, warming each
    # OpenML cache before many jobs for the same dataset accumulate.
    for model_index, model in enumerate(models):
        for seed in seeds:
            for index, spec in enumerate(specs):
                destination = unit_root / model / spec["key"] / f"seed_{seed}.json"
                if complete(destination):
                    print(f"[matrix skip] {args.kind} {spec['key']} {model} seed={seed}", flush=True)
                    continue
                jobs.append(
                    {
                        "script": script,
                        "dataset": spec["key"],
                        "model": model,
                        "seed": seed,
                        "device": f"cuda:{(index + model_index + seed) % 2}",
                        "destination": str(destination),
                    }
                )
    print(
        f"[matrix start] kind={args.kind} jobs={len(jobs)} workers={args.workers} "
        f"finalist_sha256={digest}",
        flush=True,
    )
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            job = future_map[future]
            try:
                finished, output = future.result()
                print(
                    f"[matrix complete] {args.kind} {finished['dataset']} {finished['model']} seed={finished['seed']}\n"
                    + output,
                    flush=True,
                )
            except Exception as error:  # preserve all other independent jobs
                failures.append(str(error))
                print(f"[matrix failure] {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} prospective units failed; rerun is cache-resumable")
    print(f"[matrix done] kind={args.kind} jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
