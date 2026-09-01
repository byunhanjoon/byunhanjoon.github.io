#!/usr/bin/env python3
"""Run resumable matched/mismatch reparameterization jobs from a frozen config."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.io import code_digest  # noqa: E402
from src.analysis.runner import build_run_snapshot, load_config, run_job, selected_jobs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--transform", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--split-seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config_path = args.config.resolve()
    config = load_config(config_path)
    run_snapshot = build_run_snapshot(ROOT, REPOSITORY, config_path)
    jobs = selected_jobs(
        config,
        datasets=set(args.dataset) if args.dataset else None,
        models=set(args.model) if args.model else None,
        transforms=set(args.transform) if args.transform else None,
        seeds=set(args.seed) if args.seed else None,
        split_seeds=set(args.split_seed) if args.split_seed else None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    command = " ".join(shlex.quote(x) for x in sys.argv)
    counts = {"complete": 0, "cached": 0, "failed": 0, "unavailable": 0}
    for index, (dataset, model, transform, value, seed, split_seed) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {dataset['dataset']} {model} {transform}={value} "
            f"seed={seed} split_seed={split_seed}",
            flush=True,
        )
        record = run_job(
            root=ROOT,
            repository=REPOSITORY,
            config_path=config_path,
            config=config,
            dataset_spec=dataset,
            model=model,
            transform_name=transform,
            transform_value=value,
            seed=seed,
            split_seed=split_seed,
            device=args.device,
            resume=not args.no_resume,
            command=command,
            run_snapshot=run_snapshot,
        )
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if record["status"] in {"failed", "unavailable"}:
            print(json.dumps({"status": record["status"], "failure": record.get("failure")}), flush=True)
            if args.fail_fast and record["status"] == "failed":
                raise SystemExit(1)
    print(json.dumps(counts, sort_keys=True), flush=True)
    ending_code_sha256 = code_digest(ROOT)
    if ending_code_sha256 != run_snapshot["code_sha256"]:
        print(
            json.dumps(
                {
                    "status": "source_changed_during_run",
                    "starting_code_sha256": run_snapshot["code_sha256"],
                    "ending_code_sha256": ending_code_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
