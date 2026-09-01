#!/usr/bin/env python3
"""Day-09 wrapper around the hardened Day-08 four-way audit runner."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
DAY8 = ROOT.parent / "road-to-iclr-day-08"
sys.path.insert(0, str(DAY8))

from src.analysis.io import code_digest  # noqa: E402
from src.analysis.runner import build_run_snapshot, load_config, run_job, selected_jobs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e0_reproduction.yaml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--transform", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    snapshot = build_run_snapshot(ROOT, REPOSITORY, config_path)
    jobs = selected_jobs(
        config,
        datasets=set(args.dataset) if args.dataset else None,
        models=set(args.model) if args.model else None,
        transforms=set(args.transform) if args.transform else None,
        seeds=set(args.seed) if args.seed else None,
        split_seeds=None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    command = " ".join(shlex.quote(value) for value in sys.argv)
    counts = {"complete": 0, "cached": 0, "failed": 0, "unavailable": 0}
    for index, (dataset, model, transform, value, seed, split_seed) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {dataset['dataset']} {model} {transform}={value} seed={seed}",
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
            resume=True,
            command=command,
            run_snapshot=snapshot,
        )
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if record["status"] in {"failed", "unavailable"}:
            print(json.dumps({"status": record["status"], "failure": record.get("failure")}), flush=True)
    print(json.dumps(counts, sort_keys=True), flush=True)
    if code_digest(ROOT) != snapshot["code_sha256"]:
        raise RuntimeError("Day-09 source changed during E0 run")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

