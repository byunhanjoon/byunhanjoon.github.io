"""Audit timing telemetry and benchmark only the design-generation overhead."""

from __future__ import annotations

import json
import time
from pathlib import Path

from analyze_disjoint_pair_cross import cover_graph
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_mixed_resolvable_packing import SHAPE, mixed_coset_resolution


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DIRECTORIES = (
    "tier1_confirmation", "tier1_menu_repeat", "tier1_subsample_repeat",
    "openml_external_cover", "openml_taskbalanced_cover", "openml_multiclass_cover",
)
TIMING_TOKENS = ("elapsed", "duration", "wall_time", "walltime", "cpu_time", "seconds")


def timing_paths(value, prefix="") -> list[str]:
    output = []
    if isinstance(value, dict):
        for key, current in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if any(token in key.lower() for token in TIMING_TOKENS):
                output.append(path)
            output.extend(timing_paths(current, path))
    elif isinstance(value, list):
        for index, current in enumerate(value):
            output.extend(timing_paths(current, f"{prefix}[{index}]"))
    return output


def timed(function, repetitions: int = 5) -> list[float]:
    values = []
    for _ in range(repetitions):
        start = time.perf_counter(); function(); values.append(time.perf_counter() - start)
    return values


def main() -> None:
    manifests, with_timing, timing_fields = 0, 0, set()
    for directory in DIRECTORIES:
        for path in (RESULTS / directory).glob("*.json"):
            manifests += 1
            fields = timing_paths(json.loads(path.read_text()))
            if fields:
                with_timing += 1; timing_fields.update(fields)
    # Warm the cached graph before measuring action sampling.
    graph_times = timed(lambda: cover_graph.cache_clear() or cover_graph(SHAPE), 3)
    cover_graph(SHAPE)
    pack_times = timed(lambda: sample_pack_and_pairs(SHAPE, "systems-audit", "timing"), 7)
    resolution_times = timed(mixed_coset_resolution, 20)
    summary = {
        "status": "complete", "principal_manifests_scanned": manifests,
        "manifests_with_timing_telemetry": with_timing,
        "timing_fields_found": sorted(timing_fields),
        "median_cold_cover_graph_seconds": sorted(graph_times)[len(graph_times) // 2],
        "median_1024_pack_actions_seconds": sorted(pack_times)[len(pack_times) // 2],
        "microseconds_per_pack_action": 1e6 * sorted(pack_times)[len(pack_times) // 2] / 1_024,
        "median_mixed_resolution_seconds": sorted(resolution_times)[len(resolution_times) // 2],
        "end_to_end_wallclock_claim_supported": bool(with_timing == manifests),
        "frozen_interpretation": "latency_supported" if with_timing == manifests else "fit_budget_only",
    }
    (RESULTS / "systems_efficiency_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
