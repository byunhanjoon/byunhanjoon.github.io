"""Rerun final analyzers and require byte-identical machine-readable outputs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import analyze_cross_perturbation as h5
import analyze_fullscale_arithmetic as h3
import analyze_fullscale_dynamics as h3d
import analyze_postbreach_attenuation as h9
import analyze_rounding_survival as h7
import analyze_semantic_acceleration as h8
import analyze_semantic_lyapunov as h6
import analyze_semantic_shadow as h4

HERE = Path(__file__).resolve().parent
SPECS = [
    ("h3", h3, "h3", ["h3_summary.json", "h3_cells.csv", "h3_trajectories.csv", "h3_references.csv", "h3_timing.csv", "h3_reference_loss_pairs.csv"]),
    ("h3_dynamics", h3d, "h3", ["h3_dynamics_summary.json", "h3_dynamics_paths.csv", "h3_dynamics_bundles.csv"]),
    ("h4", h4, "h4", ["h4_summary.json", "h4_seed_bundles.csv", "h4_config_cells.csv"]),
    ("h5", h5, "h4", ["h5_summary.json", "h5_seed_scores.csv", "h5_config_cells.csv", "h5_correlations.csv", "h5_top_quartile_auc.csv"]),
    ("h6", h6, "h3", ["h6_summary.json", "h6_prospective_bundles.csv"]),
    ("h7", h7, "h3", ["h7_summary.json", "h7_survival_pairs.csv", "h7_dataset_summary.csv"]),
    ("h8", h8, "h3", ["h8_summary.json", "h8_prospective_bundles.csv"]),
    ("h9", h9, "h3", ["h9_summary.json", "h9_prospective_pairs.csv", "h9_dataset_summary.csv", "h9_canonical_loss_pairs.csv"]),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results" / "analysis_reproducibility_summary.json",
    )
    args = parser.parse_args()
    all_outputs = [name for _, _, _, names in SPECS for name in names]
    missing = [name for name in all_outputs if not (args.results_root / name).exists()]
    if missing:
        result = {"status": "fail", "missing_outputs": missing, "mismatches": []}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    before = {name: digest(args.results_root / name) for name in all_outputs}
    analyzer_log = {}
    for name, module, input_family, _ in SPECS:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            module.analyze(args.results_root / input_family, args.results_root)
        analyzer_log[name] = captured.getvalue()[-500:]
    after = {name: digest(args.results_root / name) for name in all_outputs}
    mismatches = [name for name in all_outputs if before[name] != after[name]]
    result = {
        "status": "pass" if not mismatches else "fail",
        "analyzers": [name for name, _, _, _ in SPECS],
        "outputs": len(all_outputs),
        "mismatches": mismatches,
        "sha256": after,
        "captured_log_tails": analyzer_log,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
