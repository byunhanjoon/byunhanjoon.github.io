"""Strict requirement-by-requirement completion audit for the Day-6 goal."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KST = dt.timezone(dt.timedelta(hours=9))
EARLIEST_FINISH = dt.datetime(2026, 8, 29, 6, 21, tzinfo=KST)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results" / "completion_audit_summary.json")
    args = parser.parse_args()
    now = dt.datetime.now(KST)
    checks: list[dict] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "pass": bool(passed), "evidence": evidence})

    check(
        "seven_hour_wall_clock_horizon",
        now >= EARLIEST_FINISH,
        {"now": now.isoformat(), "earliest_finish": EARLIEST_FINISH.isoformat()},
    )

    summary_specs = {
        "h1_confirmation": ("h1_summary.json", "confirmation_complete", 72),
        "h2_falsification": ("h2_summary.json", "complete", 18),
        "h3_fullscale": ("h3_summary.json", "complete", 36),
        "h4_shadow": ("h4_summary.json", "complete", 324),
        "h5_transfer": ("h5_summary.json", "complete", 324),
        "h6_screen": ("h6_summary.json", "complete", 33),
        "h7_survival": ("h7_summary.json", "complete", 31),
        "h8_acceleration": ("h8_summary.json", "complete", 29),
        "h9_attenuation": ("h9_summary.json", "complete", 25),
    }
    for name, (filename, status, artifacts) in summary_specs.items():
        value = read_json(HERE / "results" / filename)
        observed_artifacts = None if value is None else value.get(
            "artifacts", value.get("prospective_bundles")
        )
        check(
            name,
            value is not None
            and value.get("status") == status
            and observed_artifacts == artifacts,
            {
                "file": filename,
                "status": None if value is None else value.get("status"),
                "artifacts": observed_artifacts,
                "required_status": status,
                "required_artifacts": artifacts,
            },
        )

    integrity = read_json(HERE / "results" / "integrity_audit_summary.json")
    required_integrity = {"h1": 72, "h2": 18, "h3": 36, "h4": 324}
    observed_integrity = {} if integrity is None else integrity.get("families", {})
    integrity_pass = bool(integrity and integrity.get("status") == "pass")
    for family, bundles in required_integrity.items():
        integrity_pass = integrity_pass and (
            observed_integrity.get(family, {}).get("bundles") == bundles
            and observed_integrity.get(family, {}).get("errors") == 0
        )
    check(
        "artifact_integrity_all_families",
        integrity_pass,
        {"required_bundles": required_integrity, "observed": observed_integrity},
    )

    reproducibility = read_json(HERE / "results" / "analysis_reproducibility_summary.json")
    check(
        "analysis_outputs_reproduce_exactly",
        bool(
            reproducibility
            and reproducibility.get("status") == "pass"
            and reproducibility.get("mismatches") == []
            and reproducibility.get("outputs") == 28
        ),
        reproducibility,
    )

    figure_stems = [
        "h3_orbit_trajectories", "h4_semantic_shadow_forecast",
        "h5_cross_perturbation_transfer", "h6_forecast_comparison",
        "h7_material_survival", "h7_material_survival_by_dataset",
        "h8_level_acceleration_screen", "h9_postbreach_attenuation",
    ]
    figure_state = {
        f"{stem}.{suffix}": (
            (HERE / "results" / "figures" / f"{stem}.{suffix}").exists()
            and (HERE / "results" / "figures" / f"{stem}.{suffix}").stat().st_size > 0
        )
        for stem in figure_stems for suffix in ("png", "pdf")
    }
    check("all_final_figures_exist", all(figure_state.values()), figure_state)

    for number in range(1, 10):
        protocol = HERE / f"HYPOTHESIS_{number:02d}_PROTOCOL.md"
        check(f"h{number}_frozen_protocol", protocol.exists(), protocol.name)

    configs = []
    matrix_ok = True
    for number in (1, 2, 3, 4):
        config = read_json(HERE / f"hypothesis_{number:02d}_config.json")
        configs.append({
            "hypothesis": number,
            "datasets": None if config is None else config.get("datasets"),
            "models": None if config is None else config.get("models"),
        })
        matrix_ok = matrix_ok and bool(
            config and len(config.get("datasets", [])) == 3
            and len(config.get("models", [])) == 3
        )
    check("three_dataset_three_model_validation", matrix_ok, configs)

    required_reports = [
        "H1_CONFIRMATION_REPORT.md", "H2_FINAL_REPORT.md", "H3_FINAL_REPORT.md",
        "H4_FINAL_REPORT.md", "H5_FINAL_REPORT.md", "H6_FINAL_REPORT.md",
        "H7_FINAL_REPORT.md", "H8_FINAL_REPORT.md", "H9_FINAL_REPORT.md",
        "DAY6_FINAL_REPORT.md",
    ]
    report_state = {}
    for name in required_reports:
        path = HERE / name
        text = path.read_text() if path.exists() else ""
        report_state[name] = bool(
            path.exists()
            and "{{" not in text
            and "Status: **DRAFT" not in text
        )
    check("all_hypothesis_and_final_reports", all(report_state.values()), report_state)

    foundational = [
        "THEORY_FOUNDATIONS.md", "RECENT_LITERATURE_AUDIT.md",
        "REVIEWER_ATTACK_AUDIT.md", "FINAL_RANKING_PROTOCOL.md",
        "FINAL_RANKING_ADDENDUM_H8.md",
        "FINAL_RANKING_ADDENDUM_H9.md",
        "ORBITCOVER_INCUMBENT_AUDIT.md",
        "STATISTICAL_SCOPE_AUDIT.md",
        "FINAL_REPORT_CHECKLIST.md",
        "EXTERNAL_CONFIRMATION_ROADMAP.md",
        "IDEA_LEDGER.md", "RESEARCH_LOG.md",
    ]
    foundation_state = {name: (HERE / name).exists() for name in foundational}
    check("theory_novelty_and_ranking_artifacts", all(foundation_state.values()), foundation_state)

    result = {
        "status": "complete" if all(item["pass"] for item in checks) else "incomplete",
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
