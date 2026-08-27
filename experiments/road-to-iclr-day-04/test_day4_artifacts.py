"""Regression checks for the quantitative claims in the Day 4 write-up."""

from __future__ import annotations

import csv
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

import pytest


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPOSITORY = HERE.parents[1]
PUBLIC_POST = REPOSITORY / "blogposts" / "road-to-iclr-day-04.html"
BLOG_INDEX = REPOSITORY / "blog.html"
SITEMAP = REPOSITORY / "sitemap.xml"
DAY4_POST = HERE / "day4.md"
DIRECTION_FREEZE = HERE / "ICLR_DIRECTION_FREEZE.md"
NOVELTY_REVIEW = HERE / "PORTFOLIO_NOVELTY_REVIEW.md"
NONFINITE_TEXT = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}
)


class _PostAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.json_ld: list[object] = []
        self._in_json_ld = False
        self._json_buffer: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        if attributes.get("href"):
            self.hrefs.append(str(attributes["href"]))
        if attributes.get("src"):
            self.srcs.append(str(attributes["src"]))
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self._in_json_ld = True
            self._json_buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_json_ld:
            self.json_ld.append(
                json.loads(
                    "".join(self._json_buffer),
                    object_pairs_hook=unique_json_object,
                )
            )
            self._in_json_ld = False


def load_summary() -> dict[str, object]:
    return json.loads((RESULTS / "day4_summary.json").read_text())


def assert_finite_json(value: object, path: Path) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), path
    elif isinstance(value, dict):
        for nested in value.values():
            assert_finite_json(nested, path)
    elif isinstance(value, list):
        for nested in value:
            assert_finite_json(nested, path)


def unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def test_complete_direct_panel_headline() -> None:
    summary = load_summary()
    assert summary["portfolio_decision"] == {
        "primary": "OrbitANOVA",
        "primary_readiness_out_of_5": 3.5,
        "primary_status": (
            "ICLR-shaped, but requires the frozen broad audit and held-out "
            "audit-guided action transfer before submission."
        ),
        "secondary": "FieldRiesz",
        "secondary_status": (
            "High-risk secondary method or targeted chart-covariant intervention "
            "inside OrbitANOVA; not a standalone submission today."
        ),
    }
    comparisons = summary["residual_vs_raple_pair_wins"]
    assert comparisons["raple_raw"]["pairs"] == 45
    assert comparisons["raple_raw"]["wins"] == 17
    assert comparisons["raple_raw"]["mean_gain_pct"] == pytest.approx(
        -0.14982434634105551
    )
    assert comparisons["anchor_only"]["wins"] == 23
    assert comparisons["anchor_mass_representer"]["wins"] == 29
    assert comparisons["anchor_wrong_representer"]["wins"] == 33
    assert "stress evidence rather than a confirmatory test" in summary[
        "performance_hypothesis"
    ]


def test_selected_case_controls_and_external_replication() -> None:
    with (RESULTS / "day4_isospectral_cell_average_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    overall = next(row for row in rows if row["model"] == "all")
    assert int(overall["unique_cells"]) == 18
    assert int(overall["correct_wins"]) == 18
    assert float(overall["mean_correct_gain_pct"]) == pytest.approx(
        0.9028122685589529
    )

    summary = load_summary()
    king = summary["king_county_spatial"]
    assert sum(row["pairs"] for row in king) == 9
    assert sum(row["wins_vs_raple_raw"] for row in king) == 2
    assert sum(row["gain_vs_raple_raw_pct"] for row in king) / 3 == pytest.approx(
        0.0723346826, abs=1e-10
    )
    king_controls = summary["king_county_controls"]
    assert sum(row["wins_vs_isospectral"] for row in king_controls) == 7
    assert sum(row["gain_vs_isospectral_pct"] for row in king_controls) / 3 == pytest.approx(
        -0.0287240688, abs=1e-10
    )


def test_reference_mass_audit_reverses_the_easy_control_claim() -> None:
    summary = load_summary()
    california = sorted(
        (
            row
            for row in summary["spatial_reference_mass"]
            if row["dataset"] == "california"
        ),
        key=lambda row: row["reference_mass_weight"],
    )
    assert [row["reference_mass_weight"] for row in california] == [0.001, 0.01, 0.1]
    assert all(row["completed_mass_rank"] == 144 for row in california)
    gains = [row["gain_vs_anchor_product_isospectral_pct"] for row in california]
    assert gains[0] < 0 < gains[1]
    assert gains[2] < 0

    rotations = {
        row["dataset"]: row
        for row in summary["spatial_reference_rotations"]
        if row["control"] == "within-cell-mean"
    }
    assert rotations["california"]["semantic_wins"] == 7
    assert rotations["california"]["mean_semantic_gain_pct"] == pytest.approx(
        0.0386565069
    )
    assert rotations["king-county-sales"]["semantic_wins"] == 6


def test_synthetic_completion_is_only_a_designed_sanity_check() -> None:
    synthetic = load_summary()["synthetic_reference_completion"]
    assert synthetic["design"]["train_support"] == (
        "Uniform([0,0.4]) union Uniform([0.6,1])"
    )
    assert synthetic["ranks"]["completed_mass_rank"] == [25]
    gap = synthetic["regions"]["unobserved-gap"]
    assert gap["empirical_correct"]["correct_wins"] == 192
    assert gap["empirical_correct"]["mean_correct_gain_pct"] == pytest.approx(
        18.333054712382484
    )
    assert gap["completed_wrong"]["correct_wins"] == 200
    assert gap["completed_isospectral"]["correct_wins"] == 200


def test_all_machine_readable_artifacts_are_structurally_valid() -> None:
    csv_paths = sorted(HERE.rglob("*.csv"))
    json_paths = sorted(HERE.rglob("*.json"))
    assert len(csv_paths) == 134
    assert len(json_paths) == 70

    row_count = 0
    for path in csv_paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames, path
            assert len(reader.fieldnames) == len(set(reader.fieldnames)), path
            expected = set(reader.fieldnames)
            for row in reader:
                assert set(row) == expected, path
                normalized_values = {
                    value.strip().lower()
                    for value in row.values()
                    if value is not None
                }
                assert normalized_values.isdisjoint(NONFINITE_TEXT), path
                row_count += 1
    assert row_count == 19_104

    for path in json_paths:
        payload = json.loads(path.read_text(), object_pairs_hook=unique_json_object)
        assert_finite_json(payload, path)


def test_public_post_metadata_and_local_links() -> None:
    parser = _PostAudit()
    html = PUBLIC_POST.read_text()
    parser.feed(html)

    assert len(parser.ids) == len(set(parser.ids))
    assert len(parser.json_ld) == 1
    metadata = parser.json_ld[0]
    assert isinstance(metadata, dict)
    assert_finite_json(metadata, PUBLIC_POST)
    assert metadata["headline"] == "A Feature Is a Domain"
    assert "OrbitANOVA remains the primary" in html
    assert "FieldRiesz" in html
    assert "Across all 45, semantic Riesz beats raw RAPLE in only 17" in html
    assert "80/90 control comparisons (+0.90%)" in html
    assert "California and Weather were selected after the broad pilot" in html

    missing = []
    for reference in [*parser.hrefs, *parser.srcs]:
        parsed = urlsplit(reference)
        if (
            parsed.scheme
            or reference.startswith(("#", "mailto:"))
            or not parsed.path
        ):
            continue
        target = (PUBLIC_POST.parent / parsed.path).resolve()
        if not target.exists():
            missing.append(reference)
    assert not missing


def test_public_post_is_listed_in_blog_and_sitemap() -> None:
    relative = "blogposts/road-to-iclr-day-04.html"
    assert relative in BLOG_INDEX.read_text()

    tree = ElementTree.parse(SITEMAP)
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    listed = [node.text for node in tree.findall(".//s:loc", namespace)]
    assert len(listed) == len(set(listed))
    locations = set(listed)
    assert "https://byunhanjoon.github.io/" + relative in locations


def test_markdown_claims_match_the_frozen_portfolio_decision() -> None:
    day4 = DAY4_POST.read_text()
    freeze = DIRECTION_FREEZE.read_text()
    review = NOVELTY_REVIEW.read_text()

    for document in (day4, freeze, review):
        assert "OrbitANOVA" in document
        assert "FieldRiesz" in document
        assert "17/45" in document

    assert "3.5/5 ready" in review
    assert "weak reject" in review
    assert "80/90" in day4
    assert "18/18" in day4
    assert "80/90" in freeze
    assert "18/18" in freeze
    assert "not a confirmatory p-value" in freeze


def test_markdown_local_links_resolve() -> None:
    markdown_paths = sorted(HERE.glob("*.md"))
    assert len(markdown_paths) == 7

    missing: list[tuple[Path, str]] = []
    for path in markdown_paths:
        for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#") or not parsed.path:
                continue
            if not (path.parent / parsed.path).resolve().exists():
                missing.append((path, target))
    assert not missing
