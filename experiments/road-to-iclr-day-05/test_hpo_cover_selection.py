from pathlib import Path

from analyze_hpo_cover_selection import RMS, analyze_cell


def test_saved_hpo_tensor_produces_all_declared_equal_compute_methods():
    path = next((Path(__file__).parent / "results" / "hpo_quotient").glob("*.npz"))
    rows = analyze_cell(path)
    assert {row["method"] for row in rows} == set(RMS.ALL_METHODS)
