import json
from pathlib import Path


def test_frozen_external_panel_has_eight_unique_sources_and_four_models():
    config = json.loads((Path(__file__).parent / "openml_external_cover_config.json").read_text())
    assert len(config["datasets"]) == len(set(config["datasets"])) == 8
    assert len(config["models"]) == 4
    assert set(config["datasets"]) == set(config["source_groups"])
