from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_core as core
from closure_designs import (
    assert_strength,
    mechanism_design,
    sample_schema_design,
    strength2_schema_base,
    trajectory_strength3,
)
from analysis_utils import full_factor_components, markdown_text
from run_experiment_b_bundle import owns_path


def test_frozen_protocol_hashes() -> None:
    config = core.load_config()
    assert config["status"] == "frozen_before_final_closure_outcomes"


def test_master_seed_domains_are_unique_and_deterministic() -> None:
    masters = [core.stable_seed("test", index) for index in range(1024)]
    assert len(masters) == len(set(masters))
    first = core.derive_subseeds(masters[0])
    assert first == core.derive_subseeds(masters[0])
    assert len(first) == len(set(first.values())) == 6


def test_schema_strength2_balance_and_unique_srs() -> None:
    cards = (4, 4, 2)
    base = strength2_schema_base(cards)
    assert len(base) == 16
    assert_strength(base, cards, 2)
    rng = np.random.default_rng(7)
    srs = sample_schema_design("SRS-JOINT", cards, 16, rng)
    assert len(np.unique(srs, axis=0)) == 16


def test_trajectory_strength3_classification_and_regression() -> None:
    classification = trajectory_strength3((4, 4, 2, 8))
    regression = trajectory_strength3((4, 4, 1, 8))
    assert classification.shape == regression.shape == (128, 4)
    assert_strength(classification, (4, 4, 2, 8), 3)
    assert_strength(regression, (4, 4, 1, 8), 3)
    assert len(np.unique(classification, axis=0)) == 128
    assert len(np.unique(regression, axis=0)) == 128
    collapsed_classification = trajectory_strength3((4, 1, 2, 8))
    collapsed_regression = trajectory_strength3((4, 1, 1, 8))
    assert_strength(collapsed_classification, (4, 1, 2, 8), 3)
    assert_strength(collapsed_regression, (4, 1, 1, 8), 3)
    assert len(np.unique(collapsed_classification, axis=0)) == 64
    assert len(np.unique(collapsed_regression, axis=0)) == 32


def test_mechanism_equal_budget_and_declared_balance() -> None:
    cards = (4, 4, 2, 4, 4)
    rng = np.random.default_rng(11)
    methods = core.CONFIG["experiment_d"]["methods"]
    for method in methods:
        design = mechanism_design(method, cards, rng)
        assert design.shape == (16, 5)
    all_factors = mechanism_design("all_factors", cards, np.random.default_rng(9))
    assert_strength(all_factors, cards, 2)


def test_canonical_method_definition_never_changes_schema() -> None:
    canonical = np.zeros((64, 3), dtype=np.int16)
    assert np.count_nonzero(canonical) == 0


def test_config_has_all_mandatory_cells_and_outputs() -> None:
    config = json.loads((Path(__file__).parent / "final_closure_config.json").read_text())
    assert len(config["all_datasets"]) == 12
    assert len(config["primary_models"]) == 4
    assert len(config["split_seeds"]) == 3
    assert config["experiment_a"]["budgets"] == [4, 8, 16, 32, 64]
    assert len(config["experiment_b"]["datasets"]) == 6
    assert config["required_figures"] == 10
    assert len(config["required_tables"]) == 5


def test_schema_transforms_preserve_row_geometry() -> None:
    config = core.completion_config()
    data = core.completion.prepare("australian_credit_approval", 2026082801, config)
    design = core.completion.views(data, config)
    canonical, _ = core.completion.render(
        data, "train", design["feature"][0], design["category"][0]
    )
    canonical_gram = canonical @ canonical.T
    for feature in design["feature"]:
        for category in design["category"]:
            transformed, coordinate_map = core.completion.render(data, "train", feature, category)
            assert sorted(coordinate_map.tolist()) == list(range(transformed.shape[1]))
            np.testing.assert_allclose(
                transformed @ transformed.T, canonical_gram, rtol=2e-5, atol=1e-5
            )


def test_nested_training_subsets_and_no_partition_leakage() -> None:
    config = core.completion_config()
    numeric, categorical, target = core.completion.raw_local(
        "bank_marketing_subscription", config
    )
    _, encoded = np.unique(target.astype(str), return_inverse=True)
    split = core.full_split_indices(encoded, "classification", 2026082801)
    order = core.nested_training_indices(split["train"], encoded, "classification", 2026082801)
    assert set(order) == set(split["train"])
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["test"]))
    small = set(order[:2048]); medium = set(order[:8192])
    assert small.issubset(medium)
    completion_small = core.completion.prepare(
        "bank_marketing_subscription", 2026082801, config
    )
    prepared, _ = core.b_prepared_datasets(
        "bank_marketing_subscription", 2026082801, config
    )
    b_small = prepared[list(prepared)[0]]
    for part in core.completion.PARTS:
        np.testing.assert_array_equal(completion_small.x_num[part], b_small.x_num[part])
        np.testing.assert_array_equal(completion_small.x_cat[part], b_small.x_cat[part])
        np.testing.assert_array_equal(completion_small.y[part], b_small.y[part])


def test_fanova_reconstruction_on_known_full_product() -> None:
    cards = (4, 4, 2)
    rows = np.asarray(list(np.ndindex(cards)), dtype=float)
    values = (
        rows[:, 0:1]
        + 2 * rows[:, 1:2]
        + 3 * rows[:, 2:3]
        + (rows[:, 0:1] - 1.5) * (rows[:, 1:2] - 1.5)
    )
    total = float(np.mean((values - values.mean(axis=0)) ** 2))
    energies = full_factor_components(values, cards)
    assert abs(sum(energies.values()) - total) < 1e-12
    assert energies[(0, 1)] > 0
    assert energies.get((0, 1, 2), 0.0) < 1e-12


def test_markdown_renderer_has_no_optional_dependency() -> None:
    rendered = markdown_text(
        pd.DataFrame({"name": ["left|right", "line\nbreak"], "value": [1.25, np.nan]})
    )
    assert "left\\|right" in rendered
    assert "line<br>break" in rendered
    assert "| 1.25 |" in rendered


def test_b_path_shards_are_disjoint_and_complete() -> None:
    assignments = [
        {index for index in range(257) if owns_path(index, shard, 3)}
        for shard in range(3)
    ]
    assert set.union(*assignments) == set(range(257))
    assert not any(assignments[left] & assignments[right] for left in range(3) for right in range(left))


def test_completed_a_pool_has_unique_independent_seeds_and_alignment() -> None:
    paths = sorted((core.RAW / "experiment_a").glob("*/manifest.json"))
    if not paths:
        return
    maximum_budget = max(core.CONFIG["experiment_a"]["budgets"])
    for manifest_path in paths:
        path = manifest_path.parent
        joint = np.load(path / "joint_master_seeds.npy")
        canonical = np.load(path / "canonical_master_seeds.npy")
        combined = np.concatenate((joint.reshape(-1)[joint.shape[1] :], canonical))
        assert len(combined) == len(np.unique(combined))
        assert joint.size >= 4 * maximum_budget
        np.testing.assert_array_equal(joint[0], canonical[: joint.shape[1]])
        manifest = json.loads(manifest_path.read_text())
        predictions = np.load(path / "joint_test.npy", mmap_mode="r")
        canonical_predictions = np.load(path / "canonical_test.npy", mmap_mode="r")
        core.validate_probabilities(predictions, manifest["task"])
        assert predictions.shape[2] == len(np.load(path / "test_y.npy"))
        np.testing.assert_array_equal(predictions[0], canonical_predictions[: joint.shape[1]])
