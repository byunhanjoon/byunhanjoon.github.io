import json

import numpy as np

from experiments.day3.analyze_external_measure_orbit import average_predictions
from experiments.day3.external_measure_orbit import (
    ROOT,
    config,
    load_external_dataset,
    verify_locked_sources,
)


def test_external_panel_is_disjoint_from_all_measure_orbit_data():
    external = set(config()["datasets"])
    screen = json.loads(
        (ROOT / "experiments/day3/configs/measure_orbit_preregistered.json").read_text()
    )
    confirmation = json.loads(
        (
            ROOT
            / "experiments/day3/configs/selective_measure_orbit_preregistered.json"
        ).read_text()
    )
    prior = set(screen["screen"]["datasets"])
    prior.update(confirmation["confirmation"]["broad_datasets"])
    prior.update(confirmation["confirmation"]["extension_datasets"])
    assert external.isdisjoint(prior)
    assert config()["chronology"]["panel_overlap_with_prior_measure_orbit_datasets"] == 0


def test_locked_external_sources_still_match_freeze():
    verify_locked_sources()


def test_external_loaders_match_frozen_tasks_and_have_numeric_input():
    for name, specification in config()["datasets"].items():
        dataset = load_external_dataset(name)
        assert dataset.task == specification["task"]
        assert dataset.x_num is not None
        assert dataset.x_num["train"].shape[1] > 0
        assert len(dataset.y["train"]) > 0
        assert len(dataset.split_fingerprint) == 16


def test_prediction_averaging_uses_probabilities_for_classification():
    binary = average_predictions(
        "binclass", [np.array([[-4.0], [2.0]]), np.array([[1.0], [-2.0]])]
    )
    expected_probability = (
        1.0 / (1.0 + np.exp(-np.array([-4.0, 2.0])))
        + 1.0 / (1.0 + np.exp(-np.array([1.0, -2.0])))
    ) / 2.0
    assert np.allclose(1.0 / (1.0 + np.exp(-binary[:, 0])), expected_probability)

    multiclass = average_predictions(
        "multiclass",
        [np.array([[3.0, 0.0]]), np.array([[0.0, 3.0]])],
    )
    assert np.allclose(np.exp(multiclass).sum(axis=1), 1.0)
    assert np.allclose(np.exp(multiclass), np.array([[0.5, 0.5]]))


def test_prediction_averaging_is_arithmetic_for_regression():
    prediction = average_predictions(
        "regression", [np.array([[1.0], [3.0]]), np.array([[3.0], [1.0]])]
    )
    assert np.array_equal(prediction, np.array([[2.0], [2.0]]))
