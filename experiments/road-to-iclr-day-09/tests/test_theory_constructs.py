import numpy as np


def binary_log_loss(probability: float, truth_probability: float) -> float:
    return -truth_probability * np.log(probability) - (1 - truth_probability) * np.log(1 - probability)


def test_t5_informative_metadata_matched_router_can_be_worse_than_fixed():
    delta, amplitude = 0.1, 0.49
    matched = binary_log_loss(0.5 + amplitude, 0.5 + delta)
    fixed = np.log(2.0)
    assert delta > 0
    assert matched > fixed
    assert binary_log_loss(0.5 + delta, 0.5 + delta) < fixed


def test_t6_convex_mixture_can_beat_every_individual():
    y = np.asarray([0.5, 0.5])
    first = np.asarray([0.0, 0.0])
    second = np.asarray([1.0, 1.0])
    mixture = 0.5 * first + 0.5 * second
    losses = [np.mean((y - prediction) ** 2) for prediction in (first, second)]
    assert np.mean((y - mixture) ** 2) < min(losses)


def test_t7_prediction_shrinkage_obeys_linear_harm_bound():
    y = np.asarray([0.0, 1.0, 1.0, 0.0])
    fixed = np.asarray([0.2, 0.7, 0.6, 0.3])
    adaptive = np.asarray([0.01, 0.99, 0.2, 0.8])
    amount = 0.1

    def log_loss(prediction):
        return np.mean(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))

    shrunk = (1 - amount) * fixed + amount * adaptive
    assert log_loss(shrunk) - log_loss(fixed) <= amount * (
        log_loss(adaptive) - log_loss(fixed)
    ) + 1e-15
