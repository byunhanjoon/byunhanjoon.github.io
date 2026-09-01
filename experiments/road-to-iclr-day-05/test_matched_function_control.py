import copy

import numpy as np
import torch

from matched_function_control import initialize, matched_state, probabilities, transform_inputs


def test_parameter_transform_preserves_aligned_function():
    training = {"embedding_dim": 3, "hidden_dim": 5}
    numerical = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    categorical = np.asarray([[0, 1], [2, -1]], dtype=np.float64)
    view = {
        "numerical": np.asarray([1, 0]),
        "categories": [np.asarray([2, 0, 1]), np.asarray([1, 0])],
        "classes": np.asarray([1, 0]),
    }
    canonical_view = {
        "numerical": np.asarray([0, 1]),
        "categories": [np.arange(3), np.arange(2)],
        "classes": np.asarray([0, 1]),
    }
    model = initialize(17, 2, [3, 2], training)
    reference = probabilities(model, *transform_inputs(numerical, categorical, canonical_view), canonical_view["classes"])
    transformed = initialize(99, 2, [3, 2], training)
    transformed.load_state_dict(matched_state(
        copy.deepcopy(model.state_dict()), view["numerical"], view["categories"], view["classes"], 2,
    ))
    actual = probabilities(transformed, *transform_inputs(numerical, categorical, view), view["classes"])
    np.testing.assert_allclose(actual, reference, atol=1e-14, rtol=0)
    torch.testing.assert_close(torch.from_numpy(actual.sum(axis=1)), torch.ones(2, dtype=torch.float64))
