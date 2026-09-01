import copy

import numpy as np
import torch

from completion_neural_panel import matched_state, output_keys


def test_dense_stem_and_output_matching_mlp():
    state = {
        "first.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "network.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "network.2.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "network.2.bias": torch.asarray([10.0, 20.0]),
    }
    actual = matched_state("mlp", copy.deepcopy(state), np.asarray([2, 0, 3, 1]), np.asarray([1, 0]))
    torch.testing.assert_close(actual["first.weight"], state["first.weight"][:, [2, 0, 3, 1]])
    torch.testing.assert_close(actual["network.0.weight"], actual["first.weight"])
    torch.testing.assert_close(actual["network.2.weight"], state["network.2.weight"][[1, 0]])
    assert output_keys("mlp", state) == ("network.2.weight", "network.2.bias")
