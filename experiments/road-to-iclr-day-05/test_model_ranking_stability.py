import numpy as np


def test_pairwise_ranking_agreement_identity_and_reverse():
    target = np.arange(5)
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    same = np.mean([(target[i] < target[j]) == (target[i] < target[j]) for i, j in pairs])
    reverse = target[::-1]
    opposite = np.mean([(reverse[i] < reverse[j]) == (target[i] < target[j]) for i, j in pairs])
    assert same == 1.0
    assert opposite == 0.0
