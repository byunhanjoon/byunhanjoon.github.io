import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("heterobag_decomp_tested", HERE / "analyze_heterobag_decomposition.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ambiguity_identity_and_gain_decomposition():
    target = np.asarray([0.2, -0.4, 0.8])
    control = np.asarray([[0.0, -0.1, 0.9], [0.4, -0.5, 0.5], [0.3, -0.2, 0.7]])
    candidate = control.copy()
    candidate[2] = np.asarray([0.1, -0.6, 0.9])
    c_member, c_loss, c_ambiguity = MODULE.ensemble_terms(control, target)
    h_member, h_loss, h_ambiguity = MODULE.ensemble_terms(candidate, target)
    assert np.isclose(c_loss - h_loss, (c_member - h_member) + (h_ambiguity - c_ambiguity))
