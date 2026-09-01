from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("highdim", HERE / "highdim_field_cover.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_oa32_supports_german_factor_count() -> None:
    binary = 14  # class plus 13 categorical fields
    design = MODULE.base_oa(binary)
    assert design.shape == (32, 16)
    MODULE.assert_pairwise_balance(design, (4, 4) + (2,) * binary)


def test_binary_form_capacity() -> None:
    assert len(MODULE.binary_forms(25)) == 25

