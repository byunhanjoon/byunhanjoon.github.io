import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("strength3_tested", HERE / "analyze_strength3_cover.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mixed_strength3_binary_classification():
    design = MODULE.strength3_base(4, 2, 4)
    assert design.shape == (64, 4)
    MODULE.S2.assert_strength(design, (4, 4, 2, 4), 3)


def test_strength3_singleton_class_and_category():
    for cardinalities in ((4, 4, 1, 4), (4, 2, 2, 4), (4, 1, 2, 4), (4, 1, 1, 4), (4, 1, 4, 4)):
        design = MODULE.strength3_base(cardinalities[1], cardinalities[2], cardinalities[3])
        MODULE.S2.assert_strength(design, cardinalities, 3)
