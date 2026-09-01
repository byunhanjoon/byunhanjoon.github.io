import numpy as np

from completion_classical_panel import render_ordinal


class Tiny:
    x_num = {"train": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    x_cat = {"train": np.asarray([[1]], dtype=np.int64)}


def test_ordinal_render_preserves_category_identity_under_map():
    values, categorical = render_ordinal(Tiny(), "train", np.asarray([2, 1, 0]), [np.asarray([1, 0])])
    np.testing.assert_array_equal(values, np.asarray([[0.0, 2.0, 1.0]], dtype=np.float32))
    assert categorical == (0,)
