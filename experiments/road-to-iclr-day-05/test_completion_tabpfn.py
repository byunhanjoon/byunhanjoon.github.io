import numpy as np

from completion_tabpfn import render_tabpfn


class Tiny:
    x_num = {"train": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    x_cat = {"train": np.asarray([[1]], dtype=np.int64)}


def test_tabpfn_render_tracks_categorical_position_and_mapping():
    values, categorical = render_tabpfn(
        Tiny(), "train", np.asarray([2, 0, 1]), [np.asarray([1, 0])]
    )
    np.testing.assert_array_equal(values, np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32))
    assert categorical == (0,)
