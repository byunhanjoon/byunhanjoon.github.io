from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "road-to-iclr-day-04"
sys.path.insert(0, str(DAY4))

from heterobag_mechanism import transformed_t_inputs
from semantic_multiview_pilot import SplitData, ple_basis
from support_identity_transfer_pilot import Encodings


def test_reversed_field_placebo_preserves_tple_basis_up_to_field_order() -> None:
    values = np.array([[0.25, 12.0], [0.75, 18.0]], dtype=np.float32)
    edges = np.array([[0.0, 0.5, 1.0], [10.0, 15.0, 20.0]], dtype=np.float32)
    data = SplitData(
        x_num={part: values.copy() for part in ("train", "val", "test")},
        x_bin=None,
        x_cat=None,
        y={part: np.zeros(2, dtype=np.float32) for part in ("train", "val", "test")},
        y_mean=0.0,
        y_scale=1.0,
        category_cardinalities=[],
        cyclic_columns=[],
        cyclic_names=[],
        cyclic_periods=[],
        cyclic_origins=[],
        split_sizes_full={part: 2 for part in ("train", "val", "test")},
    )
    empty_codes = {part: np.empty((2, 0), dtype=np.int64) for part in data.x_num}
    encoding = Encodings(edges, edges.copy(), [], [], empty_codes, empty_codes)
    transformed_data, transformed_encoding = transformed_t_inputs(data, encoding)
    original = ple_basis(torch.from_numpy(values), torch.from_numpy(edges))
    transformed = ple_basis(
        torch.from_numpy(transformed_data.x_num["train"]),
        torch.from_numpy(transformed_encoding.tple_edges),
    )
    torch.testing.assert_close(transformed, original.flip(1))
