"""Bridge frozen TabArena official splits to the Day 1/2 benchmark objects."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent.parent / "road-to-iclr-day-01"
DEFAULT_JPLE = Path("/home/byunhanjoon/2027ICLR/projects/multi_ple/jple_tabarena")
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DEFAULT_JPLE))

import real_data_benchmark as benchmark  # noqa: E402


def to_dataset(split) -> benchmark.Dataset:
    parts_num = {
        "train": split.x_num_train,
        "val": split.x_num_val,
        "test": split.x_num_test,
    }
    parts_cat = {
        "train": split.x_cat_train,
        "val": split.x_cat_val,
        "test": split.x_cat_test,
    }
    if split.problem_type == "regression":
        target = {
            "train": split.y_train.astype(np.float64) * split.y_std + split.y_mean,
            "val": split.y_val.astype(np.float64),
            "test": split.y_test.astype(np.float64),
        }
        task = "regression"
    elif split.problem_type == "binary":
        target = {
            "train": split.y_train.astype(np.float32),
            "val": split.y_val.astype(np.float32),
            "test": split.y_test.astype(np.float32),
        }
        task = "binclass"
    else:
        raise ValueError(
            f"The current binary/regression bridge does not support {split.problem_type!r}"
        )
    return benchmark.Dataset(
        name=split.dataset,
        task=task,
        x_num=parts_num,
        x_bin=None,
        x_cat=parts_cat if split.x_cat_train.shape[1] else None,
        y=target,
    )
