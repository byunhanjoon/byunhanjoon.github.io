"""Data loader for the frozen five-dataset prospective breadth extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Dataset, PARTS


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/broad_extension_preregistered.json"


def extension_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _indices(length: int, limit: int, seed: int) -> np.ndarray:
    values = np.arange(length, dtype=np.int64)
    if length <= limit:
        return values
    return np.sort(np.random.default_rng(seed).choice(values, limit, replace=False))


def load_extension_dataset(name: str) -> Dataset:
    cfg = extension_config()
    if name not in cfg["datasets"]:
        raise KeyError(f"Dataset {name!r} is absent from the frozen extension")
    spec = cfg["datasets"][name]
    directory = Path(spec["path"])
    info = json.loads((directory / "info.json").read_text())
    task = str(info["task_type"]).lower()
    if task == "binary":
        task = "binclass"
    if task not in ("binclass", "multiclass", "regression"):
        raise ValueError(f"Unsupported task {task!r} for {name}")

    indices: dict[str, np.ndarray] = {}
    for offset, part in enumerate(PARTS):
        length = len(np.load(directory / f"y_{part}.npy", mmap_mode="r", allow_pickle=True))
        limit = int(cfg["max_train_rows"] if part == "train" else cfg["max_eval_rows"])
        indices[part] = _indices(length, limit, int(cfg["sample_seed"]) + offset)

    def arrays(stem: str | None, *, allow_pickle: bool = False):
        if stem is None or not (directory / f"{stem}_train.npy").exists():
            return None
        return {
            part: np.asarray(
                np.load(
                    directory / f"{stem}_{part}.npy",
                    mmap_mode=None if allow_pickle else "r",
                    allow_pickle=allow_pickle,
                )[indices[part]]
            )
            for part in PARTS
        }

    x_num = arrays(spec["numeric_stem"])
    x_cat = arrays(spec["categorical_stem"], allow_pickle=True)
    y_raw = arrays("y", allow_pickle=True)
    assert y_raw is not None
    if task == "regression":
        y = {part: values.astype(np.float32) for part, values in y_raw.items()}
        n_classes = 1
    else:
        classes = sorted(set(y_raw["train"].tolist()), key=str)
        lookup = {value: index for index, value in enumerate(classes)}
        y = {
            part: np.asarray([lookup[value] for value in values.tolist()], dtype=np.int64)
            for part, values in y_raw.items()
        }
        n_classes = 2 if task == "binclass" else len(classes)

    digest = hashlib.sha256()
    digest.update(str(directory.resolve()).encode())
    for part in PARTS:
        digest.update(indices[part].tobytes())
    return Dataset(
        name=name,
        task=task,
        x_num=x_num,
        x_bin=None,
        x_cat=x_cat,
        y=y,
        n_classes=n_classes,
        split_fingerprint=digest.hexdigest()[:16],
    )
