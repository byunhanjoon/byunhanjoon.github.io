"""Fixed row-random resplits for the preregistered distribution-shift audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Dataset, PARTS


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/distribution_shift_preregistered.json"


def shift_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _load_pooled(directory: Path, stem: str, *, allow_pickle: bool = False):
    paths = [directory / f"{stem}_{part}.npy" for part in PARTS]
    if not paths[0].exists():
        return None
    return np.concatenate(
        [np.load(path, allow_pickle=allow_pickle) for path in paths], axis=0
    )


def _limited(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    selected = np.sort(np.random.default_rng(seed).choice(len(indices), limit, replace=False))
    return indices[selected]


def load_random_shift_dataset(name: str) -> Dataset:
    cfg = shift_config()
    if name not in cfg["datasets"]:
        raise KeyError(f"Dataset {name!r} is absent from the frozen shift audit")
    spec = cfg["datasets"][name]
    split_cfg = cfg["resplit"]
    directory = Path(spec["path"])
    info = json.loads((directory / "info.json").read_text())
    task = str(info["task_type"]).lower()
    if task == "binary":
        task = "binclass"

    y_all = _load_pooled(directory, "y", allow_pickle=True)
    assert y_all is not None
    order = np.random.default_rng(int(split_cfg["seed"])).permutation(len(y_all))
    train_end = int(round(float(split_cfg["train_fraction"]) * len(order)))
    val_end = train_end + int(round(float(split_cfg["validation_fraction"]) * len(order)))
    raw_indices = {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }
    indices = {
        part: _limited(
            raw_indices[part],
            int(split_cfg["max_train_rows"] if part == "train" else split_cfg["max_eval_rows"]),
            int(split_cfg["subsample_seed"]) + offset,
        )
        for offset, part in enumerate(PARTS)
    }

    def split(stem: str, *, allow_pickle: bool = False):
        pooled = _load_pooled(directory, stem, allow_pickle=allow_pickle)
        if pooled is None:
            return None
        return {part: np.asarray(pooled[indices[part]]) for part in PARTS}

    x_num = split("N")
    x_cat = split("C", allow_pickle=True)
    y_raw = {part: np.asarray(y_all[indices[part]]) for part in PARTS}
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
    digest.update(str(split_cfg["kind"]).encode())
    for part in PARTS:
        digest.update(indices[part].astype(np.int64).tobytes())
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
