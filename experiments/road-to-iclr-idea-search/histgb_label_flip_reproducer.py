"""Minimal class-ID equivariance failure in sklearn native categorical HistGB.

The fifth category has fewer than the categorical splitter's fixed support
threshold.  Swapping binary class IDs should complement the fitted predictor,
but the low-support category is forced to one side of the candidate splits.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder


def data(rare_count: int = 9) -> tuple[np.ndarray, np.ndarray]:
    counts = (80, 80, 80, 80, rare_count)
    positive_rates = (0.05, 0.25, 0.50, 0.75, 1.00)
    x, y = [], []
    for category, (count, rate) in enumerate(zip(counts, positive_rates)):
        positive_count = round(count * rate)
        x.extend([[category]] * count)
        y.extend([1] * positive_count + [0] * (count - positive_count))
    return np.asarray(x, dtype=np.float64), np.asarray(y)


def aligned_gap(mode: str, rare_count: int = 9) -> dict[str, object]:
    x, y = data(rare_count)
    categorical_features = [0] if mode == "native" else None
    if mode == "onehot":
        x = OneHotEncoder(sparse_output=False).fit_transform(x)
    model_options = dict(
        categorical_features=categorical_features,
        max_iter=5,
        max_leaf_nodes=5,
        min_samples_leaf=2,
        learning_rate=0.3,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=0,
    )
    reference = HistGradientBoostingClassifier(**model_options).fit(x, y)
    flipped = HistGradientBoostingClassifier(**model_options).fit(x, 1 - y)
    reference_probabilities = reference.predict_proba(x)
    aligned_flipped_probabilities = flipped.predict_proba(x)[:, ::-1]
    result: dict[str, object] = {
        "maximum_probability_gap": float(
            np.max(np.abs(reference_probabilities - aligned_flipped_probabilities))
        ),
        "mean_squared_probability_gap": float(
            np.mean(
                np.sum(
                    (reference_probabilities - aligned_flipped_probabilities) ** 2,
                    axis=-1,
                )
            )
        ),
    }
    if mode == "native":
        def root_left_categories(model: HistGradientBoostingClassifier) -> list[int]:
            bitset = model._predictors[0][0].raw_left_cat_bitsets[0]
            return [
                category
                for category in range(5)
                if int(bitset[category // 32]) & (1 << (category % 32))
            ]

        result["reference_root_left_categories"] = root_left_categories(reference)
        result["flipped_root_left_categories"] = root_left_categories(flipped)
    return result


def main() -> None:
    print(
        json.dumps(
            {
                "rare_category_support_9": {
                    mode: aligned_gap(mode, 9)
                    for mode in ("native", "ordinal", "onehot")
                },
                "native_category_support_20": aligned_gap("native", 20),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
