from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.phase3 import DESCRIPTOR_COLUMNS, aggregate_descriptors, cross_dataset_meta_models


def test_descriptor_aggregation_and_grouped_meta_models():
    rng = np.random.default_rng(4)
    feature_rows = []
    for dataset in range(6):
        for split in (1, 2):
            for feature in range(3):
                row = {"dataset": f"d{dataset}", "split_seed": split, "feature": str(feature)}
                row.update({column: float(dataset + rng.normal(scale=.1)) for column in DESCRIPTOR_COLUMNS})
                feature_rows.append(row)
    descriptors = aggregate_descriptors(pd.DataFrame(feature_rows))
    assert len(descriptors) == 12
    descriptors["target"] = descriptors["mean_skewness"] * 0.5
    metrics, details = cross_dataset_meta_models(descriptors, "target", seed=8)
    assert set(metrics.meta_model) == {"ridge", "random_forest"}
    assert metrics.group_folds.eq(5).all()
    assert metrics["fold_mean_baseline_mae"].gt(0).all()
    assert np.isfinite(metrics["mae_improvement_fraction"]).all()
    assert details["prediction_ridge"].notna().sum() == len(descriptors)
