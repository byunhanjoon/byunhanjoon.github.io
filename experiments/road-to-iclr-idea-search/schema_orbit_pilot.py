"""Pilot representation-risk audit for TabPFN.

Jointly permute feature positions and class labels in train and test data, map
probabilities back to the reference labels, and measure the risk attributable
only to the equivalent schema representative.  For multiclass Brier loss,

    mean_g R(p_g) - R(mean_g p_g) = mean_{x,g} ||p_g(x) - mean_g p_g(x)||^2.

The right side uses no test labels.  This script is a diagnostic, not a broad
benchmark.  It requires the public ``tabpfn`` package and a local checkpoint.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.eye(probabilities.shape[1])[y]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def load_dataset(
    name: str,
    adult_root: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if name == "breast_cancer":
        x, y = load_breast_cancer(return_X_y=True)
        return x, y, ()
    if name == "wine":
        x, y = load_wine(return_X_y=True)
        return x, y, ()
    if name == "adult":
        numerical = np.load(adult_root / "x_num.npy").astype(np.float64)
        binary = np.load(adult_root / "x_bin.npy").astype(np.float64)
        raw_categorical = np.load(adult_root / "x_cat.npy")
        categorical = np.column_stack([
            np.unique(raw_categorical[:, column], return_inverse=True)[1]
            for column in range(raw_categorical.shape[1])
        ]).astype(np.float64)
        y = np.load(adult_root / "y.npy").astype(int)
        x = np.column_stack((numerical, binary, categorical))
        categorical_indices = tuple(range(numerical.shape[1] + binary.shape[1], x.shape[1]))
        return x, y, categorical_indices
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("breast_cancer", "wine", "adult"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adult-root",
        type=Path,
        default=Path("experiments/road-to-iclr-day-01/data/adult"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-feature-orbit", type=int, default=8)
    parser.add_argument("--n-category-orbit", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, nargs="+", default=(1, 8))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("default", "feature_only", "class_only", "none"),
        default=("default",),
    )
    args = parser.parse_args()

    from tabpfn import TabPFNClassifier

    x, y, categorical_indices = load_dataset(args.dataset, args.adult_root)
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=20_260_826,
        stratify=y,
    )
    if len(train_y) > 1_000:
        train_x, _, train_y, _ = train_test_split(
            train_x,
            train_y,
            train_size=1_000,
            random_state=31_415,
            stratify=train_y,
        )
    if len(test_y) > 1_000:
        test_x, _, test_y, _ = train_test_split(
            test_x,
            test_y,
            train_size=1_000,
            random_state=27_182,
            stratify=test_y,
        )
    classes = int(np.max(y) + 1)
    output: dict[str, object] = {
        "dataset": args.dataset,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "n_features": x.shape[1],
        "n_classes": classes,
        "n_feature_orbit": args.n_feature_orbit,
        "n_category_orbit": args.n_category_orbit if categorical_indices else 1,
        "results": {},
    }
    label_permutations = [np.asarray(item) for item in itertools.permutations(range(classes))]
    output["n_label_orbit"] = len(label_permutations)
    category_rng = np.random.default_rng(821_571)
    category_maps: list[dict[int, np.ndarray]] = []
    for category_orbit_index in range(args.n_category_orbit if categorical_indices else 1):
        mappings = {}
        for column in categorical_indices:
            size = int(max(np.max(train_x[:, column]), np.max(test_x[:, column])) + 1)
            mappings[column] = (
                np.arange(size)
                if category_orbit_index == 0
                else category_rng.permutation(size)
            )
        category_maps.append(mappings)
    policy_configs = {
        "default": None,
        "feature_only": {
            "FEATURE_SHIFT_METHOD": "shuffle",
            "CLASS_SHIFT_METHOD": None,
        },
        "class_only": {
            "FEATURE_SHIFT_METHOD": None,
            "CLASS_SHIFT_METHOD": "shuffle",
        },
        "none": {
            "FEATURE_SHIFT_METHOD": None,
            "CLASS_SHIFT_METHOD": None,
        },
    }
    for n_estimators, policy in itertools.product(args.n_estimators, args.policies):
        predictions = []
        member_risks = []
        rng = np.random.default_rng(91_773)
        for feature_index in range(args.n_feature_orbit):
            if feature_index == 0:
                feature_permutation = np.arange(x.shape[1])
            else:
                feature_permutation = rng.permutation(x.shape[1])
            feature_predictions = []
            feature_risks = []
            transformed_categorical_indices = tuple(
                index
                for index, original_index in enumerate(feature_permutation)
                if original_index in categorical_indices
            )
            for mappings in category_maps:
                category_train_x = train_x.copy()
                category_test_x = test_x.copy()
                for column, mapping in mappings.items():
                    category_train_x[:, column] = mapping[train_x[:, column].astype(int)]
                    category_test_x[:, column] = mapping[test_x[:, column].astype(int)]
                category_predictions = []
                category_risks = []
                for label_permutation in label_permutations:
                    transformed_train_y = label_permutation[train_y]
                    model = TabPFNClassifier(
                        n_estimators=n_estimators,
                        categorical_features_indices=transformed_categorical_indices,
                        model_path=args.checkpoint,
                        device=args.device,
                        random_state=4_201,
                        inference_config=policy_configs[policy],
                        show_progress_bar=False,
                    )
                    model.fit(category_train_x[:, feature_permutation], transformed_train_y)
                    transformed_probabilities = model.predict_proba(
                        category_test_x[:, feature_permutation]
                    )
                    aligned_probabilities = transformed_probabilities[:, label_permutation]
                    category_predictions.append(aligned_probabilities)
                    category_risks.append(brier(test_y, aligned_probabilities))
                feature_predictions.append(category_predictions)
                feature_risks.append(category_risks)
            predictions.append(feature_predictions)
            member_risks.append(feature_risks)

        prediction_array = np.asarray(predictions)
        risk_array = np.asarray(member_risks)
        orbit_mean = prediction_array.mean(axis=(0, 1, 2))
        mean_member_risk = float(np.mean(member_risks))
        orbit_mean_risk = brier(test_y, orbit_mean)
        representation_variance = float(
            np.mean(np.sum((prediction_array - orbit_mean) ** 2, axis=4))
        )
        feature_means = prediction_array.mean(axis=(1, 2))
        category_means = prediction_array.mean(axis=(0, 2))
        label_means = prediction_array.mean(axis=(0, 1))
        feature_component = float(
            np.mean(np.sum((feature_means - orbit_mean) ** 2, axis=2))
        )
        category_component = float(
            np.mean(np.sum((category_means - orbit_mean) ** 2, axis=2))
        )
        label_component = float(
            np.mean(np.sum((label_means - orbit_mean) ** 2, axis=2))
        )
        feature_category_means = prediction_array.mean(axis=2)
        feature_label_means = prediction_array.mean(axis=1)
        category_label_means = prediction_array.mean(axis=0)
        feature_category_interaction = (
            feature_category_means
            - feature_means[:, None]
            - category_means[None]
            + orbit_mean
        )
        feature_label_interaction = (
            feature_label_means
            - feature_means[:, None]
            - label_means[None]
            + orbit_mean
        )
        category_label_interaction = (
            category_label_means
            - category_means[:, None]
            - label_means[None]
            + orbit_mean
        )
        triple_interaction = (
            prediction_array
            - feature_means[:, None, None]
            - category_means[None, :, None]
            - label_means[None, None]
            - feature_category_interaction[:, :, None]
            - feature_label_interaction[:, None, :]
            - category_label_interaction[None]
            + 2.0 * orbit_mean
        )
        feature_category_component = float(
            np.mean(np.sum(feature_category_interaction**2, axis=3))
        )
        feature_label_component = float(
            np.mean(np.sum(feature_label_interaction**2, axis=3))
        )
        category_label_component = float(
            np.mean(np.sum(category_label_interaction**2, axis=3))
        )
        triple_component = float(
            np.mean(np.sum(triple_interaction**2, axis=4))
        )
        output["results"][f"{n_estimators}:{policy}"] = {
            "reference_brier": float(risk_array[0, 0, 0]),
            "mean_member_brier": mean_member_risk,
            "worst_member_brier": float(np.max(risk_array)),
            "best_member_brier": float(np.min(risk_array)),
            "member_brier_std": float(np.std(risk_array)),
            "orbit_mean_brier": orbit_mean_risk,
            "representation_variance_label_free": representation_variance,
            "feature_position_component": feature_component,
            "category_code_component": category_component,
            "class_label_component": label_component,
            "feature_category_interaction": feature_category_component,
            "feature_class_interaction": feature_label_component,
            "category_class_interaction": category_label_component,
            "three_way_interaction": triple_component,
            "anova_absolute_error": abs(
                representation_variance
                - feature_component
                - category_component
                - label_component
                - feature_category_component
                - feature_label_component
                - category_label_component
                - triple_component
            ),
            "brier_reduction_by_averaging": mean_member_risk - orbit_mean_risk,
            "decomposition_absolute_error": abs(
                mean_member_risk - orbit_mean_risk - representation_variance
            ),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
