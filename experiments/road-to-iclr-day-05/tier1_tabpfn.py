"""Supporting TabPFN-v2.5 audit on the frozen Day-5 exact schema product."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("day5_tier1_orbit", HERE / "tier1_orbit.py")
TIER1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TIER1
SPEC.loader.exec_module(TIER1)


DATASETS = (
    "australian_credit_approval",
    "bank_marketing_subscription",
    "german_credit_risk",
    "lendingclub_loan_default",
)
SETTINGS = ((1, "none"), (1, "default"), (8, "none"), (8, "default"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/home/byunhanjoon/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt"),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "tabpfn")
    args = parser.parse_args()
    from tabpfn import TabPFNClassifier

    config = TIER1.read_config()
    config["subsample"] = {
        "max_train_rows": 1000, "max_validation_rows": 1000,
        "max_test_rows": 1000, "seed": 20260827,
    }
    data = TIER1.load_dataset(Path(config["data_root"]) / args.dataset, config)
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    shape = (len(views["feature"]), len(views["category"]), len(views["class"]))
    outputs = {}
    for estimators, policy in SETTINGS:
        validation = np.empty(shape + (len(data.validation_y), 2), dtype=np.float32)
        test = np.empty(shape + (len(data.test_y), 2), dtype=np.float32)
        completed = 0
        for fi, feature in enumerate(views["feature"]):
            for ci, category in enumerate(views["category"]):
                train_x, categorical = TIER1.render(data.train_n, encoded["train"], feature, category)
                validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], feature, category)
                test_x, _ = TIER1.render(data.test_n, encoded["test"], feature, category)
                for li, class_map in enumerate(views["class"]):
                    inference_config = None if policy == "default" else {
                        "FEATURE_SHIFT_METHOD": None, "CLASS_SHIFT_METHOD": None,
                    }
                    model = TabPFNClassifier(
                        n_estimators=estimators,
                        categorical_features_indices=categorical,
                        model_path=args.checkpoint,
                        device=args.device,
                        random_state=4201,
                        inference_config=inference_config,
                        fit_mode="fit_preprocessors",
                    )
                    model.fit(train_x, class_map[data.train_y])
                    joined = np.concatenate((validation_x, test_x), axis=0)
                    raw = model.predict_proba(joined)[:, class_map]
                    validation[fi, ci, li] = raw[: len(validation_x)]
                    test[fi, ci, li] = raw[len(validation_x) :]
                    completed += 1
                    print(f"{args.dataset} {estimators}:{policy} {completed}/{np.prod(shape)}", flush=True)
        outputs[f"validation__{estimators}__{policy}"] = validation
        outputs[f"test__{estimators}__{policy}"] = test
    outputs["validation_y"] = data.validation_y
    outputs["test_y"] = data.test_y
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}.npz"
    np.savez_compressed(output, **outputs)
    manifest = {
        "status": "complete", "dataset": args.dataset,
        "factor_shape": list(shape), "settings": [list(item) for item in SETTINGS],
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": TIER1.sha256(args.checkpoint),
        "source_hashes": {
            name: TIER1.sha256(Path(config["data_root"]) / args.dataset / name)
            for name in ("N_train.npy", "N_val.npy", "N_test.npy", "y_train.npy", "y_val.npy", "y_test.npy")
        },
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
