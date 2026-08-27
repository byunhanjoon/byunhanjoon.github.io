#!/usr/bin/env python3
"""Equal-compute PLE-view additivity after explicit Adult atom handling."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from adult_identity_mechanism_pilot import (
    classifier_tree_edges,
    exact_support_codes,
    build_model,
    load_adult,
    parameter_count,
    quantile_edges,
    subset_encoding,
    train_one,
)


HERE = Path(__file__).resolve().parent
PARTS = ("validation", "test")


def probability(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))


def score(logit: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prob = probability(logit)
    return {
        "log_loss": float(log_loss(target, prob)),
        "auc": float(roc_auc_score(target, prob)),
    }


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def anchor_path(model: str) -> Path | None:
    directory = {
        "mlp": "adult_identity_mechanism_mlp_predictions",
        "resnet": "adult_identity_mechanism_resnet_predictions",
        "ft_transformer": "adult_identity_mechanism_ft_a_predictions",
    }.get(model)
    if directory is None:
        return None
    return HERE / "results" / directory / f"{model}__tple_exact_supervised_additive.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "adult_exact_multiview_config.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/adult_exact_multiview.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = load_adult()
    qple = quantile_edges(data.x_num["train"], config["qple_bins"])
    tple = classifier_tree_edges(
        data.x_num["train"], data.y["train"], config["tple_bins"],
        config["tple_min_samples_leaf"],
    )
    columns, cards, codes = exact_support_codes(data.x_num, 128)
    encoding = subset_encoding(
        data, qple, tple, columns, cards, codes,
        config["supervised_residual_columns"],
    )
    source = __import__("pandas").read_csv(HERE / "results/adult_identity_mechanism.csv")
    rows: list[dict[str, object]] = list(read(args.output))
    done = {row["model"] for row in rows}
    member_config = dict(config, seed=config["member_seed"])
    for architecture in config["architectures"]:
        if architecture in done:
            continue
        prior = source.query(
            "model == @architecture and method == 'tple_exact_supervised_additive'"
        )
        if len(prior):
            target_parameters = int(prior.iloc[0].target_parameters)
            anchor_parameters = int(prior.iloc[0].parameters)
        else:
            reference = build_model(
                data, encoding, config, "tple_exact_supervised_additive",
                architecture, config["width"], config["ft_feedforward_width"],
            )
            target_parameters = parameter_count(reference)
            anchor_parameters = target_parameters
        q_result, q_val, q_test = train_one(
            data, encoding, member_config, "qple_exact_supervised_additive",
            architecture, target_parameters, args.device,
        )
        t_result, t_val, t_test = train_one(
            data, encoding, member_config, "tple_exact_supervised_additive",
            architecture, target_parameters, args.device,
        )
        stored_path = anchor_path(architecture)
        if stored_path is not None and stored_path.exists():
            with np.load(stored_path) as stored:
                anchor_val = stored["validation"]
                anchor_test = stored["test"]
        else:
            anchor_config = dict(config, seed=config["anchor_seed"])
            anchor_result, anchor_val, anchor_test = train_one(
                data, encoding, anchor_config, "tple_exact_supervised_additive",
                architecture, target_parameters, args.device,
            )
            anchor_parameters = int(anchor_result["parameters"])
        output: dict[str, object] = {
            "dataset": "adult",
            "model": architecture,
            "anchor_seed": config["anchor_seed"],
            "member_seed": config["member_seed"],
            "anchor_parameters": anchor_parameters,
            "q_member_parameters": q_result["parameters"],
            "t_member_parameters": t_result["parameters"],
        }
        for split, anchor, q_member, t_member, target in (
            ("val", anchor_val, q_val, t_val, data.y["val"]),
            ("test", anchor_test, q_test, t_test, data.y["test"]),
        ):
            candidate = score(0.5 * (anchor + q_member), target)
            control = score(0.5 * (anchor + t_member), target)
            for metric, value in candidate.items():
                output[f"tq_exact_{split}_{metric}"] = value
            for metric, value in control.items():
                output[f"tt_exact_{split}_{metric}"] = value
            output[f"relative_{split}_logloss_gain_pct"] = (
                100.0 * (control["log_loss"] - candidate["log_loss"])
                / control["log_loss"]
            )
        rows.append(output)
        done.add(architecture)
        write(args.output, rows)
        print(json.dumps(output, sort_keys=True), flush=True)
    gains = np.array([float(row["relative_val_logloss_gain_pct"]) for row in rows])
    evaluable = len(gains) >= 3
    decision = {
        "validation_wins": int((gains > 0).sum()),
        "architectures": len(gains),
        "mean_relative_validation_logloss_gain_pct": float(gains.mean()),
        "gate_evaluable": evaluable,
        "gate_passed": (
            bool((gains > 0).sum() >= 2 and gains.mean() > 0)
            if evaluable
            else None
        ),
        "test_role": "descriptive_only",
    }
    args.output.with_name(args.output.stem + "_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
