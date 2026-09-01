#!/usr/bin/env python3
"""Small real-row endpoint-role panel using existing airline/taxi/bike tables."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

import run_direction2_extended as d2


REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "experiments" / "mpe_iclr" / "processed"


def pair_table(source, destination):
    a = pd.read_parquet(DATA / source / "rows.parquet", columns=["row_id", "field_state", "target"]).rename(columns={"field_state": "source", "target": "target_source"})
    b = pd.read_parquet(DATA / destination / "rows.parquet", columns=["row_id", "field_state", "target"]).rename(columns={"field_state": "destination", "target": "target_destination"})
    x = a.merge(b, on="row_id", validate="one_to_one")
    x["target"] = (x.target_source + x.target_destination) / 2
    return x[["row_id", "source", "destination", "target"]]


def bike_table():
    x = pd.read_parquet(DATA / "citibike_start_station" / "rows.parquet", columns=["row_id", "field_state", "end_station_id", "target"])
    return x.rename(columns={"field_state": "source", "end_station_id": "destination"})


def encoded_split(frame, seed):
    x = frame.dropna().sample(min(120_000, len(frame)), random_state=seed).reset_index(drop=True)
    cal, train, test = np.split(x, [40_000, 80_000])
    mu, sd = cal.target.mean(), cal.target.std()
    source_map = cal.groupby("source").target.agg(["mean", "count"])
    destination_map = cal.groupby("destination").target.agg(["mean", "count"])
    # Ten-row empirical-Bayes shrinkage, fit only on the calibration partition.
    source_map["smooth"] = (source_map["mean"] * source_map["count"] + mu * 10) / (source_map["count"] + 10)
    destination_map["smooth"] = (destination_map["mean"] * destination_map["count"] + mu * 10) / (destination_map["count"] + 10)
    def transform(part):
        s = part.source.map(source_map.smooth).fillna(mu)
        d = part.destination.map(destination_map.smooth).fillna(mu)
        return np.c_[((s-mu)/sd).to_numpy(), ((d-mu)/sd).to_numpy()], ((part.target-mu)/sd).to_numpy()
    return transform(train), transform(test)


def main():
    started = time.time()
    domains = {
        "aviation": encoded_split(pair_table("airline_origin_airport", "airline_destination_airport"), 11),
        "taxi": encoded_split(pair_table("tlc_pickup_zone", "tlc_dropoff_zone"), 29),
        "bike_share": encoded_split(bike_table(), 47),
    }
    audit = pd.read_csv(d2.OUT / "real_workspace_vocabulary_audit.csv")
    audit = audit[audit.condition == "clean"]
    methods = ["structure_only", "tfidf", "bge", "e5", "gte", "oracle"]
    records, coefficient_records = [], []
    # First quantify whether the endpoint effect is asymmetric at all.
    for domain, ((xtr, ytr), (xte, yte)) in domains.items():
        m = Ridge(alpha=1.0).fit(xtr, ytr)
        coefficient_records.append({"domain": domain, "source_coefficient": m.coef_[0], "destination_coefficient": m.coef_[1], "absolute_asymmetry": abs(m.coef_[0]-m.coef_[1]), "within_domain_r2": r2_score(yte, m.predict(xte))})
    for layout_seed in range(30):
        rng = np.random.default_rng(90_000 + layout_seed)
        positions = {domain: int(rng.integers(0, 2)) for domain in domains}
        for held in domains:
            train_domains = [x for x in domains if x != held]
            for method in methods:
                xtrain, ytrain = [], []
                for domain in train_domains:
                    x, y = domains[domain][0]; pos = positions[domain]
                    physical = x if pos == 0 else x[:, ::-1]
                    if method == "structure_only": canonical = physical
                    elif method == "oracle": canonical = physical if pos == 0 else physical[:, ::-1]
                    else:
                        correct = float(audit[(audit.domain == domain) & (audit.method == method)].correct_orientation.iloc[0]) >= .5
                        predicted_pos = pos if correct else 1-pos
                        canonical = physical if predicted_pos == 0 else physical[:, ::-1]
                    xtrain.append(canonical); ytrain.append(y)
                model = Ridge(alpha=1.0).fit(np.vstack(xtrain), np.concatenate(ytrain))
                xte, yte = domains[held][1]; pos = positions[held]; physical = xte if pos == 0 else xte[:, ::-1]
                if method == "structure_only": canonical = physical
                elif method == "oracle": canonical = physical if pos == 0 else physical[:, ::-1]
                else:
                    correct = float(audit[(audit.domain == held) & (audit.method == method)].correct_orientation.iloc[0]) >= .5
                    predicted_pos = pos if correct else 1-pos
                    canonical = physical if predicted_pos == 0 else physical[:, ::-1]
                pred = model.predict(canonical)
                records.append({"layout_seed": layout_seed, "held_domain": held, "method": method, "rmse": np.sqrt(mean_squared_error(yte, pred)), "r2": r2_score(yte, pred)})
    pd.DataFrame(records).to_csv(d2.OUT / "real_endpoint_transfer_metrics.csv", index=False)
    pd.DataFrame(coefficient_records).to_csv(d2.OUT / "real_endpoint_asymmetry.csv", index=False)
    result = {"records": records, "endpoint_coefficients": coefficient_records, "calibration_rows_per_domain": 40_000, "train_rows_per_domain": 40_000, "test_rows_per_domain": 40_000, "runtime_seconds": time.time()-started, "errors": []}
    (d2.OUT / "real_panel_results.json").write_text(json.dumps(d2.jsonify(result), indent=2, allow_nan=False)+"\n")
    print(f"real endpoint panel complete in {(time.time()-started)/60:.1f} minutes")


if __name__ == "__main__": main()
