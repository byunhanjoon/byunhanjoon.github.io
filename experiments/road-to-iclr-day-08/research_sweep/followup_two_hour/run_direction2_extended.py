#!/usr/bin/env python3
"""Extended Direction 2: frozen language embeddings for schema-role transfer."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direction_2"
FIG = ROOT / "figures"
CACHE = Path("/home/byunhanjoon/.cache/huggingface/hub")


TRAIN_ROLES = [
    ("commerce", "buyer", "seller", "purchases merchandise and sends payment", "offers merchandise and accepts payment", "selects goods in a transaction", "fulfills the purchase order"),
    ("messaging", "sender", "receiver", "transmits a message", "gets the transmitted message", "dispatches the communication", "is addressed by the communication"),
    ("payments", "payer", "payee", "remits money", "is credited with the money", "authorizes the outgoing transfer", "collects the transferred amount"),
    ("shipping", "shipper", "consignee", "dispatches a shipment", "takes delivery of the shipment", "hands the parcel to a carrier", "is named to obtain the parcel"),
    ("telephone", "caller", "callee", "places a telephone call", "answers the telephone call", "dials another party", "is reached by the dialed connection"),
    ("publishing", "author", "reader", "writes material for an audience", "consumes the written material", "creates the document", "reads the resulting document"),
    ("charity", "donor", "beneficiary", "contributes resources", "benefits from the contribution", "gives support voluntarily", "obtains the donated support"),
    ("trade", "exporter", "importer", "sends products out of a country", "brings products into a country", "supplies goods across the border", "acquires foreign goods"),
    ("media", "publisher", "subscriber", "distributes recurring content", "signs up to obtain the content", "issues each edition", "is enrolled for each edition"),
    ("insurance", "insurer", "policyholder", "underwrites financial protection", "owns the protection contract", "covers specified losses", "holds the issued coverage"),
    ("housing", "landlord", "tenant", "leases a property", "occupies the leased property", "grants temporary use of a dwelling", "rents and uses the dwelling"),
    ("retail", "vendor", "customer", "supplies a product for sale", "acquires the supplied product", "lists inventory for purchase", "shops from the inventory"),
    ("hospitality", "host", "guest", "provides accommodation", "stays in the accommodation", "welcomes a visitor", "is welcomed for a stay"),
    ("manufacturing", "producer", "consumer", "makes a product for use", "uses the finished product", "creates the market offering", "demands the market offering"),
    ("debt", "creditor", "debtor", "is owed an obligation", "owes the obligation", "extends a claim for repayment", "must satisfy the claim"),
    ("broadcast", "broadcaster", "viewer", "airs a program", "watches the aired program", "transmits audiovisual content", "observes the transmitted content"),
    ("software", "maintainer", "user", "releases and supports software", "operates the released software", "publishes code updates", "runs the installed application"),
    ("auction", "bidder", "auctioneer", "submits an offer", "conducts the sale and receives offers", "proposes a purchase price", "administers the bidding process"),
]

HELD_ROLES = [
    ("medicine", "doctor", "patient", "prescribes treatment and clinical care", "undergoes the prescribed care", "selects a medical intervention", "is treated through that intervention"),
    ("lending", "lender", "borrower", "provides funds under a loan", "receives and later repays the funds", "advances capital", "takes out the advance"),
    ("education", "teacher", "student", "instructs a learner", "learns through the instruction", "delivers a lesson", "studies the delivered lesson"),
    ("employment", "employer", "employee", "hires and compensates a worker", "works for compensation", "offers a position", "fills the offered position"),
    ("family", "parent", "child", "raises a younger family member", "is raised by the older family member", "provides guardianship", "is under that guardianship"),
    ("mentoring", "mentor", "mentee", "provides professional guidance", "develops through the guidance", "advises a less experienced person", "is advised while developing skills"),
    ("aviation", "origin_airport", "destination_airport", "is where a flight departs", "is where the flight arrives", "marks the beginning of the route", "marks the endpoint of the route"),
    ("taxi", "pickup_zone", "dropoff_zone", "is where the passenger enters the vehicle", "is where the passenger leaves the vehicle", "starts the hired ride", "finishes the hired ride"),
    ("bike_share", "start_station", "end_station", "is where the bicycle is checked out", "is where the bicycle is returned", "opens the rental journey", "closes the rental journey"),
    ("cybersecurity", "attacker", "target", "initiates a hostile operation", "is affected by the hostile operation", "launches the intrusion", "is subjected to the intrusion"),
    ("sports", "coach", "athlete", "plans training for a competitor", "performs the planned training", "directs preparation", "is prepared for competition"),
    ("research", "investigator", "participant", "conducts a study involving a person", "takes part in the study", "administers the research protocol", "is enrolled under the protocol"),
    ("law", "prosecutor", "defendant", "brings a criminal case", "answers the criminal allegation", "presents charges to a court", "is charged before the court"),
    ("prescriptions", "prescriber", "recipient", "orders a medication", "is intended to take the medication", "issues the drug instruction", "receives the instructed drug"),
    ("supply_chain", "manufacturer", "retailer", "fabricates goods for distribution", "stocks the fabricated goods for sale", "builds the inventory item", "resells the inventory item"),
    ("networking", "server", "client", "responds to service requests", "issues requests for a service", "hosts a network resource", "connects to the hosted resource"),
    ("navigation", "departure_port", "arrival_port", "is the port a vessel leaves", "is the port the vessel reaches", "begins the voyage", "terminates the voyage"),
    ("legal_transfer", "grantor", "grantee", "conveys a property right", "receives the conveyed right", "executes the transfer", "is vested by the transfer"),
]


def jsonify(v):
    if isinstance(v, dict): return {str(k): jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonify(x) for x in v]
    if isinstance(v, np.ndarray): return jsonify(v.tolist())
    if isinstance(v, (np.floating, float)):
        x = float(v); return x if math.isfinite(x) else None
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v


def role_texts(role, condition):
    domain, src, dst, src_def, dst_def, src_para, dst_para = role
    if condition == "name_only":
        return f"Database column {src}_id in a {domain} table.", f"Database column {dst}_id in a {domain} table."
    if condition == "clean":
        return f"Database column {src}_id. The {src} {src_def}.", f"Database column {dst}_id. The {dst} {dst_def}."
    if condition == "paraphrase":
        return f"Attribute {src}_key in {domain}: this party {src_para}.", f"Attribute {dst}_key in {domain}: this party {dst_para}."
    if condition == "opaque_informative":
        return f"Database column field_x. This entity {src_def}.", f"Database column field_x. This entity {dst_def}."
    if condition == "opaque_paraphrase":
        return f"Field field_x: this entity {src_para}.", f"Field field_x: this entity {dst_para}."
    if condition == "negated_description":
        return f"Field field_x. This is not the receiving party; it {src_para}.", f"Field field_x. This party does not initiate the relation; it {dst_para}."
    if condition == "ambiguous":
        return "Database column field_x. A participant in the recorded relation.", "Database column field_x. A participant in the recorded relation."
    if condition == "shuffled_description":
        return f"Database column field_x. This entity {dst_def}.", f"Database column field_x. This entity {src_def}."
    raise ValueError(condition)


class FrozenEncoder:
    def __init__(self, key, device):
        patterns = {"bge": "models--BAAI--bge-base-en-v1.5", "e5": "models--intfloat--e5-base-v2", "gte": "models--Alibaba-NLP--gte-base-en-v1.5"}
        snaps = list((CACHE / patterns[key] / "snapshots").glob("*"))
        if not snaps: raise FileNotFoundError(patterns[key])
        self.key, self.device, self.path = key, device, snaps[0]
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=True).to(device).eval()

    def encode(self, texts, batch_size=64):
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if self.key == "e5": batch = ["query: " + x for x in batch]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**toks).last_hidden_state
                if self.key in {"bge", "gte"}:
                    emb = hidden[:, 0]
                else:
                    mask = toks["attention_mask"].unsqueeze(-1)
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.vstack(out)

    @property
    def parameters(self): return sum(p.numel() for p in self.model.parameters())


def build_field_corpus(roles, conditions):
    texts, labels, meta = [], [], []
    for rid, role in enumerate(roles):
        for condition in conditions:
            src, dst = role_texts(role, condition)
            texts += [src, dst]; labels += [1, 0]
            meta += [{"role_id": rid, "domain": role[0], "field": "source", "condition": condition}, {"role_id": rid, "domain": role[0], "field": "destination", "condition": condition}]
    return texts, np.asarray(labels), meta


def train_probes(encoders):
    train_conditions = ["name_only", "clean", "paraphrase", "opaque_informative"]
    texts, labels, _ = build_field_corpus(TRAIN_ROLES, train_conditions)
    probes, vectorizer = {}, FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    ])
    xt = vectorizer.fit_transform(texts)
    probes["tfidf"] = LogisticRegression(C=1.0, max_iter=1000).fit(xt, labels)
    for key, encoder in encoders.items():
        emb = encoder.encode(texts)
        probes[key] = LogisticRegression(C=0.5, max_iter=1000).fit(emb, labels)
    return probes, vectorizer


def method_scores(roles, condition, encoders, probes, vectorizer):
    texts, _, meta = build_field_corpus(roles, [condition])
    scores = {"tfidf": probes["tfidf"].predict_proba(vectorizer.transform(texts))[:, 1]}
    for key, encoder in encoders.items():
        emb = encoder.encode(texts)
        scores[key] = probes[key].predict_proba(emb)[:, 1]
        proto_text = ["entity that initiates, sends, provides, or causes the relation", "entity that receives, is acted upon, or is the destination of the relation"]
        proto = encoder.encode(proto_text)
        scores[key + "_zero_shot"] = emb @ proto[0] - emb @ proto[1]
    scores["oracle"] = np.tile([1.0, 0.0], len(roles))
    return scores, texts, meta


def orientation_records(encoders, probes, vectorizer):
    conditions = ["name_only", "clean", "paraphrase", "opaque_informative", "opaque_paraphrase", "negated_description", "ambiguous", "shuffled_description"]
    records, score_cache = [], {}
    for condition in conditions:
        scores, texts, meta = method_scores(HELD_ROLES, condition, encoders, probes, vectorizer)
        score_cache[condition] = scores
        for method, values in scores.items():
            for rid, role in enumerate(HELD_ROLES):
                ss, ds = values[2 * rid], values[2 * rid + 1]
                tied = abs(ss - ds) < 1e-8
                records.append({"condition": condition, "method": method, "role_id": rid, "domain": role[0], "source_name": role[1], "destination_name": role[2], "source_score": ss, "destination_score": ds, "correct_orientation": 0.5 if tied else float(ss > ds), "abstained_tie": tied, "margin": ss - ds})
    return records, score_cache


def generate_numeric(role, n, rng, source_pos, signal=1.0):
    source, destination, context = rng.normal(size=(3, n))
    prob = expit(signal * (2 * source - destination + 0.35 * context))
    y = rng.binomial(1, prob)
    v0, v1 = (source, destination) if source_pos == 0 else (destination, source)
    return v0, v1, context, y


def canonical_features(v0, v1, context, predicted_source_pos):
    source, dest = (v0, v1) if predicted_source_pos == 0 else (v1, v0)
    return np.c_[source, dest, context]


def downstream_records(score_cache, encoders, probes, vectorizer, numeric_seeds):
    methods = ["structure_only", "tfidf", *encoders.keys(), *[k + "_zero_shot" for k in encoders], "oracle"]
    records = []
    train_scores, _, _ = method_scores(TRAIN_ROLES, "clean", encoders, probes, vectorizer)
    for seed in numeric_seeds:
        rng = np.random.default_rng(50_000 + seed)
        train_parts = []
        for rid, role in enumerate(TRAIN_ROLES):
            pos = int(rng.integers(0, 2))
            v0, v1, c, y = generate_numeric(role, 1200, rng, pos)
            train_parts.append((rid, pos, v0, v1, c, y))
        for condition in ["name_only", "clean", "paraphrase", "opaque_informative", "opaque_paraphrase", "negated_description", "ambiguous", "shuffled_description"]:
            held_scores = score_cache[condition]
            held_parts = []
            for rid, role in enumerate(HELD_ROLES):
                pos = int(rng.integers(0, 2))
                v0, v1, c, y = generate_numeric(role, 1800, rng, pos)
                held_parts.append((rid, pos, v0, v1, c, y))
            for method in methods:
                xtrain, ytrain = [], []
                for rid, pos, v0, v1, c, y in train_parts:
                    if method == "structure_only": pred_pos = 0
                    elif method == "oracle": pred_pos = pos
                    else:
                        ss, ds = train_scores[method][2 * rid], train_scores[method][2 * rid + 1]
                        if abs(ss - ds) < 1e-8: pred_pos = 0
                        else: pred_pos = pos if ss > ds else 1 - pos
                    xtrain.append(canonical_features(v0, v1, c, pred_pos)); ytrain.append(y)
                clf = LogisticRegression(C=1.0, max_iter=500).fit(np.vstack(xtrain), np.concatenate(ytrain))
                yy, pp = [], []
                for rid, pos, v0, v1, c, y in held_parts:
                    if method == "structure_only": pred_pos = 0
                    elif method == "oracle": pred_pos = pos
                    else:
                        ss, ds = held_scores[method][2 * rid], held_scores[method][2 * rid + 1]
                        if abs(ss - ds) < 1e-8: pred_pos = 0
                        else: pred_pos = pos if ss > ds else 1 - pos
                    yy.append(y); pp.append(clf.predict_proba(canonical_features(v0, v1, c, pred_pos))[:, 1])
                yy, pp = np.concatenate(yy), np.concatenate(pp)
                records.append({"numeric_seed": seed, "condition": condition, "method": method, "auroc": roc_auc_score(yy, pp), "accuracy": accuracy_score(yy, pp >= .5), "log_loss": log_loss(yy, pp)})
    return records


def real_vocabulary_audit(orientation):
    real_domains = {"aviation", "taxi", "bike_share"}
    return [r for r in orientation if r["domain"] in real_domains and r["condition"] in {"name_only", "clean", "opaque_informative"}]


def make_figures(orientation, downstream):
    odf, ddf = pd.DataFrame(orientation), pd.DataFrame(downstream)
    order = ["tfidf", "bge", "e5", "gte", "bge_zero_shot", "oracle"]
    order = [x for x in order if x in set(odf.method)]
    q = odf[odf.condition.isin(["name_only", "clean", "opaque_informative", "ambiguous"])].groupby(["condition", "method"]).correct_orientation.mean().unstack()
    fig, ax = plt.subplots(figsize=(9.2, 4.6)); x = np.arange(len(q.index)); width = .8 / len(order)
    for j, method in enumerate(order): ax.bar(x - .4 + width/2 + j*width, q[method], width, label=method)
    ax.set_xticks(x, [z.replace("_", "\n") for z in q.index]); ax.set_ylim(0, 1.05); ax.set_ylabel("held-role orientation accuracy"); ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "direction2_frozen_embedding_orientation.png", dpi=180); plt.close(fig)

    methods = ["structure_only", "tfidf", "bge", "e5", "gte", "oracle"]
    methods = [x for x in methods if x in set(ddf.method)]
    q = ddf[ddf.condition.isin(["clean", "opaque_informative", "ambiguous", "shuffled_description"])].groupby(["condition", "method"]).auroc.mean().unstack()
    fig, ax = plt.subplots(figsize=(9.2, 4.6)); x = np.arange(len(q.index)); width = .8 / len(methods)
    for j, method in enumerate(methods): ax.bar(x - .4 + width/2 + j*width, q[method], width, label=method)
    ax.set_xticks(x, [z.replace("_", "\n") for z in q.index]); ax.set_ylim(.2, 1.0); ax.set_ylabel("downstream held-schema AUROC"); ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "direction2_frozen_embedding_downstream.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--numeric-seeds", type=int, default=12)
    parser.add_argument("--encoders", nargs="+", default=["bge", "e5", "gte"])
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    started = time.time(); errors, encoders, encoder_meta = [], {}, {}
    for key in args.encoders:
        try:
            t0 = time.time(); encoders[key] = FrozenEncoder(key, args.device)
            encoder_meta[key] = {"path": str(encoders[key].path), "parameters": encoders[key].parameters, "load_seconds": time.time() - t0}
            print(f"loaded {key}: {encoder_meta[key]}", flush=True)
        except Exception as exc:
            errors.append({"stage": "load_encoder", "encoder": key, "exception": repr(exc), "traceback": traceback.format_exc()})
    if not encoders: raise RuntimeError("No frozen encoder loaded")
    probes, vectorizer = train_probes(encoders)
    orientation, score_cache = orientation_records(encoders, probes, vectorizer)
    downstream = downstream_records(score_cache, encoders, probes, vectorizer, list(range(args.numeric_seeds)))
    real_audit = real_vocabulary_audit(orientation)
    pd.DataFrame(orientation).to_csv(OUT / "orientation_metrics.csv", index=False)
    pd.DataFrame(downstream).to_csv(OUT / "downstream_metrics.csv", index=False)
    pd.DataFrame(real_audit).to_csv(OUT / "real_workspace_vocabulary_audit.csv", index=False)
    make_figures(orientation, downstream)
    result = {
        "study": "Extended Direction 2 frozen-embedding schema transfer",
        "parameters": vars(args), "encoder_metadata": encoder_meta,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu": torch.cuda.get_device_name(1 if args.device.endswith("1") else 0) if torch.cuda.is_available() else None},
        "train_roles": TRAIN_ROLES, "held_roles": HELD_ROLES,
        "orientation_records": orientation, "downstream_records": downstream, "real_workspace_vocabulary_audit": real_audit,
        "errors": errors, "runtime_seconds": time.time() - started,
    }
    (OUT / "results.json").write_text(json.dumps(jsonify(result), indent=2, allow_nan=False) + "\n")
    print(f"direction2 complete in {(time.time()-started)/60:.1f} minutes; encoders={list(encoders)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
