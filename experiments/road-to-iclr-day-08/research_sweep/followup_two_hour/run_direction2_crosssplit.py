#!/usr/bin/env python3
"""Cross-split robustness for the frozen schema-role embedding result."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion

import run_direction2_extended as d2


ALL_ROLES = d2.TRAIN_ROLES + d2.HELD_ROLES
CONDITIONS = ["name_only", "clean", "paraphrase", "opaque_informative", "opaque_paraphrase", "negated_description", "ambiguous", "shuffled_description"]
TRAIN_CONDITIONS = ["name_only", "clean", "paraphrase", "opaque_informative"]
SPLIT_SEEDS = [7, 19, 31, 43, 59, 71, 83, 97]


def corpus_for_roles(indices, conditions):
    texts, labels, role_ids = [], [], []
    for rid in indices:
        for condition in conditions:
            src, dst = d2.role_texts(ALL_ROLES[rid], condition)
            texts.extend([src, dst]); labels.extend([1, 0]); role_ids.extend([rid, rid])
    return texts, np.asarray(labels), np.asarray(role_ids)


def score_pair(values, rid_local):
    return values[2 * rid_local], values[2 * rid_local + 1]


def main():
    start = time.time(); device = "cuda:1"
    encoders = {key: d2.FrozenEncoder(key, device) for key in ["bge", "e5", "gte"]}
    # Cache every encoder/condition once so fold robustness does not repeatedly
    # run the language model.
    texts_by_condition, embedding_cache = {}, {key: {} for key in encoders}
    for condition in CONDITIONS:
        texts, _, _ = corpus_for_roles(range(len(ALL_ROLES)), [condition])
        texts_by_condition[condition] = texts
        for key, encoder in encoders.items():
            embedding_cache[key][condition] = encoder.encode(texts)

    orientation_records, downstream_records = [], []
    for split_seed in SPLIT_SEEDS:
        rng = np.random.default_rng(split_seed)
        train_ids = np.sort(rng.choice(len(ALL_ROLES), len(ALL_ROLES) // 2, replace=False))
        held_ids = np.asarray([i for i in range(len(ALL_ROLES)) if i not in set(train_ids.tolist())])
        train_texts, train_y, _ = corpus_for_roles(train_ids, TRAIN_CONDITIONS)
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        tf_probe = LogisticRegression(C=1.0, max_iter=1000).fit(vectorizer.fit_transform(train_texts), train_y)
        probes = {}
        train_positions = np.concatenate([[2 * rid, 2 * rid + 1] for rid in train_ids])
        # Reorder cached embeddings to match role-major, condition-minor corpus.
        for key in encoders:
            chunks, labels = [], []
            for rid in train_ids:
                for condition in TRAIN_CONDITIONS:
                    chunks.append(embedding_cache[key][condition][[2 * rid, 2 * rid + 1]])
                    labels.extend([1, 0])
            probes[key] = LogisticRegression(C=.5, max_iter=1000).fit(np.vstack(chunks), labels)

        held_scores, train_scores = {}, {}
        for condition in CONDITIONS:
            held_text = []
            for rid in held_ids:
                held_text.extend(d2.role_texts(ALL_ROLES[rid], condition))
            held_scores[("tfidf", condition)] = tf_probe.predict_proba(vectorizer.transform(held_text))[:, 1]
            for key in encoders:
                emb = np.vstack([embedding_cache[key][condition][[2 * rid, 2 * rid + 1]] for rid in held_ids])
                held_scores[(key, condition)] = probes[key].predict_proba(emb)[:, 1]
        clean_train_text = []
        for rid in train_ids: clean_train_text.extend(d2.role_texts(ALL_ROLES[rid], "clean"))
        train_scores["tfidf"] = tf_probe.predict_proba(vectorizer.transform(clean_train_text))[:, 1]
        for key in encoders:
            emb = np.vstack([embedding_cache[key]["clean"][[2 * rid, 2 * rid + 1]] for rid in train_ids])
            train_scores[key] = probes[key].predict_proba(emb)[:, 1]

        for condition in CONDITIONS:
            for method in ["tfidf", "bge", "e5", "gte"]:
                vals = held_scores[(method, condition)]
                for local, rid in enumerate(held_ids):
                    ss, ds = score_pair(vals, local); tied = abs(ss - ds) < 1e-8
                    orientation_records.append({"split_seed": split_seed, "condition": condition, "method": method, "role_id": int(rid), "domain": ALL_ROLES[rid][0], "correct_orientation": .5 if tied else float(ss > ds), "margin": ss - ds, "abstained_tie": tied})

        for numeric_seed in range(5):
            nrng = np.random.default_rng(split_seed * 10_000 + numeric_seed)
            train_parts, held_parts_by_condition = [], {}
            for local, rid in enumerate(train_ids):
                pos = int(nrng.integers(0, 2)); v0, v1, c, y = d2.generate_numeric(ALL_ROLES[rid], 800, nrng, pos)
                train_parts.append((local, pos, v0, v1, c, y))
            for condition in ["clean", "opaque_informative", "opaque_paraphrase", "negated_description", "ambiguous", "shuffled_description"]:
                parts = []
                for local, rid in enumerate(held_ids):
                    pos = int(nrng.integers(0, 2)); v0, v1, c, y = d2.generate_numeric(ALL_ROLES[rid], 1200, nrng, pos)
                    parts.append((local, pos, v0, v1, c, y))
                held_parts_by_condition[condition] = parts
            for method in ["structure_only", "tfidf", "bge", "e5", "gte", "oracle"]:
                xtr, ytr = [], []
                for local, pos, v0, v1, c, y in train_parts:
                    if method == "structure_only": pred_pos = 0
                    elif method == "oracle": pred_pos = pos
                    else:
                        ss, ds = score_pair(train_scores[method], local)
                        pred_pos = 0 if abs(ss-ds)<1e-8 else (pos if ss > ds else 1-pos)
                    xtr.append(d2.canonical_features(v0, v1, c, pred_pos)); ytr.append(y)
                clf = LogisticRegression(C=1.0, max_iter=500).fit(np.vstack(xtr), np.concatenate(ytr))
                for condition, parts in held_parts_by_condition.items():
                    yy, pp = [], []
                    vals = held_scores.get((method, condition))
                    for local, pos, v0, v1, c, y in parts:
                        if method == "structure_only": pred_pos = 0
                        elif method == "oracle": pred_pos = pos
                        else:
                            ss, ds = score_pair(vals, local)
                            pred_pos = 0 if abs(ss-ds)<1e-8 else (pos if ss > ds else 1-pos)
                        yy.append(y); pp.append(clf.predict_proba(d2.canonical_features(v0, v1, c, pred_pos))[:, 1])
                    yy, pp = np.concatenate(yy), np.concatenate(pp)
                    downstream_records.append({"split_seed": split_seed, "numeric_seed": numeric_seed, "condition": condition, "method": method, "auroc": roc_auc_score(yy, pp), "log_loss": log_loss(yy, pp)})
        print(f"completed semantic split {split_seed}", flush=True)

    odf, ddf = pd.DataFrame(orientation_records), pd.DataFrame(downstream_records)
    odf.to_csv(d2.OUT / "crosssplit_orientation_metrics.csv", index=False)
    ddf.to_csv(d2.OUT / "crosssplit_downstream_metrics.csv", index=False)
    result = {"split_seeds": SPLIT_SEEDS, "roles": ALL_ROLES, "orientation_records": orientation_records, "downstream_records": downstream_records, "runtime_seconds": time.time()-start, "errors": []}
    (d2.OUT / "crosssplit_results.json").write_text(json.dumps(d2.jsonify(result), indent=2, allow_nan=False)+"\n")
    print(f"cross-split direction2 complete in {(time.time()-start)/60:.1f} minutes", flush=True)


if __name__ == "__main__": main()
