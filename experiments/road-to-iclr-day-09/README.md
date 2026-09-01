# Road to ICLR — Day 09

Day 09 began as a coordinate–marginal factorization program. The original adaptive M6
method failed its frozen G3 gate and remains stopped. An authorized fallback then found a
different, narrower result: generator metadata can be identifiable yet unsafe for expert
routing, while context-loss-aligned soft aggregation gives strong regression performance
and a small shrinkage-confirmed classification improvement.

## Start here

- Four-hour continuation outcome: `reports/FOUR_HOUR_CONTINUATION_SUMMARY.md`
- Exact final claim boundary: `reports/FINAL_AUDIT.md`
- Final executable hashes: `reports/CODE_HASHES.md`
- Hostile reviewer audit: `reports/REVIEWER_ATTACK_AUDIT.md`
- Claim-by-claim evidence: `reports/CLAIMS_EVIDENCE_MATRIX.md`
- Chronological freezes and decisions: `reports/DECISION_LOG.md`
- Current novelty boundary: `reports/NOVELTY_LEDGER.md`
- Paper-ready result text: `paper/RESULTS_SENTENCES.md`
- Paper-safe abstract and title options: `paper/ABSTRACT_DRAFT.md`
- Contribution order and four-figure plan: `paper/FRAMING.md`

## Main continuation results

- Untouched PriorDial competence routing:
  `reports/LOSS_ALIGNED_ROUTING_RESULTS.md`
- Real regression breadth and independent confirmation:
  `reports/OPENML_COMPETENCE_RESULTS.md`
- All-panel dataset-balanced sensitivity:
  `reports/REAL_DATA_SYNTHESIS_RESULTS.md`
- Classification failure and opposite-sign regression tail:
  `reports/CLASSIFICATION_FAILURE_RESULTS.md` and
  `reports/TAIL_RISK_CONTRAST_RESULTS.md`
- Synthetic-to-real adaptation-strength shift:
  `reports/SHRINKAGE_TRANSFER_RESULTS.md`
- Independent 10% classification confirmation:
  `reports/CLASSIFICATION_SHRINKAGE_CONFIRMATION_RESULTS.md`
- Context-rescaled preprocessing robustness:
  `reports/CONTEXT_RESCALED_CONFIRMATION_RESULTS.md`

## Independent semantic-orbit kill experiment

Kill Experiment 2 is reported separately in `results.md`. Its frozen six-dataset grid finds
a GO-level representation-sensitivity signal under information-identical, well-conditioned
basis changes, replicated across TabICLv2, TabPFN-2.6, CatBoost, three seeds, and a bounded
TabM follow-up. Raw bundles, processed tables, and eight figures are under
`results/semantic_orbits/` and `figures/semantic_orbits/`.

Reproduce its audits and report with:

```bash
conda run -n base python scripts/analyze_semantic_orbits.py
conda run -n base python -m pytest -q
```

## Reproduce the integrity checks

Run from this directory with the repository's base Conda environment:

```bash
conda run -n base python scripts/audit_manifest.py
conda run -n base python scripts/audit_claim_state.py
conda run -n base python -m pytest -q
conda run -n base python -m py_compile scripts/*.py src/*.py
```

Expected context-selection state: 150 manifest records, 49 unique referenced artifacts, zero missing
artifacts/config drift/non-finite arrays/duplicate run keys/episode-shape mismatches, and
30 passing top-level tests after including the semantic-orbit checks.

## Context-selection program scope

The supported real results use numeric features and a six-expert lightweight panel. Full
synthetic-tuned competence is confirmed for regression under the predeclared outer-fold
normalization. A real-development-tuned 10% step is independently confirmed for binary
classification and survives context-rescaled affine preprocessing. Categorical,
multiclass, modern-TFM, temporal/grouped, and state-of-the-art claims are not supported.
