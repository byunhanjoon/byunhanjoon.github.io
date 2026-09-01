# Dataset-balanced real-data synthesis

This retrospective synthesis pools every completed numeric transfer panel without
outcome-based exclusion. It is not a new confirmation experiment; its purpose is to
measure the aggregate task asymmetry and audit sensitivity to individual datasets.

| Task | Datasets | Mean fixed-minus-competence gain | Dataset-bootstrap 95% CI | Median | Positive | Gate |
|---|---:|---:|---:|---:|---:|---|
| classification log loss | 9 | -0.000266 | [-0.002976, 0.003167] | -0.001196 | 3/9 | fail |
| standardized regression MSE | 16 | +0.217965 | [0.041682, 0.459988] | +0.023386 | 14/16 | pass |

Regression passes every frozen sensitivity condition. Its 10%-trimmed mean is
+0.134626, and its leave-one-dataset-out mean ranges from +0.124652 to +0.233495, so
neither House nor Physiochemical Protein individually creates the conclusion. The three
panel means are independently positive: +0.236997 on the small panel, +0.291028 on the
unseen breadth panel, and +0.100451 on the deterministic confirmation panel. The two
preserved regression negatives are Black Friday and Auction Verification.

Classification does not merely lack power: its median is negative, only three of nine
datasets improve, and its leave-one-out range crosses zero. This rejects a task-general
transfer story and makes regression the only supported external-performance scope.

The most defensible empirical statement is therefore narrow: a synthetic-tuned,
context-loss-aligned soft mixture improves a fixed six-expert mixture on these numeric
regression panels, with dataset-level robustness across 16 identities. The evidence
does not cover categorical inputs, multiclass classification, a stronger model roster,
or generic tabular superiority.

Artifacts:

- protocol: `REAL_DATA_SYNTHESIS_PROTOCOL.md`
- dataset gains: `results/processed/real_data_synthesis_dataset_gains_v1.csv`
- frozen audit: `results/processed/real_data_synthesis_audit_v1.json`
- dataset forest plot: `figures/real_data_synthesis_v1.png`
