# Day 3 completion audit

This file maps the requested seven finishing requirements to evidence. The
machine-readable fail-closed version is generated at
`results/day3/broad_benchmark/day3_goal_completion_audit.json`.

| Requirement | Status | Evidence |
| --- | --- | --- |
| 1. Broad 30--50 dataset benchmark | Complete at 30 | 25-dataset frozen screen plus separately frozen five-dataset prospective replication; 2,520 primary cells, no failures |
| 2. MLP, ResNet, FT-Transformer/TabM, strong preprocessing | Complete | Four architectures; raw-standard, quantile-standard, PLE, and exact natural representation comparisons |
| 3. K-FAC, Shampoo/SOAP, whitening, first-layer natural | Complete | Nine-remedy Phase-1 screen and seven-method, ten-dataset, four-model, five-seed confirmation |
| 4. Natural encodings, not only synthetic κ | Complete | 150 paired natural cumulative/local encoding cells across 25 datasets; ordinary preprocessing comparisons |
| 5. Runtime, memory, rank deficiency, distribution shift | Complete | Remedy efficiency exports; 504-cell duplication/ridge stress test; 18-cell same-table chronological/random comparison |
| 6. Theory and prior-art distinction | Complete | `THEORY_DAY3.md` and `RELATED_WORK_DAY3.md`; K-FAC/natural-gradient/canonicalization explicitly treated as prior art |
| 7. Systematic phenomenon framing and ICLR verdict | Complete | `BROAD_BENCHMARK_REPORT.md`; empirical-audit framing, limitations, and final borderline/ICLR-plausible verdict |

The benchmark completion claims are valid only when the final syntax, test, and
freeze audits pass. Their exact command outputs and timestamps are stored in
`final_verification.json`; the seven checks are recomputed rather than asserted
manually by `experiments/day3/audit_day3_goal.py`.
