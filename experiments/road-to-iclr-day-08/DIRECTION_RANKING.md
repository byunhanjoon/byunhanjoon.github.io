# DAY 8 DIRECTION RANKING

Scales are `/5`; for **Prior-art risk**, larger is worse.

| Direction | Novelty | Theory | Signal | Simplicity | Prior-art risk | ICLR potential |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval Risk Geometry | 2.5 | 5.0 | 2.5 | 4.0 | 4.5 | 2.0 |
| Nonlinear Feature Metric | 1.5 | 3.0 | 2.5 | 4.0 | 5.0 | 2.0 |
| Transformer Geometry | 2.5 | 2.0 | 1.0 | 2.5 | 4.0 | 1.5 |
| OrbitCover Extension | 1.0 | 3.0 | 2.5 | 3.0 | 5.0 | 1.5 |

WINNER = Retrieval Risk Geometry

WHY = It remains the most informative analysis direction because the exact decomposition exposed why a plausible reliability intervention fails. The post-hoc aggregate-risk QP has a substantial real-panel signal, but its mismatch-only ablation matches the full method and its ModernNCA synthetic transfer gate fails. It is not an ICLR-ready method.

KILL CONDITION = MET. On the prospectively frozen panel, lower proxy risk did not predict gains and true reliability underperformed its permutation control in dataset-balanced score for both models. The one allowed post-hoc correction passed both real-data subgates but failed the frozen joint gate on S3 ModernNCA.

NEXT DECISIVE EXPERIMENT = NONE IN THIS RUN. The post-hoc check is complete; per protocol, stop rather than launch a larger benchmark. Revisiting mismatch-aware aggregation would require a new literature audit, new claim, and newly frozen prospective study.
