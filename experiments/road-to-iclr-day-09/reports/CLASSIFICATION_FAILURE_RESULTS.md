# Real-classification failure diagnostic

The real binary failure is concentrated in the loss tail rather than explained by a
broad ranking collapse.

Across all nine datasets, competence-minus-fixed AUC is +0.000285
[-0.003372, 0.004642], so there is no detectable ranking shift. Brier gain is -0.000102
[-0.001323, 0.001285], calibration-in-the-large gain is -0.000525
[-0.001316, 0.000246], and ECE gain is +0.001782 [-0.000485, 0.004119]. These mixed,
mostly inconclusive summaries do not support a simple global-miscalibration account.

The frozen tail addendum is sharper. On the six-dataset unseen breadth panel, where the
parent log-loss result was significantly negative:

| Metric, fixed minus competence | Gain | 95% hierarchical CI |
|---|---:|---:|
| total log loss | -0.002130 | [-0.004659, -0.000291] |
| bottom 90% mean pointwise NLL | +0.000469 | [-0.000620, +0.001497] |
| worst-decile mean pointwise NLL | -0.025122 | [-0.045252, -0.008502] |
| fraction with pointwise NLL > 2 | -0.001497 | [-0.002930, -0.000228] |

Thus most predictions are not detectably worse, but competence creates roughly 0.15
percentage points more high-loss predictions and materially worsens the worst decile.
Coarse ECE does not reveal this proper-loss tail. The most defensible explanation is
rare confident-error amplification under episode-adaptive weights, not degraded ranking
or a uniform calibration shift. This is a post-result mechanism diagnostic and does not
authorize classification tuning.

Artifacts: `CLASSIFICATION_FAILURE_PROTOCOL.md`,
`results/processed/classification_failure_detail_v1.csv`, and
`results/processed/classification_failure_audit_v1.json`.
