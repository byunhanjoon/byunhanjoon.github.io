# Final OrbitCover closure audit

Status: **PASS**.

- Mandatory manifests: `{'experiment_a': 144, 'experiment_a_classical': 24, 'experiment_b': 360, 'matched_convergence': 24, 'experiment_d': 48}`.
- Prediction arrays checked finite/aligned: 1128.
- Represented fits from manifests: 140,592.
- Persistent complete registry keys: 140,592.
- Summed fit telemetry: 232.005 hours
  (231.802 GPU-fit-hours and
  0.203 CPU-fit-hours).
- End-to-end closure wall clock: 44.055 hours.
- Independent seed records checked: 125,616; no undeclared duplicates.
- Protocol/config hashes: both match `PROTOCOL_HASH.txt`.
- Recorded deviations: 12; none is unrecorded.
- Tests: 116 / 116 passed (`116 passed in 19.84s`).
- Figures: 10 concepts / 20 files regenerate.
- Tables: 5 CSV+Markdown pairs regenerate.

The audit found no missing mandatory cell, corrupt prediction, unequal declared
fit budget, duplicate supposedly-independent seed, protocol-hash change, or
missing final figure/table.  The final report may now be regenerated from the
audited summaries.
