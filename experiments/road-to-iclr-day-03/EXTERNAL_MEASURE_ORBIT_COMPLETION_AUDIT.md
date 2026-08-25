# External Measure-Orbit completion audit

## Status

**Experiment complete; external claim rejected.** Completion means the frozen
test was executed without missing cells or silent protocol changes. It does not
mean the scientific gate passed.

| Check | Result |
| --- | --- |
| Frozen protocol/config/source hashes | PASS |
| Expected result cells | 63 |
| Completed result cells | 63 |
| Missing / duplicate / unexpected cells | 0 / 0 / 0 |
| Stored prediction artifacts | 63/63 |
| Failed or non-finite fits | 0 |
| Parameter matching | PASS |
| Exact portfolio gradient-update matching | PASS |
| Primary gate versus two-seed ensemble | FAIL |
| Preservation gate versus one baseline | FAIL |
| Negative result retained in report | PASS |
| Full repository test suite | 35 passed |

## Frozen-gate outcome

- Versus the update-matched two-seed ensemble: −0.521% mean relative
  proper-loss reduction, 95% dataset-bootstrap interval [−0.831%, −0.195%],
  2/7 positive dataset means, and 7/21 positive cells.
- Versus one ordinary baseline: −0.079%, interval [−0.366%, +0.184%], 3/7
  positive dataset means, and 7/21 positive cells.

All performance clauses in both gates failed. No panel member, seed, threshold,
or selector was changed after outcomes became visible.

Machine-readable audit:
`results/day3/external_measure_orbit/completion_audit.json`.
