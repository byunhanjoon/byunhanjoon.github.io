# Measure-Orbit completion audit

## Objective audit

The requested loop was: test the residual-atlas hypothesis; if it fails, keep
forming and testing paper-level hypotheses until one validates.

| Requirement | Evidence | Status |
| --- | --- | --- |
| Test residual-atlas transfer | Frozen Day-2 follow-up selected a structure on 1/6 untouched datasets, below its 2/6 gate | Rejected as headline |
| Test next Adult-derived hypothesis | `mixed_measure_ple`: 216/216 cells, zero failures, parameter matched | Completed; gate failed |
| Continue after failure | `measure_orbit`: 48/48 cells, zero failures, parameter matched | Completed; gate narrowly failed |
| Continue after second failure | `selective_measure_orbit`: prospectively frozen 21-dataset confirmation | Completed |
| Validate a paper-level hypothesis | +1.103% mean proper-loss reduction, 95% interval [+0.333%, +2.154%], 17/21 dataset means positive, 39/63 paired wins | All frozen gates passed |
| Preserve Adult-sized performance | Measure-Orbit Adult screen: +1.013 accuracy points and +4.931% proper-loss reduction | Passed |
| Leakage control | Atom discovery is target-free; model selection uses validation proper loss only; test targets are aggregate endpoints only | Passed by construction and code inspection |
| Integrity | 390/390 new matrix fits, no duplicates, finite outcomes, no failures, equal parameters within paired cells | Passed by analyzers |
| Reproduction | Nine relevant unit tests pass; all three analyzers rerun successfully | Passed |
| Honest claim boundary | Reports state prior art, reused-dataset status, two-fit cost, missing equal-compute prediction ensemble, and remaining external replication | Passed |

## Authoritative artifacts

- Frozen protocols:
  - `experiments/day3/configs/mixed_measure_ple_preregistered.json`
  - `experiments/day3/configs/measure_orbit_preregistered.json`
  - `experiments/day3/configs/selective_measure_orbit_preregistered.json`
- Machine decisions:
  - `results/day3/mixed_measure_ple/analysis_summary.json`
  - `results/day3/measure_orbit/analysis_summary.json`
  - `results/day3/selective_measure_orbit/analysis_summary.json`
- Human-readable protocol and report:
  - `MEASURE_ORBIT_PROTOCOL.md`
  - `MEASURE_ORBIT_REPORT.md`

## Final decision

The original residual-atlas and universal mixed-measure-replacement stories
are falsified as broad methods. Selective Measure-Orbit is validated as a
paper-level candidate under its frozen method-specific confirmation protocol.
It is ready for the next submission stage—external replication and
equal-compute ensemble controls—but should not yet be described as a universal
encoder or compute-matched state-of-the-art method.
