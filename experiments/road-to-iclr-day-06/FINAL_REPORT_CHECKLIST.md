# Day-6 final-report evidence checklist

Status: living execution checklist; no verdict or score is encoded here.

Every report must use the final, complete versions of the following artifacts.
Partial values in README or the research log are provenance, not final evidence.

| Report | Required machine-readable evidence | Required interpretation |
|---|---|---|
| H3 | `h3_summary.json`, `h3_cells.csv`, `h3_dynamics_summary.json`, `h3_dynamics_paths.csv`, `h3_timing.csv`, `h3_reference_loss_pairs.csv`, integrity audit | all five frozen gates; logical falsification; horizon/architecture counterexamples; timing-arm-order caveat |
| H4 | `h4_summary.json`, `h4_config_cells.csv`, `h4_seed_bundles.csv`, integrity audit | all five gates; within-dataset ranks; constant epoch-zero convention; three-dataset cap |
| H5 | `h5_summary.json`, `h5_config_cells.csv`, `h5_correlations.csv`, `h5_top_quartile_auc.csv` | transfer versus same-source prediction; no test labels; average-tie convention; new-seed confirmation requirement |
| H6 | `h6_summary.json`, `h6_prospective_bundles.csv` | exactly 33 tests after three exclusions; AUROC and fixed-decision gates; raw-level comparator; uncalibrated-score warning |
| H7 | `h7_summary.json`, `h7_survival_pairs.csv`, `h7_dataset_summary.csv` | exactly 31 bundles / 93 pairs after five exclusions; dataset heterogeneity; delay, exactness, final-win, and failure gates separately |
| H8 | `h8_summary.json`, `h8_prospective_bundles.csv` | exactly 29 tests after seven exclusions; both branches; delayed-positive recall; unchanged H6 comparator; selected-successor caveat |
| H9 | `h9_summary.json`, `h9_prospective_pairs.csv`, `h9_dataset_summary.csv`, `h9_canonical_loss_pairs.csv` | exactly 25 bundles / 75 pairs after eleven exclusions; eligibility denominator; development calibration and runtime-order split caveats |
| Day 6 | all reports above, H1/H2 reports, final integrity/completion audits, ranking protocol/addenda, incumbent/literature/reviewer/statistical audits | numeric frozen-rubric ranking after 06:21 KST; lead and alternative; discard list; strongest claim, collision, decisive result, failure boundary, and next experiment |

## Cross-report nonnegotiables

- `hypothesis_supported` is true only when the summary is complete and every
  frozen gate passes.  A promising partial value never becomes a pass.
- Dataset is the scientific replication scope.  Seeds, views, bundles, and
  optimizer configurations are labeled as repeated measurements.
- Exact prediction equality is distinguished from the one-update state audit;
  unrecorded intermediate GPU states are not claimed bitwise equal.
- Canonical gathering, hardware specificity, lack of mean accuracy benefit,
  constant-learning-rate stress horizon, and close-work subtraction appear in
  the final synthesis regardless of which gate passes.
- H8/H9 are selected successors.  Their development exclusions and selection
  pressure appear next to any positive result.
- Every displayed figure is regenerated from final CSVs after H4/H5 finish.
- The strict audit requires nonempty PNG and PDF versions of all eight declared
  H3–H9 figures; final visual inspection is still required because existence is
  not evidence that labels, scales, and panels render correctly.
- The completion audit, full tests, hashes/shapes/finiteness audit, and detached
  process logs are checked after the last bundle and before completion.
- `analysis_reproducibility_summary.json` must pass after rerunning every final
  H3–H9 analyzer and comparing all 28 declared JSON/CSV outputs byte-for-byte.
