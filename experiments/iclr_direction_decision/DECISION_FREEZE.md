# ICLR direction decision freeze

Freeze time: 2026-08-28T00:04:28+09:00 (Asia/Seoul).

This file freezes the minimum decision sprint before any new HeteroBag
second-panel outcome is run. Existing Day-3/Day-5 and Day-4 outcomes are prior
evidence; their original protocol/config timestamps and hashes, rather than
this retrospective umbrella file, determine prospective status. The umbrella
freeze must not upgrade exploratory evidence to confirmatory evidence.

## Candidate A — Day 3 / OrbitANOVA

Reuse the checksum-audited Tier-1 exact-schema panel, HPO panel, canonical
closure, joint orthogonal-cover initial panel, and six-source-group conditional
confirmation. Do not rerun fitted cells. Verify raw manifests, hashes, exact
schema-risk identities, fANOVA reconstruction, canonical input closure,
strength-1 balancing, and strength-2 balancing.

Frozen primary gates are those in `DAY5_TIER1_PROTOCOL.md`,
`HPO_QUOTIENT_PROTOCOL.md`, `JOINT_ORTHOGONAL_COVER_PROTOCOL.md`,
`JOINT_COVER_CONFIRMATION_PROTOCOL.md`, and `STRENGTH2_COVER_PROTOCOL.md`.
Failed gates remain failed. Adaptive experiments retain their explicit
adaptive status.

## Candidate B — HeteroBag-3

Reuse the original four-dataset successful prospective panel only as prior
evidence. Run the frozen conditional second-panel Phase-1 screen in
`road-to-iclr-day-05/HETEROBAG_PHASE1_PROTOCOL.md`. Dataset IDs, splits,
architectures, seed triplet, representation rules, metric, model budget, and
screen gate are fixed in `heterobag_phase1_config.json`.

If Phase 1 fails, do not run more triplets or post-hoc variants. If it passes,
the additional triplets and all representation/placebo/diversity controls must
be frozen in a new addendum before running; without those controls HeteroBag
cannot be promoted to primary.

## Candidate C — FieldRiesz

Reuse Day-4 semantic geometry results. The predeclared King County spatial
replication failed the frozen hierarchy, the Bike cyclic result did not
separate correct from permuted geometry, and the direct broad panel did not
beat raw RAPLE broadly. Therefore FieldRiesz fails Phase 1 and receives no new
performance-method compute. Its status is `STOP_PERFORMANCE_METHOD` and
`KEEP_AS_MECHANISM/APPENDIX`.

## Final comparison

The dataset is the replication unit. Use paired dataset-level means, win rates,
and frozen proper-loss/RMSE effects. Seeds and architecture cells are repeated
measurements. Label every input as `DEVELOPMENT`, `PROSPECTIVE_CONFIRMATORY`,
`PROSPECTIVE_CONFIRMATORY_CONDITIONAL`, `ADAPTIVE_CONFIRMATORY`, or
`POST_GATE_EXPLORATORY`.

Score Novelty, Prospective replication, Dataset breadth, Architecture breadth,
Mechanistic clarity, Effect size/practical importance, Reviewer-defensible
controls, and Paper-story simplicity on 1–5 with weights
25/20/15/10/10/10/5/5 percent. Do not change these scores after computing the
weighted totals except to correct a documented factual or arithmetic error.

The final files are `FINAL_DIRECTION_DECISION.md`,
`final_direction_summary.csv`, and `final_direction_summary.json` under this
directory. The Markdown must follow the exact answer and terminal-decision
format in the sprint brief.

