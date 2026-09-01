# Road to ICLR — Day 6: Native Feature Geometry

This directory studies a theory-first question: when a tabular feature has an
intrinsic metric or topology, should its neural interface respect that geometry
rather than treat its stored schema as the semantic object?

The proposed direction is **Native Feature Geometry (NFG)**.  A schema is a
coordinate chart (integer codes, a flat category, or a hierarchical
factorization).  The semantic object is a typed finite metric space.  NFG
compiles that object into a spectral embedding and asks three separate
questions:

1. do unconstrained neural embeddings recover the native geometry?
2. does geometry defect predict schema-dependent risk?
3. can the geometry reduce schema risk and causally transport a learned chart
   to categories absent from training?

`THEORY_FOUNDATIONS.md`, `RECENT_LITERATURE_AUDIT.md`, and
`PILOT_PROTOCOL.md` were frozen before any outcome-bearing pilot run.  The
pilot deliberately includes a nominal/random negative control and a corrupted-
metric control so that ordinary regularization or mere schema invariance cannot
count as support for the geometry claim.

## Status

Complete: 24 pilot bundles / 720 paths plus 12 H6 replay bundles / 120 paths and
600 causal dose interventions.  H2 and H4 fail; H5 and prospective H6 pass.
The surviving direction is narrowed to **Metric Chart Transport**, a synthetic
mechanism candidate rather than a paper lead.  See `PILOT_FINAL_REPORT.md`.

A pre-outcome construction audit changed the common rank from 8 to the complete
15-dimensional centered space so no tied eigenspace is split.  A later H1
analyzer precision correction is recorded in `ANALYSIS_CORRECTION_LOG.md`.

## Authoritative outputs

- `PILOT_FINAL_REPORT.md`: final verdict and numbers;
- `THEORY_FOUNDATIONS.md`: propositions and hypotheses;
- `PILOT_PROTOCOL.md` and `H6_TRANSPORT_DOSE_PROTOCOL.md`: frozen gates;
- `RECENT_LITERATURE_AUDIT.md` and `REVIEWER_ATTACK_AUDIT.md`: novelty and
  adversarial boundaries;
- `results/pilot_summary.json`, `results/h6_summary.json`, and
  `results/posthoc_diagnostics.json`: machine-readable evidence;
- `results/integrity_summary.json` and
  `results/analysis_reproducibility_summary.json`: audits.
