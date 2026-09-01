# Leakage Audit — acs_industry

- Status: `RUN`
- Source: ACS
- Metric definition: unweighted shortest path on frozen official-code prefix hierarchy
- Information used to define the metric: NAICSP, official Census crosswalk.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: PERNP, PINCP, other earnings, target-field aliases, allocation flags.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

