# Leakage Audit — employee_salaries

- Status: `RUN`
- Source: STRING_BENCHMARK
- Metric definition: one minus padded character-trigram Jaccard similarity
- Information used to define the metric: employee_position_title string.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: full name, gross pay received, overtime pay, underfilled job title.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

