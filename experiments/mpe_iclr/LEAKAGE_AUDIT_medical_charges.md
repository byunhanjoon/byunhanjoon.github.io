# Leakage Audit — medical_charges

- Status: `RUN`
- Source: STRING_BENCHMARK
- Metric definition: one minus padded character-trigram Jaccard similarity
- Information used to define the metric: drg_definition string.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: provider identity/address/city/zip, Medicare payments, total discharges.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

