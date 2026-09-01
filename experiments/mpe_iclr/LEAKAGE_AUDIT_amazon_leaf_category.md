# Leakage Audit — amazon_leaf_category

- Status: `NOT RUN`
- Source: AMAZON_2023
- Metric definition: unavailable
- Information used to define the metric: .
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: .
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

Schema failure: Frozen raw_meta_All_Beauty snapshot contains no nonempty categories path, so the declared external hierarchy does not exist.

