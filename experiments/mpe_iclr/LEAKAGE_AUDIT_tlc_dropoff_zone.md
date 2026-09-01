# Leakage Audit — tlc_dropoff_zone

- Status: `RUN`
- Source: NYC_TLC
- Metric definition: haversine distance between official taxi-zone polygon centroids
- Information used to define the metric: official taxi zone polygon, EPSG:2263 to EPSG:4326 transform.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: actual trip distance, fares, tips, tolls, payment type, dropoff time components.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

