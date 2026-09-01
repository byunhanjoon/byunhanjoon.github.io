# Leakage Audit — citibike_start_station

- Status: `RUN`
- Source: CITI_BIKE
- Metric definition: haversine distance between target-independent pooled published station coordinates
- Information used to define the metric: start_station_id, published start_lat/start_lng across frozen periods.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: end time, duration, future target values, post-trip measurements.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

