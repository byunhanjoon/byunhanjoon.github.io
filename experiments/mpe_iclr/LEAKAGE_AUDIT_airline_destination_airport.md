# Leakage Audit — airline_destination_airport

- Status: `RUN`
- Source: BTS
- Metric definition: haversine distance between official FAA airport coordinates
- Information used to define the metric: Origin/Dest airport code, FAA LAT_DECIMAL/LONG_DECIMAL.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: actual departure/arrival, taxi/wheels times, cancellation/diversion, delay causes.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.

