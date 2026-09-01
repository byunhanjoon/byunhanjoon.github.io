# Prospective audit

Status: **PASS; SEPARATELY FROZEN HIERARCHY GAP ALSO PASSES**

- Frozen hashes match:
  - protocol `f75dfede8138098463df60902484262a576a0b89f9796cf8a65065e4e569a7ba`;
  - config `36b2b06903f58d2a09866d88bd1f550fcc8250ae99a4328dfa2e38e66304241b`.
- Nine prediction seals exist (three sources × three splits). Every seal hashes
  correctly, marks `outer_test_outcomes_accessed=false`, and has disjoint state
  IDs.
- Twenty-seven outer cells are retained (nine seals × three operators).
- Nested predictions retrain the base, row-OOF residuals, state effects, Sigma,
  and operator inside every inner state fold.
- Outer-test outcomes are joined only after the prediction file is atomically
  written.
- NOAA GHCN (30 states), Beijing air quality (12 states), and Chicago Divvy
  (80 states) ran. The BLS hierarchy source is explicitly `NOT RUN`: both its
  legacy frozen path and current official path returned HTTP 403. It was not
  replaced.
- The UCI wrapper-archive loader correction occurred before outcome evaluation
  and is recorded in `PROTOCOL_DEVIATIONS.md`.
- Aggregate predicted-versus-actual Spearman is `0.9167`; direct sign accuracy
  is `1.000`; all frozen P1–P5 rules evaluate true.

P5 is vacuous because no source/operator aggregate was harmful. P4 has exactly
three rather than four runnable families, and there is no prospective hierarchy
result inside the original protocol. Those facts remain part of its immutable
audit trail.

The post-program hierarchy addendum was frozen separately at protocol/config
hashes `36b20009...bb2e28` and `3bdec9c0...68e7fd`, before the Census outcome
archive was acquired. It adds three valid prediction seals and nine retained
cells over 922 six-digit NAICS industries. Split-level Spearman is `0.8333`,
sign accuracy is `100%`, and all frozen G1–G5 criteria pass. The combined
four-family, 12-aggregate prospective result has Spearman `0.9091` and `100%`
direct sign accuracy. The hierarchy/breadth limitation is therefore closed by
an explicitly separate evidence layer. P5 remains observationally vacuous
because Census also produced no harmful aggregate.
