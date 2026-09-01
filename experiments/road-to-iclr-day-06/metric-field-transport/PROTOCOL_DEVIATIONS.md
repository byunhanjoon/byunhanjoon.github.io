# Protocol deviations

## 2026-08-29 — duplicate baseline execution

The first E1b launch retrained nine `weights_m32` cells that are identical to
already-complete E0 `weights_direct` cells: all six ACS occupation cells and
the three TLC dropoff split-0 cells. Their initial scores, best scores, chosen
epochs, stopping epochs, parameter counts, and full learning curves match the
E0 artifacts exactly; only timing and device-memory telemetry can differ.

The duplication was detected during execution. Both E1b runners were
interrupted, leaving no partial JSON artifact because writes are atomic. The
nine completed artifacts are retained. All remaining overlapping E1b baseline
cells are materialized as explicit references to their E0 source artifacts and
are not retrained. No score, protocol condition, or promotion criterion changed.
