# Prospective anytime nested-cover protocol

Status: frozen before outcomes.

Post-gate scope addendum (2026-08-28): after evaluating the frozen
confirmation and external panels, apply the identical schedule to the
task-balanced classification/regression panel and the four-class panel.
These additional panels do not alter the original gate.

## Question

Can one random nuisance schedule be stopped after 4, 16, or 64 fits while
retaining exact strength 1, 2, or 3 respectively? This addresses a practical
weakness of separately constructed covers: a user should not have to discard
early fits when increasing the robustness budget.

## Construction

Use the `GF(4)^3` strength-3 array. Its 16 rows satisfying `w=u+v` form a
strength-2 subarray. Within that subarray, the rows

`(u,v) in {(0,0),(1,3),(2,1),(3,2)}`

form a strength-1 subarray. Order these four rows first, the remaining
strength-2 rows next, and the remaining strength-3 rows last. Apply one set of
independent, uniformly random level permutations to the complete ordered
array. Projection handles singleton or binary category/class factors.

The construction must pass exhaustive margin checks at every checkpoint and
the prefixes must be literal subsets, not separately generated designs.

## Frozen evaluation

Reuse saved full nuisance tensors only. Evaluate held-out test residuals on:

- the 25 validation-material confirmation cells;
- the 12 validation-material untouched OpenML cells.

At each checkpoint compare the exact expected prefix residual against equal-
budget IID. Also compare against the corresponding separately randomized
strength cover to quantify any cost of nesting. The prospective gate passes
if, on both panels, every checkpoint has lower pooled residual than IID and
the 16- and 64-fit checkpoints are lower than IID on at least 75% of cells.
Report all cell results even if the gate fails. No predictive tensor is
regenerated or selected after observing test outcomes.
