# Frozen expert-assignment permutation control

Frozen: 2026-09-01, before computing any cyclically misassigned routing loss.

## Question

Could competence routing help merely because it produces concentrated, episode-varying
weights, or must each context-CV loss be attached to the correct expert?

## Control

Reuse immutable predictions from the synthetic untouched test and the 9-classification/
16-regression real dataset identities (small real panel, unseen Day-8 OpenML panel, and
independent regression confirmation). Exclude context-scaling repeats to avoid duplicate
identity weighting.

For every episode, compute the frozen aligned soft weights. Then cyclically rotate the
six CV losses by offsets 1 through 5 before converting them to weights, preserving the
exact multiset, entropy, temperature, and concentration but breaking expert assignment.
The permutation-null loss is the mean query loss across those five nonidentity rotations.

Primary contrast is permutation-null minus aligned loss. Use the original equal-cell
bootstrap for synthetic tasks and a 10,000-draw hierarchical dataset/episode bootstrap
for real tasks. Recomputed aligned loss must match parent tables within `1e-5` (float32
storage bound).

A positive interval supports expert-specific loss alignment rather than generic dynamic
weight variation. This is an immutable-prediction mechanism control, not a new method or
independent performance confirmation.
