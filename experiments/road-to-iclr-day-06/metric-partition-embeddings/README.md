# Road to ICLR — Day 6: Metric Partition Embeddings

This directory tests whether a raw feature can be embedded through its declared
metric rather than a linear stored code.  The proposed Metric Partition
Embedding (MPE) maps distances to 16 landmarks into normalized weights and
mixes learned landmark tokens.

The result is conditional.  MPE gives exact schema invariance and strong
unseen-state interpolation on synthetic circles and trees, confirmed by an
equal-parameter neural model.  On UCI Bike Sharing it beats Q-PLE and a
code-distance RBF on average and beats a corrupted metric in 6/6 cells, but
fixed periodic features are better in both backbone means.  Multiscale MPE is
rejected.  This is a mechanism candidate for irregular typed metric spaces,
not a universal PLE replacement.

Start with `FINAL_REPORT.md`.  Frozen hypotheses and gates are in
`THEORY_FREEZE.md` and `PROTOCOL_FREEZE.md`.  Machine-readable summaries and
the main figure are under `results/`.
