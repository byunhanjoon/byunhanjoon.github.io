# Basis Dependence Confirmation — Prospective Protocol Lock

This protocol was written before any outcome from the new confirmation experiment was run.
The earlier six-dataset semantic-orbit experiment motivated the hypotheses but is not part of
the prospective confirmation sample.

## Dataset assignment

Eighteen OpenML datasets were selected using dataset metadata only. Dataset keys were sorted,
then permuted with `numpy.random.default_rng(20260901)`. The first eleven are development and
the final seven are prospective. The exact IDs, versions, task types, assignment order, and
panel labels are in `configs/dataset_panel.json`.

No prospective model outcome may be read until `configs/FROZEN_METHOD_CONFIG.json` exists and
its SHA-256 is recorded. Development failures may change implementation but not dataset panel,
model families, seed set, orbit size, headline transforms, or holdout membership.

## Primary estimand

For each dataset × model, the primary estimand is mean prediction disagreement across eight
condition-one, within-feature-block orthogonal transforms. One-block and independently rotated
all-block orbits are reported separately. Condition≤3 transforms are a secondary control.
The dataset × model cell—not an orbit member—is the inferential unit.

## Frozen implementation choices

- Continuous features have at least 16 training-unique values. Each is replaced by an 8-RBF
  block with centers at training quantiles and width equal to the median positive adjacent gap.
- Other numeric and categorical values are represented by target-free training-fitted one-hot
  blocks, keeping a common numeric input across model families.
- The one-block feature is the valid continuous feature with greatest training variance, ties
  broken lexicographically.
- Each orthogonal matrix is obtained from seeded Gaussian QR, has deterministic QR signs and
  positive determinant. General transforms have singular values in `[1/sqrt(3), sqrt(3)]`.
- Train/validation/test row identities are fixed by split seed 20260901; model seeds are 0, 1,
  and 2; training rows are capped at 1,024, validation at 256, and test at 512.
- TabICLv2 and TabPFN-2.6 use pinned official checkpoints and one estimator. Trainable models
  are refit independently under each representation.
- The controlled MLP is globally fixed at three GELU layers of width 256, no batch norm, and
  zero dropout. Its default optimizer is AdamW.
- Development remedies are raw, per-coordinate standardization, within-block whitening,
  deterministic PCA/SVD coordinates, AnchorCanonical, lambda=1 orbit consistency, and a labeled
  oracle inverse ceiling. Holdout receives only the frozen primary method plus required controls.
- Natural pairs are one-hot/Helmert, local/spectral hat, and Fourier-origin representations.
- Polynomial pairs, expanded HPO, and dual-view refinement are the first optional items dropped
  if compute runs long. Prospective validation is never dropped.

## Decision criteria

The verdict and useful-repair thresholds are exactly those in the authoritative 1,300-line
agent specification. In particular, a useful repair targets at least 70% median disagreement
reduction with at most 1% median relative task degradation and must succeed prospectively.

