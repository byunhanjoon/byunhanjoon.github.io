# Day 6 Metric Partition Embeddings frozen pilot protocol

Frozen before outcome-bearing runs on 2026-08-29.

## Candidates and equal interface

Every main representation has 16 coordinates and uses the same ridge or neural
predictor and split.  The primary screen compares:

1. `linear`: standardized stored scalar padded to 16 coordinates;
2. `ple`: exact quantile piecewise-linear encoding;
3. `periodic`: eight sine/cosine pairs on the stored scalar;
4. `code_rbf`: normalized Gaussian landmark weights using stored-code distance;
5. `mpe_native`: the same weights using the declared native metric;
6. `mmpe_native`: equal mixture at half, one, and twice the cover scale;
7. `mpe_corrupt`: identical MPE after a frozen random relabelling of the metric.

Landmarks are selected without labels by farthest-point traversal under the
relevant metric.  Ridge strength is selected on validation loss from a frozen
grid.  Test labels never choose a representation, bandwidth, or ridge penalty.

## Domains

- `interval`: a continuous scalar with smooth and threshold components;
- `cycle`: a 32-state circle, with every fourth state absent from training;
- `tree`: a balanced 31-node hierarchy, with a subset of leaves absent from
  training;
- `nominal`: 16 unrelated states with random state effects and no held-out
  states.

The interval uses monotone equivalent storage charts.  Cycle, tree, and nominal
use random one-to-one integer codebooks.  Native metrics are transported with
the schema.  Code-space methods see only the stored scalar.

## Stage A: deterministic construction audit

Required before outcome analysis:

- finite, row-normalized MPE/MMPE weights;
- exact 16-coordinate interface;
- native MPE predictions invariant, to numerical tolerance, after relabelling
  when fitted by the deterministic ridge procedure;
- corrupt metric differs from native metric on structured domains;
- no target values enter landmarks or feature maps.

## Stage B: frozen ridge screen

- 12 data seeds;
- 8 equivalent schemas per seed;
- all four domains and seven methods;
- proper loss is MSE on a standardized target;
- primary unit is a seed x domain aggregate over schemas, not an individual
  row or schema.

Primary summaries:

- mean relative MSE reduction against PLE;
- wins against PLE across seed x domain units;
- unseen-state MSE for cycle/tree;
- schema risk: coefficient of variation and worst/best loss ratio across the
  eight equivalent schemas;
- native-minus-corrupt and native-minus-code-RBF gaps.

## Gates

### H1: interval safety

`mmpe_native` loses no more than 2% mean MSE versus PLE on the interval.

### H2: topology interpolation

On both cycle and tree, `mmpe_native` must beat PLE, code-RBF, and the corrupted
metric in at least 9/12 seed aggregates and by at least 10% mean MSE on unseen
states.

### H3: schema stability

On cycle and tree, MMPE must reduce median schema CV by at least 75% versus PLE,
and its maximum numerical prediction discrepancy across transported schemas
must be below `1e-8` in the deterministic audit.

### H4: nominal negative control

MMPE must not beat PLE by more than 2% mean MSE on nominal data, and native and
corrupt metrics must be effectively tied.  Otherwise generic capacity rather
than geometry could explain the result.

### H5: multiscale residual

MMPE must beat single-scale MPE on both structured domains by positive mean
gain without violating H1/H4.  If not, retain the simpler MPE.

## Stage C: neural confirmation

Only the best geometry candidate from Stage B is eligible.  Confirm on cycle
and tree with a learned 16-to-16 landmark-token layer and a shared two-layer
MLP, comparing PLE, periodic, code-RBF, candidate, and corrupt metric.  Use
three fresh seeds and four frozen schemas.  Promotion requires positive mean
gain and at least 8/12 wins versus every control on each domain.

## Interpretation boundary

Passing synthetic gates establishes a mechanism and a usable typed tokenizer
prototype.  It does not establish a general PLE replacement or an ICLR-ready
method.  A prospective real-data panel with declared cyclic, spatial, ordinal,
or hierarchical metadata remains mandatory.
