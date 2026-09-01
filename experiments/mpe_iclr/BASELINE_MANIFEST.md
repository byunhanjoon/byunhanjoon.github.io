# MPE ICLR Baseline Manifest

Frozen prospectively on 2026-08-29. No baseline may be removed because it wins,
is inconvenient, or makes the MPE story less novel. A failed implementation is
debugged; an objectively unavailable dependency is named in
`PROTOCOL_DEVIATIONS.md` and remains visible in every table.

## Shared comparison contract

All methods in a `dataset x task x setting x state split x backbone` cell use
the same rows, target transform, ordinary covariates, optimizer trial list,
early-stopping rule, and training seeds. Landmarks and every unsupervised
transform are fit on training states/rows only unless the method explicitly
uses target-independent public schema metadata for unseen nodes. Test labels
are inaccessible to construction and selection code.

The natural view uses the method's usual feature dimension. The parameter-
matched view inserts or narrows only learned projections so total trainable
parameters differ from MPE by at most 5% when mathematically meaningful. Fixed
features are not distorted to manufacture equality. Every result records
tokenizer, backbone, and total parameter counts separately.

`m=32` and `D=32` are primary. If fewer than 32 distinct training states exist,
`m` is their count with no duplicate landmark. Farthest-point traversal starts
at the training-state medoid (minimum sum distance) and breaks every tie by the
normalized state ID. Bandwidth is selected only on unseen validation states
from the frozen training-distance grid. All same-metric methods reuse the exact
same landmark IDs, pairwise distances, bandwidth candidate, and selected
bandwidth unless the method has no bandwidth.

## Generic mandatory baselines

### MPE (candidate, not a baseline)

For Gaussian affinities `a_j=exp(-d(x,l_j)^2/(2h^2))`, MPE normalizes
`w=a/sum(a)` and emits `wV`, with trainable `V in R^(m x D)`. Stable row-wise
log-sum-exp is used. Primary landmarks are training-only farthest-point
prototypes. Multiscale MPE is not run as a candidate; its preserved negative
Day-8 result is cited.

### A/B — categorical lookup and support-complete categorical

- `unknown_embedding`: a train-state lookup table of width `D`; every unseen
  validation/test state maps to one shared learned UNK vector.
- `support_complete_categorical`: one-hot plus an unseen column for ridge/tree
  models and the identical lookup+UNK construction for neural models. It is
  listed separately to make the linear and neural formulations explicit.

### C/D — code-based PLE

- `q_ple`: exact quantile piecewise-linear encoding of arbitrary stored integer
  codes with `D` bins, trained on row values.
- `uniform_ple`: `D` support-complete uniform intervals from training code
  minimum to maximum. For cardinality below `D`, a deterministic full-rank
  support-complete basis is used and zero-padded. Neither code method sees the
  semantic metric.

The real codebook intervention applies eight arbitrary bijections. MPE and all
metric-aware methods receive transported metric metadata; code methods do not.

### E — Similarity Encoding showdown

`similarity_same_metric` exposes the normalized affinity vector
`[w_1,...,w_m]` directly. `similarity_unnormalized` exposes `[a_1,...,a_m]`.
The metric, landmarks, bandwidth, rows, and backbone are identical to MPE.
Natural comparison uses all `m` coordinates. The D-dimensional view is identity
when `m=D`; otherwise a training-only randomized SVD projection is frozen.

The report must state the architectural fact that, when `m=D` and a backbone
begins with an unconstrained linear map, normalized Similarity Encoding can
represent the same first linear composition as MPE. Optimization differences
are empirical; they are not described as added representational information.

### F — RBF and Nyström

- `rbf_normalized`: the same normalized Gaussian affinity vector.
- `rbf_unnormalized`: the same unnormalized Gaussian affinities.
- `nystrom`: `K_XL (K_LL + 1e-8 I)^(-1/2)`, with the same Gaussian kernel,
  landmarks, and bandwidth. The eigendecomposition clips eigenvalues at
  `1e-8 * max_eigenvalue` and records effective rank.
- `random_kernel_features`: Gaussian random Fourier features only when the
  metric has an explicit Euclidean coordinate chart; it is never presented as
  an arbitrary-metric method.

### G — metric kNN

`knn_metric` uses `k in {1,3,5,10}` chosen on validation. The isolated ridge
predictor is the inverse-distance weighted mean of training-state mean targets,
where state means use training labels only. In full tables this scalar and a
same-neighbor ordinary-feature centroid are appended to the unchanged ordinary
covariates. Zero distances are averaged exactly. No validation/test target is
ever a neighbor label.

### H/I — causal and no-information controls

- `mpe_corrupt`: ten frozen permutations of the state-to-geometry association,
  preserving state count, matrix dimensions, parameter count, and, for graph/
  hierarchy fields, the graph itself. Geography permutes coordinates. The
  primary corrupt result is the mean across the ten instances; all ten remain
  in raw results.
- `mpe_equality`: replaces `d` with `0` on the diagonal and `1` off-diagonal.
  Gaussian weights for every genuinely unseen non-landmark state must be
  identical. This is a theorem validation, not a straw-man competitor.

Partial corruption at 10%, 25%, 50%, and 100% permutes that frozen proportion
of state IDs while holding the rest fixed. The representative subset is ACS
occupation, TLC pickup, Citi start station, BTS origin, Amazon category, and
medical charges.

## Hierarchy-specific mandatory baselines

Applied to ACS occupation, ACS industry, and Amazon category.

- `ancestor_multihot`: binary root-to-node ancestors plus normalized depth.
- `path_to_root`: inverse-depth weighted path coordinates; its natural sparse
  dimension is the number of schema nodes.
- `hierarchy_shortest_path_similarity`: similarities to the same landmarks.
- `wu_palmer`: `2*depth(LCA)/(depth(x)+depth(l))`, with root depth one.
- `lch_path`: `-log((shortest_path+1)/(2*max_depth+1))`, avoiding undefined
  zero values. The exact convention is recorded.
- `laplacian`: leading nonconstant symmetric-normalized-Laplacian eigenvectors
  of the public full hierarchy graph. Eigenvector signs are fixed by making the
  largest-magnitude entry positive. Unseen test nodes may use the public graph.
- `node2vec`: graph-topology-only skip-gram coordinates with dimensions 32,
  walk length 20, ten walks/node, window five, `p=q=1`, and frozen seed.
- `tree_rbf`: unnormalized and normalized shortest-path RBF/Nyström features,
  using MPE's exact landmarks and bandwidth.

No hierarchy ancestor/path feature appears among MPE's ordinary covariates.

## Geographic/network mandatory baselines

Applied to TLC zones, Citi stations, and BTS airports.

- `raw_coordinates`: longitude and latitude represented as unit-sphere
  `(x,y,z)` coordinates; the report also displays raw two-coordinate results.
- `coordinate_mlp`: a two-layer ReLU map to `D`, with width chosen for the 5%
  parameter-matched view.
- `coordinate_fourier`: sine/cosine features at powers-of-two frequencies on
  normalized longitude/latitude, truncated or padded to D.
- `spatial_rbf`: Euclidean/unit-sphere RBFs at the exact MPE geographic centers.
- `graph_laplacian`: leading nonconstant eigenvectors where the frozen
  adjacency/route graph exists.
- `node2vec`: the same graph-only settings as above.
- `shortest_path_similarity`: normalized and unnormalized similarities using
  the exact shortest-path metric, landmarks, and bandwidth given to graph MPE.

Coordinates are available to every coordinate baseline and are never included
only for MPE. Results under geographic and graph distance are distinct metric
variants, not independent sources.

## String-specific mandatory baselines

On all three secondary tasks, prototype 3-gram Similarity Encoding is the main
baseline and uses the exact MPE prototypes. Additional controls are character
3-gram hashing, one-hot/UNK, Jaro-Winkler prototype similarity, normalized
Levenshtein prototype similarity, and their MPE counterparts. Strings are
lower-cased, Unicode NFKC-normalized, and internal whitespace is collapsed.

## Cyclic, interval, and ordinal boundaries

The preserved UCI Bike hour result is carried forward: fixed Fourier features
beat MPE in both backbones. Synthetic and any optional real cyclic ablation must
include sine/cosine Fourier features. Interval validation compares Q-PLE,
uniform PLE, triangular ordered-landmark MPE, and Gaussian MPE. An ordinal task,
if added, must include raw/normalized rank and monotonic embedding; no ordinal
task is required for the primary source gate.

## Tree models and neural backbones

CatBoost with its native categorical field and LightGBM are run on every real
task. On one hierarchy, one geographic, and one string task, CatBoost/LightGBM
also receive MPE weights to test complementarity. They use the same split and
ordinary covariates but are reported outside neural parameter matching.

Primary neural backbones are MLP, residual MLP, FT-Transformer, and official
TabM. Stable repository/reference implementations are used. For fixed dense
representations, FT-Transformer receives one metric-field token plus ordinary
field tokens; TabM receives a dense stem. Representation comparisons within a
backbone share every backbone hyperparameter. Ridge/linear heads are run for
every dataset and representation as the mechanism view.

## Landmark, kernel, normalization, and dimension ablations

On ACS occupation, TLC pickup, Citi start station, Amazon leaf category, and
medical charges:

- landmarks: farthest point, k-medoids, uniform random training states, and
  frequency-weighted training states at `m=16,32,64`;
- kernels: Gaussian, Laplacian, triangular compact, inverse distance;
- normalization: partition weights, unnormalized RBF, softmax `(-d/h)`;
- dimensions: 16, 32, 64;
- budgets: 8, 16, 32, 64, 128 and 256 where cardinality permits.

Primary results always remain Gaussian, farthest point, normalized, `m=D=32`.
No post-outcome MPE variant can replace them.

## Efficiency accounting

Every fit records trainable parameters, preprocessing/metric/landmark time,
training wall time, inference rows/second, peak host memory, peak GPU memory,
and serialized representation bytes. Dense MPE lookup is reported as `O(m)`;
the frozen `k={4,8}` sparse variants as `O(k)` after state-to-weight precompute.
Spectral/node2vec preprocessing is charged rather than hidden.
