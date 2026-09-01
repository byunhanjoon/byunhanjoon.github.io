# Native Feature Geometry — frozen pilot protocol

Status: **FROZEN BEFORE OUTCOME-BEARING RUNS**

Freeze timestamp: 2026-08-29 KST.  No pilot result existed when these gates
were written.

## Matrix

Semantic domains:

1. `cycle16`: a 16-cycle with periodic smooth effects;
2. `ordinal16`: a 16-path with ordinal smooth effects;
3. `tree16`: 16 leaves of a balanced depth-4 hierarchy with depth-decaying
   branch effects;
4. `nominal16`: equality geometry with outcome-independent random category
   effects (negative control).

Regimes:

- `interpolation`: every category occurs in training;
- `category_holdout`: four prospectively selected categories never occur in
  training or validation and are evaluated separately at test time.

Frozen held-category menus are cycle `[1,5,9,13]`, ordinal `[2,6,10,13]`,
tree leaves `[1,6,9,14]`, and nominal `[0,5,10,15]`.

Interfaces, all with representation width 15 and the same downstream MLP:

- `label`: schema-dependent scalar code lifted by a learned affine layer;
- `learned`: unconstrained trainable lookup table;
- `random_fixed`: chart-independent fixed random semantic table;
- `native_fixed`: chart-independent frozen native spectral table;
- `native_tuned`: native spectral initialization followed by unconstrained
  table training;
- `corrupt_fixed`: frozen native table with outcome-independent semantic-row
  corruption.

Replication:

- learner/data seeds: `7301, 7302, 7303`;
- four independently generated nonidentity label charts plus identity;
- 4 domains × 2 regimes × 3 seeds = 24 write-once bundles;
- each bundle contains all 6 interfaces × 5 charts = 30 trained paths;
- total planned paths: 720.

Charts change only stored integer codes.  Native and random semantic tables
are reindexed through the chart; `label` and `learned` receive raw chart codes.
The same semantic rows, targets, minibatch/full-batch order, optimizer, and
downstream initialization are used within a seed.

## Data-generating process

Each example contains one typed categorical value and three independent
standard-normal covariates.  The response combines a category main effect, a
category-by-continuous interaction, a shared nonlinear continuous term, and
Gaussian noise.  Structured category effects are analytic low-frequency
functions of their declared geometry.  Nominal effects are drawn before
training from an outcome-independent seeded generator and have no declared
neighborhood smoothness.

Training/validation/test sizes are 384/384/1024.  Test rows are balanced over
all categories and reuse the same semantic examples across charts and
interfaces.  Targets are standardized using training statistics for
optimization and reported in original units.

## Training

- deterministic PyTorch execution;
- full-batch AdamW;
- 400 epochs, no outcome-adaptive early stopping;
- learning rate `3e-3`, weight decay `1e-4`;
- SiLU MLP with two width-64 hidden layers;
- MSE objective;
- float32 parameters and activations.

The 15-dimensional interface retains the full centered space of each 16-value
domain.  This pre-outcome correction replaces an initial rank-8 draft that
would have split tied eigenspaces for the balanced tree and nominal controls.
Geometry remains nontrivial because spectral coordinates retain their native
eigenvalue weights; only the arbitrary truncation was removed.

No gate, domain, interface, seed, holdout menu, or training hyperparameter may
be changed after inspecting results.  A successor must receive a new protocol.

## Primary estimands

- original-unit test MSE, separately for seen and held categories;
- per-example prediction variance across schema charts;
- centered-kernel alignment (CKA) between an aligned learned embedding Gram
  and the declared native Gram, before and after training;
- H5 held-category MSE after original, native-transport, mean, random, and
  category-shuffled patches;
- maximum absolute seen-category prediction change after each held-row patch.

The independent unit for confirmatory summaries is a domain × seed cell.
Charts and test rows are repeated measurements, not independent replicates.

## Frozen gates

### H1 — Native Gram Equivariance

Pass only if all domain/chart construction tests satisfy maximum aligned Gram
error `<= 1e-10` in float64 and no retained rank splits a tied eigenvalue
block.  A construction failure blocks H2–H5 interpretation.

### H2 — Native Geometry Emergence

For each structured domain × seed × chart learned path, define CKA gain as
`CKA_final - CKA_initial`.  Pass only if:

The primary population is the `interpolation` regime, where every embedding
row receives training signal.  Holdout-regime CKA is descriptive because four
rows are untrained.

1. pooled median gain is at least `0.10`;
2. all three structured domains have positive median gain;
3. at least 80% of structured domain × seed cells have positive chart-median
   gain; and
4. final CKA exceeds alignment to the corrupted native Gram in at least 80%
   of structured paths.

The nominal control is descriptive and cannot rescue failure.

### H3 — Geometry–Schema Risk Coupling

Within the unconstrained learned and native-tuned paths, aggregate by domain ×
seed × chart in the `category_holdout` regime.  Chart-level orbit damage is the
mean squared difference from the across-chart quotient on held-category test
rows.  Pass only if:

1. pooled Spearman correlation between native CKA and held-category MSE is at
   most `-0.60`;
2. at least two of three structured domains have correlation at most `-0.40`;
3. pooled Spearman correlation between native CKA and chart-level orbit damage
   is at most `-0.50`.

Correlations are mechanism diagnostics, not independent-sample p-values.

### H4 — Native Geometry Mitigation

For each structured domain × seed cell, compare chart-mean metrics.  Pass only
if all hold:

1. `native_fixed` held-category MSE beats each of `label`, `learned`, and
   `random_fixed` in at least 7/9 structured domain × seed cells;
2. its pooled median relative held-MSE reduction against the best of those
   three baselines is at least 20%;
3. its pooled schema-orbit variance is at least 90% below both `label` and
   `learned`;
4. in interpolation, its median MSE is no more than 5% worse than the best of
   `learned` and `random_fixed`;
5. on `nominal16`, its median held-MSE improvement over the best baseline is
   less than 5% (negative-control boundary).

The corrupted interface cannot count as a baseline win.

### H5 — Native Chart Transport

Apply patches to both `learned` and `native_tuned` holdout paths, but freeze
`native_tuned` as the primary population.  Pass only if:

1. correct native transport beats every control patch in at least 7/9
   structured domain × seed cells after chart aggregation;
2. pooled median held-MSE reduction relative to the best control patch is at
   least 20%;
3. maximum seen-category prediction change is exactly zero in stored float32
   predictions for every patch;
4. shuffled/corrupted native transport does not achieve a comparable rescue
   (less than half the correct median relative reduction); and
5. the nominal control does not pass criteria 1–2.

## Verdict policy

Each hypothesis is independently kept, changed, or discarded.  H4 may pass
while H2 or H5 fails; that would support a useful fixed prior, not spontaneous
geometry recovery or chart transport.  Synthetic evidence alone caps the
direction at “mechanism candidate,” regardless of effect size.
