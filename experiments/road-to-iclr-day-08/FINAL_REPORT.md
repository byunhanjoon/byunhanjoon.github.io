# Day 8 final report: can a metric-aware tokenizer beat PLE?

## Bottom line

Yes, a coherent new embedding can be built:

```text
raw typed value x
  -> distances d(x, landmark_j)
  -> normalized local weights w_j(x)
  -> learned token sum_j w_j(x) v_j.
```

I call this **Metric Partition Embedding (MPE)**.  It is nonlinear in the raw
value, accepts interval, cyclic, tree, or equality metrics, and is exactly
invariant to equivalent code relabellings when the metric is transported.

The experiment supports MPE as a mechanism for **interpolating unseen states
of a declared metric space**.  It does not support a universal PLE replacement
or a new best cyclic encoder.  The frozen primary protocol formally fails
because its nominal negative control was too strict and its multiscale
hypothesis failed.  Corrective controls explain the nominal failure but cannot
retroactively turn the primary screen into a full pass.

## The hypotheses, simply

1. **A schema is a chart, not the feature.** Integer `0` and `23` are far apart
   on a line but adjacent as hours.  The tokenizer should see hour distance,
   not storage-code distance.
2. **Nearby values should share a token trace.** MPE mixes landmark vectors by
   metric proximity, so a value absent from training can borrow from semantic
   neighbors.
3. **Equivalent schemas should give the same answer.** Relabel the categories
   and transport the metric; MPE is unchanged exactly.
4. **Wrong geometry must hurt.** A randomly permuted metric has the same
   dimensions and model capacity.  Correct MPE must beat it for geometry to be
   causal rather than decorative.
5. **Nominal data are the no-free-lunch case.** An equality metric says nothing
   about an unseen distinct state.  Any nominal gain must come from resolving
   states or conditioning, not topology.
6. **More scales are not automatically better.** The proposed equal
   half/base/double-bandwidth mixture should win only if multiple resolutions
   are useful.  It did not.

## What was tested

The fixed 16-coordinate screen compared raw linear input, exact quantile PLE,
periodic features, code-space RBFs, native MPE, multiscale MPE, and a corrupted
metric.  It used 12 data seeds, eight equivalent schemas, and four domains:
interval, 32-state cycle, 31-node tree, and 16-state nominal.  Cycle and tree
test rows came only from states absent during training.

The primary screen contains 2,688 parameter-matched ridge fits.  Post-hoc
controls add 1,152 fits for local Q-PLE, whitened Q-PLE, and uniform PLE.  The
neural confirmation adds 144 equal-parameter fits (5,569 parameters each), and
the Bike diagnostic adds 36 MLP/ResNet fits.  Total: 4,020 reported fits.

## Frozen ridge results

Mean reduction in standardized test MSE relative to quantile PLE:

| Domain | native MPE | multiscale MPE | native vs corrupt | MPE wins vs PLE |
|---|---:|---:|---:|---:|
| Interval | +8.52% | +9.80% | +93.89% | 10/12 |
| Cycle, unseen states | **+98.87%** | +98.85% | **+99.38%** | 12/12 |
| Tree, unseen leaves | **+53.66%** | +45.49% | **+76.49%** | 12/12 |
| Nominal | +77.64% | +77.64% | 0.00% | 12/12 |

The nominal row is deliberately alarming.  Correct and corrupted equality
metrics are identical, so geometry cannot explain the 77.64% gain.  Q-PLE
occasionally collapses discrete states when empirical quantiles repeat under a
codebook.  A post-hoc 16-bin uniform PLE that can isolate all 16 states reduces
MPE's mean nominal advantage to **0.094%**.  That is the right negative result:
state resolution explains the nominal gain.

The corrective control is still strong on structured unseen states.  Native
MPE beats uniform PLE in 12/12 cycle and 12/12 tree seed aggregates, by 99.73%
and 85.56% mean MSE respectively.  It also beats local and whitened Q-PLE in
12/12 on both domains.  Hence basis conditioning and discrete resolution do
not explain the structured interpolation result.

MPE predictions are numerically identical across all eight transported
schemas; PLE's median schema-loss coefficient of variation is 0.316 on cycle
and 0.304 on tree.  However, the predeclared global pairwise-distance
distortion is not a useful risk score for arbitrary codebooks: median
within-seed Spearman correlations with PLE loss are -0.167 on cycle and +0.036
on tree.  Exact invariance passes; the simple scalar risk predictor fails.

## Neural confirmation

The learned interface was a 16-to-16 landmark-token layer followed by the same
two-layer MLP for every method.  On three fresh seeds and four schemas:

| Unseen domain | MPE gain vs PLE | vs periodic | vs code-RBF | vs corrupt | wins in each comparison |
|---|---:|---:|---:|---:|---:|
| Cycle | **+97.32%** | +97.63% | +97.85% | +98.85% | 12/12 |
| Tree | **+62.27%** | +59.03% | +66.36% | +61.16% | 12/12 |

This passes the frozen Stage C gate.  It is nonetheless a synthetic mechanism
test whose targets were constructed to be smooth in the native metric.

## Real cyclic diagnostic

On UCI Bike Sharing, only the hour field changed and every representation had
16 hour coordinates.  The split was chronological and all methods shared the
other 86 input features.

| Backbone | MPE gain vs Q-PLE | vs code-RBF | vs periodic | vs corrupt |
|---|---:|---:|---:|---:|
| MLP | +5.24% | +11.37% | **-6.22%** | +40.27% |
| ResNet | +3.98% | +8.52% | **-8.20%** | +7.57% |

Correct ring MPE beats the corrupted ring in 6/6 cells and passes its narrow
diagnostic gate.  But fixed Fourier features have lower mean loss in both
backbones and win 6/6 head-to-head.  On a fully observed clean cycle, Fourier
coordinates are simpler and better.  MPE's plausible niche is irregular
metric spaces and unseen-state interpolation, not ordinary hour encoding.

## Ranking after novelty subtraction

### 1. Single-scale MPE for typed discrete metric spaces — highest potential

Why it survives: exact chart invariance, 12/12 ridge and 12/12 neural wins on
both unseen structured domains, correct-over-corrupt causality, and a clean
nominal capacity control.  It extends the PLE tokenizer interface to circles,
trees, and arbitrary finite metrics.

Why it is not yet a paper: Gaussian partitions and landmark interpolation are
classical; synthetic targets favor the declared metric; the sole real cyclic
diagnostic loses to Fourier features; real tables rarely supply reliable field
metrics.

Best next test: a prospective panel where meaningful states are genuinely
missing from training—new locations, products in an ontology, road-network
nodes, medical-code descendants, or device states—and where the metric is
available from schema metadata alone.

### 2. Geometry-aware safe chart bank — medium potential, crowded

Use PLE for intervals, Fourier for clean cycles, equality/uniform PLE for
nominal low-cardinality fields, and MPE for irregular graphs/hierarchies.  A
zero-start residual or validation gate could preserve the best conventional
path.

This is likely the strongest practical system, but its research novelty is
lower: Day 4 TriChart already showed multi-chart complementarity and failed a
broad confirmation gate.  A successor must learn when MPE has an extrapolation
advantage without using test-only unseen states.

### 3. Support-complete uniform PLE for low-cardinality coded fields — useful,
low novelty

It removes the nominal Q-PLE collapse and ties MPE within 0.094%.  This is a
good engineering fallback and schema-risk control, not a new representation
paper; exact-state and support-aware variants were already explored on Days
1--4.

## Ideas rejected or demoted

- **Multiscale MPE:** worse than single-scale MPE on all 12 cycle and all 12
  tree ridge aggregates; not better in both Bike backbones.
- **Code-space RBF:** severe schema sensitivity and worse than native MPE in
  every structured ridge/neural aggregate and on Bike.
- **Universal better-than-PLE claim:** rejected.  The advantage is conditional
  on a trustworthy metric and an interpolation problem.
- **Global metric-distortion risk score:** rejected for arbitrary relabellings;
  a task-weighted local distortion measure is needed.

## Why the residual could still be novel

[Numerical PLE and periodic embeddings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html),
[random kernel features](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html),
function bases, and graph positional encodings are established.  The possible
novelty is not any ingredient.  It is the exact combination of a per-field
tabular tokenizer, arbitrary declared feature metrics, transported-schema
equivariance, unseen-state evaluation, and a capacity-matched corrupted-metric
intervention.  That is differentiated enough for a prospective experiment,
not enough to claim novelty before a broader search and real evidence.

## Reproduction map

- `THEORY_FREEZE.md`, `PROTOCOL_FREEZE.md`: pre-outcome theory and gates;
- `metric_partition_benchmark.py`: frozen ridge screen;
- `POSTHOC_BASIS_CONTROL.md`, `POSTHOC_CAPACITY_CONTROL.md`: transparent
  corrective controls;
- `neural_confirmation.py`: equal-parameter neural confirmation;
- `BIKE_CONFIRMATION_FREEZE.md`, `bike_mpe_confirmation.py`: real cyclic
  diagnostic;
- `results/ridge_summary.json`, `basis_control_summary.json`,
  `neural_summary.json`, `bike_summary.json`: machine-readable decisions;
- `results/figures/method_screen.png`: screen overview;
- `audit_integrity.py`, `test_metric_partition.py`: integrity and construction
  checks.
