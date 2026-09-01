# Final Prospective Protocol — Metric Partition Embeddings for ICLR

Frozen on 2026-08-29 before acquisition or inspection of any new target-bearing
outcome. This protocol tests and may falsify MPE; it is not a rescue program.
The preserved Day-8 conclusions under
`experiments/road-to-iclr-day-06/metric-partition-embeddings` are inputs and are
not rerun for better seeds.

## 1. Question and immutable candidate

The primary question is whether one metric-space tokenizer can use externally
supplied feature geometry to generalize to genuinely unseen tabular states more
effectively than PLE, categorical/UNK encodings, Similarity Encoding, kernel/
Nyström features, kNN, hierarchy/graph encodings, and coordinate specialists.

The candidate MPE is immutable:

```text
a_j(x) = exp(-0.5 * (d(x,l_j)/h)^2)
w_j(x) = a_j(x) / sum_k a_k(x)
MPE(x) = sum_j w_j(x) v_j
```

Primary choices are 32 training-state-only farthest-point landmarks, token
dimension 32, one Gaussian scale, and validation-only bandwidth selection.
Fewer than 32 training states imply one landmark per state and no duplicates.
Multiscale MPE remains rejected and cannot return as the primary method.

## 2. Frozen evidence boundaries

The following are carried into every conclusion:

1. synthetic unseen cycle/tree interpolation was strong;
2. support-complete PLE explained the nominal Q-PLE result;
3. multiscale MPE failed;
4. Fourier beat MPE on observed cyclic Bike hour;
5. transported chart invariance was exact;
6. arbitrary global code distortion was not a useful scalar risk predictor.

The new real panel is decisive. Synthetic data validate mechanisms and
theorems but cannot establish usefulness because their smooth targets are
constructed from the supplied metric.

## 3. Files frozen with this protocol

`final_config.json`, `DATASET_MANIFEST.md`, `BASELINE_MANIFEST.md`, and
`THEORY_PLAN.md` are normative. SHA-256 digests in `PROTOCOL_HASHES.txt` define
the freeze. Later edits to any normative file are forbidden. Clarifications or
implementation corrections go only in `PROTOCOL_DEVIATIONS.md`, with timestamp,
reason, affected cells, whether outcomes had been seen, and impact on claims.

## 4. Datasets, source units, and tasks

The primary external-metric panel contains eight field tasks from five
independent public sources:

| Source unit | Field tasks | Metric family |
|---|---|---|
| ACS | occupation, industry | official SOC/NAICS hierarchy |
| NYC TLC | pickup zone, dropoff zone | official-zone geography/adjacency |
| Citi Bike | naturally new start station | published geography/train-only network |
| BTS | origin airport, destination airport | FAA geography/train-only route graph |
| Amazon 2023 | leaf product category | published category hierarchy |

This exceeds six primary field tasks and spans hierarchy, geography, and graph
geometry. The secondary string panel contains Employee Salaries, Medical
Charges, and Open Payments. MIMIC-III is prospectively `NOT RUN — CONTROLLED
ACCESS UNAVAILABLE`; no credentials/local data were present at freeze.

Exact sources, targets, filters, row caps, covariates, and split constructions
are in `DATASET_MANIFEST.md`. No target can be switched after performance is
seen. No source/task can be silently removed. An unavailable required public
file stays in tables as not run and weakens the final decision.

## 5. Split and sampling rules

Primary controlled experiments use five independent state partitions and three
neural training seeds. The metric-field state sets are pairwise disjoint:

```text
S_train intersect S_validation = empty
S_train intersect S_test       = empty
S_validation intersect S_test  = empty.
```

Rows inherit state membership. Validation therefore also consists of unseen
states. Natural Citi Bike states follow the frozen temporal rule and use three
training seeds; five controlled state partitions are a replication. Hard
hierarchy/spatial blocks are secondary and label-free. Every task also has a
seen-state IID row control.

Primary state eligibility is at least 50 rows; secondary strings use 20. State
filters happen before assignment and can never depend on target performance.
Row caps use stable source IDs hashed with the global seed. For huge sources,
sampling is balanced by frozen month/file before the hash rule. The primary
unseen-state metric is state-balanced; row-weighted, worst quartile, and worst
decile results are also mandatory.

## 6. Metrics and landmark construction

All geometry exists independently of the target. Hierarchies use official
paths; geography uses published coordinates/polygons; graphs use topology or
training-period non-target edges; strings use the raw string and are explicitly
secondary; nominal controls use equality.

Farthest-point landmarks are selected from unique training states only. The
initial landmark is the training-state medoid and ties use normalized state ID.
Secondary landmark choices are k-medoids, uniform random training prototypes,
and frequency-weighted training prototypes. No test frequency, target, or
performance enters selection.

Bandwidth candidates are exactly those in `final_config.json`, computed from
training-state distances. The bandwidth minimizing state-balanced validation
loss is selected, ties favor the smaller bandwidth, and it is then fixed before
test evaluation. Same-metric baselines share it. A construction with zero
denominator fails loudly rather than silently changing the kernel.

## 7. Models, fairness, and validation budget

Every dataset uses ridge/linear heads. Primary neural backbones are MLP,
residual MLP, FT-Transformer, and official TabM. CatBoost native categorical
and LightGBM run wherever the prepared table is valid. A representative
hierarchy/geography/string subset also appends MPE features to each tree model.

The eight frozen hyperparameter trials are an equal maximum for every
`dataset-task-setting-backbone-representation` cell. Eight, rather than the
recommended 20, is prospectively chosen because all five state sources,
15 final seed/split fits per cell, and two settings are retained under the
available 19 GB disk budget. This reduces tuning resolution but does not favor
MPE. Trials are generated once from the Cartesian space in `final_config.json`
using the HPO seed and reused by index. Selection minimizes the primary
state-balanced unseen-validation loss averaged over tuning replicates. Test
states remain sealed.

Maximum training is 300 epochs with patience 30 and recorded best/stop epoch,
loss curves, wall time, and seed. ACS occupation, TLC pickup, and Amazon category
repeat representative winning validation configurations at 600 epochs; a
material validation improvement is defined prospectively as more than 1%.
OOM handling may reduce batch size only, must preserve optimizer/epochs, and is
logged as a deviation.

Natural and parameter-matched comparisons follow `BASELINE_MANIFEST.md`.
Representation output dimension is 32 primary, with 16/64 ablations. Tokenizer,
backbone, and total parameters are always separate. Fixed methods are not given
artificial trainable layers solely to match counts.

## 8. Mandatory comparisons

All tasks include MPE, lookup/UNK, support-complete categorical, Q-PLE, uniform
PLE, same-landmark Similarity Encoding, normalized/unnormalized RBF, Nyström,
metric kNN, ten corrupted MPE metrics, and equality MPE. Hierarchies additionally
include ancestors, paths, Wu-Palmer/LCH/shortest-path similarities, Laplacian,
node2vec, and tree RBF. Geographic tasks include coordinates, coordinate MLP,
2-D Fourier, spatial RBF, spectral, node2vec, and graph similarity as applicable.
String tasks designate trigram Similarity Encoding as the main baseline.

The primary comparator per source is
`BEST_NON_MPE_METRIC_BASELINE`, selected only with validation data from all
applicable similarity, kernel, hierarchy, spectral, graph, Fourier, and
coordinate methods. PLE is never the primary paper comparator.

## 9. Causal and boundary controls

Each primary split has ten target-independent corrupt state-to-metric
associations. Correct-versus-mean-corrupt is the primary causal comparison.
Partial corruption at 10/25/50/100 percent runs on the representative subset.
Eight real codebook bijections test predictive schema sensitivity; 32
relabelings validate the exact theorem in both synthetic and real fields.

Seen-state IID controls test cold-state specificity. Three real equality-metric
nominal fields test no-free-lunch. Interval/triangular MPE is checked against
ordinary piecewise-linear interpolation. The preserved Fourier boundary is
reported, not repaired. Declared metrics that have weak target relevance and
cases outside useful support remain failures.

## 10. Support-distance and train-only smoothness mechanism

For every unseen test state `s`, compute

```text
r(s)   = min_{t in S_train} d(s,t)
R_w(s) = sum_j w_j(s)d(s,l_j).
```

Near/medium/far bins are state-level test tertiles fixed without labels. For
each baseline, calculate state-level loss difference `loss_baseline-loss_MPE`,
its Spearman relationship with support distance, and source-clustered
uncertainty. A positive relationship is not required at extreme gaps; any
turning point is reported descriptively, not selected to maximize a claim.

The conditional smoothness diagnostic is training-only: cross-fit a model on
ordinary covariates excluding the metric field, aggregate cross-fitted residual
means by training state, then report nearest-neighbor residual agreement,
distance-versus-absolute-residual-difference Spearman, a graph Moran-style
score, and empirical variogram slope. The prespecified scalar is negative
distance/residual-difference Spearman (higher means smoother). Its ability to
predict MPE benefit is evaluated with leave-one-source-out linear prediction;
failure rejects the practitioner diagnostic.

## 11. Ablations, scalability, and efficiency

The representative tasks and exact ablations are fixed in both manifests.
Landmark budgets, cover radius, kernel, selection, normalization, dimension,
bandwidth, sparse-k, and metric corruption are tested without replacing the
primary definition. Scaling uses 10k, 50k, and full-cap rows on ACS occupation,
TLC pickup, and Amazon, plus 100/1k/10k-state synthetic/high-cardinality sweeps
where source support permits.

Every run records preprocessing, metric, landmark, training and inference time;
host/GPU memory; model parameters; serialized bytes; and accuracy. Dense versus
sparse MPE, full similarity, Nyström, and lookup are the scalability showdown.

## 12. Synthetic theorem-validation program

The nine frozen spaces, seven targets, support-gap sweep, and 0/5/10/25/50/100
percent metric-association corruption are defined in `THEORY_PLAN.md`. They use
multiple deterministic seeds but are summarized by space/target, not inflated
as real sources. Assumption-violating discontinuous/random/misaligned targets
remain in all tables.

## 13. Statistics

The independent primary unit is public source. Each source first averages tasks,
settings, state splits, and neural seeds equally. Report source-balanced mean,
median, source wins, state-split wins, and a 10,000-replicate paired source
bootstrap confidence interval. With only five primary sources, the report must
display all source effects and state that the interval has low discrete
resolution. ACS tasks and all within-source variants are clustered together.
Rows and neural seeds are never treated as independent evidence.

Secondary uncertainty bootstraps source clusters and, within source, state
splits. Spearman mechanism analyses operate on state aggregates and report a
source-balanced summary. Multiplicity is descriptive; the predeclared gates,
not cellwise p-values, determine the verdict.

## 14. Frozen ICLR success gates

Gate A: MPE beats the validation-selected strongest non-MPE metric-aware
baseline on at least 4/5 primary source means, and the source-balanced mean
relative improvement is positive with paired source-bootstrap 95% interval
excluding zero.

Gate B: correct-metric MPE beats mean corrupt MPE on at least 80% of primary
dataset-task by state-split aggregates and on all or nearly all source means.

Gate C: source-balanced MPE advantage is materially greater for unseen-state
than seen-state evaluation. Material means at least two percentage points of
relative loss or a ratio of unseen/seen advantage above 1.5, with direction
consistent on a majority of sources.

Gate D: a majority (at least 3/5) of metric-structured source means show a
positive state-level association between support distance and MPE advantage in
the near-to-medium range. Extreme-far reversal is allowed and reported.

Gate E: equality MPE shows no systematic advantage over support-complete,
capacity-matched nominal controls: no more than 1/3 nominal source-field means
may favor it by over 2% relative loss.

Gate F: MPE shows over 2% source-mean relative improvement against at least one
classical same-metric representation (Similarity Encoding or RBF/Nyström) on at
least three primary sources. Otherwise the tokenizer itself lacks support even
if geometry helps.

Fourier or coordinate specialists beating MPE on suitable analytic spaces does
not alone fail the project, but is included in the strongest-baseline gate.

## 15. Verdict rules

- `SUPPORTED`: Gates A, B, E, and F pass, with C/D at least directionally
  supported and no fatal leakage/integrity failure.
- `PARTIALLY SUPPORTED`: geometry causality and real unseen-state value survive
  on multiple sources, but Gate A or F narrowly fails or evidence is confined
  to one irregular metric family.
- `NOT SUPPORTED`: real value does not survive strongest same-information
  baselines, causal geometry fails broadly, integrity is fatal, or evidence is
  too incomplete to evaluate the primary gate.

The ICLR action is mechanically aligned: supported -> `READY TO WRITE ICLR` and
normally `COMMIT`; partial -> `ONE TARGETED GAP REMAINS` only when exactly one
predeclared, decision-relevant gap remains, otherwise pivot/second paper; not
supported -> `PIVOT METHOD` or `ABANDON MPE AS MAIN PAPER`. No new MPE-v2 is
invented after outcomes.

## 16. Registry, raw evidence, and fail-closed execution

SQLite registry keys contain dataset, task, setting, state/row split, metric,
corruption, representation, landmarks, bandwidth, backbone, hyperparameters,
and seed. A completed unique key is never rerun. Each result is written
atomically with source checksum, protocol hash, code commit/diff hash, device,
and timestamps. Failures remain registered and resume from the last safe unit.

Automated integrity tests cover all 20 items in the mandate: metric axioms,
disjoint states, target-independent metric/landmarks, corrupt controls,
relabeling/equality identities, dimensions, leakage, covariate/backbone/HPO
parity, sealed tests, coordinate/ancestor parity, state-balanced arithmetic,
and raw-to-figure/table regeneration. Tests fail loudly.

Every source receives `LEAKAGE_AUDIT_<dataset>.md`. Transductive knowledge of a
future ontology/graph node and its target-independent metadata is allowed;
future outcomes are not. The public literature audit searches through 2026 and
subtracts PLE, similarity, hierarchy, kernel, spectral/graph, interpolation,
and cold-start prior work before any novelty claim.

## 17. Required outputs

Raw results, registry, environment lock, checksums, logs, and reproducible
scripts live under `experiments/mpe_iclr`. Ten numbered Markdown tables plus
CSV and Parquet, Figures 1–11 plus optional Figure 12, `THEORY.md`, all leakage
audits, `LITERATURE_AUDIT.md`, `PROTOCOL_DEVIATIONS.md`, and `FINAL_AUDIT.md`
must be present.

The final `results.md` uses exactly the 28-section structure in the governing
AGENT brief, includes every important failure, chooses one allowed scientific
verdict, ICLR decision, thesis, and recommendation, and never substitutes logs
for a decision.
