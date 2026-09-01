# AGENT.md — FINAL ICLR PROGRAM FOR METRIC PARTITION EMBEDDINGS

## Mission

Determine whether **Metric Partition Embedding (MPE)** is strong enough for an ICLR paper.

Do not optimize for confirming MPE.

Do not stop after synthetic experiments.

Do not stop after implementation.

Do not stop when one dataset or baseline fails.

Do not silently remove unfavorable datasets.

Complete the full mandatory experimental program, theory validation, controls, ablations, statistical analysis, and integrity audit.

The final mandatory deliverable is:

```text
results.md
```

The purpose is to answer:

> Can a single metric-space tokenizer use externally supplied feature geometry to generalize to genuinely unseen tabular states better than PLE, categorical embeddings, similarity encodings, hierarchy-specific encodings, graph/spectral embeddings, and specialized geometric representations?

The intended paper thesis is broader than:

> MPE beats PLE.

The candidate thesis is:

> Tabular feature types are better described by their state space and geometry than by a binary numerical/categorical distinction. A metric-partition tokenizer can exploit known within-feature geometry to interpolate unseen states while remaining invariant to arbitrary storage-code relabelings.

This thesis must be falsified if the real experiments do not support it.

---

# 0. Read existing work first

Before coding, read all Day-8 artifacts, especially:

```text
FINAL_REPORT.md
THEORY_FREEZE.md
PROTOCOL_FREEZE.md
POSTHOC_BASIS_CONTROL.md
POSTHOC_CAPACITY_CONTROL.md
BIKE_CONFIRMATION_FREEZE.md
```

and all:

```text
ridge_summary.json
basis_control_summary.json
neural_summary.json
bike_summary.json
```

Preserve the following existing conclusions:

- synthetic cycle/tree unseen-state interpolation is strong;
- uniform/support-complete PLE explains the nominal result;
- multiscale MPE failed and is not the primary method;
- MPE loses to Fourier features on ordinary observed cyclic hour encoding;
- exact chart invariance works;
- arbitrary global code-distortion is not a useful scalar risk predictor.

Do not rerun these merely to obtain more positive seeds.

---

# 1. Freeze the final program prospectively

Before examining outcomes on the new real-data panel, create:

```text
experiments/mpe_iclr/FINAL_PROTOCOL.md
experiments/mpe_iclr/final_config.json
experiments/mpe_iclr/DATASET_MANIFEST.md
experiments/mpe_iclr/BASELINE_MANIFEST.md
experiments/mpe_iclr/THEORY_PLAN.md
```

Hash them.

Write hashes to:

```text
experiments/mpe_iclr/PROTOCOL_HASHES.txt
```

After freezing:

- do not replace unfavorable datasets;
- do not alter the primary MPE definition;
- do not change the primary metric;
- do not remove strong baselines;
- do not change the success criteria.

Implementation errors go to:

```text
experiments/mpe_iclr/PROTOCOL_DEVIATIONS.md
```

---

# 2. Formal problem definition

A feature is not merely:

```text
numerical
categorical
```

Represent a typed field as:

```text
(X, d)
```

where:

- `X` is its semantic state space;
- `d` is a target-independent metric or pseudometric supplied by schema/domain metadata.

Examples:

```text
age            -> real interval
hour           -> circle
occupation     -> hierarchy/tree
airport        -> geographic metric
station        -> geographic/network metric
taxi zone      -> adjacency/geographic metric
product class  -> taxonomy
nominal ID     -> equality metric
```

MPE should consume the declared geometry without requiring the storage representation to carry it.

---

# 3. Primary MPE definition

For field state `x`, landmarks:

```text
L = {l_1, ..., l_m}
```

and target-independent metric:

```text
d(x, l_j)
```

define:

```text
a_j(x) = kappa(d(x,l_j) / h)

w_j(x) = a_j(x) / sum_k a_k(x)
```

and:

```text
MPE(x) = sum_j w_j(x) v_j
```

where:

```text
v_j ∈ R^D
```

are trainable landmark tokens.

Primary kernel:

```text
Gaussian:
kappa(r) = exp(-r² / 2)
```

Also implement:

```text
compact triangular kernel
sparse k-nearest-landmark MPE
```

for theory/scalability ablations.

Do NOT make multiscale MPE primary.

---

# 4. Strict inductive landmark rule

This is mandatory.

Primary MPE landmarks must come only from:

```text
TRAINING STATES
```

Never select landmarks using:

- test frequencies;
- test targets;
- test state performance;
- target-dependent geometry.

For unseen test state `x`, MPE must interpolate from landmarks whose trainable tokens were actually learned.

Primary landmark selection:

```text
farthest-point sampling in metric space
```

Secondary:

```text
k-medoids
random train-state prototypes
frequency-weighted prototypes
```

Landmark selection must be target independent.

---

# 5. Landmark budgets

Use:

```text
m ∈ {8, 16, 32, 64, 128}
```

when state cardinality permits.

Primary:

```text
m = 32
embedding dimension D = 32
```

If fewer than 32 training states exist:

```text
m = number of training states
```

Do not create duplicate landmarks.

For very large fields, additionally test:

```text
m = 256
```

on representative datasets.

---

# 6. Bandwidth

Bandwidth must be selected using training/validation only.

Candidate grid should be derived from training-state distances:

```text
h ∈ {
0.5 × median NN distance,
1 × median NN distance,
2 × median NN distance,
4 × median NN distance,
25th/50th/75th distance quantiles
}
```

Use one frozen rule across datasets where possible.

Do not tune on test states.

---

# 7. THEORETICAL PROGRAM

Create:

```text
THEORY.md
```

containing formal definitions, propositions/theorems, proofs, assumptions, and empirical validation.

At minimum prove the following.

---

# THEOREM 1 — Exact chart/relabeling invariance

Let:

```text
π : X -> X'
```

be a bijective relabeling.

Transport the metric:

```text
d'(π(x), π(y)) = d(x,y)
```

and transport landmarks:

```text
l_j -> π(l_j).
```

Show exactly that:

```text
w'_j(π(x)) = w_j(x)
```

and therefore:

```text
MPE'(π(x)) = MPE(x)
```

when semantic landmark tokens are correspondingly identified.

This establishes that arbitrary integer/string codebooks are charts rather than semantics.

Validate numerically under at least:

```text
32 random code relabelings
```

on synthetic and real discrete fields.

Tolerance:

```text
max aligned representation difference < 1e-7
```

or an explicitly justified floating-point tolerance.

---

# THEOREM 2 — Partition-of-unity interpolation bound

Let:

```text
f : X -> R^q
```

be `L`-Lipschitz under `d`.

For landmark values `a_j` approximating `f(l_j)`, define:

```text
f_hat(x) = sum_j w_j(x) a_j.
```

Prove:

```text
||f_hat(x) - f(x)||
<=
L * sum_j w_j(x) d(x,l_j)
+
sum_j w_j(x) ||a_j - f(l_j)||
```

Therefore if:

```text
||a_j - f(l_j)|| <= epsilon
```

then:

```text
error(x)
<=
L * weighted_metric_radius(x)
+
epsilon.
```

For compact-support kernels with all active landmarks within radius `h`:

```text
error(x) <= Lh + epsilon.
```

This is the core unseen-state interpolation theorem.

Do NOT claim this proves optimization/generalization of the full neural network.

It proves that the MPE function class contains an interpolant controlled by metric support distance.

---

# THEOREM 3 — Linear-head realizability

Show that when token dimension/output rank is sufficient, a linear downstream head can realize the landmark interpolant in Theorem 2 by choosing landmark token/head combinations appropriately.

The purpose is to connect the approximation theorem to the actual tokenizer architecture.

State rank/dimension assumptions explicitly.

---

# THEOREM 4 — Equality-metric impossibility

For equality metric:

```text
d_eq(x,y) =
0 if x=y
1 otherwise
```

and unseen states `x,x'` not equal to any training landmark:

```text
d_eq(x,l_j) = d_eq(x',l_j) = 1
```

for every landmark.

Therefore prove:

```text
w(x) = w(x')
```

and:

```text
MPE(x) = MPE(x').
```

Thus a metric-only MPE has no basis for distinguishing genuinely unseen nominal states under the equality metric.

This theorem must be emphasized.

It theoretically explains the corrected Day-8 nominal result.

---

# THEOREM 5 — Metric perturbation stability

Let two supplied metrics satisfy:

```text
|d(x,l_j) - d_tilde(x,l_j)| <= delta
```

for all relevant state-landmark pairs.

Assume:

- kernel `kappa` is Lipschitz;
- normalization denominator has lower bound `z0`;
- landmark token norms are bounded by `V`.

Derive a bound of the form:

```text
||MPE_d(x) - MPE_dtilde(x)||
<=
C * delta / h
```

with the full constant stated.

For a Lipschitz kernel, an acceptable bound to derive/check is proportional to:

```text
2 * V * m * L_kappa * delta / (h * z0)
```

subject to the exact assumptions used in the proof.

Do not state the constant until verified algebraically.

---

# THEOREM 6 — Landmark coverage / metric complexity

Let landmarks form an `r`-cover of the relevant metric support.

Using Theorem 2, derive an approximation guarantee based on:

```text
covering radius r
```

rather than raw category count.

Relate required landmark count to the metric covering number:

```text
N(X, r).
```

The intended conclusion is:

> MPE complexity should depend on intrinsic metric coverage, not merely the number of storage codes.

Do not overclaim rates without assumptions.

---

# PROPOSITION 7 — Interval special case

For uniformly spaced landmarks on a line and a compact triangular hat kernel, show that the MPE partition becomes ordinary piecewise-linear interpolation between adjacent landmark tokens.

Use this to explain the relationship to:

```text
piecewise-linear / bin-based numerical embeddings
```

rather than pretending MPE invented piecewise-linear interpolation.

---

# OPTIONAL THEOREM 8 — Metric corruption adversary

If rigorously possible, prove that when a corrupted metric maps a target state's local neighborhood to states that are far under the true metric, there exists an `L`-Lipschitz target under the true metric for which corrupted-metric interpolation incurs error proportional to that mismatch.

Only include if the proof is clean.

Do not force it.

---

# 8. Synthetic theorem-validation suite

The synthetic experiments now exist to validate theory, not to establish real-world usefulness.

Generate:

```text
interval
cycle
path graph
balanced tree
unbalanced tree
2-D grid
random geometric graph
small-world graph
nominal equality space
```

For each generate several target types:

```text
Lipschitz smooth
piecewise smooth
high-frequency
localized bump
discontinuous
random labels
metric-misaligned target
```

---

# 9. Synthetic support-gap sweep

Create unseen states with controlled:

```text
r(x) = min_{s in S_train} d(x,s)
```

Sweep support gaps from near-zero to large.

Test whether:

```text
MPE error
```

scales with the weighted metric radius predicted by Theorem 2.

Plot:

```text
x = weighted metric radius
y = absolute prediction error
```

and:

```text
x = nearest training support distance
y = MPE advantage over identity/PLE
```

---

# 10. Synthetic metric corruption sweep

Use:

```text
0%
5%
10%
25%
50%
100%
```

metric/state association corruption.

Corruption must preserve:

- state count;
- model parameter count;
- distance-matrix dimensions.

Where appropriate preserve:

- distance distribution;
- graph degree distribution.

Measure whether performance degrades with corruption.

Do not assume monotonicity; report it.

---

# 11. REAL-DATA PANEL

Real data is the decisive evidence.

The primary panel must contain metrics that exist independently of the prediction target.

Use the following mandatory public datasets/sources.

---

# DATASET 1 — ACS PUMS OCCUPATION

Source:

```text
American Community Survey Public Use Microdata
```

Field:

```text
occupation code
```

Geometry:

```text
Standard Occupational Classification hierarchy
```

Use official Census occupation/SOC crosswalks.

Task:

```text
regression:
log1p(annual wage income)
```

Suggested target:

```text
WAGP
```

among working-age employed respondents with valid occupation/wage.

Use common non-target covariates such as:

```text
age
education
hours worked
weeks worked
state
sex
class of worker
other non-leaking demographic/employment variables
```

Do not include descendants/aliases of the target.

Primary unseen-state split:

- identify sufficiently frequent leaf occupations;
- group leaves by higher-level SOC parent;
- allocate disjoint occupation states to train/validation/test;
- no test occupation may appear in training;
- ensure test occupations have metric neighbors in train.

Also create a harder:

```text
held-out occupational subgroup
```

split where nearest support is farther away.

---

# DATASET 2 — ACS PUMS INDUSTRY

Use the same raw public PUMS source but treat this as a separate field experiment.

Field:

```text
industry code
```

Geometry:

```text
NAICS-derived hierarchy
```

Target:

```text
log1p(annual wage income)
```

Use an analogous state-disjoint split.

Do not count OCCUPATION and INDUSTRY as two independent source datasets in significance calculations.

Cluster them under:

```text
ACS
```

for source-level inference.

---

# DATASET 3 — NYC TLC TAXI

Use official NYC TLC trip records.

Metric field:

```text
pickup taxi zone
```

and separately:

```text
dropoff taxi zone
```

Available metadata includes official taxi-zone geometry.

Construct two target-independent geometries:

### Primary

```text
great-circle / centroid geographic distance
```

### Secondary

```text
shortest-path distance on taxi-zone adjacency graph
```

where adjacency comes only from zone geometry.

Task:

```text
log trip-duration regression
```

Use only features available under the declared prediction setting.

Possible covariates:

```text
pickup time
day of week
passenger count
vendor/rate type
known origin/destination field
```

Do not use actual trip duration components as predictors.

Do not use future information.

Primary cold-state split:

```text
entire taxi zones held out from training
```

with state-disjoint train/val/test zones.

Additionally test a spatial-block split holding out contiguous regions.

---

# DATASET 4 — CITI BIKE NYC

Use public Citi Bike trip histories.

Field:

```text
station ID
```

Metadata:

```text
station latitude
station longitude
```

Primary metric:

```text
haversine/geodesic distance
```

Secondary metric:

```text
station network shortest-path distance
```

where graph edges are constructed only from allowed non-target trip/network metadata.

Task:

```text
log trip-duration regression
```

Use features such as:

```text
start time
day
member/casual
rideable type
known start/end location field
```

Critical split:

### Natural new-station split

Use time ordering to identify stations appearing after the training period.

Train only on earlier stations/rows.

Evaluate on genuinely new stations introduced later.

Also use controlled state-holdout splits for replication.

---

# DATASET 5 — US AIRLINE ON-TIME PERFORMANCE

Use Bureau of Transportation Statistics on-time flight data.

Use a fixed historical year, preferably:

```text
2024
```

unless data-access constraints make another predeclared year more reproducible.

Fields:

```text
origin airport
destination airport
```

Metric:

```text
geodesic airport distance
```

using authoritative airport coordinates.

Secondary:

```text
route-network shortest-path geometry
```

constructed without target labels.

Targets:

Primary:

```text
arrival-delay regression
```

Secondary:

```text
arrival delay >= 15 minutes
```

Use scheduled/pre-departure covariates only.

No actual-arrival information.

Cold-airport split:

```text
hold entire airports out from training
```

while retaining enough neighboring train airports for meaningful interpolation.

---

# DATASET 6 — AMAZON REVIEWS 2023 PRODUCT METADATA

Use product metadata, not review text, for the primary tabular task.

Field:

```text
leaf/fine-grained product category
```

Geometry:

```text
hierarchical `categories` metadata
```

Task:

Primary:

```text
log product-price regression
```

Alternative only if price quality makes the primary impossible:

```text
another frozen non-derived product-level target
```

Do not switch target after viewing MPE performance.

Do not use category-path ancestors as ordinary covariates in the primary MPE comparison because they directly expose the hierarchy.

Instead include:

```text
ancestor multi-hot
```

as a strong explicit baseline.

Use leaf-category-disjoint train/validation/test splits.

Ensure held-out leaf categories have represented parents/siblings in training.

---

# 12. OPTIONAL DIRECT PRIOR-WORK REPLICATION — MIMIC-III

If valid PhysioNet/MIMIC access is ALREADY available in the environment, add:

```text
MIMIC-III mortality prediction
```

using ICD-9 diagnosis hierarchy.

Reproduce a state-holdout setting close to hierarchy-based semantic embedding prior work:

```text
unseen diagnosis leaves/siblings at test time.
```

Directly compare against the published hierarchy-based semantic representation.

If credentials/access are not already available:

```text
DO NOT block the project.
```

Mark this dataset:

```text
NOT RUN — CONTROLLED ACCESS UNAVAILABLE
```

and continue.

---

# 13. SECONDARY STRING-METRIC PANEL

MPE overlaps with similarity encoding when the metric is string similarity.

Therefore test it in the home territory of similarity encoding rather than avoiding that comparison.

Use at least THREE of:

```text
Employee Salaries
Medical Charges
Traffic Violations
Road Safety
Beer Reviews
Open Payments
```

using the same or closely reproduced targets/high-cardinality fields from the similarity-encoding benchmark.

Primary metric:

```text
3-gram string distance/similarity
```

Secondary:

```text
Jaro-Winkler
Levenshtein-based distance
```

Similarity Encoding is the main baseline here.

These datasets are SECONDARY because the metric is derived from the raw code/string itself rather than external semantic metadata.

---

# 14. Minimum real-data breadth

The final report must contain at least:

```text
6 primary field tasks
from >=5 independent public data sources
```

and preferably:

```text
9+ total real tasks
```

including the string-metric panel.

The experiment must span at least:

```text
hierarchy
graph/geography
string/similarity
```

metric families.

Do not use ordinary cycle/hour as headline real evidence.

Bike-hour already established that Fourier is superior on a clean cycle.

---

# 15. UNSEEN-STATE SPLIT RULES

For every primary task:

```text
S_train ∩ S_val = empty
S_train ∩ S_test = empty
S_val ∩ S_test = empty
```

where `S` denotes states of the metric-aware feature.

Rows inherit the split from the field state.

Validation hyperparameters therefore generalize to unseen states too.

This is essential.

Do NOT use:

```text
random row split
```

as the main unseen-state evaluation.

---

# 16. Minimum frequency rules

Before assigning states:

Require enough observations for reliable per-state evaluation.

Suggested:

```text
>= 50 rows/state
```

for ordinary tasks.

Preferred:

```text
>= 100
```

where data permits.

Freeze exact thresholds by dataset.

Do not remove low-performing states later.

---

# 17. State-split replications

Use:

```text
5 independent state partitions
```

where random state splitting is appropriate.

For each state split:

```text
3 neural training seeds
```

minimum.

Natural temporal splits such as new Citi Bike stations can use fewer state partitions but must include several training seeds and multiple time windows where feasible.

---

# 18. Equal-state and equal-row evaluation

Report both.

### Row-weighted

Ordinary test loss.

### State-balanced

Compute test loss separately for each unseen state and then average states equally.

State-balanced performance is primary for the unseen-state question.

This prevents frequent cold states from dominating.

Also report:

```text
worst-quartile state performance
worst-decile state performance
```

when enough test states exist.

---

# 19. BASELINES — MANDATORY

MPE must not be compared only with PLE.

The following are mandatory.

---

# Baseline A — Unknown categorical embedding

Standard learned category embedding for train states.

All unseen states map to:

```text
UNK token
```

This is the simplest cold-category baseline.

---

# Baseline B — One-hot / support-complete categorical

For linear models:

```text
one-hot + unseen fallback
```

For neural models:

```text
learned lookup embedding + UNK
```

---

# Baseline C — Q-PLE

Quantile PLE using integer/category codes.

Include because it is the historical baseline.

Do not treat it as semantically appropriate for arbitrary categories.

---

# Baseline D — Uniform/support-complete PLE

Use enough bins to isolate observed low-cardinality states where feasible.

This prevents the Day-8 repeated-quantile artifact.

---

# Baseline E — Similarity Encoding

Implement the Cerda/Varoquaux/Kégl-style prototype similarity representation.

For every metric task, create:

```text
SimilarityVector(x)
=
[k(d(x,l_1)), ..., k(d(x,l_m))]
```

using the SAME landmarks and metric as MPE.

This is one of the most important baselines.

It tests whether MPE's learned token mixing improves on simply exposing the metric similarities to the downstream model.

---

# Baseline F — Metric RBF / Nyström features

Use the same landmarks.

Compute fixed kernel features:

```text
k(d(x,l_j)/h)
```

and normalized and unnormalized variants.

Pass them through the same backbone.

This distinguishes:

```text
learned MPE token mixture
```

from:

```text
ordinary metric kernel features.
```

---

# Baseline G — kNN metric interpolation

Construct training-only nearest-neighbor predictors or representations using the declared metric.

For the full tabular task, combine metric-neighbor representation with other features.

This is a simple and important nonparametric baseline.

---

# Baseline H — random/corrupted metric MPE

Same:

```text
m
D
kernel
backbone
parameter count
training
```

but scramble state-to-metric association.

Use:

```text
>= 10 independent corrupted metrics
```

per main dataset/split.

This is the causal geometry control.

---

# Baseline I — equality-metric MPE

Replace declared structure with equality metric.

Expected to fail on unseen distinct states.

This validates Theorem 4.

---

# 20. HIERARCHY-SPECIFIC BASELINES

Mandatory on ACS/Amazon/MIMIC-style hierarchy tasks.

### H1. Ancestor multi-hot

Encode every state's ancestors.

Include depth information if available.

### H2. Path-to-root representation

Binary or weighted path encoding.

### H3. Hierarchy semantic similarity embedding

Implement direct hierarchy-based similarity features inspired by prior hierarchy-based semantic embeddings.

At minimum test:

```text
shortest path
Wu-Palmer-style similarity where applicable
LCH/path-based structural similarity where applicable
```

Do not require corpus-derived information content.

### H4. Graph Laplacian spectral encoding

Build the hierarchy graph.

Use leading nontrivial Laplacian eigenvectors as fixed state coordinates.

Allow unseen test nodes access to the target-independent hierarchy graph.

### H5. node2vec / DeepWalk-style unsupervised graph embedding

Use graph topology only.

No target labels.

### H6. Tree distance RBF/Nyström

Same tree metric as MPE.

This is essential.

---

# 21. GEOGRAPHIC/GRAPH BASELINES

Mandatory for stations, airports, taxi zones.

### G1. Raw coordinates

Use:

```text
latitude
longitude
```

when available.

MPE must not receive credit for information already trivially available as coordinates.

### G2. Coordinate MLP

Map lat/lon through a small learned MLP with parameter count comparable to MPE.

### G3. 2-D Fourier features

Use multi-frequency Fourier features on normalized coordinates.

### G4. Spatial RBF features

Centers selected with the same landmark budget.

### G5. Laplacian/spectral graph coordinates

Where adjacency/network graph exists.

### G6. node2vec

Where meaningful.

### G7. shortest-path similarity encoding

Use the exact same graph metric as MPE.

---

# 22. CYCLE-SPECIFIC BASELINE

For any cyclic ablation:

```text
sin/cos Fourier features
```

are mandatory.

Do not claim MPE should beat Fourier on simple cycles.

The prior Bike experiment says otherwise.

Use cycles as a known-specialist boundary.

---

# 23. ORDINAL BASELINES

If an ordinal field is included:

```text
raw rank
normalized rank
Q-PLE
uniform PLE
monotonic embedding if available
```

are mandatory.

Again, MPE need not beat specialized 1-D methods to be useful on irregular metric spaces.

---

# 24. TREE-MODEL BASELINES

Include:

```text
CatBoost
LightGBM or XGBoost
```

on every real dataset where computationally practical.

For CatBoost:

```text
native categorical field
```

is the primary tree baseline.

Also test:

```text
CatBoost/GBDT + MPE features
```

on a representative subset to determine whether MPE's information is useful beyond neural networks.

---

# 25. Neural backbones

Primary:

```text
MLP
ResNet
FT-Transformer
TabM
```

Use stable official/reference implementations where possible.

Do not add more architectures merely to inflate breadth.

Representation comparisons must use identical downstream architectures.

---

# 26. Mechanism model

Also use:

```text
ridge / linear head
```

for every dataset.

The ridge experiment asks:

> Is the metric representation itself useful before deep nonlinear modeling?

Do not judge final predictive performance from ridge alone.

---

# 27. Fair representation comparison

Primary tokenizer output dimension:

```text
D = 32
```

Also test:

```text
D = 16
D = 64
```

on the ablation subset.

For trainable representation methods:

- report tokenizer parameters;
- report backbone parameters;
- report total parameters.

Create two fairness views:

### Natural configuration

Each method uses its normal formulation.

### Parameter-matched configuration

Adjust projection dimensions so total trainable parameter counts differ by no more than approximately:

```text
±5%
```

where meaningful.

Do not distort fixed feature methods solely to create fake parameter equality.

---

# 28. Hyperparameter fairness

Use an equal validation budget.

For each:

```text
dataset × backbone × representation
```

allow the same maximum number of hyperparameter trials.

Recommended:

```text
20 trials
```

for the primary experiments.

Tune only using:

```text
training states
validation unseen states
```

Never test states.

Freeze search spaces in `FINAL_PROTOCOL.md`.

---

# 29. Training to convergence

Do not repeat the OrbitCover problem of compute-capped undertraining.

Use:

```text
max epochs = 300
```

with validation early stopping.

Suggested:

```text
patience = 30
```

but freeze final values.

Record:

```text
training loss
validation loss
best epoch
stop epoch
wall clock
```

On three representative datasets verify that doubling the maximum budget does not materially improve the best validation result.

---

# 30. Dataset size

Use realistically large samples.

Do not cap every dataset at 2,048 rows.

Preferred maximum:

```text
100k–500k rows
```

depending on source and compute.

For huge sources, use a prospectively fixed subsample large enough to retain many states.

For at least three datasets, run a scaling check:

```text
10k
50k
100k/full
```

---

# 31. Two evaluation settings per real dataset

Where practical, create:

## Setting A — Isolated field

Other features are limited.

This asks:

> Does the metric field itself support interpolation?

## Setting B — Full realistic table

Use normal non-leaking covariates.

This asks:

> Does MPE still matter when the rest of the table is informative?

Do not let the paper rely solely on artificially isolated one-feature tasks.

---

# 32. Primary metrics

Regression:

```text
standardized MSE
RMSE
MAE
```

Primary:

```text
standardized MSE
```

Classification:

```text
Brier
log loss
AUROC
accuracy
```

Primary:

```text
Brier
```

or predeclare another proper loss if class structure requires it.

State-balanced primary metric must also be reported.

---

# 33. SUPPORT-DISTANCE ANALYSIS — CRITICAL

For every unseen test state define:

```text
r(s)
=
min_{t in S_train} d(s,t)
```

Also calculate MPE's:

```text
R_w(s)
=
sum_j w_j(s) d(s,l_j)
```

which appears directly in Theorem 2.

For every test row attach its state's support-distance values.

Bin unseen states into:

```text
near
medium
far
```

tertiles or quartiles prospectively.

Report MPE advantage versus each baseline as a function of support distance.

---

# 34. Core mechanism hypothesis

Test:

> MPE's relative advantage should increase when test states are unseen but remain within meaningful metric support.

This is stronger than reporting average gains.

Compute:

```text
Spearman(
    support distance,
    baseline loss - MPE loss
)
```

at the STATE level.

Use source-clustered uncertainty.

Do not treat rows as independent samples.

---

# 35. Conditional target smoothness diagnostic

A metric is useful only if the target contains structure along it.

Construct a TRAIN-ONLY diagnostic.

Procedure:

1. fit a model using all ordinary features EXCEPT the metric field;
2. cross-fit residuals on training data;
3. aggregate residual means by state;
4. measure state residual smoothness against `d`.

Candidate diagnostics:

```text
distance-vs-residual-difference correlation
graph Moran-like autocorrelation
nearest-neighbor residual agreement
empirical variogram slope
```

Do not use test labels.

Test whether this training-only smoothness measure predicts MPE benefit.

This could provide an answer to:

> When should a practitioner use MPE?

Do not turn it into a complex learned gate unless the simple diagnostic is clearly predictive.

---

# 36. Real metric corruption intervention

For every primary metric dataset generate:

```text
10
```

frozen target-independent corrupted metrics.

Examples:

### Hierarchy

Randomly reassign leaf states to leaves while preserving the tree itself.

### Geography

Permute station/airport/zone coordinates among state IDs.

### Graph

Permute semantic node IDs while keeping graph topology fixed.

This keeps representation capacity constant while destroying the meaningful state-to-geometry relationship.

Primary causal comparison:

```text
correct MPE
vs
mean corrupted MPE
```

---

# 37. Partial metric noise

Additionally run corruption levels:

```text
10%
25%
50%
100%
```

on a representative subset.

This should connect empirical degradation to Theorem 5.

---

# 38. Codebook relabeling

For each discrete real field generate:

```text
8 arbitrary storage-code permutations
```

Compare:

```text
MPE
Q-PLE
uniform PLE
code-space RBF
embedding lookup
metric-aware baselines
```

MPE must be numerically invariant when metric metadata is transported.

Do not confuse codebook invariance with predictive superiority.

---

# 39. Seen-state control

For every real dataset also create a normal row-wise split in which important states occur in both training and test.

This is a boundary experiment.

Question:

> Does MPE retain an advantage when interpolation to unseen semantic states is unnecessary?

Do not require MPE to dominate.

A desirable result is:

```text
large gain on unseen states
small/tied gain on seen states
```

which supports the proposed mechanism.

---

# 40. NOMINAL NEGATIVE CONTROLS

Add at least three real low-cardinality nominal fields with no defensible geometry.

Use equality metric.

Compare:

```text
lookup embedding
support-complete one-hot
uniform PLE
MPE equality
random geometry
```

Expected:

```text
MPE should not systematically outperform capacity-matched nominal controls.
```

This is an important no-free-lunch result.

---

# 41. Metric-space family generalization

The paper should answer whether ONE MPE implementation works across:

```text
line
cycle
tree
general graph
geographic metric
string metric
equality metric
```

without changing the neural tokenizer architecture.

Only:

```text
d
landmarks
bandwidth
```

should change.

This is central to the "types as geometry" thesis.

---

# 42. Landmark selection ablation

On at least four representative tasks compare:

```text
farthest-point
k-medoids
random prototypes
frequency-based prototypes
```

at:

```text
m = 16, 32, 64
```

Determine whether MPE's result is robust to landmark choice.

---

# 43. Kernel ablation

Compare:

```text
Gaussian
Laplacian
triangular compact
inverse-distance with stabilization
```

Do NOT exhaustively tune dozens of kernels.

The point is robustness.

---

# 44. Normalization ablation

Compare:

```text
normalized partition weights
unnormalized RBF features
softmax(-d/h)
```

This isolates whether partition-of-unity normalization is important.

---

# 45. Learned metric ablation — SECONDARY ONLY

Do NOT make metric learning the main contribution.

On a small subset optionally test:

```text
fixed declared metric
fixed metric + learned global scale
fixed metric + learned monotone distance transform
```

No arbitrary target-supervised metric network in the primary result.

Otherwise the claim "schema metadata supplies geometry" becomes muddled.

---

# 46. PLE relationship experiment

On ordinary interval-valued features compare:

```text
Q-PLE
uniform PLE
triangular MPE with ordered landmarks
Gaussian MPE
```

Validate Proposition 7 numerically.

Expected:

```text
triangular interval MPE behaves similarly to piecewise-linear interpolation.
```

Do not claim superiority if it simply reproduces PLE behavior.

---

# 47. Fourier boundary experiment

Keep the Day-8 Bike-hour result.

Optionally reproduce on one additional clean cyclic feature.

Expected:

```text
Fourier >= MPE
```

on globally smooth observed cycles.

Use this to say:

> MPE is not intended to replace analytic bases when an ideal basis is already known.

---

# 48. Graph-specific strong comparison

For hierarchy/general graph tasks, compare MPE against:

```text
Laplacian eigenvectors
node2vec
shortest-path RBF
ancestor/path encoding
similarity encoding
```

This comparison determines whether the method is more than a generic graph embedding rediscovery.

If MPE loses consistently to graph-specific representations, report that.

---

# 49. Similarity Encoding showdown

This deserves a dedicated table.

For every metric space compute:

```text
SIM(x) = metric similarities to landmarks
MPE(x) = learned mixture of landmark tokens
```

using IDENTICAL:

```text
metric
landmarks
bandwidth
backbone
training split
```

Compare at identical output dimensions where possible.

The central question:

> Does learning token vectors on top of a metric partition add value beyond directly exposing similarity coordinates?

If not, MPE novelty is weak.

---

# 50. Nyström/kernel showdown

Use the same kernel.

Compare:

```text
MPE
full normalized RBF vector
Nyström approximation
random kernel features where meaningful
```

MPE must demonstrate either:

```text
better accuracy
better parameter efficiency
better dimensional efficiency
```

to justify itself over classical kernel machinery.

---

# 51. Hierarchy semantic embedding showdown

On hierarchy tasks compare directly against:

```text
hierarchy similarity vector
ancestor representation
path embedding
spectral embedding
MPE
```

MPE must not be presented as the first unseen-hierarchy encoding method.

The contribution must be:

```text
one generic tokenizer across metric spaces
```

if that is what survives.

---

# 52. Efficiency analysis

Record:

```text
parameter count
training time
inference time
GPU memory
preprocessing time
metric computation time
landmark selection time
```

For discrete finite spaces precompute:

```text
state -> sparse landmark weight vector
```

so inference need not recompute all distances.

Report complexity approximately as:

```text
dense: O(m)
sparse-k landmark: O(k)
```

per field lookup after distance preprocessing.

---

# 53. Scalability experiment

Use one high-cardinality field.

Sweep:

```text
number of states:
100
1k
10k
```

where data permits.

Sweep landmarks:

```text
16
32
64
128
256
```

Compare:

```text
full similarity encoding
Nyström
MPE dense
MPE sparse
lookup embedding
```

Measure accuracy vs memory/time.

---

# 54. Statistical analysis

Primary independent unit:

```text
dataset/source
```

not rows.

Secondary:

```text
state split
```

Report:

```text
source-balanced mean
median
source wins
state-split wins
paired bootstrap interval
```

Cluster ACS occupation/industry as one source.

Cluster multiple variants of the same trip dataset as one source.

Do not count 5 seeds as 5 independent datasets.

---

# 55. Primary comparison

For every source construct:

```text
BEST_NON_MPE_METRIC_BASELINE
```

using validation data only.

This includes the strongest applicable method among:

```text
similarity encoding
Nyström/RBF
hierarchy encoding
spectral embedding
raw coordinates
Fourier
node2vec
ancestor encoding
```

Primary paper comparison:

```text
MPE
vs
BEST_NON_MPE_METRIC_BASELINE
```

not merely MPE vs PLE.

---

# 56. ICLR PRIMARY SUCCESS GATE

The MPE paper receives strong empirical support only if all or nearly all of these hold.

## Gate A — real unseen-state benefit

Across independent primary sources:

```text
MPE beats the strongest non-MPE metric-aware baseline
on >= 4/5 independent public sources
```

or an equivalently strong threshold if the frozen source count differs.

AND:

```text
source-balanced mean improvement > 0
with paired/source-bootstrap 95% interval excluding 0.
```

---

# 57. Gate B — causal geometry

Correct-metric MPE must beat corrupted-metric MPE on:

```text
>= 80% of primary dataset × state-split aggregates
```

and on:

```text
all or nearly all source means.
```

---

# 58. Gate C — unseen-state specificity

MPE improvement should be materially greater in:

```text
unseen-state
```

evaluation than:

```text
seen-state IID
```

evaluation.

This supports interpolation rather than generic extra capacity.

---

# 59. Gate D — support-distance mechanism

At least a majority of metric-structured sources should show:

```text
greater MPE advantage as support distance increases
```

within the meaningful interpolation range.

Do not demand monotonic improvement indefinitely; very distant states may be outside useful support.

---

# 60. Gate E — nominal no-free-lunch

On equality-metric nominal controls:

```text
MPE must NOT show a systematic unexplained advantage
over support-complete capacity-matched baselines.
```

Otherwise investigate a confound.

---

# 61. Gate F — similarity/kernel baseline

MPE must show a meaningful advantage over at least one classical way of exposing the SAME metric:

```text
similarity encoding
RBF/Nyström
```

on several real sources.

If it does not, the new tokenizer itself is not sufficiently justified.

---

# 62. Gate G — specialized representation honesty

Do NOT fail the project because:

```text
Fourier beats MPE on simple cycles
```

or:

```text
raw coordinate methods beat MPE on some low-dimensional geographic tasks.
```

Instead determine whether MPE's value is:

```text
generic irregular metric spaces
```

rather than known analytic geometries.

---

# 63. Metric usefulness diagnostics

For each source report:

```text
conditional train-only smoothness
state cardinality
metric diameter
median support gap
landmark cover radius
estimated metric dimension / covering growth
```

Then test correlations with MPE benefit.

Candidate hypothesis:

```text
benefit increases with
target-relevant metric smoothness
×
unseen support gap
```

but decreases once states lie too far beyond train coverage.

---

# 64. Possible phase diagram

If evidence supports it, create a 2-D analysis:

X:

```text
training-only metric smoothness
```

Y:

```text
test support distance
```

Color:

```text
MPE improvement over best baseline
```

Potential regimes:

```text
low smoothness -> no benefit
high smoothness + near support -> small benefit
high smoothness + moderate unseen gap -> largest benefit
extreme gap -> all interpolation methods fail
```

This could become a central paper figure.

Do not force the pattern.

---

# 65. Main figures

Generate publication-quality figures.

## Figure 1 — MPE concept

Diagram:

```text
typed state
-> metric distances
-> partition weights
-> learned landmark token mixture
-> tabular backbone
```

Show line, cycle, tree, graph examples.

---

## Figure 2 — Real benchmark

Dataset/source-balanced MPE performance versus:

```text
PLE
UNK embedding
Similarity Encoding
best metric-aware baseline
```

---

## Figure 3 — Support-distance effect

X:

```text
distance to training support
```

Y:

```text
MPE gain over strongest baseline
```

---

## Figure 4 — Correct vs corrupted metric

Per source.

---

## Figure 5 — Hierarchy showdown

```text
MPE
ancestor
hierarchy similarity
spectral
node2vec
Nyström
```

---

## Figure 6 — Geographic showdown

```text
MPE
raw coordinates
Fourier
RBF
spectral
```

---

## Figure 7 — Similarity-encoding showdown

Same metric + same landmarks.

---

## Figure 8 — Theoretical interpolation bound

Observed synthetic error versus:

```text
weighted metric radius
```

with bound behavior.

---

## Figure 9 — Metric noise

Metric corruption magnitude versus representation/prediction degradation.

---

## Figure 10 — Equality no-free-lunch

Show unseen nominal states collapse under equality metric.

---

## Figure 11 — Landmark budget

Performance vs:

```text
m
```

and cover radius.

---

## Figure 12 — Phase diagram

Only if supported.

---

# 66. Main tables

Create:

```text
TABLE_1_DATASETS.md
TABLE_2_BASELINES.md
TABLE_3_MAIN_REAL_RESULTS.md
TABLE_4_SUPPORT_DISTANCE.md
TABLE_5_CORRUPT_METRIC.md
TABLE_6_HIERARCHY_BASELINES.md
TABLE_7_GEOGRAPHIC_BASELINES.md
TABLE_8_ABLATIONS.md
TABLE_9_THEOREM_VALIDATION.md
TABLE_10_EFFICIENCY.md
```

Also save CSV/Parquet versions.

---

# 67. Integrity tests

Implement automated tests for:

1. metric symmetry where appropriate;
2. metric diagonal equals zero;
3. triangle inequality when the declared object is claimed to be a metric;
4. train/validation/test state sets are disjoint;
5. no test target enters metric construction;
6. landmarks use training states only;
7. corrupted metrics preserve intended controls;
8. code relabelings preserve semantic distances;
9. MPE relabeling invariance;
10. equality-metric unseen states receive identical weights;
11. all representation output dimensions are correct;
12. no accidental target leakage;
13. all baselines receive the same ordinary covariates;
14. hyperparameter budgets are equal;
15. same backbone configuration is used for representation comparisons;
16. test set is evaluated only after validation selection;
17. raw coordinates are not accidentally included only for MPE;
18. hierarchy ancestors are not leaked into ordinary features in the primary comparison;
19. state-balanced metrics are computed correctly;
20. figures/tables regenerate from raw results.

Fail loudly.

---

# 68. Data leakage audit

For every dataset create:

```text
LEAKAGE_AUDIT_<dataset>.md
```

Answer:

```text
What information defines the metric?
Was any target used?
Are test states known structurally at inference?
Are test labels ever used during representation construction?
Which covariates would be unavailable at prediction time?
```

For transductive schema metadata:

It is acceptable to know that a future state exists in an ontology/graph and know its metadata.

It is not acceptable to use its target outcomes.

State this explicitly.

---

# 69. Literature audit

Before writing the final novelty claim, create:

```text
LITERATURE_AUDIT.md
```

Search at minimum for work on:

```text
piecewise linear numerical embeddings
similarity encoding
categorical entity embeddings
hierarchy-based categorical embeddings
unseen category generalization
graph embeddings
Laplacian eigenmaps
node2vec/DeepWalk
Nyström/kernel embeddings
RBF features
metric-space interpolation
partition-of-unity networks/features
cold-start categorical variables
ontology-based embeddings
tabular positional/structural encodings
```

Search through 2026.

For each close paper record:

```text
method
input structure
whether metric is externally declared
whether unseen states are evaluated
whether it is a tabular tokenizer
whether it works across arbitrary metrics
closest overlap with MPE
remaining distinction
```

Do not claim:

```text
first metric embedding
first semantic categorical embedding
first hierarchy-aware unseen-category method
first RBF landmark method
```

unless exhaustive evidence genuinely supports it.

---

# 70. The likely novelty claim

Only if experiments support it, the defensible novelty should be approximately:

> A single trainable partition tokenizer that treats heterogeneous tabular fields as metric spaces, is exactly invariant to equivalent code relabelings, and supports inductive prediction on unseen states using externally declared geometry.

The novelty is NOT:

```text
Gaussian kernels
RBFs
landmarks
hierarchy similarity
piecewise interpolation
```

individually.

---

# 71. Avoid target-engineered metrics

Absolutely do not construct the main metric using:

```text
target correlation
supervised embedding
test performance
gradient similarity
label similarity
```

The metric must exist independently.

A supervised learned metric may be a secondary ablation only.

---

# 72. Negative findings to preserve

The final paper should explicitly preserve these if they remain true:

```text
MPE does not replace Fourier on simple cycles.
MPE does not help arbitrary nominal unseen states.
Multiscale MPE is not automatically better.
Some declared real metrics may be irrelevant to the target.
Specialized graph/hierarchy embeddings may beat generic MPE.
Very distant unseen states may be outside interpolation support.
```

These make the claim more credible.

---

# 73. Do not build a "safe chart bank" yet

Do NOT shift the project into:

```text
learn which encoder to use
```

unless MPE itself survives the primary real benchmark.

The first question is:

```text
Does MPE deserve to exist as a new tokenizer?
```

Only then consider automatic routing.

---

# 74. Compute execution

Use all available GPUs efficiently.

Maintain a registry keyed by:

```text
dataset
state_split
row_split
metric
metric_corruption
representation
landmarks
bandwidth
backbone
hyperparameters
seed
```

Never rerun identical completed jobs.

Resume failures.

Do not stop midway.

If OOM:

```text
reduce batch size
preserve optimization recipe
record deviation
continue
```

If a baseline is difficult to implement:

```text
debug it
```

Do not silently omit a strong baseline.

---

# 75. Stage execution

Stages are for organization only.

Do NOT stop after a stage.

## Stage 1

Theory implementation/unit tests.

## Stage 2

Dataset acquisition and leakage audits.

## Stage 3

Linear/ridge real benchmark.

## Stage 4

Neural main benchmark.

## Stage 5

Strong metric-specific baselines.

## Stage 6

Support-distance + corruption experiments.

## Stage 7

Ablations/scalability.

## Stage 8

Final integrity audit.

## Stage 9

Generate `results.md`.

Proceed automatically through all stages.

---

# 76. No adaptive rescue

If MPE loses to:

```text
Similarity Encoding
Nyström
spectral embeddings
ancestor encoding
raw coordinates
```

do not invent MPE-v2 after seeing the results.

Record the failure.

A new method may be studied later, but it is not part of this frozen ICLR test.

---

# 77. FINAL RESULTS.MD

At completion create:

```text
results.md
```

It must be a scientific decision report, not a dump of logs.

Use the following structure exactly.

---

# RESULTS — MPE ICLR VERDICT

## 1. Executive verdict

Choose exactly one:

```text
SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED
```

Then state in 10–15 sentences:

- whether MPE survives real unseen-state testing;
- whether it beats metric-aware baselines;
- where it works;
- where it fails;
- whether the "features as metric spaces" thesis survives.

---

## 2. Final dataset panel

Table:

```text
source
task
metric field
metric type
# rows
# train states
# val states
# test states
median support gap
```

Clearly separate:

```text
PRIMARY EXTERNAL-METRIC
SECONDARY STRING-METRIC
OPTIONAL CONTROLLED-ACCESS
```

---

## 3. Main real-world result

For each primary source report:

```text
MPE
best non-MPE metric-aware baseline
Similarity Encoding
PLE
UNK embedding
corrupt MPE
```

Report:

```text
state-balanced metric
row-balanced metric
relative improvement
paired uncertainty
```

---

## 4. Strongest-baseline showdown

This is the most important table.

| Source | MPE | Best baseline | Baseline name | Relative gain | Winner |
|---|---:|---:|---|---:|---|

Then report:

```text
source wins
source-balanced mean
median
95% source-bootstrap interval
```

---

## 5. Similarity Encoding vs MPE

Answer explicitly:

> Does learned landmark-token mixing outperform directly exposing similarities to the same landmarks?

If not, state that the MPE architecture itself lacks sufficient evidence of novelty.

---

## 6. Nyström/kernel comparison

Answer explicitly:

> Does MPE outperform classical kernel landmark representations using the same metric?

Report accuracy, dimension, memory, and compute.

---

## 7. Hierarchy tasks

Report:

```text
MPE
ancestor multi-hot
path
hierarchy semantic embedding
spectral
node2vec
Nyström
```

Discuss ACS, Amazon, and MIMIC if available.

---

## 8. Geographic/network tasks

Report:

```text
MPE
raw lat/lon
coordinate MLP
Fourier
RBF
spectral
node2vec
```

State whether generic metric interpolation adds anything beyond coordinates.

---

## 9. Support-distance mechanism

For each source report:

```text
Spearman(
support distance,
MPE advantage
)
```

and near/medium/far bins.

Answer:

> Does MPE become more useful when the test state is farther from observed support?

Also report where the relationship breaks because states are too distant.

---

## 10. Theoretical validation

For every theorem/proposition provide:

```text
statement
proof status
empirical validation
violations/assumption failures
```

At minimum cover Theorems 1–6 and Proposition 7.

---

## 11. Exact chart invariance

Report maximum numerical differences across codebooks.

Compare schema sensitivity of:

```text
MPE
PLE
code-RBF
lookup
metric-aware baselines
```

---

## 12. Correct vs corrupted metric

Report:

```text
correct metric performance
mean corrupt performance
10 corruption distribution
win counts
```

This is the causal geometry section.

---

## 13. Nominal negative controls

Report that equality geometry supplies no unseen-state discrimination.

If MPE still wins, identify why.

Do not hide it.

---

## 14. Seen vs unseen states

Compare effect sizes under:

```text
seen-state random-row split
unseen-state state-disjoint split
```

State whether MPE's advantage is specifically cold-state interpolation.

---

## 15. Target-smoothness diagnostic

Report whether the training-only smoothness diagnostic predicts MPE gains.

If useful, give leave-one-source-out predictive performance.

If not useful, reject it.

---

## 16. Landmark/metric complexity

Report MPE performance as a function of:

```text
landmark count
cover radius
metric-space size
```

Test whether cover radius explains performance better than raw cardinality.

---

## 17. Ablations

Summarize:

```text
kernel
landmark selection
normalization
embedding dimension
bandwidth
metric corruption
metric scale
```

---

## 18. Efficiency

Report:

```text
parameters
training time
inference time
precompute time
memory
```

Compare especially to:

```text
full similarity encoding
Nyström
lookup embedding
spectral embedding
```

---

## 19. Failure cases

Mandatory.

List every important case where:

```text
MPE loses to PLE
MPE loses to Similarity Encoding
MPE loses to Nyström
MPE loses to hierarchy-specific methods
MPE loses to raw coordinates
MPE loses to Fourier
correct metric fails to beat corrupt metric
support-distance mechanism fails
real metric has no target relevance
```

Do not compress this into one sentence.

---

## 20. What the synthetic results did and did not prove

Explicitly acknowledge:

> The synthetic cycle/tree targets were constructed to be smooth in the supplied metric.

Explain whether real results validate that mechanism without synthetic target construction.

---

## 21. Final claim after novelty subtraction

Write the strongest claim that remains AFTER considering:

```text
PLE
similarity encoding
hierarchy semantic embeddings
kernel/Nyström methods
spectral/graph embeddings
specialized Fourier/coordinate encodings
```

Do not exaggerate.

---

## 22. ICLR contribution assessment

Score 1–5:

```text
conceptual novelty
method novelty
theoretical contribution
synthetic mechanism evidence
real-world evidence
dataset breadth
baseline strength
unseen-state relevance
statistical rigor
reproducibility
story coherence
```

---

## 23. Reviewer simulation

Write the FIVE strongest rejection arguments.

For each:

```text
objection
evidence supporting the objection
evidence against it
remaining weakness
best defensible response
```

No strawmen.

---

## 24. ICLR decision

Choose exactly:

```text
READY TO WRITE ICLR
ONE TARGETED GAP REMAINS
PIVOT METHOD
ABANDON MPE AS MAIN PAPER
```

Do not invent more experiments simply because results are imperfect.

---

## 25. Best final thesis

Choose exactly one of:

### Thesis A

```text
Tabular fields are metric spaces, not merely numeric/categorical types,
and MPE provides a generic tokenizer for unseen-state interpolation.
```

### Thesis B

```text
MPE is specifically useful for irregular hierarchy/graph-valued fields
with externally supplied geometry.
```

### Thesis C

```text
Known geometry helps unseen categories, but existing similarity/kernel/
graph methods are sufficient and MPE itself adds little.
```

### Thesis D

```text
Real-world metric smoothness is too weak for the approach to justify
a standalone paper.
```

---

## 26. Best paper titles

Give five titles ranked strongest to weakest.

If Thesis A survives, consider titles in the spirit of:

```text
Beyond Numerical and Categorical:
Metric-Space Tokenization for Tabular Learning
```

or:

```text
Metric Partition Embeddings:
Generalizing Tabular Features to Unseen States
```

Do not use these automatically if results support only Thesis B/C.

---

## 27. Paper outline

Provide an ICLR paper outline:

```text
1 Introduction
2 Typed metric fields
3 Metric Partition Embedding
4 Theory
5 Experimental protocol
6 Real unseen-state benchmark
7 Mechanism and support-distance analysis
8 Boundaries and failures
9 Related work
10 Conclusion
```

Adjust based on evidence.

---

## 28. Final recommendation

Choose exactly:

```text
COMMIT TO MPE PAPER
KEEP MPE AS SECOND PAPER / CONTINUE LATER
PIVOT TO DIFFERENT METRIC METHOD
STOP THIS DIRECTION
```

Explain why.

---

# 78. Final completion audit

Before writing `results.md`, create:

```text
FINAL_AUDIT.md
```

Verify:

```text
all mandatory datasets attempted
all required public datasets completed unless objectively unavailable
all mandatory baselines completed
all state splits are disjoint
no target leakage
all metric definitions frozen before outcomes
all theorem tests pass
all raw results are present
all tables regenerate
all figures regenerate
all statistical summaries reproduce
no unfavorable cells silently omitted
protocol deviations documented
```

Report:

```text
tests passed / total tests
```

---

# 79. Final decision principle

The project should survive only if reality confirms the synthetic mechanism.

The decisive question is NOT:

> Can MPE crush PLE on a synthetic tree?

That is already known.

The decisive question is:

> Given a real feature whose geometry exists independently of the target, and a real unseen state at test time, does MPE use that geometry more effectively than the strongest existing ways to expose exactly the same information?

If YES across several unrelated metric spaces, this can become a strong ICLR paper.

If hierarchy-specific, spectral, similarity, kernel, or coordinate methods consistently match or beat MPE, state that clearly.

The purpose of this program is to discover whether MPE is genuinely a new useful abstraction—not to prove that it is.