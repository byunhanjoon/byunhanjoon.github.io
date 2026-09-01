# AGENT.md — 8-HOUR ICLR 2027 THEORY-FIRST DIRECTION SEARCH

## Mission

Spend approximately **8 hours of wall-clock time** identifying and stress-testing the strongest next research direction for an ICLR 2027 paper.

The main candidate is:

> **Retrieval Risk Geometry:** understand what geometry a retrieval-based tabular learner should use to decide which training rows are useful neighbors.

But this is a direction-search program, not a mandate to make that hypothesis succeed.

Also test:

1. nonlinear per-feature geometry without retrieval;
2. whether geometry preservation inside a Transformer matters;
3. whether any genuinely new OrbitCover consequence is stronger than these alternatives.

The final goal is not to accumulate model wins.

The goal is to find a direction with:

```text
simple theoretical statement
+
nontrivial derived insight
+
clear falsifiable mechanism
+
empirical signal on real tabular data
+
defensible novelty after 2026 literature
```

Do not stop midway.

Do not ask for confirmation.

Use the available compute until the program is complete or the eight-hour budget is exhausted.

At the end, create:

```text
results.md
DIRECTION_RANKING.md
```

`results.md` must contain the complete scientific findings.

`DIRECTION_RANKING.md` must choose the best next ICLR direction.

---

# 0. Existing project boundary

Read the current OrbitCover paper and existing experimental reports first.

Treat OrbitCover as a separate mature project.

Important existing findings to preserve:

```text
coupled schema×RNG balancing strongly reduces finite-budget quotient error;

schema balancing with fresh independent randomness does not;

canonical and symmetrized targets can differ;

the OC2 advantage over SRS disappears on average near convergence;

matched-function controls nearly close schema effects at convergence;

interaction order does not reliably predict which cells benefit.
```

Do NOT spend the eight-hour run merely adding more OrbitCover datasets.

Only pursue an OrbitCover successor if a genuinely new theoretical principle emerges.

---

# 1. Literature boundary — maximum 45 minutes

Immediately search literature through August 2026.

At minimum inspect:

```text
TabR
ModernNCA
On Embeddings for Numerical Features in Tabular Deep Learning
learnable spline / learned-knot numerical embeddings
Unveiling the Role of Data Uncertainty in Tabular Deep Learning
metric learning / NCA
local Mahalanobis metric learning
adaptive nearest-neighbor metrics
kernel regression bandwidth/metric theory
Tab-PET / tabular structural positional encodings
GGPL or other 2026 embedding methods used inside ModernNCA/TabR-like models
```

Create:

```text
LITERATURE_BOUNDARY.md
```

For each close paper record:

```text
what geometry is learned
whether geometry is global or input-dependent
whether retrieval is involved
whether numerical features are nonlinearly warped
whether a local risk theorem is derived
whether neighbor noise enters the theory
whether the method distinguishes retrieval representation from prediction representation
closest overlap with each hypothesis below
```

Reject any proposed novelty of the form:

```text
"nonlinear embedding before kNN is new"

"deep retrieval metric is new"

"target-consistent embeddings are new"

"geometry-aware tabular attention is generically new"

"learnable knots or splines are new"
```

---

# 2. Candidate directions

Evaluate FOUR directions.

## Direction A — Retrieval Risk Geometry

Priority:

```text
PRIMARY
```

Question:

> What makes one training row a statistically good neighbor for another row?

Core idea:

TabR and ModernNCA learn a distance space, but Euclidean closeness is only a proxy for actual retrieval usefulness.

Derive the risk of retrieving candidate rows directly.

---

## Direction B — Local Feature Metric / nonlinear warp

Question:

> Can a simple nonlinear per-feature warp induce a better local metric for retrieval than a globally linear representation?

Treat this as a possible consequence of Direction A.

Do NOT treat arbitrary nonlinear embeddings as inherently novel.

---

## Direction C — Geometry preservation in Transformers

Question:

> If a tokenizer creates useful value geometry, does standard FT-Transformer attention preserve, exploit, or destroy it?

This is diagnostic.

Do not invent complex geometry-aware attention unless there is empirical evidence that standard attention specifically destroys a useful geometry.

---

## Direction D — OrbitCover theoretical successor

Question:

> Is there a new general principle of optimal stochastic coupling that survives the limitations of the existing OrbitCover result?

Time-cap this branch aggressively.

Do not duplicate generic antithetic sampling, OA theory, or covariance reduction.

---

# 3. THEOREM A1 — exact retrieval risk identity

This is the first major theoretical target.

Consider a query `x`.

Candidate training targets satisfy:

```text
Y_i = m_i + eps_i
```

where:

```text
m_i = E[Y_i | X_i]
E[eps_i | X_i] = 0
```

Let:

```text
m_x = E[Y | X=x]
```

A retrieval predictor uses weights:

```text
w_i >= 0
sum_i w_i = 1
```

and predicts:

```text
m_hat(x) = sum_i w_i Y_i.
```

Assume candidate noises are conditionally independent initially.

Derive:

```text
E[(m_hat(x) - m_x)^2 | X]
=
(w^T d)^2
+
w^T Sigma w
```

where:

```text
d_i = m_i - m_x
Sigma = diag(sigma_i^2)
```

Call:

```text
B_retrieval = (w^T d)^2
```

the:

```text
neighbor mismatch / transfer bias
```

and:

```text
V_retrieval = w^T Sigma w
```

the:

```text
neighbor noise cost.
```

This is the basic Retrieval Risk Law.

Fresh query outcome noise is irreducible and cancels in method comparisons.

Verify algebra symbolically and by Monte Carlo.

---

# 4. THEOREM A2 — oracle optimal retrieval weights

Let:

```text
H = d d^T + Sigma.
```

Ignoring nonnegativity but enforcing:

```text
1^T w = 1,
```

derive the minimum-risk weights:

```text
w* = H^{-1} 1 / (1^T H^{-1} 1)
```

when `H` is invertible.

If singular, state the pseudoinverse version carefully.

With:

```text
w >= 0
```

show that the problem becomes a convex quadratic program:

```text
min_w w^T H w
s.t. 1^T w = 1
     w >= 0.
```

Interpretation:

> The optimal neighbor aggregation depends jointly on conditional target mismatch and candidate uncertainty.

This gives a concrete ideal that ordinary Euclidean retrieval only approximates.

---

# 5. COROLLARY A3 — one-neighbor ideal geometry

For one-neighbor retrieval:

```text
w_i = 1
```

the excess risk of candidate `i` is:

```text
r_i(x)
=
(m_i - m_x)^2
+
sigma_i^2.
```

Thus the statistically best neighbor is NOT generally the Euclidean nearest neighbor.

It minimizes:

```text
target mismatch² + candidate noise.
```

This is a key simple insight.

---

# 6. COROLLARY A4 — local target metric

Assume:

```text
m(x)
```

is differentiable.

For:

```text
x_i = x + delta_i
```

derive:

```text
m(x_i) - m(x)
≈
grad m(x)^T delta_i.
```

Therefore candidate mismatch is locally:

```text
(m_i - m_x)^2
≈
delta_i^T
[
grad m(x) grad m(x)^T
]
delta_i.
```

Define the local signal metric:

```text
G_signal(x)
=
grad m(x) grad m(x)^T.
```

For multi-output regression or probability-vector prediction, derive if clean:

```text
G_signal(x)
=
J_m(x)^T J_m(x).
```

Interpretation:

> Locally, the geometry relevant to retrieval is induced by how the conditional target changes, not by raw Euclidean feature distance.

Candidate uncertainty contributes a separate cost.

Do NOT claim this is a new general theorem of differential geometry or metric learning.

Its value is the tabular retrieval interpretation and experimental validation.

---

# 7. THEOREM A5 — induced geometry of TabR-like models

Let a differentiable row encoder be:

```text
Phi(x)
```

and TabR-like retrieval keys be:

```text
k(x) = W_K Phi(x).
```

Squared retrieval distance is:

```text
d_theta²(x,x')
=
||k(x)-k(x')||².
```

For small perturbations:

```text
x' = x + delta,
```

derive:

```text
d_theta²(x,x+delta)
≈
delta^T G_theta(x) delta
```

with:

```text
G_theta(x)
=
J_Phi(x)^T W_K^T W_K J_Phi(x).
```

Important consequences:

### Linear representation

If:

```text
Phi(x) = Bx + c
```

then:

```text
G_theta(x)
```

is constant.

Retrieval uses a global Mahalanobis metric.

### Nonlinear representation

If:

```text
Phi
```

is nonlinear:

```text
G_theta(x)
```

can vary over the input space.

Retrieval therefore implements a local metric field.

This is the formal bridge from numerical embeddings to TabR.

---

# 8. HYPOTHESIS A

The important mechanism is NOT:

```text
nonlinear embeddings are more expressive.
```

It is:

```text
a learned retrieval metric is good when it ranks candidates
similarly to the theoretical retrieval-risk quantity.
```

Define candidate oracle risk on synthetic data:

```text
r_i(x)
=
(m_i-m_x)^2 + sigma_i².
```

For a trained retrieval model calculate:

```text
Spearman(
    model_distance(x, x_i),
    r_i(x)
)
```

over candidate sets.

Lower-distance candidates should have lower retrieval risk.

Also report:

```text
mean oracle risk among top-k retrieved rows

oracle top-k overlap

retrieved target mismatch

retrieved candidate noise

test prediction loss
```

The mechanism is supported only if improved prediction accompanies improved neighborhood risk.

---

# 9. Synthetic dataset S1 — rotating local metric

Construct a 2-D regression problem where the locally relevant feature direction changes with position.

Example structure:

```text
for one region:
    target varies rapidly with x1 and slowly with x2

for another region:
    target varies rapidly with x2 and slowly with x1
```

Make the transition smooth.

Use heteroscedastic noise.

This construction makes a single global diagonal/Mahalanobis metric suboptimal.

Ground-truth:

```text
grad m(x)
G_signal(x)
sigma²(x)
```

are exactly known.

Compare:

```text
raw Euclidean kNN
global Mahalanobis metric
TabR baseline
ModernNCA
PLE/PLR + retrieval
nonlinear local-warp retrieval
oracle G_signal neighbor ranking
oracle risk weights
```

The purpose is NOT to make the new method win.

The purpose is to determine whether learned retrieval geometry tracks the known target metric.

---

# 10. Synthetic dataset S2 — global metric suffices

Construct a target with constant relevant direction.

A global Mahalanobis metric should be enough.

A nonlinear local metric should provide little/no advantage.

This is a negative control.

---

# 11. Synthetic dataset S3 — geometry vs noise

Hold:

```text
m(x)
```

fixed.

Create regions with identical conditional means but different:

```text
sigma²(x).
```

The theory predicts that equally close candidates need not be equally useful.

Compare:

```text
ordinary distance retrieval
uncertainty-aware oracle retrieval
TabR
ModernNCA
```

Determine whether trained retrieval models implicitly avoid high-noise candidates.

This directly tests a claim adjacent to, but more precise than, the 2026 tabular uncertainty work.

---

# 12. Synthetic dataset S4 — nonlinear value warp

Use one or two numerical dimensions where the target changes at strongly nonuniform rates.

Construct a monotone nonlinear coordinate:

```text
tau_j(x_j)
```

whose local derivative expands high-target-gradient regions.

Compare:

```text
raw values
PLE
periodic/PLR
B-spline / learned-knot spline
simple monotone warp
inverse/wrong warp
```

inside:

```text
MLP
TabR
ModernNCA
```

Critical comparison:

> Does the same nonlinear warp help retrieval significantly more than it helps a retrieval-free MLP?

If no, demote the retrieval-geometry story.

---

# 13. Branch B method prototype — LocalWarp

Only after the theory and synthetic checks are working.

Implement a deliberately SIMPLE per-feature warp:

```text
z_j = tau_j(x_j)
```

with:

```text
tau_j'(x) > 0
```

for numerical fields.

Use either:

```text
piecewise-linear monotone warp
or
small monotone spline
```

with very few parameters.

Do NOT claim the warp parameterization is novel.

Its role is to realize a simple input-dependent metric.

Retrieval metric:

```text
d²(x,x')
=
sum_j alpha_j
[
tau_j(x_j)-tau_j(x'_j)
]^2
```

optionally followed by TabR's lightweight learned key transform.

Compare against:

```text
raw
PLE
PLR
learned-knot spline
```

because nonlinear numerical embedding is crowded.

---

# 14. Critical representation-vs-retrieval ablation

For every promising embedding/warp in TabR implement four conditions:

```text
A. raw prediction branch + raw retrieval branch

B. nonlinear prediction branch + raw retrieval branch

C. raw prediction branch + nonlinear retrieval branch

D. nonlinear prediction branch + nonlinear retrieval branch
```

This is mandatory.

Interpretation:

### If B helps but C does not

The effect is ordinary representation learning.

Not a retrieval-geometry paper.

### If C helps materially

Strong evidence for the retrieval-geometry mechanism.

### If D > B and C

There may be complementarity.

---

# 15. Key-network capacity ablation

A deep key encoder may already absorb input warps.

Test:

```text
linear key encoder
shallow key encoder
standard TabR key encoder
more expressive key encoder
```

with and without nonlinear feature geometry.

Question:

> Does explicit feature geometry remain useful once the retrieval network can learn an arbitrary nonlinear metric?

If the gain disappears with a sufficiently expressive key model, the new embedding is mostly an inductive-bias/optimization aid.

Report that honestly.

This is potentially an important insight by itself.

---

# 16. Real-data screening panel

Use cached/local public datasets to avoid wasting the eight-hour budget on downloads.

Freeze approximately:

```text
8 datasets
```

before results.

Target:

```text
4 classification
4 regression
```

with diversity in:

```text
sample size
numerical feature fraction
feature count
noise level
retrieval usefulness
```

Prefer locally available TabArena / TabReD / TALENT datasets.

Good candidates if available:

```text
California Housing
Adult
Bank Marketing
Credit Card Default
Covertype
FREMtpl Claim Count
Kin8nm
KDD17 Stock Return
```

If some are unavailable, replace BEFORE model results with the next cached dataset satisfying the balance criteria.

Write the frozen list to:

```text
REAL_PANEL_FREEZE.md
```

Do not swap datasets based on outcomes.

---

# 17. Real-data models

Mandatory:

```text
MLP
TabR
ModernNCA
```

Optional if cheap:

```text
FT-Transformer
```

Do NOT run a huge architecture zoo.

The key comparison is:

```text
retrieval-free
vs
two independent retrieval paradigms.
```

---

# 18. Real-data representations

At minimum compare:

```text
raw standardized numerical inputs
PLE
PLR/periodic embedding
learned-knot spline if implementation already available
LocalWarp prototype
```

Do not spend hours implementing exotic new encodings.

Use the same preprocessing for categoricals across relevant comparisons.

---

# 19. Compute policy for real screen

Use:

```text
3 seeds
```

for the first-pass core methods where affordable.

Use moderate training budgets sufficient to expose direction signal.

Do not pursue full SOTA hyperparameter optimization.

Use published/default settings for:

```text
TabR
ModernNCA
MLP
```

and freeze them.

If a dataset is too large for the eight-hour budget:

```text
use a predeclared training subsample
```

based on dataset size, not performance.

---

# 20. Real-data retrieval diagnostics

For every retrieval model record its top-k candidates on a fixed test subset.

Calculate:

```text
neighbor target consistency

neighbor residual consistency after a base predictor

neighbor label entropy for classification

within-neighborhood target variance

retrieval distance

candidate frequency

candidate prediction uncertainty proxy
```

Use cross-fitting where target-derived diagnostics might otherwise leak.

The central question is:

> Do better models actually retrieve statistically better neighbors?

---

# 21. Cross-fitted real retrieval-risk proxy

For each training row obtain out-of-fold predictions:

```text
m_hat_i
```

and residual/noise estimates.

For query/candidate pairs define a proxy:

```text
r_hat_i(x)
=
(m_hat_i - m_hat_x)^2
+
sigma_hat_i².
```

Do not use final test labels in training.

Use this only as an analysis metric on test queries.

Measure how well each trained model's distance ranks:

```text
r_hat.
```

Compare this geometric-alignment statistic to test-performance improvements across:

```text
dataset × model × representation
```

cells.

If alignment improves while prediction does not, note the mismatch.

---

# 22. Test the local-metric theorem directly

For low-dimensional numerical datasets or synthetic slices estimate:

```text
G_theta(x)
```

from automatic differentiation:

```text
J_Phi(x)^T W_K^T W_K J_Phi(x).
```

Compare with:

```text
G_signal(x)
```

or a proxy based on the gradient of a strong cross-fitted target model.

Metrics:

```text
Frobenius cosine similarity
top eigenvector angle
featurewise diagonal correlation
```

Question:

> Does TabR learn a metric field aligned with local conditional-target geometry?

This is one of the most interesting possible findings.

---

# 23. Direction C — Transformer diagnostic

Time budget:

```text
<= 60 minutes of experiments
```

Do not let this branch consume the run.

Use:

```text
FT-Transformer
```

on:

```text
2 synthetic tasks
2 real datasets
```

Compare:

```text
raw embedding
PLE
best nonlinear warp from Branch A/B
```

Measure whether the embedding's within-feature distance geometry survives:

```text
tokenization
first attention block
final representation
```

through pairwise-distance correlation/Jacobian diagnostics.

Do NOT build geometry-aware attention unless BOTH hold:

```text
1. the nonlinear geometry strongly helps MLP/retrieval models;

2. standard FT-Transformer specifically destroys that geometry
   and loses the corresponding performance advantage.
```

If those conditions do not hold:

```text
DEMOTE TRANSFORMER-GEOMETRY.
```

---

# 24. Optional Transformer intervention

Only if the previous gate passes.

Try ONE minimal intervention:

```text
orthogonal / near-isometric QK projection
```

or:

```text
residual geometry-preserving branch
```

Do not invent a complicated new Transformer.

The question is mechanistic:

> Does preserving the tokenizer geometry rescue the lost effect?

---

# 25. Direction D — OrbitCover successor audit

Time budget:

```text
<= 30 minutes theory/literature
<= 30 minutes analysis
```

Investigate only one question:

> Can the same-target coupling result be reframed as a general covariance-optimal coupling theorem that is not already generic antithetic Monte Carlo?

Search recent 2026 antithetic/randomized integration literature.

If the answer appears occupied or merely classical:

```text
STOP THIS BRANCH.
```

Do not run new OrbitCover training.

Use existing tensors only.

---

# 26. Strong novelty gate for Direction A

The direction is NOT novel enough if its final claim is merely:

```text
"nonlinear embeddings improve TabR"

"learned distances are useful"

"neighbors with similar labels are better"

"high-noise neighbors are worse"

"deep metric learning helps tabular data"
```

To survive, the eight-hour program should identify at least ONE deeper contribution such as:

### Candidate contribution 1

Exact Retrieval Risk Law:

```text
risk = target-mismatch² + propagated neighbor noise.
```

### Candidate contribution 2

A local target metric:

```text
G_signal(x)=J_m(x)^T J_m(x)
```

that predicts which retrieval geometries succeed.

### Candidate contribution 3

Evidence that:

```text
G_theta(x)
```

learned by TabR/ModernNCA aligns with:

```text
G_signal(x)
```

when retrieval works.

### Candidate contribution 4

A simple derived metric/warp that improves retrieval specifically, not merely feature representation.

The ideal paper contains contributions 1–3 and optionally 4.

---

# 27. Falsification experiments

The following negative controls are mandatory.

## F1 — globally linear task

Local metric should add little.

## F2 — wrong nonlinear warp

Same parameter count, deliberately inverse local stretching.

Should worsen neighborhood quality if geometry matters.

## F3 — prediction-only warp

Tests ordinary representation effect.

## F4 — retrieval-only warp

Tests actual metric effect.

## F5 — expressive key network

Tests whether explicit feature geometry is redundant.

## F6 — random-label / high-noise synthetic region

Tests whether the metric chases noise.

## F7 — same representation dimension and comparable parameter count

Prevents simple capacity explanations.

---

# 28. Simple insight target

Prefer conclusions a reader can remember.

Potential examples, only if supported:

```text
"A good neighbor is not the closest row; it is the row with
low conditional target mismatch and low label uncertainty."

"TabR's Euclidean retrieval is a learned Riemannian metric."

"Linear embeddings give a global metric; nonlinear embeddings
allow the neighborhood geometry to change across the table."

"Nonlinear embeddings help retrieval only when the desired
neighbor geometry is spatially nonstationary."

"An expressive retrieval network can make explicit nonlinear
feature warps redundant."
```

One or two of these are better than ten minor benchmark wins.

---

# 29. Eight-hour schedule

Use this as a target, not a hard stop per command.

## Hour 0:00–0:45

```text
read existing artifacts
literature audit
freeze hypotheses
```

## Hour 0:45–2:00

```text
derive A1–A5
implement theorem checks
build synthetic tasks
```

## Hour 2:00–3:15

```text
run synthetic metric experiments
test local metric field
test noise-aware retrieval law
```

## Hour 3:15–6:15

```text
run 8-dataset real screening
MLP / TabR / ModernNCA
representation/retrieval ablations
```

Use parallel GPUs/jobs where available.

## Hour 6:15–7:00

```text
mechanism analysis
G_theta alignment
retrieval-neighborhood diagnostics
```

## Hour 7:00–7:40

```text
Transformer diagnostic
OrbitCover-successor novelty audit
```

## Hour 7:40–8:00

```text
aggregate
audit
write results.md
write DIRECTION_RANKING.md
```

If training finishes early, spend extra time on:

```text
mechanism
theory validation
stronger controls
```

not on random extra datasets.

---

# 30. Runtime discipline

Do NOT finish after only implementing code.

The task ends when:

```text
theory derivations are written
synthetic results exist
real screening results exist
mechanism diagnostics exist
all directions are ranked
results.md exists
DIRECTION_RANKING.md exists
```

If an experiment crashes:

```text
debug
retry
continue
```

If one method is impossible to run:

```text
document why
continue with all remaining mandatory work
```

Do not spend >30 minutes rescuing one optional dependency.

---

# 31. Theoretical integrity

Create:

```text
THEORY.md
```

Include:

```text
Theorem A1 retrieval risk identity
Theorem A2 optimal weights
Corollary A3 one-neighbor risk
Corollary A4 local signal metric
Theorem A5 induced TabR metric
assumptions
proofs
failure boundaries
relationship to metric-learning literature
```

Numerically verify every formula.

Do not call standard NCA/metric-learning results novel.

---

# 32. Main result tables

Generate:

```text
table_synthetic_theory.csv
table_real_performance.csv
table_retrieval_quality.csv
table_metric_alignment.csv
table_branch_ablation.csv
table_direction_ranking.csv
```

---

# 33. Required figures

Generate at least:

## Figure 1 — Retrieval Risk Law

Theory vs Monte Carlo retrieval risk.

---

## Figure 2 — What is a good neighbor?

Show candidates with:

```text
feature distance
target mismatch
noise
total theoretical retrieval risk
```

---

## Figure 3 — Rotating metric synthetic

Visualize:

```text
G_signal(x)
```

and learned:

```text
G_theta(x)
```

over space.

---

## Figure 4 — Retrieval-only vs prediction-only

Across real datasets.

---

## Figure 5 — Metric alignment vs performance gain

X:

```text
G_theta / retrieval-risk alignment
```

Y:

```text
retrieval performance improvement
```

---

## Figure 6 — Key capacity interaction

Embedding benefit vs key-network expressiveness.

---

## Figure 7 — Direction comparison

Summary chart of:

```text
novelty
theory
empirical signal
prior-art risk
simplicity
ICLR potential
```

for all four directions.

---

# 34. Statistical treatment

For real datasets, report:

```text
dataset-balanced averages
per-dataset differences
seed uncertainty
wins/losses
```

Do not treat seeds as independent datasets.

This is an eight-hour screening experiment, so do not overclaim significance from a small panel.

The question is:

```text
is there enough signal to justify a full prospective study?
```

---

# 35. results.md structure

At the end create:

```text
results.md
```

Use this exact structure.

# RESULTS — 8-HOUR ICLR 2027 DIRECTION SEARCH

## 1. Executive verdict

Rank:

```text
Retrieval Risk Geometry
General Feature Geometry
Transformer Geometry
OrbitCover Successor
```

Give each:

```text
ICLR potential /5
novelty /5
theory clarity /5
empirical signal /5
prior-art risk /5
```

Select ONE primary next direction.

---

## 2. Literature subtraction

Explain what is already occupied.

Explicitly discuss:

```text
TabR
ModernNCA
2026 uncertainty analysis
PLE/PLR
learned spline/knots
recent retrieval embedding modifications
tabular structural Transformer work
```

State exactly what novelty remains.

---

## 3. Retrieval Risk Law

State A1–A5.

Mark:

```text
proved
partially proved
failed
```

for each.

---

## 4. Synthetic theory validation

Report:

```text
theory vs Monte Carlo
oracle retrieval quality
local metric alignment
noise experiment
negative controls
```

---

## 5. Does nonlinear geometry improve retrieval specifically?

Main ablation:

| Prediction branch | Retrieval branch | Result |
|---|---|---|
| raw | raw | |
| nonlinear | raw | |
| raw | nonlinear | |
| nonlinear | nonlinear | |

Answer clearly.

---

## 6. TabR results

Report:

```text
performance
retrieval quality
metric alignment
representation variants
key capacity
```

---

## 7. ModernNCA results

Same analysis.

Does the insight transfer beyond TabR?

This is important.

---

## 8. MLP control

Does the same embedding help even without retrieval?

If yes, quantify how much.

---

## 9. Learned metric field

Report evidence for:

```text
G_theta(x)
```

being input-dependent and aligned/misaligned with:

```text
G_signal(x).
```

---

## 10. Neighbor quality mechanism

Answer:

> When TabR/ModernNCA improves, does it actually retrieve lower-risk candidates according to the theory?

---

## 11. Key-network redundancy

Answer:

> Does a sufficiently expressive key network make explicit nonlinear feature geometry unnecessary?

This may determine the paper thesis.

---

## 12. Transformer diagnostic

State whether:

```text
standard FT-Transformer exploits
preserves
or destroys
```

the tested feature geometry.

Choose:

```text
PROMOTE
DEMOTE
```

for custom Transformer geometry.

---

## 13. OrbitCover successor

State whether any truly new theoretical extension was found.

Default to:

```text
KEEP CURRENT PAPER SEPARATE
```

unless evidence strongly says otherwise.

---

## 14. Failed hypotheses

List every failure.

Examples:

```text
nonlinear retrieval geometry gives no benefit
metric alignment does not correlate with performance
explicit warp is redundant with deep keys
Transformer does not destroy geometry
noise-aware theory does not explain neighborhoods
```

---

## 15. Best simple scientific insight

Write ONE central sentence.

It should be something memorable such as:

```text
"A statistically good neighbor is defined by conditional target
mismatch and uncertainty, not raw feature distance."
```

but only if supported.

---

## 16. Candidate ICLR thesis

Write the strongest supported thesis.

Examples:

### Thesis A

```text
Retrieval-based tabular models can be understood as learned local
metric fields; their success is predicted by alignment with a
conditional target-and-noise retrieval risk.
```

### Thesis B

```text
Nonlinear feature embeddings improve retrieval by allowing local
rather than global tabular metrics.
```

### Thesis C

```text
Deep key encoders already learn the needed local geometry, making
explicit nonlinear embeddings largely redundant.
```

A negative Thesis C can still motivate an analysis paper if the mechanism is strong.

---

## 17. Method consequence

If a new method survived, describe it.

Do NOT invent a method merely because the theory is interesting.

Report:

```text
method name
equations
parameter overhead
why it follows from theory
which baseline it beats
```

---

## 18. ICLR readiness

Choose:

```text
STRONG NEW DIRECTION
PROMISING — NEED PROSPECTIVE PANEL
INTERESTING THEORY ONLY
TOO CROWDED
FAILED
```

---

## 19. Next 3-day experiment plan

Give a concrete follow-up ONLY for the winning direction.

Specify:

```text
datasets
baselines
theorems
prospective freeze
architectures
compute
success gates
```

---

# 36. DIRECTION_RANKING.md

Create a separate concise file.

Use:

| Direction | Novelty | Theory | Signal | Simplicity | Prior-art risk | ICLR potential |
|---|---:|---:|---:|---:|---:|---:|

Directions:

```text
Retrieval Risk Geometry
Nonlinear Feature Metric
Transformer Geometry
OrbitCover Extension
```

Then write:

```text
WINNER =
```

and exactly one direction.

Also state:

```text
WHY =
KILL CONDITION =
NEXT DECISIVE EXPERIMENT =
```

---

# 37. Decision criteria

## Promote Retrieval Risk Geometry if

At least three of these hold:

```text
1. exact theory cleanly predicts synthetic retrieval risk;

2. learned TabR/ModernNCA distances correlate strongly with
   low theoretical neighbor risk;

3. better metric alignment tracks real performance;

4. retrieval-only nonlinear geometry provides gains beyond
   prediction-only embedding;

5. the phenomenon transfers across TabR and ModernNCA;

6. a simple theory-derived modification improves equal-budget performance.
```

---

## Demote generic nonlinear embeddings if

```text
their gains are similar in MLP and retrieval models;

deep key networks absorb the effect;

learned splines/PLE match them;

mechanism reduces to additional nonlinear capacity.
```

---

## Promote Transformer geometry only if

```text
a useful tokenizer geometry is empirically destroyed by standard
attention and a minimal preservation intervention restores the effect.
```

Otherwise do not pursue it.

---

## Keep OrbitCover separate unless

a genuinely new theorem emerges that:

```text
is not generic antithetic sampling;
uses the existing same-target coupling evidence;
survives convergence or explains why convergence changes the optimum;
has clear broader relevance beyond the current finite nuisance product.
```

---

# 38. Final principle

Do not optimize for:

```text
another +1% benchmark method.
```

Search for:

```text
one equation
+
one causal mechanism
+
one surprising empirical fact
```

that together explain something important about modern tabular learning.

The best candidate is currently:

> **A good retrieved neighbor is not the geometrically closest row; it is one whose conditional target is compatible with the query and whose label is sufficiently reliable. TabR and ModernNCA can be interpreted as attempts to learn that geometry.**

The eight-hour run must determine whether this is actually true.