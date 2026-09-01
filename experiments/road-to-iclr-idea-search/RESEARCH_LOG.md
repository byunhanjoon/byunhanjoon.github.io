# ICLR idea search — working research log

Date: 2026-08-26

This is a chronological research notebook, not the final recommendation. It
records evidence boundaries early so that later literature or experiments do
not quietly rewrite failed hypotheses.

## 1. Local evidence audit

### Day 1–2: what survived

- The universal multi-view encoder failed against a tuned PLE baseline.
- Adult contains a large, repeatable exact-value effect: identity views for
  `capital-gain` and `capital-loss` add about 1.2 accuracy points across MLP,
  ResNet, and TabM. The gain is concentrated in rare-but-seen values rather
  than the dominant zero atom.
- The conservative exact-state selector discovered structure on only one of
  six untouched datasets. This is a mechanism/case study, not a broad method.
- Cumulative and local PLE have the same affine span and dimension. Their
  two-model ensemble improved proper loss on all nine development datasets,
  but an ordinary two-seed ensemble was stronger on average.

### Day 3: what survived

- On 30 datasets and four dense-stem neural architectures, a scale-controlled
  invertible basis change at condition number 1,000 harmed AdamW in 336/360
  paired cells; mean normalized sensitivity was about -8.4%.
- Function matching removed 94.66% of the severe kappa=3,000 endpoint harm in
  a three-dataset trajectory study. Ordinary initialization is therefore the
  dominant mechanism in that stress test; AdamW still becomes non-equivariant
  after one matched update.
- Exact anchor coordinates and ideal input-natural updates close the
  controlled orbit. These are causal controls, not new invariance ideas.
- Rank deficiency is a real boundary: an undamped inverse was unstable on the
  123-column/rank-116 Adult representation.
- Natural cumulative/local differences are much smaller (median absolute
  normalized gap about 0.38%) and have no universal ordering.
- Temporal deployment splits did not amplify the effect in the three finance
  tasks tested.
- Selective Measure-Orbit passed an internal 21-dataset gate but failed the
  required seven-dataset untouched test: -0.521% proper-loss reduction versus
  an exactly update-matched two-seed TabM ensemble, with the interval entirely
  below zero. It is not a broad performance method.

### Unresolved observations with real information value

1. Conditioning is not a complete explanation. At equal condition number,
   local sparse and dense orthogonal charts can still induce different
   solutions.
2. The large controlled harm is mostly an initialization-prior mismatch, but
   the smaller optimizer-only residual is causal and immediate.
3. Robustness to arbitrary bases may conflict with the useful axis-aligned
   sparsity/locality prior of tabular data.
4. Current tabular leaderboards report one schema per dataset even though
   modern pipelines and foundation models use materially different
   preprocessing ensembles.

## 2. Breadth literature map

| Area | Established by prior work | Gap that remains potentially useful |
| --- | --- | --- |
| Affine-invariant optimization | Natural gradient and ideal K-FAC invariance; inverse-free/stable K-FAC; isometric and preconditioned-norm optimizers | A new generic invariant optimizer is crowded. The finite tabular trade-off between invariance and useful axis bias is not answered by these results. |
| Rotation/axis bias | Grinsztajn et al. show that preserving feature orientation is important in typical tabular learning. VectorAdam shows coordinatewise Adam is not rotation equivariant. | No checked work traces a fixed-information, fixed-spectrum, function-matched rotation path through *tabular feature-function bases* and separates semantic locality from conditioning. |
| Sparse targets vs invariance | Warmuth et al. show rotation-invariant learners can be suboptimal for noisy sparse targets. | This creates a directly testable tension with Day 3: maximal basis invariance may erase a beneficial tabular prior. |
| Numerical embeddings | PLE/periodic embeddings and general function-basis encodings are established. Stretch transformations optimize target smoothness. | Basis choice inside one exact function space is under-studied as an optimization variable after spectrum and function prior are controlled. |
| Tabular ensembling | TabM, TabPFN preprocessing ensembles, TabArena portfolio results, and Rotation Forest establish multiple forms of diversity. | Day 3's performance portfolio failed externally. Do not claim representation diversity without an equal-compute seed/checkpoint baseline. |
| Benchmark design | TabArena is a living benchmark; data-centric tabular evaluation shows expert preprocessing changes model rankings; general benchmark rankings can be unstable. | Existing studies do not define a schema-equivalence orbit or report ranking uncertainty over transformations that retain exactly the same information. This is narrower than generic preprocessing sensitivity. |
| Feature shifts | TabFSBench changes the available feature set between environments. | Jointly recoding train and test by an invertible map is not distribution/feature shift; it audits the learner, not deployment drift. |
| Tabular foundation models | TabPFN, TabDPT, TabICL, and the new TabFM make synthetic-pretrained in-context learners central baselines; TabPFN already ensembles preprocessing configurations. | Their learned-algorithm robustness to exact schema orbits, and the stability of cross-family rankings under those orbits, remains unclear. |
| Mixed atomic/continuous columns | Mixed discrete/continuous modeling is established in generative modeling; supervised numerical embedding work does not appear to target within-column atoms directly. | Atom/tail decomposition remains a plausible niche mechanism, but the local external evidence is too weak for a lead paper without a sharply pre-specified dataset family. |

## 3. Candidate directions after the breadth pass

Scores are provisional, 1–5 (higher is better except risk).

| Candidate | Novelty | Scientific value | Local evidence | Feasibility | Risk | Current status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A. FieldRiesz: chart-covariant tabular stems with semantic field metrics | 2.5 | 4.0 | 3.5 | 4.0 | 4.5 | Conditional secondary; fixed semantics selective, adaptive topology search fails nominal admissibility |
| B. OrbitANOVA: schema-representation risk and factor attribution | 3.5 | 5.0 | 4.5 | 4.5 | 3.0 | Current lead after literature falsification and three-dataset pilot |
| A0. Same-spectrum rotation/locality audit as a standalone paper | 2.5 | 4.0 | 4.0 | 4.5 | 3.5 | Keep as an experiment inside A, not the headline |
| C. Mixed-measure numerical embedding | 3.5 | 3.0 | 2.0 | 3.0 | 4.5 | Keep only as targeted side study |
| D. New invariant initializer/canonicalizer | 2.0 | 3.0 | 4.0 | 3.0 | 4.0 | Deprioritize; prior-art crowding |
| E. Orbit/Measure-Orbit performance ensemble | 3.0 | 3.0 | 2.0 | 2.5 | 5.0 | Drop as headline after external failure |

## 4. Candidate A0 — the mechanism experiment, not the paper

Working title: **Same Geometry, Different Bias: Axis Alignment in Tabular
Neural Optimization**

Start with a whitened local feature-function basis `X0`. Construct an
orthogonal path `Q(t)` from the local chart to a dense Haar-like chart and use
`Xt = X0 Q(t)`. Every point on the path has:

- identical information and dimension;
- condition number one and the same covariance spectrum;
- identical pairwise Euclidean distances;
- identical hypothesis class after the first affine layer;
- an exactly matched initial function via `Wt = W0 Q(t)` (orientation adjusted
  to implementation convention).

The only controlled quantity that changes is alignment of coordinate axes with
the local feature-state basis. Compare AdamW with SGD, VectorAdam/Iso-style
updates, and the existing input-natural closure. If only coordinatewise AdamW
tracks rotation density, the optimizer imposes an axis prior. If all methods
change similarly, locality enters elsewhere (architecture, regularization, or
finite nonlinear feature learning). If no method changes, the proposed Day 3
locality story is falsified.

This can no longer carry a paper alone. DePavia et al. (2025) already show that
small input rotations can reverse Adam's implicit-bias advantage, and Singh
(2026) gives a gauge-equivariance account of Adam versus shared-scalar and
spectral updates. The rotation path remains the decisive ablation for the new
field-semantic formulation below.

## 5. Candidate A — original lead, now conditional

Working title: **Features Are Function Spaces, Not Coordinates:
Chart-Covariant Tabular Learning**

Represent field `j` by a finite function space `H_j` and a centered coordinate
chart `phi_j(x_j) in R^{d_j}`. A chart change inside the same field is
`phi'_j = B_j phi_j`, with `B_j` invertible. The legitimate nuisance group is
therefore the direct product `G = product_j GL(d_j)`, not a global `GL(d)` that
mixes semantically different fields.

Each field carries two coordinate-covariant tensors:

- a train-distribution mass matrix `M_j = E[phi_j phi_j^T]`;
- a semantic stiffness matrix `S_j`, such as derivative energy for numerical
  intervals, a path Dirichlet form for ordinal fields, a ring form for cyclic
  fields, or no additional smoothness for nominal fields.

Use `K_j = M_j + lambda S_j` as a Riesz map for the first-layer block. If
`W'_j = W_j B_j^{-1}`, then `G'_j = G_j B_j^T` and

`G'_j K_j'^{-1} = G_j K_j^{-1} B_j^{-1}`,

because `K'_j = B_j K_j B_j^T`. Thus initialization from covariance
`K_j^{-1}`, the penalty `tr(W_j K_j W_j^T)`, and metric-gradient updates all
describe the same function prior and training trajectory in every chart. A
single invariant second-moment scalar per field can recover Adam-like temporal
adaptivity without coordinatewise anisotropy.

This construction preserves a semantic smoothness/locality prior while being
indifferent to the arbitrary basis used to express it. It is deliberately not
invariant to cross-field rotations, which erase the schema structure that makes
tabular learning sample-efficient.

### Exact pilot

`chart_covariance_pilot.py` uses centered cumulative and local bases of the
same ten-knot piecewise-linear space, matches the initial function, and trains a
nonlinear one-hidden-layer predictor for 25 full-batch steps.

- chart reconstruction relative error: `5.70e-15`;
- matched initial prediction gap: `1.02e-15`;
- field-metric prediction gap after 25 steps: `5.77e-15`;
- ordinary function-matched AdamW gap after 25 steps: `0.7064`.

This is an algebraic mechanism check, not evidence of a performance gain.

### Honest novelty boundary

The paper must not claim the first basis-sensitive Adam result, the first
symmetry-compatible optimizer, the first block preconditioner, or the first
spline roughness penalty. The potentially new combination is:

1. the field-wise chart group for supervised tabular representations;
2. the distinction between nuisance coordinates and semantic topology;
3. an exactly chart-covariant first-layer training rule that preserves that
   topology; and
4. a broad empirical test of whether this selective invariance closes chart
   orbits without sacrificing the useful tabular inductive bias.

## 6. Candidate B — original formulation

Working title: **One Dataset, Many Leaderboards: Schema-Orbit Uncertainty in
Tabular Model Evaluation**

Treat a learner as the full mapping from a represented training table to a
predictor. For each dataset, evaluate the learner over a frozen family of
joint train/validation/test recodings that preserve all information. Report:

- orbit-average, orbit-worst, and reference-schema proper loss;
- within-model orbit spread versus seed/HPO spread;
- pairwise winner-flip rate and Kendall rank stability;
- robustness of tuned preprocessing ensembles versus single pipelines;
- results split by semantically mild transformations and deliberately strong
  GL/orthogonal stress tests.

The essential novelty boundary is exact equivalence. Generic expert feature
engineering changes the information/hypothesis interface and is already known
to change rankings. This benchmark asks whether a leaderboard conclusion is
identifiable when the dataset content and split are literally unchanged.

The primary orbit must be semantically defensible: feature order, categorical
label permutations, unit/origin changes, and within-field function-basis
changes. Cross-feature rotations should be reported only as stress tests. A
global rotation preserves information but destroys field identity and changes
the useful axis-aligned hypothesis class of trees, so treating it as a mandatory
invariance would overstate the benchmark claim.

Strong current baselines should include CatBoost/XGBoost/LightGBM, RealMLP,
TabM, TabPFN, TabICL or TabDPT, and (if licensing/compute permits) TabFM.

## 7. Sources checked so far

- Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022), *Why do tree-based models
  still outperform deep learning on typical tabular data?*
- Gorishniy et al. (NeurIPS 2022), *On Embeddings for Numerical Features in
  Tabular Deep Learning*.
- Martens and Grosse (ICML 2015), *Optimizing Neural Networks with
  Kronecker-factored Approximate Curvature*.
- Ling, Sharp, and Jacobson (NeurIPS 2022), *VectorAdam for Rotation
  Equivariant Geometry Optimization*.
- Jackson (2023), *An Isometric Stochastic Optimizer*.
- Lin et al. (ICML 2024), *Structured Inverse-Free Natural Gradient Descent*.
- Dym, Lawrence, and Siegel (ICML 2024), *Equivariant Frames and the
  Impossibility of Continuous Canonicalization*.
- Warmuth et al. (ALT 2025), *How rotation invariant algorithms are fooled by
  noise on sparse targets*.
- Xie et al. (ICML 2025), *Structured Preconditioners in Adaptive
  Optimization*.
- Yun, Lozano, and Yang (AISTATS 2022), *AdaBlock*.
- Bahamou, Goldfarb, and Ren (AISTATS 2023), *A Mini-Block Fisher Method*.
- DePavia, Charisopoulos, and Willett (2025), *How do simple rotations affect
  the implicit bias of Adam?*
- Lau and Su (2026), *Symmetry-Compatible Principle for Optimizer Design*.
- Singh (2026), *The Loss Does Not See the Basis, but Adam Does*.
- Rodríguez, Kuncheva, and Alonso (TPAMI 2006), *Rotation Forest*.
- Gorishniy et al. (ICLR 2025), *TabM*.
- Hollmann et al. (Nature 2025), *Accurate predictions on small data with a
  tabular foundation model*.
- Erickson et al. (NeurIPS 2025), *TabArena*.
- Cheng et al. (ICML 2025), *TabFSBench*.
- Schalz et al. (NeurIPS 2024), *A Data-Centric Perspective on Evaluating
  Machine Learning Models for Tabular Data*.
- Google Research (2026), *Introducing TabFM*.
- Shaheen et al. (2026), *Understanding the Surprising Generalization
  Properties of Tabular Foundation Models*.
- Anonymous TMLR submission (2026), *From Uniform to Learned Knots: A Study of
  Spline-Based Numerical Encodings for Tabular Deep Learning*.
- Luo et al. (SIGIR 2020), *Network On Network for Tabular Data Classification
  in Real-world Applications* (field-wise processing precedent, not chart
  covariance).
- Zhang, Maekawa, and Bhutani (ICLR 2026), *Same Content, Different
  Representations* (Table QA; adjacent evaluation precedent, not supervised
  tabular prediction).

## 8. Next depth-search questions

1. Does FieldRiesz retain competitive proper loss after the exact trajectory
   closure, or does invariance merely make every chart equally mediocre?
2. Does the correct semantic stiffness form outperform `S=0` and deliberately
   wrong path/ring/nominal forms on local, smooth, and cyclic targets?
3. Can a rank-reduced quotient implementation retain covariance at practical
   tolerances on Adult's deficient blocks?
4. Is a per-field shared-scalar adaptive rule enough, or is momentum metric-SGD
   the only exact and stable implementation available in time?
5. Which parts of the Schema-Orbit model matrix can be executed credibly before
   the ICLR 2027 paper deadline on 2026-09-25?

## 9. Initial time-sensitive decision

The ICLR 2027 abstract and paper deadlines are 2026-09-18 and 2026-09-25. A
large new benchmark spanning trees, trained neural models, and several closed
or heavyweight foundation models is unlikely to reach publication quality in
that window. Candidate A reuses the exact Day 2 bases, the Day 3 trajectory and
input-natural code, and the frozen 30-dataset suite. Candidate B is best treated
as a backup paper or a reduced secondary analysis unless existing TabArena
infrastructure makes the full model matrix nearly turnkey.

## 10. Depth-pass falsification of Candidate A

Two nearby 2025–2026 papers materially reduce the optimizer novelty.

- Veprikov et al., *Preconditioned Norms*, give necessary and sufficient
  transformation rules for affine-invariant matrix-parameter updates. The
  FieldRiesz update is an application/special case, not a new generic
  trajectory-equivariance theorem.
- Lau and Su state a symmetry-compatible optimizer-design principle directly,
  while Singh analyzes gauge-equivariant rules and shared-scalar Adam. A claim
  that each parameter block should receive an update matching its symmetry is
  occupied.

The remaining possible novelty was the semantic stiffness tensor. A small
classical diagnostic crossed a mass metric, correct interval stiffness, and a
rotated/wrong stiffness. Interval smoothing helped the smooth numerical target,
but validation-selected smoothing also helped a deliberately false adjacency
on an unordered categorical target. Only stronger fixed regularization exposed
the expected harm. This shows that the first synthetic mostly measures an
ordinary bias-variance trade-off, not a decisive semantic-topology mechanism.

FieldRiesz remains algebraically valid: the cumulative/local pilot closes to
`5.77e-15` after 25 steps while matched AdamW separates by `0.7064`. But exact
closure plus a classical roughness prior is not yet an ICLR contribution. The
idea is now conditional on a frozen smooth/ordinal/cyclic/nominal suite showing
selective benefits from *declared* field metadata. Section 53 later shows that
unconstrained validation can mimic a nominal benefit with false topology, so
wrong graphs cannot be treated as ordinary hyperparameter candidates.

## 11. OrbitANOVA formulation

The backup became sharper after distinguishing representation risk from generic
uncertainty.

For an aligned prediction `p_g(x)` under schema representative `g`, define

`SR = E_x E_g ||p_g(x) - E_g p_g(x)||^2`.

For Brier classification or squared regression, the bias-variance identity
gives exactly

`E_g R(p_g) - R(E_g p_g) = SR`.

Thus `SR` is the average risk removable by orbit averaging and is measurable
without evaluation labels. It is not total uncertainty: a constant predictor
has zero `SR`.

For a product distribution over feature position, class numbering, category
codes, units, and within-field charts, a Hoeffding/functional ANOVA decomposes
`SR` orthogonally into every main effect and interaction. Functional ANOVA is
prior art; the potential contribution is the supervised tabular schema product,
the exact audit quantity, and cross-model/ranking consequences.

EquiTabPFN already defines and proves a target-permutation equivariance gap.
Therefore target-label permutations alone cannot carry OrbitANOVA. The 2026
mechanistic TFM study likewise covers feature/row/class permutation behavior.
Nominal category-code relabeling, unit choices, within-field charts, and the
multi-factor interaction/ranking audit are the differentiated scope.

## 12. Three-dataset TabPFN pilot

`schema_orbit_pilot.py` ran against the cached public TabPFN v2.5 classifier
checkpoint. It applies schema actions jointly to context and query data, aligns
class probabilities back, and evaluates balanced feature/class or
feature/category/class factorials.

| Dataset | Passes | Total schema risk | Dominant component |
| --- | ---: | ---: | --- |
| Breast Cancer | 1 | 0.0010086 | feature position, 90.4% |
| Breast Cancer | 8 | 0.0001279 | feature position, 83.0% |
| Wine | 1 | 0.0022868 | feature position, 76.3% |
| Wine | 8 | 0.0001806 | feature position, 64.6% |
| Adult 1k/1k | 1 | 0.0024538 | nominal category code, 53.5% |
| Adult 1k/1k | 8 | 0.0008847 | nominal category code, 83.1% |

The empirical Brier decomposition error was at most about `3e-9`; the ANOVA
closure error was at most about `2e-10`. The eight-pass default ensemble reduced
schema risk by 87%, 92%, and 64% on the three datasets but did not eliminate it.

An equal-pass internal-shift ablation supports causal interpretation. Removing
feature shifts increased residual risk roughly fourfold on Breast Cancer and
tenfold on Wine; removing both feature and class shifts increased Wine risk
roughly twelvefold. The default joint policy remained best, so this is not yet
evidence for a new ensemble policy.

Adult is the important novelty signal. After correct categorical metadata, a
bijection of the seven nominal code sets dominated residual risk even under
the default ensemble. This is not target-label order or column order. It must
still survive more datasets, pipelines, and checks that textual category values
are not being stripped of genuine semantics.

## 13. Revised decision

OrbitANOVA is now the lead because it meets four conditions that FieldRiesz
currently does not:

1. a precise object with an exact, testable identity;
2. a successful pilot whose dominant factor changes across datasets;
3. a differentiated category-code signal beyond the closest permutation work;
4. feasibility with the existing two H100 NVL GPUs, cached TabPFN checkpoints,
   and 30-dataset harness.

The decisive next gate is cross-model rather than more TabPFN-only examples.
Invariant one-hot linear controls must close numerically; CatBoost/boosted trees,
MLP/ResNet, TabM, and TabPFN must show different factor profiles; and at least
one natural schema factor must affect worst-schema risk, compute allocation, or
a model ranking. If not, the pilot is an implementation note rather than an
ICLR paper.

## 14. Adult cross-model factor gate

The first cross-model gate uses the same official Adult source split and a
fixed 1,000/1,000 subsample. Every model receives a balanced `4 x 4 x 2` grid
of feature permutations, within-field nominal code bijections, and binary
class-ID swaps. A separate 16-run reference-schema seed orbit measures ordinary
randomness.

| Pipeline | Schema risk | Seed risk | Main profile |
| --- | ---: | ---: | --- |
| one-hot logistic | 2.7e-22 | 4.9e-32 | exact negative control |
| LightGBM native categorical | 2.1e-31 | 5.5e-32 | exact negative control |
| CatBoost native categorical | 0.000721 | 0.001127 | feature + feature×class |
| ordinal-code random forest | 0.002034 | 0.000673 | category + feature×category |
| sklearn native categorical HistGB | 0.001372 | 5.4e-32 | class only |
| XGBoost native categorical | 0.002346 | numerical zero | category + category×class + class |
| one-hot Adam MLP | 0.023889 | 0.025359 | every main effect and interaction |

This passes three parts of the E1 gate: invariant controls close, several
nontrivial families exceed numerical noise, and factor profiles differ sharply.
It does not yet pass the practical-consequence gate because these diagnostic
hyperparameters do not establish a model-ranking flip or a targeted
intervention win.

The sklearn HistGradientBoosting class-label result was stress-tested. It
vanishes on all-numerical Breast Cancer and on Adult when categorical fields
are treated as ordinal, but persists with native categorical splits and early
stopping disabled. Local source inspection gives a mechanism: the categorical
splitter orders high-support levels by gradient/Hessian, removes low-support
levels from the scan, and always sends removed levels to the right child. A
binary target flip reverses the gradient order but not the low-support-side
convention, so the candidate split family is asymmetric.

The native tree comparison prevents overgeneralization. LightGBM is exact on
the grid; CatBoost is nominal-code invariant but position/class sensitive;
XGBoost shows large category and category×class components. The object is the
full library pipeline, not a theorem that all categorical boosters share one
failure.

The next statistical step should use standard Sobol/pick-freeze estimators for
vector outputs. A Saltelli-style design can estimate total and first-order
factor effects in `O(KN)` pipeline evaluations instead of enumerating the full
product. Exact factorial ANOVA remains the validation reference on small
groups. This is established global-sensitivity machinery, not proposed
mathematical novelty.

## 15. Scalable estimator, proper-loss extension, and Churn gate

`orbit_anova.py` now includes a vector-output pick-freeze estimator. In a
100,000-pair synthetic test, its total-effect estimates were within `1.9e-5`
absolute of exact factorial values for all three factors; the largest
first-order error was `3.2e-5`. This makes the planned five-factor audit
`O(KN)` rather than exponential in factor count. The estimator is standard
Sobol machinery and should be cited as such.

The Brier identity is one instance of a broader established result. For a
proper loss represented by `D_Phi(e_y,p)`, use the left Bregman centroid
`grad Phi(q)=E grad Phi(p_g)`. Then

`E_g ell(y,p_g) - ell(y,q) = E_g D_Phi(q,p_g)`,

which is label-free. For log loss, `q` is the normalized geometric mean and
the gap is `E KL(q||p_g)`. The numerical validator closed this identity to
`6.6e-17`. Gruber and Buettner (2023), Wood et al. (2023), and the older
Bregman-centroid literature occupy the theorem; the paper contribution is its
schema-orbit use. Euclidean functional ANOVA is retained for Brier/squared
loss rather than overclaiming an orthogonal log-loss decomposition.

A second `4 x 4 x 2` cross-model gate on Churn replicated the model-specific
structure while changing the profiles:

| Pipeline | Churn schema risk | Dominant profile |
| --- | ---: | --- |
| one-hot logistic | 1.6e-31 | invariant control |
| LightGBM native categorical | 2.0e-31 | invariant control |
| CatBoost native categorical | 0.000598 | feature + feature×class |
| ordinal-code random forest | 0.001083 | category + feature + interaction |
| sklearn native categorical HistGB | 0.001015 | class + feature×class + feature |
| XGBoost native categorical | 5.6e-14 | numerical noise |

This is a useful cross-dataset result: CatBoost's feature/class profile
persists, whereas XGBoost's large Adult category/class profile disappears on
Churn. Harmless schema representatives changed the LightGBM-versus-HistGB
winner in 4 of 32 cells, but paired 95% intervals for the extremal loss gaps
crossed zero widely. The pilot therefore does **not** yet establish a
statistically meaningful ranking reversal.

Predictive multiplicity is an adjacent established field. The sharper framing
is structured schema-induced multiplicity: competing predictors arise from one
pipeline and explicitly equivalent representatives, not an unconstrained
Rashomon set, and the product action identifies which arbitrary schema choice
caused the disagreement.

## 16. Instance-level consequence and an isolated library mechanism

The Adult grid has operationally visible disagreement despite modest average
risks. Harmless schema choices alter hard predictions for `4.4%` of rows with
CatBoost, `7.0%` with the ordinal forest, `2.5%` with native HistGB, and `6.9%`
with XGBoost. Their 95th-percentile maximum probability spans are `0.125`,
`0.193`, `0.124`, and `0.243`; maxima reach `0.274`, `0.380`, `0.422`, and
`0.391` respectively. This gives the paper a stronger consequence than a
leaderboard-only statistic: one person's decision can depend on arbitrary
schema spelling.

The HistGB effect survives the full official Adult split. Swapping only binary
class IDs changes `1.79%` of 16,281 test decisions and reaches a `0.434`
probability gap. Its exact removable risk is `0.000654` Brier and `0.001048`
log loss. This is not a small-subsample artifact.

`histgb_label_flip_reproducer.py` isolates the categorical-split mechanism.
With four 80-sample levels and one nine-sample level, native categorical HistGB
has a `0.100811` aligned probability gap after a target-ID swap. Treating the
same codes as ordinal or one-hot closes to `1.1e-16`; raising the rare level to
20 samples closes native handling to `2.2e-16`. The installed sklearn 1.9.0
source explicitly filters levels below fixed effective support 10, keeps them
out of the sorted scan, and maps them to the right child. Label reversal
reverses the gradient ordering but cannot produce the complementary candidate
partition because the excluded level remains on the fixed side. In the minimal
case, the reference and flipped root bitsets are `{2,3}` and `{0,1}` while rare
category `4` remains right in both. At support 20 they become `{2,3,4}` and
`{0,1}`, true complements, and the prediction gap closes. One hundred random
category-rate trials failed label closure in 100/100 cases at rare support 5
and 99/100 at support 9, versus 0/100 at support 20 or 40. This supports the
mechanism beyond the symmetric hand-built example.

This is exactly the kind of implementation artifact the product audit is meant
to uncover, but the paper must avoid becoming a bug report. The broad value is
that the same audit distinguishes exact LightGBM behavior, CatBoost's
feature/class profile, XGBoost's dataset-dependent category profile, and
TabPFN's residual feature/category profiles without model-specific probes.

## 17. Joint schema-by-seed decomposition

The earlier comparison of one fixed-seed schema orbit with a separate
reference-schema seed orbit was statistically incomplete. A schema action can
change which stochastic model a seed selects, producing a schema×seed
interaction even if its average direction vanishes. The harness now crosses
the `4 x 4 x 2` schema grid with eight seed levels and decomposes all four
factors exactly.

| Model | Marginal schema | Seed main | Schema×seed | Conditional schema | Conditional seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinal forest | 0.001464 | 0.000085 | 0.000532 | 0.001996 | 0.000617 |
| CatBoost | 0.000079 | 0.000462 | 0.000599 | 0.000678 | 0.001062 |

For the ordinal forest, the arbitrary category geometry persists after seed
averaging and marginal schema variance is over 17 times the seed main effect.
For CatBoost, most apparent fixed-seed schema sensitivity is schema×seed
interaction; seed averaging removes the directional schema effect but a
realized training run remains schema-sensitive. Reporting only two separate
variance numbers would miss this distinction.

Formally, with product factors `G` and `S`, `Var_g(E_s p)` is the sum of ANOVA
components containing schema factors but not seed. `E_s Var_g(p|s)` adds every
schema×seed component. Conversely, `E_g Var_s(p|g)` is seed main plus those
same interactions. This is standard product ANOVA, but its use prevents a
major confound in the proposed audit and directly connects to Day 3's finding
that seed ensembles can dominate representation ensembles.

## 18. Older metamorphic-testing and multiverse boundary

A targeted search found closer historical prior art than the recent TFM
papers. Xie et al. (2011) explicitly formulate affine feature transformations,
class-label permutations, and attribute permutations as metamorphic relations
for supervised classifiers. OrbitANOVA therefore cannot claim that these are
new invariance tests or that label-free comparison of transformed predictions
is new in spirit.

The remaining distinction is quantitative and pipeline-level: metamorphic
testing asks whether an expected relation passes, often at the hard-label
level; OrbitANOVA defines the exact proper-risk price of failure, aligns
probabilistic outputs, decomposes a product of nuisance relations and their
interactions, crosses them with seed, and audits current tabular pipelines.
Nominal code bijections and within-field charts also go beyond the three old
relations, although novelty must rest on the framework and evidence rather
than on any one transformation.

PRESTO (Wayland et al., ICML 2024) is the closest multiverse-style framing. It
maps latent representations over methods, hyperparameters, and datasets using
topological summaries. Those branches are alternative analytical choices, not
necessarily semantics-equivalent representatives with an invariance target.
OrbitANOVA should explicitly present itself as a *quotient multiverse*: every
branch is in one declared semantic equivalence class, so dispersion is
identifiable arbitrariness rather than generic pipeline sensitivity.

## 19. OrbitCover: an audit-guided method, not only a benchmark

The fANOVA object yields a direct intervention theorem. Exact averaging over a
factor set `J` deletes precisely every component whose index set intersects
`J`. If `V_S` are component risks, the removable amount is

`F(J)=sum_{S: S intersects J} V_S`.

This is a weighted coverage function and therefore monotone submodular. With
generic full-factor wrappers, member counts multiply and their logarithms give
additive knapsack costs. With native model mechanisms, costs can be measured
directly. Standard submodular optimization—not a novel optimizer—is enough to
select the factors with maximal measured removable risk under a pass budget.
The new algorithmic proposal is to feed schema-risk components into that
selection, tentatively named **OrbitCover**.

`symmetrization_frontier` now verifies the deletion identity numerically. On a
three-factor vector predictor, residual variance computed after every subset
average agrees with the sum of untouched fANOVA components to `3.5e-18`.

The Adult components make the policy concrete. At two passes, class-ID
averaging removes `52.1%` of CatBoost risk, nearly `100%` of HistGB risk, and
`55.3%` of XGBoost risk, but essentially none of ordinal-forest risk. At four
passes, the best factor is feature position for CatBoost (`85.3%` removed),
category code for the forest (`93.0%`), class ID for HistGB (`~100%`), and
category code for XGBoost (`83.8%`). A universal policy is demonstrably
wasteful before any further tuning.

The guarantee is deliberately narrow. Orbit averaging improves proper loss
relative to the average transformed member under the declared nuisance
distribution; it need not beat a canonical member that happens to be lucky.
The method targets identifiability, average/worst-schema risk, and efficient
invariance closure. It must be compared against seed ensembles and
validation-selected TTA at equal compute, especially because Day 3 found seed
ensembles stronger on average.

The action set should include generic iid schema sampling. For `m` independent
orbit members, expected residual squared schema variance is `SR/m`, giving
`1-1/m` expected coverage. This fixes the forest's two-pass case: class-ID
averaging covers zero, while two generic schema samples cover `50%` in
expectation. At four passes, the targeted factor still beats generic `75%`
expected coverage for all four Adult pipelines (`85%`, `93%`, `100%`, `84%`).

An exploratory equal-compute labeled check remains mixed, as expected from the
guarantee. On CatBoost, OrbitCover Brier was `0.18917`, `0.18866`, and
`0.18860` at budgets 2/4/8 versus seed ensembles `0.18950`, `0.18940`, and
`0.18915`; generic schema samples were slightly better at budgets 4/8. On the
forest, the canonical reference was unusually favorable, so category averaging
reduced representation dependence but did not beat that member. The paper
must foreground expected/worst-schema closure and treat raw reference gains as
secondary, not quietly switch objectives.

## 20. Nine-class Otto gate: factor profiles change again

A `4 x 4` feature-position/class-ID gate on a fixed Otto 1,000/1,000 split
uses four random permutations of 93 numerical columns and four random
permutations of nine target IDs. The class permutations close for every tree
pipeline; feature position does not.

| Pipeline | Total schema risk | Hard-label flip fraction | P95 probability span |
| --- | ---: | ---: | ---: |
| one-hot logistic | 1.44e-13 | 0% | 1.3e-6 |
| LightGBM | 0.002529 | 8.0% | 0.207 |
| CatBoost | 0.002686 | 11.6% | 0.195 |
| random forest | 0.001367 | 9.2% | 0.101 |
| sklearn HistGB | 0.002272 | 7.7% | 0.198 |
| XGBoost | 0.000878 | 4.4% | 0.113 |

Every non-control component is feature position up to numerical precision.
This reverses the Adult picture: LightGBM was exact there; XGBoost was driven
by category/class interactions; HistGB was driven by binary class IDs. On
Churn, XGBoost closed. The profile is a property of model, data geometry,
metadata, randomness, and implementation—not a permanent badge attached to an
algorithm name.

Same-seed coupling explains much but not all of the stochastic models' Otto
effect. In an eight-seed joint grid, CatBoost has marginal feature variance
`0.000334`, seed main `0.001002`, and feature×seed `0.002328`; the forest has
`0.000170`, `0.000433`, and `0.001163`. For a realized seed, 22.7% and 17.4%
of hard predictions vary over the joint orbit, respectively. Deterministic
LightGBM, HistGB, and XGBoost have no such escape: ordered feature search,
histogram ties, and implementation conventions remain plausible mechanisms.

The full official Otto split falsifies the strongest version of that last
sentence. At 39,601/12,376 rows, LightGBM retains feature-position risk
`0.001151` with `5.11%` hard flips, and XGBoost retains `0.000339` with `3.03%`
hard flips. sklearn HistGB closes to `2.4e-32` and zero flips. Thus some
deterministic position effects persist at scale, while HistGB's small-sample
effect is likely a tie- or regime-specific phenomenon. Sample size must be a
benchmark stratum and mechanism claims must remain library/dataset specific.

## 21. Representation independence must be restricted

Halpern and Koller (JAIR 2004) formalize representation dependence for
probabilistic inference and show that demanding independence under overly broad
representation changes is incompatible with useful non-deductive defaults.
This is a helpful conceptual guardrail. OrbitANOVA should not imply that every
invertible encoding is a nuisance or that a learner should ignore all such
changes. A logarithm can reveal a linear law; semantic text can carry external
meaning; ordinal codes encode topology.

The paper's defensible object is therefore a declared restricted quotient:
whole-field permutations with metadata, opaque nominal-ID bijections,
class-coordinate relabeling with output alignment, explicit physical-unit
changes, and within-field basis changes whose function space is fixed. The
distribution over each action is part of the estimand. This declaration is not
administrative detail—it is what avoids the impossibility/triviality of broad
representation independence.

## 22. Monte Carlo feasibility is conditional, not solved

Four hundred repeated synthetic audits at each sample size compared the
pick-freeze estimator with exact three-factor ANOVA. For a five-factor design,
`N=16` costs `(K+2)N=112` pipeline evaluations yet gives `16.2%` relative RMSE
for total variance and about `29%`–`33%` for individual total effects. Even
`N=128` costs 896 evaluations and leaves roughly `11%`–`12%` total-effect
RMSE. A naive 30-dataset × 10-model × fixed-large-N audit is not credible by
the September deadline.

The decision problem is easier than exact estimation. The synthetic's feature
and category total effects differ by only `3.2%`, so top-factor identification
is intrinsically unstable: it rises from `56.8%` at `N=16` to only `66%` at
`N=512`. Nevertheless, choosing the estimated winner has only `1.4%` mean and
`3.2%` P95 relative coverage regret at `N=16`; the estimator reliably avoids
the substantially worse third factor.

The feasible benchmark is sequential. First estimate only total risk with
16–32 iid orbit members, which costs far less than factor pick-freeze. Stop
near-zero controls. Attribute only material model/dataset pairs, using exact
small factorials where possible and increasing pair count until policy regret
or component uncertainty reaches a frozen threshold. Report close factors as
ties. This turns estimator uncertainty into part of the method rather than
hiding a prohibitively noisy broad matrix.

## 23. Day-3 chart orbit closes the local-to-paper loop

Day 3 already supplied a particularly clean schema nuisance: five ordinal
bases that were verified to span the same within-field function spaces. Its
registered runs stored loss and accuracy but not predictions, so they could
not answer whether the learned functions were actually stable. The companion
[`chart_orbit_pilot.py`](chart_orbit_pilot.py) reuses the exact Day-3 feature
constructors and training loop and retains aligned probabilities on Adult.

For the expanded `5 chart x 32 seed` standard MLP grid, total Brier prediction
variance is `0.00186664`. Orthogonal plug-in decomposition gives chart main
`0.00024856`, seed main `0.00061389`, and chart×seed `0.00100418`. Hence
expected conditional chart variance is `0.00125274`, and exact five-chart
averaging removes `67.11%` of joint chart/seed variance. Across the grid,
`11.81%` of test rows change hard prediction, P95 maximum class-probability
range is `0.30691`, and the label-free log gap is `0.00305834`.

[`seed_coupling_analysis.py`](seed_coupling_analysis.py) corrects the
finite-seed bias in the chart-main plug-in with a paired Hilbert U-statistic.
Persistent mean-predictor chart risk is `0.00021617`, jackknife SE `1.13e-5`,
and normal 95% interval `[0.00019412, 0.00023822]`. Disjoint 16-versus-16 seed
energy tests reject equal randomized prediction distributions in both
directions for nine of ten chart pairs. The exception is local versus path
spectral, the pair related by an orthogonal within-field basis change. A
split-null-corrected empirical transport diagnostic is positive for all 500
balanced seed splits (median excess `0.0002334`), but remains diagnostic rather
than a calibrated Wasserstein estimate.

This is not caused by a different hypothesis space. A negative-control
pipeline performs full-rank sample whitening of each complete design matrix,
after which equivalent charts differ only orthogonally, then fits an L2
logistic model. All five fits converge in 83 iterations with retained rank 98;
chart risk is `1.90e-18`, maximum probability range `5.93e-8`, and there are no
hard-label changes. This is the cleanest direct bridge from Day 3's coordinate
geometry to the OrbitANOVA audit.

The pilot also corrects terminology. The five chart levels are a declared
finite representative set, not a closed group. Functional ANOVA needs a
product probability space, not group closure. The formal paper object should
therefore be a product of nuisance factors, with group structure invoked only
where it actually exists.

## 24. Seed interactions require a coupling declaration

Crossing schema with a seed integer gives a useful operational experiment, but
the schema×seed component is not intrinsic. The same pseudorandom array is
attached to different coordinate functions after a feature or basis change.
Relabeling seed outcomes independently across schema representatives leaves
every marginal randomized pipeline unchanged but changes the paired
interaction variance.

Let `P_z` be the distribution of prediction vectors under training randomness
at schema representative `z`, and let `pi` be any joint coupling of these
marginals. Define its coupled dispersion

`C(pi) = E_pi |Z|^-1 sum_z ||P_z - Pbar||^2`.

The Hilbert pairwise identity and Jensen yield

`Var_z(E P_z) <= |Z|^-2 sum_{z<z'} W2^2(P_z,P_z') <= inf_pi C(pi) <= C(pi_seed)`.

The left term measures persistent differences between mean predictors. The
middle term is a coupling-free lower bound on differences between the complete
randomized prediction distributions. The right term is the feasible
same-integer-seed coupling already measured by factorial ANOVA. For two schema
levels the transport term equals the optimal coupled dispersion; with more
levels, separately optimal pairwise transports may be incompatible.

On the 32-seed chart pilot, unbiased mean-predictor chart risk is `0.0002162`
while same-seed conditional chart risk is `0.0012527`. The naive full empirical
pairwise-Wasserstein bound is `0.0009603`, but remains positively biased. A
within-chart split correction reduces the median excess to `0.0002334`, close
to the persistent-mean term. This suggests that most of the huge paired
interaction is coupling-specific, while a smaller but nonzero difference in
randomized predictor distributions persists. A formal transport claim still
needs a principled estimator and more than one dataset.

## 25. A chart-covariant closure intervention

The 32-seed pattern suggested a mechanism: local and path-spectral bases are
already related orthogonally and are the only pair whose marginalized
prediction distributions do not separate. The nonorthogonal basis charts can
be reduced to the same case by sample-whitening each declared ordinal field.
If `Z_chart = Z_ref O` with block-orthogonal `O`, transporting the first-layer
weight as `W_chart = W_ref O` aligns the initial function. SGD with momentum,
isotropic weight decay, identical minibatches, and identical dropout masks
then preserves the relation at every update.

[`chart_covariant_training_pilot.py`](chart_covariant_training_pilot.py)
implements exactly this intervention. The measured coordinate maps have
orthogonality error at most `1.34e-13`; float32 coordinate residual is at most
`1.40e-6`. After 100 epochs on Adult, five chart trajectories have maximum
training-curve range `7.54e-9`, Brier chart risk `1.93e-15`, zero hard-label
changes, and maximum probability range `6.39e-6`. Test Brier is `0.19545184`,
slightly better than the 32-seed raw-chart AdamW orbit mean `0.19547932`.

The optimizer ablation is causal. With the same whitening and transported
initialization, ordinary AdamW produces chart risk `0.00010415` after 40 fixed
epochs because its coordinatewise second moment is not rotation equivariant.
A field-block VectorAdam update—ordinary coordinate moments on unchanged
schema coordinates and one scalar second moment per transforming field
block—reduces chart risk to `2.88e-13`. This matches established VectorAdam
theory rather than creating a generic optimizer theorem. What may be new is the
schema-semantic declaration of which parameter blocks transform, paired with
an audit that verifies closure on predictive functions.

This intervention materially raises the secondary idea's value. It is now a
clean, competitive Adult result with exact trajectory closure, not merely the
earlier mass/stiffness toy. It is still not ready to displace OrbitANOVA:
rotation-equivariant optimization and whitening are crowded, one ordinal
dataset is insufficient, and the benefit over simply choosing the canonical
whitened chart has not been benchmarked. It can serve either as OrbitANOVA's
E3 design closure or as a second paper idea conditional on transfer.

## 26. Diamond transfer: exact closure, qualified performance, unstable claim

[`chart_regression_transfer_pilot.py`](chart_regression_transfer_pilot.py)
extends the probability-orbit experiment to standardized squared regression
loss on Diamond. Sixteen seeds for each of five exactly equivalent ordinal
charts give raw-AdamW MLP total prediction variance `0.00115048`: chart main
`0.00022144`, seed main `0.00030173`, and chart×seed `0.00062731`. Mean
prediction range is `0.1366` original target units, P95 `0.3295`, maximum
`2.377`. A full-rank-whitened ridge control closes at `1.5e-30`.

Within-field whitening plus transported SGD again closes the chart orbit:
coordinate-map error is at most `1.07e-12`, curve range `3.26e-9`, and chart
risk `5.22e-15`. Field-block VectorAdam with validation early stopping selects
epoch 36 in all five charts and closes at `5.71e-13`. Transfer therefore
supports the covariance theorem and implementation. It does not yet support a
performance claim. Field-VectorAdam RMSE is `0.15315`, versus `0.14665` for the
80-member raw MLP orbit mean and roughly `0.145`–`0.153` across ordinary
single-chart/seed choices. SGD at 100 epochs is worse (`0.16853`); even a
reference-chart 400-epoch diagnostic reaches only `0.15195`. Do not market the
closure method as accuracy-improving without better optimizer design.

[`diamond_architecture_chart_pilot.py`](diamond_architecture_chart_pilot.py)
adds 16-seed ResNet predictions. ResNet is even more chart-coupled: total
variance `0.00236612`, chart main `0.00029547`, seed main `0.00052403`, and
chart×seed `0.00154662`. The architecture conclusion depends on the equivalent
chart. Under cumulative coding, ResNet beats MLP by paired MSE `0.00140377`
with 95% interval `[0.00087048, 0.00193706]` and unadjusted `p=4.97e-5`; under
the other four charts every interval covers zero widely. This is not evidence
that MLP wins elsewhere. It is evidence that the binary scientific statement
“ResNet significantly outperforms MLP” is not identified by the represented
dataset alone. Under the declared uniform chart-and-seed orbit, the averaged
comparison favors ResNet (`0.0201008` versus MLP `0.0216088`).

## 27. Instability is concentrated in rows and subgroups

The Adult `5 chart x 32 seed` prediction archive makes rowwise conditional
chart variance observable. It is extremely concentrated: the top 1%, 5%, 10%,
20%, and 50% of rows account for 25.63%, 52.64%, 64.11%, 78.12%, and 98.12%
of the total. This is the first result in the search that directly suggests
conditional inference compute rather than a uniform orbit size.

[`chart_subgroup_audit.py`](chart_subgroup_audit.py) stratifies this label-free
quantity using archived Adult fields. The only binary-coded feature has groups
of 10,860 and 5,421 rows. Group 0 versus group 1 has persistent chart risk
`0.0002937` versus `0.0001582`, conditional risk `0.001466` versus `0.0008264`,
and hard-flip probability `14.77%` versus `5.88%`. Bootstrap intervals for all
three differences exclude zero. The preprocessing archive does not prove the
level-to-name mapping, so the result cannot be labeled as sex or interpreted
as a fairness-harm estimate without recovering provenance.

Rarity is a plausible mechanism. Across 14 education codes with at least 100
rows, Spearman correlation between log training frequency and persistent
chart risk is `-0.7758` (`p=0.001108`). Decision-boundary ambiguity is even
more predictive: centroid Bernoulli uncertainty correlates `0.923` with
conditional row risk and `0.850` with persistent row risk. These are
exploratory associations selected after inspecting the orbit, so the paper
needs frozen replications rather than a table of nominal p-values.

## 28. OrbitCascade passes a held-out equal-compute pilot, but is not generic novelty

[`orbit_cascade_pilot.py`](orbit_cascade_pilot.py) uses seeds 0--15 and one
random half of Adult rows to choose a two-chart probe. It freezes
standardized-cumulative plus path-spectral. On seeds 16--31, each seed obtains
a disagreement threshold from the unlabeled calibration half; evaluation is
on the disjoint 8,141 rows. Rows above the threshold receive the remaining
three chart passes and the exact five-chart mean; other rows retain the probe
mean.

At realized average costs `2.619`, `3.036`, `3.538`, and `4.022`, residual
squared distance to the full centroid is respectively `0.0001227`,
`0.00007482`, `0.00003526`, and `0.00001289`. A row-independent escalation
from the identical pair, matched to each seed's realized cost, has expected
residual `0.0002681`, `0.0002212`, `0.0001647`, and `0.0001101`. Adaptive
allocation therefore removes 54.2%, 66.2%, 78.6%, and 88.3% of that residual.
Two-way bootstraps over seeds and rows put every advantage above zero. At cost
2.62 the interval is `[0.0001055, 0.0001891]`.

This result survives the cleanest possible comparator and uses no labels for
policy selection, thresholding, or the approximation endpoint. However,
adaptive inference is crowded. Kim et al. (NeurIPS 2020) learned
instance-specific augmentation loss; Inoue (AISTATS 2019) stopped ensemble
evaluation by confidence; AdapTTA dynamically changed pass counts; SA-TTS
(CVPRW 2026) combines entropy, margin, and augmentation disagreement. The
novelty boundary is therefore not adaptive TTA. OrbitCascade is valuable only
as an audit-to-action bridge: the target is the declared schema quotient, the
error is directly label-free, and the candidate passes come from OrbitANOVA.
It should remain an intervention inside the main paper unless broad tabular
replication reveals a much stronger phenomenon.

## 29. Adult supplies a schema-identifiable architecture comparison

The Diamond architecture result could be criticized as a generic noisy
comparison in which one chart happened to cross a significance threshold. A
paired Adult replication gives a needed negative control.
[`adult_architecture_chart_pilot.py`](adult_architecture_chart_pilot.py) trains
ResNet on the five equivalent charts for the same 16 seeds already stored for
MLP. MLP-minus-ResNet Brier gaps are `-0.001322`, `-0.001341`, `-0.001287`,
`-0.001719`, and `-0.001726`; all five paired intervals exclude zero in favor
of MLP. Averaging chart differences within seed gives a quotient gap
`-0.001479`, interval `[-0.001878, -0.001080]`.

The conclusion is robust even though ResNet predictions are not. ResNet total
chart×seed prediction variance is `0.0035746`, with chart main `0.0003112`,
seed main `0.0011570`, and interaction `0.0021064`; 13.73% of rows change hard
label. This falsifies the simplistic claim that high schema risk must imply an
unstable architecture ordering.

[`claim_identifiability_analysis.py`](claim_identifiability_analysis.py) makes
the comparison explicit. Adult is schema-identifiable over the five charts.
Diamond has quotient gap `0.0002924` with interval crossing zero, while chart
means span `-0.0000401` to `0.0014038`; only cumulative is detectably positive.
The claim layer should therefore report quotient effect, representative range,
and a frozen ROPE. A broad paper should count robust, direction-changing, and
detection-changing comparisons rather than cherry-pick reversals.

## 30. PREF is the most dangerous direct neighbor found so far

An anonymous TMLR submission posted May 2026, *Preprocessing Robustness in
Heterogeneous Tabular Learning*, defines a modular preprocessing evaluation
framework (PREF), a knob-wise Preprocessing Sensitivity index, and aggregate
Model Volatility across trees, tabular MLPs, and TFMs. Its transformations
include numerical scaling, categorical/text encoding, dimensionality
reduction, feature selection, and augmentation. It uses single-knob absolute
performance deltas in mismatched and curated best-practice operator spaces.

This kills any pitch of “the first preprocessing robustness benchmark for
tabular models.” It does not kill the current object. PREF deliberately varies
choices that can change information, inductive appropriateness, and achievable
performance. OrbitANOVA admits only branches declared to spell the same
learning problem, aligns outputs, and asks for a zero-risk invariance target.
Its primary metric is prediction-space proper-loss dispersion, exactly
label-free for Brier/squared and appropriate Bregman centroids; its product
design attributes interactions; its random learner audit separates persistent,
distributional, and coupled effects; and OrbitCover/OrbitCascade approximate
the quotient. These differences must be visible in the title, abstract,
benchmark transformations, and main figures. Adding generic scaling/encoding
ablations to inflate breadth would erase the novelty boundary.

## 31. Schema radius makes the nuisance-weight objection testable

The audit distribution `mu` is necessarily part of schema risk. Uniform over
five hand-chosen charts is defensible but still contestable. For a finite set
of aligned prediction vectors, optimizing variance over every probability
weight has a clean answer:

`sup_w [sum_i w_i ||p_i||^2 - ||sum_i w_i p_i||^2]`

is the dual of the minimum-enclosing-ball problem and equals its squared
radius. This is known convex geometry, not a theorem contribution. It supplies
a distribution-free worst-reweighting diagnostic while retaining uniform
product measures for the attributed estimand.

[`schema_radius_analysis.py`](schema_radius_analysis.py) solves the five-point
quadratic dual from prediction Gram matrices. Persistent radius/uniform ratios
are 1.174 for Adult MLP, 1.072 for Adult ResNet, 1.171 for Diamond MLP, and
1.211 for Diamond ResNet. Mean same-seed conditional ratios are respectively
1.147, 1.112, 1.107, and 1.043. Split-half persistent ratios lie between 1.043
and 1.219. Numerical primal/dual closure is below `4.2e-9` across conditional
runs.

The optimal weights are meaningfully nonuniform—some persistent solutions
drop local or path-spectral entirely—but the worst possible weighting raises
risk by only 4%--21%. Thus the current uniform chart result is not an artifact
of averaging away one isolated spelling. The radius loses the product
semantics and can concentrate on a few extremes, so it is a secondary endpoint
and adversarial check rather than a replacement for `mu` or OrbitANOVA.

## 32. A June 2026 PNAS paper removes the broad discovery claim

Liu, Yang, and Adomavicius, *Robustness is important: Limitations of LLMs for
predictions on tabular data* (PNAS Nexus, June 2026), is closer than the earlier
metamorphic-testing references. It studies explicitly task-irrelevant changes:
variable names/order, row order, decimal precision, and serialization format.
It evaluates closed/open general LLMs, TabPFN, and LimiX, repeats local TFM
experiments over 100 synthetic datasets, compares change-induced prediction
differences with repeat-generation randomness, and investigates attention as
a mechanism. Ten row permutations produce sensitivity 2.5--10 times larger
than a two-order comparison.

This fully removes the claim that the project discovers tabular prediction
sensitivity, or even that it first establishes residual TabPFN sensitivity
after internal ensembling. The paper is still differentiated, but only if the
following are central rather than appendix flourishes:

1. The phenomenon extends materially beyond language-like inference. Adult,
   Churn, Otto, and the chart experiments show large category-ID, class-ID,
   feature-position, and coordinate-chart effects in forests, boosters, MLPs,
   and ResNets. This directly qualifies Liu et al.'s view that supervised
   tabular techniques are invariant by design or have only minor RNG effects.
2. The primary object is aligned prediction dispersion with an exact
   label-free proper-score value, not relative MAE between a base and one
   changed pipeline.
3. The nuisance product identifies main effects and interactions, and the
   stochastic extension distinguishes persistent mean effects from
   implementation-specific seed coupling.
4. OrbitCover, OrbitCascade, and chart-covariant training use the audit to
   allocate compute or close the diagnosed component.

The working title should move from the generic *Same Table, Different
Predictor* toward *Learning on the Schema Quotient: Attributing
Arbitrary-Representation Risk Across Tabular Pipelines*. If the broad study
does not establish conventional-pipeline effects and factor-specific remedies,
this new prior art makes the ICLR project too incremental.

## 33. Black Friday transfers chart risk and adaptive quotient approximation

A two-GPU run retained predictions for five Day-3 equivalent ordinal charts
and 16 seeds on Black Friday's official 100,000/25,000 split. The dataset has
two declared ordinal fields and is substantially larger than Adult or Diamond.
[`chart_regression_orbit_pilot.py`](chart_regression_orbit_pilot.py) wrote
disjoint seed shards, and
[`merge_chart_regression_orbits.py`](merge_chart_regression_orbits.py) checked
chart/target consistency before concatenation.

Total standardized prediction variance is `0.00631994`: chart main
`0.00073758`, seed main `0.00175298`, and chart×seed `0.00382938`. Hence
expected conditional chart variance is `0.00456696`, 72.26% of total. The mean
member MSE is `0.497350`; the full chart-and-seed centroid MSE is `0.491030`, an
exact reduction of `0.00631994`. Prediction range has mean `0.370`, P95
`0.654`, and maximum `2.168` target standard deviations. The effect persists
at 100k rows and cannot be dismissed as small-sample representation noise.

The row geometry differs from Adult. Black Friday's top 1%, 5%, 10%, 20%, and
50% of rows account for 7.56%, 21.22%, 32.23%, 48.00%, and 76.83% of
conditional chart risk—still concentrated, but far less extremely. Development
selects cumulative plus whitened probes. On held-out seeds 8--15 and 12,500
held-out rows, label-free escalation beats cost-matched row-independent
escalation by 18.6%, 22.5%, 25.3%, and 26.4% at approximately 2.6, 3.0, 3.5,
and 4.0 passes. Every two-way bootstrap interval excludes zero; at 2.6 passes
the advantage is `[0.0001799, 0.0002856]`.

This is a valuable transfer because it is weaker than Adult's 54%--88% but
still clear. OrbitCascade is not a one-dataset artifact, while its gain is
dataset-dependent. Black Friday schema-radius squared is `0.00085485` versus
uniform persistent risk `0.00073758` (ratio 1.159); mean conditional ratio is
1.048. Uniform weighting again does not conceal an isolated catastrophic
chart.

A covariant follow-up reuses the retained split and evaluates one common
best epoch selected across the five charts. Ridge closes chart risk to
`2.32e-30`; sample-whitened, initialization-transported SGD closes it to
`7.27e-14` but has standardized MSE `0.499504`. Field-block VectorAdam has
chart risk `1.54e-6`, a 99.97% reduction from raw conditional chart risk, and
MSE `0.497309`—roughly the raw-member mean (`0.497350`) but worse than the
80-member quotient centroid (`0.491030`). The residual likely reflects
float32 coordinate error accumulated over many adaptive updates. This is a
strong causal closure at scale, but not a performance-dominance result.

## 34. Formal identity audit and another fANOVA neighbor

[`test_orbit_anova_identities.py`](test_orbit_anova_identities.py) now checks
the Brier/Hilbert identity, exact balanced fANOVA reconstruction, the
label-free reverse-KL log-loss gap, two-point schema-radius duality, and the
`K^-2` pairwise-distance scaling used in the coupling hierarchy. The audit
confirms the current formulas and preserves two qualifications: nonquadratic
proper scores do not automatically inherit orthogonal squared-space ANOVA,
and pairwise optimal transports need not form a compatible multi-marginal
coupling.

An August 2026 remote-sensing paper applies fANOVA to architecture,
initialization, fine-tuning, and learning-strategy performance, including
interactions. This further kills any generic novelty claim for
interaction-attributed benchmark choices. It does not use semantically
equivalent representatives, aligned predictions, a label-free proper-risk
total, or a zero invariance target, which keeps the quotient-specific
combination differentiated.

The audit also tightened an overbroad word. Factor averaging is always an
ANOVA marginalization, but it is true symmetrization only for a closed group
orbit. A finite chart menu needs access to the raw semantic field and a
canonical renderer; it does not automatically make an already charted input
invariant. The concept memo now separates exact group symmetries, invertible
within-field bases, and semantics-backed unit/charts. This makes the strongest
benchmark claim rest on the first tier and treats the latter two as declared
coordinate-dependence audits.

## 35. A 25-dataset claim-identifiability reuse

The frozen Day-3 natural encoding grid contains cumulative/local results for
MLP and ResNet over 25 datasets and three paired seeds. A new read-only
analysis finds point-estimate architecture-winner changes on 7/25 datasets
(`28%`, Wilson 95% interval `[14.3%, 47.6%]`). The median normalized
representative span is `0.00380`, P90 `0.01516`, and maximum `0.03595`.

At a post-hoc normalized ROPE of `0.001`, 16 comparisons are identifiable,
four chart ranges cross both ROPE boundaries, two lie entirely within the
ROPE, and three touch one boundary. Only Compustat direction has nonzero
three-seed bootstrap intervals on opposite sides under the two charts. This is
therefore useful breadth support for the claim layer, but three seeds and
scalar test metrics are insufficient for a strong reversal headline. The
future frozen audit should retain aligned row predictions and use at least
8--16 seeds for shortlisted model comparisons.

## 36. Diamond transfers OrbitCascade across architectures

The held-out cascade analysis was run unchanged on the retained Diamond MLP
and ResNet grids. MLP selects local plus standardized-cumulative probes and
beats row-independent escalation by 39.6%--43.4% across 2.57--3.97 realized
passes. ResNet selects local plus whitened and improves by 51.1%--61.0%
across 2.59--4.00 passes. All eight two-way seed/row bootstrap intervals are
positive.

The architecture-specific probe pair strengthens the audit-to-action story.
It also provides an honest negative detail: for MLP at the largest budget,
smaller residual to the semantic quotient does not yield lower realized MSE
than random escalation. OrbitCascade guarantees/optimizes quotient
approximation, not label-dependent accuracy. Adaptive TTA and
disagreement-based compute allocation are already established; the only
differentiated role here is as an operational consequence of a declared
schema quotient.

## 37. Coupling-free risk transfers, but empirical OT is not ready

The many-seed transport diagnostic now runs on Adult MLP, Diamond MLP/ResNet,
and Black Friday MLP. Persistent mean-predictor risk is only 17.3%, 21.2%,
10.4%, and 10.6% of same-seed conditional chart risk, respectively. Thus the
large chart×seed component is reproducible and cannot be hidden in one Adult
decomposition.

Naive pairwise empirical Wasserstein lower bounds remain close to the large
same-seed costs at 16--32 seeds. A split within-representation null correction
instead yields `0.000233`, `0.000193`, `0.000219`, and `0.000488`, close to
the corresponding persistent mean risks. High-dimensional empirical OT bias
therefore dominates at feasible seed counts. Keep the coupling hierarchy as a
conceptual/statistical qualification and report mean plus same-seed endpoints;
do not promote multi-marginal transport estimation into a second ICLR idea
without a separate estimator contribution.

## 38. Churn and Otto provenance reconstruction

The original cross-model gates printed JSON but did not save it. The script
now accepts `--output`, and both grids were regenerated in the preserved TabM
environment (scikit-learn 1.4.2, LightGBM 4.7.0, CatBoost 1.2.10, XGBoost
2.1.4). Churn reproduces numerical-zero logistic/LightGBM/XGBoost controls and
distinct CatBoost feature, forest category, and HistGB class profiles. Otto
reproduces pure feature-position risk for all five tree families and numerical
closure for all sampled nine-class ID permutations.

The current Otto XGBoost risk is `0.000804` with 4.1% hard flips, versus
`0.000878` and 4.4% in the earlier log; other deviations are similarly small.
The machine-readable artifacts preserve the new environment-specific values
rather than implying bitwise cross-version reproducibility.

The full official Otto split was also regenerated for LightGBM and XGBoost
with four feature positions. LightGBM exactly reproduces risk `0.00115125` and
5.11% hard flips. XGBoost yields `0.00038280` and 3.34% flips (the earlier log
had `0.000339` and 3.03%). Both seed-control orbits close exactly. The central
scale conclusion is unchanged and now has a saved artifact.

## 39. OrbitCover becomes audit-guided Rao--Blackwellization

The action space now combines exact factor marginalization with independent
draws of the complementary factors. If marginalizing `J` costs `c_J`, a budget
`B` yields `floor(B/c_J)` conditional draws and expected residual
`SR(Q_J p)/floor(B/c_J)`. The empty subset is ordinary iid schema sampling.
This is a conditional-Monte-Carlo identity, not itself a novelty claim; the
potential contribution is using OrbitANOVA to choose what to
Rao--Blackwellize. Seven formal tests pass, including exact enumeration of the
hybrid residual, and realized/unused cost is now explicit.

The corrected Churn row-split pilot chooses actions without labels on 500
query rows and evaluates on a disjoint 500. All 18 decisions across six
pipelines and budgets 2/4/8 match the evaluation-row oracle. CatBoost chooses
iid, feature, then feature+class (50.0%, 87.4%, 100% removal); HistGB chooses
class, repeated class marginalization, then feature+class (76.1%, 88.0%, 100%);
the forest chooses iid throughout. This repairs the prior 1.07-point HistGB
regret at budget four. It is internal evidence only: the predeclared gate still
requires factor choices frozen on development dataset/model cases and tested
on unseen cases against iid schema and ordinary seed ensembling.

The novelty audit then found two direct predecessors. Liu et al. (ICML 2019)
constructively compare budgeted Rao--Blackwellization with iid minibatching,
while Liu and Owen (2023) and Liu (2024) choose preintegrated directions using
Sobol/variance importance and tractability. OrbitCover therefore loses its
method-novelty score. It survives only as an operational-validity experiment:
an OrbitANOVA schema profile should predict which arbitrary representation
axis is worth deployment compute on unseen tabular cases.

## 40. The selected Compustat ranking reversal fails confirmation

The only broad Day-3 case with opposite nonzero three-seed MLP-versus-ResNet
intervals was rerun unchanged for seeds 3--15 under both equivalent bases. On
these 13 conditionally prospective seeds, cumulative-chart AUC difference is
`+0.00004` with bootstrap interval `[-0.00179,+0.00206]`; local is
`+0.00265` with `[+0.00072,+0.00461]`; the quotient is `+0.00134` with
`[-0.00013,+0.00297]`. The original persistent opposite-ranking story is
falsified, and no broad architecture reversal is currently confirmed.

The negative result sharpens the randomness layer: 9/13 new paired seeds
change the architecture winner between charts, and the chart-by-architecture
contrast itself has interval `[-0.00491,-0.00019]` (exact sign-flip `p=.064`).
This is operational chart×seed instability, not evidence that the
seed-marginal architecture ordering reverses. Future claim audits require at
least 8--16 seeds for shortlisted cases and must report persistent and
same-seed conclusions separately. Artifacts:
[`compustat_claim_confirmation.py`](compustat_claim_confirmation.py) and
[`compustat_claim_confirmation.json`](compustat_claim_confirmation.json).

## 41. All four selected broad reversals disappear at 13 new seeds

The same conditional extension was run for Churn, HELOC, and Polish-2, covering
all four Day-3 cases whose three-seed representative means crossed both sides
of the `0.1%` ROPE. Zero of four retains opposite seed-marginal chart means.
HELOC robustly favors MLP under both charts (quotient `+0.00513`, exact paired
sign-flip `p=0.00024`), while Polish-2 strongly favors ResNet (quotient
`-0.06954`, `p=0.00024`). Churn and Compustat quotient intervals cross zero.
The mean within-seed chart-winner-change fraction remains 30.8%, separating an
operational chart×seed effect from a persistent architecture-ranking effect.

The deterministic exact Tier-1 grids offer only limited ranking evidence:
Adult has no pairwise ordering change, Churn has one CatBoost/forest change,
and small Otto has two tree-pair changes, but the Otto row-bootstrap intervals
overlap zero at both extrema. Ranking reversal should therefore leave the
headline and remain a predeclared consequence to test. The primary claim is
measurement and attribution of quotient risk, not that arbitrary schemas
usually reverse leaderboards. Artifact:
[`claim_confirmation_panel.json`](claim_confirmation_panel.json).

## 42. Effect-size calibration

Raw Brier variance is hard to compare across tasks and against the PNAS claim
that conventional pipelines are much less sensitive than foundation models.
Across the saved Adult, Churn, and small-Otto exact grids, 12/17 conventional
pipeline cells are material. Quotient averaging removes 0.21%--1.13% of mean
member Brier (median 0.53%); root schema risk is 0.024--0.052 aligned
probability-vector units, and hard-label changes span 2.3%--11.6%.

For Adult MLP/ResNet, Diamond MLP/ResNet, and Black Friday MLP, same-seed chart
tax is 0.63%--8.20% of mean proper loss (median 1.22%). Chart×seed accounts for
54%--65% of joint prediction variance. The fraction of total loss is modest on
classification/Black Friday but large on Diamond; it must always be paired
with raw/root risk because it is not excess risk over Bayes and can be unstable
near zero. The protocol already freezes raw, root, relative-loss, hard-flip,
and seed-relative endpoints. Artifact:
[`effect_size_calibration.json`](effect_size_calibration.json).

## 43. Model selection can improve loss while amplifying schema risk

The fixed-recipe/selection-rule distinction was tested rather than left as a
protocol caveat. A semantic train/validation split was frozen before rendering
a `4 feature x 4 category x 2 class` orbit. Four predeclared candidates were
compared by validation Brier inside every representative, with test labels
hidden until the two predictive orbits were complete. The comparator selects
once on the identity representation and freezes that configuration.

The completed panel crosses Adult, Churn, and Otto with ordinal forest,
native HistGB, and native CatBoost. Selection is stable in 6/9 cells and the
two orbit estimands then coincide. It is unstable for Adult CatBoost, Churn
forest, and Churn CatBoost. Their selected-pipeline schema-risk ratios are
`1.26`, `1.77`, and `1.54`; all paired-row intervals exclude one. This 3/3
direction is descriptive (minimum exact two-sided sign `p=.25`), not evidence
that selection generally increases schema risk. In Adult and Churn CatBoost,
arbitrary target-ID swaps change which depth/regularization setting wins,
locating non-equivariance in the tuning path itself.

Churn forest is the informative two-objective counterexample: it selects three
configurations, differs from the identity choice in 12/32 representatives,
and has 1.20 bits of selection entropy. Representation-wise selection lowers
orbit-mean Brier by `0.0012946` but raises label-free schema risk from
`0.0013665` to `0.0024126`, a 76.6% increase. The CatBoost loss contrasts are
small and their intervals cross zero, so they support tuning-path sensitivity
but not an accuracy--stability tradeoff.

The selected-versus-frozen risk difference was then decomposed exactly. With
`d_z=p_{z,h(z)}-p_{z,h_0}`, the identity
`Delta SR=Var(d)+2Cov(p_frozen,d)` reconstructs to machine precision. Switch
dispersion is `0.000395`, `0.001747`, and `0.000736` in Adult CatBoost, Churn
forest, and Churn CatBoost, respectively. Negative cross-covariance cancels
71%, 40%, and 69%, so selection is partly corrective but still net amplifying.
An fANOVA of one-hot configuration decisions attributes 60% of Adult-CatBoost
decision variance to class ID, 79% of Churn-forest decision variance to
feature×category, and 58% of Churn-CatBoost decision variance to
feature×class. This turns selection instability into an attributable pipeline
mechanism rather than an undifferentiated HPO caveat.

A 5,000-draw paired query-row bootstrap, conditional on the fitted orbit,
places the risk difference at `[0.000942,0.001154]`, the ratio at
`[1.685,1.850]`, and the Brier difference at
`[-0.002422,-0.000173]`. The result rules out treating HPO as an automatic
invariance repair and reveals a two-objective accuracy--quotient-stability
tradeoff. It remains conditional on three datasets, three four-point searches,
and one fitted orbit per cell, not evidence about arbitrary selection rules.
Artifacts:
[`selection_rule_orbit_pilot.py`](selection_rule_orbit_pilot.py),
[`selection_rule_panel.py`](selection_rule_panel.py), and
[`selection_rule_panel.json`](selection_rule_panel.json).

## 44. Selection-path novelty boundary

A focused literature pass found adjacent but distinct HPO-stability work.
[Overtuning](https://proceedings.mlr.press/v293/schneider25a.html) formalizes
over-optimization of noisy validation scores and measures test regret; it does
not use equivalent representations. [Stability-based tuning for penalized
regression](https://www.jmlr.org/papers/v14/sun13b.html) targets reproducible
variable sets under data resampling. [SmoothDARTS](https://proceedings.mlr.press/v119/chen20f.html)
regularizes perturbations of a differentiable architecture-search landscape.
[Clinical model-stability work](https://proceedings.mlr.press/v182/markus22a/markus22a.pdf)
varies populations, phenotype definitions, and databases—scientifically
different datasets rather than task-preserving schema spellings.

Consequently, neither generic HPO instability nor stability-aware selection is
novel. The defensible addition to OrbitANOVA is the end-to-end selection-rule
estimand on an explicit schema quotient, output-aligned switch/covariance
decomposition, and factor attribution of discrete tuning decisions. Keep it
inside the primary paper and do not spin it out as a second stable-HPO method.

## 45. Selection amplification survives seven unseen validation splits

The baseline 3×3 panel still conditioned on one semantic train/validation
split. The three cells selected as unstable there were therefore rerun on
seven conditionally prospective split seeds, saving full predictive tensors.
All three remain selection-unstable in every new split. Selected-path schema
risk exceeds identity-select-then-freeze on 7/7 Adult-CatBoost splits
(magnitude/binomial-sign `p=.015625/.015625`), 6/7 Churn-forest splits
(`.046875/.125`), and 7/7 Churn-CatBoost splits (`.015625/.015625`). Mean risk differences are `0.000206`,
`0.000762`, and `0.000177`, respectively. Mean Brier differences favor
reselection slightly but are nonsignificant (`p=.172`, `.344`, and `.219`).

The joint schema×split ANOVA changes the interpretation. Persistent schema
variance moves only modestly between the comparator and selected paths. The
schema×split interaction fraction, however, rises from 39.0% to 62.3% for
Adult CatBoost, 15.7% to 42.6% for Churn forest, and 32.1% to 53.6% for Churn
CatBoost. Decision fANOVA is likewise dominated by feature/class/category
interactions with split (largest components 34.9%, 37.3%, and 33.9%). Ordinary
validation selection therefore amplifies representation-dependent search
randomness in these confirmed cases; it does not establish a persistent
schema bias or an accuracy benefit.

This is stronger than the baseline row bootstrap but remains conditional
confirmation: the three cases were selected for baseline instability, and the
seven tests assess persistence rather than population prevalence. Artifacts:
[`selection_split_confirmation.py`](selection_split_confirmation.py),
[`selection_split_confirmation.json`](selection_split_confirmation.json), and
the saved grids in [`selection_split_repeats`](selection_split_repeats/).

## 46. Schema-pooled validation selection repairs coupling, at a possible cost

The confirmed schema×split mechanism predicts a direct intervention: average
each candidate's validation loss over the declared schema menu, choose one
configuration, and freeze it across representatives. This uses no test labels.
It is invariant to the starting representative for a uniform complete group
orbit, but the present four-level feature/category menus are sampled and not
closed; the pilot supports only menu-relative pooling.

Across the same seven unseen splits, pooled selection lowers schema risk versus
per-representative selection on 7/7 Adult-CatBoost splits
(magnitude/binomial-sign `p=.015625/.015625`), 6/7 Churn-forest splits
(`.046875/.125`), and 7/7 Churn-CatBoost splits (`.015625/.015625`).
The joint same-split risk drops from `0.000655` to `0.000479` (27%), `0.002251`
to `0.001453` (35%), and `0.000560` to `0.000358` (36%). Schema×split fractions
fall from 62.3% to 37.9%, 42.6% to 15.7%, and 53.6% to 27.2%, matching the
audit's diagnosis.

The repair is not a free accuracy gain. Mean orbit-Brier changes relative to
per-representative selection are `+0.000416`, `+0.000490`, and `+0.000232`;
their exact split-level tests are unresolved (`p=.0625`, `.34375`, `.0625`).
The defensible result is a measurable stability--accuracy frontier and an
audit-predicted action, not dominance. A frozen experiment must choose on a
development sub-orbit and test unseen representatives to rule out menu
overfitting. Results are consolidated in
[`selection_split_confirmation.json`](selection_split_confirmation.json).

## 47. Pooled selection transfers to held-out nuisances but relocates variance

To remove the in-menu leakage, configurations were chosen on a `2 feature x 2
category x 2 class` development sub-orbit and evaluated on disjoint feature
and category levels with both class IDs retained. Across the seven prospective
split seeds, development-pooled selection lowers held-out schema risk on 6/7
Adult-CatBoost splits (magnitude/binomial-sign `p=.03125/.125`), 6/7
Churn-forest splits (`.03125/.03125`), and 5/7 Churn-CatBoost splits
(`.0625/.21875`). Held-out same-split schema risk falls
from `0.000572` to `0.000411` (28%), `0.001027` to `0.000664` (35%), and
`0.000436` to `0.000294` (32%). Brier differences are unresolved (`p=.406`,
`.625`, `.438`). This passes a small unseen-nuisance action gate in two cases
and is borderline in the third.

The joint partition prevents an overclaim. Split-main variance increases from
`0.000135` to `0.000309` for Adult CatBoost and from `0.001329` to `0.002211`
for Churn forest, while schema×split falls. A single pooled configuration makes
split-dependent configuration changes coherent over all representatives;
per-representative selection can partially average this axis. Pooled selection
therefore targets quotient stability but can relocate randomness rather than
reduce total instability. The frozen benchmark must report all components and
must not call the repair globally stabilizing.

## 48. Selection-substudy compute correction

The earlier protocol row `9 cells x 6 configurations x 8 reps = 432` silently
omitted schema representatives and final refits, and became invalid once split
randomness and three selection paths were explicit. A balanced held-out action
design needs two eight-member nuisance products, not one eight-member menu.
The sign-test audit subsequently raises the split count to twelve: 10/12
same-direction differences attain two-sided binomial-sign `p≈.039`, whereas
7/8 attain only `p≈.070`. The corrected cost is therefore `9 cells x 16 schema
reps x 12 split seeds x (6 candidate fits + 3 final paths) = 15,552` fits
before caching coincident configurations. The total study ceiling is about
24,992 fits/passes, not 9,872.
The pooled repair must choose on the frozen development product and evaluate
on the held-out product; otherwise it demonstrates only in-menu optimization.

## 49. Field topology clears a principled synthetic selectivity test

The original semantic-metric pilot used one fixed nominal target and could not
separate false adjacency from ordinary shrinkage. A replacement suite draws
latent 24-state functions from Gaussian priors with path, ring, or isotropic
precision, then fits the same one-hot function space with isotropic, path,
ring, or permuted-path ridge metrics. Regularization strength is selected on
independent validation data. Candidate topology strength is fixed at four,
while true strength `1/4/16` is crossed with three sample/noise regimes and 200
paired trials per cell.

For ordinal targets, the path metric beats a permuted path with bootstrap
interval excluding zero in 9/9 cells, isotropic in 6/9, and ring in 7/9. For
cyclic targets, ring beats permuted path in 9/9, isotropic in 6/9, and path in
6/9. For nominal targets, isotropic beats every imposed topology in all 9/9
cells. This is the selective semantic pattern FieldRiesz needed and is robust
to strong prior misspecification.

The weak-topology (`true strength=1`) cases remain unresolved versus isotropic
and sometimes favor isotropic in mean. Thus topology type can be metadata, but
stiffness magnitude is a genuine hyperparameter that should be regularized or
validated. The result is deliberately Bayes-structured and does not clear the
real-data/multi-architecture/noninferiority gates for a standalone paper.
Artifacts: [`field_topology_bayes_suite.py`](field_topology_bayes_suite.py) and
[`field_topology_bayes_suite.json`](field_topology_bayes_suite.json).

## 50. Baseline-stable selection cells are prospectively heterogeneous

The three confirmation cases were selected for baseline instability and could
not estimate prevalence. The complementary baseline-stable binary cells—Adult
forest, Adult HistGB, and Churn HistGB—were therefore run unchanged on the
seven unseen validation splits. Adult forest becomes unstable on 2/7 splits:
one changes configuration on 28/32 representatives and the other on 2/32;
schema risk increases in both. The remaining five splits reproduce one choice
throughout the orbit.

Adult and Churn HistGB remain exactly selection-stable on all 7/7 splits, even
though the fitted HistGB predictor itself can retain class-ID schema risk.
Across the six repeated binary cells, four ever show selection-path
sensitivity and the two HistGB cells do not. This was the binary-only screen;
Section 58 later completes the original 3×3 panel with Otto and sharpens the
family pattern to forest 3/3, CatBoost 2/3, and HistGB 0/3. Artifacts:
[`selection_stable_screen.py`](selection_stable_screen.py),
[`selection_stable_screen.json`](selection_stable_screen.json), and the cell
records in [`selection_split_stable_screen`](selection_split_stable_screen/).

## 51. FieldRiesz novelty narrows after topology/symmetry prior art

A focused search found three additional boundaries. Classical smoothing spline
and Galerkin methods already encode derivative penalties as basis-coordinate
quadratic forms, so mass and stiffness tensors are established numerical
machinery. [Otto et al. (JMLR
2025)](https://www.jmlr.org/papers/v26/24-1315.html) unify symmetry enforcement,
discovery, and promotion using Lie derivatives and explicitly cover
basis-function regression, neural networks, and fields. [PH-Reg (ICML
2024)](https://proceedings.mlr.press/v235/zhang24z.html) already argues that
representation topology should match regression target topology.

FieldRiesz therefore cannot claim function-space regularization, topology
matching, or general symmetry-compatible optimization. Its only plausible
standalone contribution is their schema-specific composition: declared
nominal/path/ring input fields, transport across equivalent within-field
charts, and a single initialization/regularization/optimizer construction that
is selective in field semantics. This keeps the direction conditional despite
the stronger synthetic result.

## 53. Adaptive topology tuning fails the nominal semantic gate

The fixed-strength suite left a natural question: can validation safely shrink
a topology toward isotropic when the semantic prior is weak or absent? A
post-frozen calibration extension lets each path, ring, and permuted-path
family choose stiffness from `{0,1,4,16}` jointly with its ridge coefficient;
the isotropic family has only the zero-stiffness member. The candidate-count
asymmetry is explicit, so this is a mechanism stress test rather than a fair
leaderboard.

For structured tasks, tuning helps the intended comparison. Path beats
isotropic in 9/9 ordinal cells, ring beats isotropic in 9/9 cyclic cells, and
the matched family beats a permuted path in 8/9 ordinal and 9/9 cyclic cells.
Cross-topology discrimination is weaker: path beats ring in 7/9 ordinal cells,
whereas ring beats path in only 4/9 cyclic cells with five unresolved.

The nominal result falsifies the stronger selectivity claim. Tuned path and
ring each significantly beat isotropic in 8/9 cells, and tuned permuted path
does so in 9/9. Their mean zero-stiffness selection rates are only 62.8%,
62.2%, and 63.1%, respectively, so validation chooses a nonzero false geometry
about 37% of the time. A finite sample can reward accidental smoothness along
an arbitrary category ordering even when the generating prior is exchangeable.

Consequently topology is not an ordinary hyperparameter and HPO cannot certify
semantic admissibility. The field type must be declared from task meaning;
only strength within that declared family may be selected. This negative result
actually sharpens OrbitANOVA: a false ordering may improve a single spelling's
loss while remaining unidentified on the quotient, exactly why schema risk
must be audited independently of mean performance. Artifacts:
[`field_topology_strength_selection.py`](field_topology_strength_selection.py)
and
[`field_topology_strength_selection.json`](field_topology_strength_selection.json).

## 54. The pooled-HPO action is sensitive to the nuisance partition

The held-out repair originally froze one development product: feature and
category levels `0:2`, evaluated on levels `2:4`. To test whether that choice
drove the result without any new fits or test labels, a post-frozen diagnostic
enumerates all 36 disjoint `2-of-4 × 2-of-4` development/complement partitions
for each of the seven prospective splits.

Across 252 partition/split cases, the development-selected configuration
matches the full-menu choice 88.1% for Adult CatBoost, 84.9% for Churn forest,
and 73.8% for Churn CatBoost. It is optimal on complement-representative
validation loss in 76.2%, 75.4%, and 57.1%. Only 4/7, 2/7, and 3/7 splits have
one configuration across all 36 partitions. Mean complement regret is small
(`0.000171`, `0.000245`, `0.000347`) but the maxima reach `0.00177`, `0.00351`,
and `0.00207`.

This neither erases the original test-prediction risk reductions nor confirms
them: the alternative partitions have no all-candidate test predictions, and
development/complement representatives reuse the same validation rows. It
does show that partition choice is another part of the selection rule. The
confirmatory study must predeclare several balanced nuisance folds, aggregate
their choices by a fixed rule, and report fold sensitivity. Picking the most
favorable partition would simply recreate the overtuning problem at the schema
level. Artifacts:
[`selection_partition_sensitivity.py`](selection_partition_sensitivity.py)
and
[`selection_partition_sensitivity.json`](selection_partition_sensitivity.json).

## 55. Sampled-menu randomness belongs in the pipeline estimand

The partition sensitivity is not merely a reporting nuisance. When pooled HPO
uses a sampled development menu `M_m`, its output is `p_{z,s,m}`. The menu seed
can change one configuration coherently over all evaluation representatives
(menu main variance) or change their relative responses (schema×menu); it can
also interact with validation split/search randomness. These are properties of
the deployed selection rule and should enter the same product fANOVA as schema
and split.

This extra factor has a clean boundary. Uniform pooling over a complete finite
group has no menu seed: changing the origin reindexes the same sum. A sampled
finite menu or non-group chart family does. The confirmatory design should
therefore draw independent development and evaluation schema menus from the
declared measure, replicate `m`, and report menu and schema×menu components.

The saved decision grid already supports a conditional estimate. Across the
three selected cases, pure feature/category-menu terms account for 3.1%, 5.3%,
and 10.1% of one-hot decision variance; split main accounts for 69.2%, 50.7%,
and 38.5%; and menu×split interactions account for 27.7%, 43.9%, and 51.4%.
Thus the major hidden term is not a stable bad menu but split-dependent menu
sensitivity. The balanced subsets overlap, so this is an fANOVA over the
declared finite subset distribution, not independent replication or a
prevalence estimate.

Balanced cross-fitting remains useful as a conditional diagnostic. Averaging
candidate loss across the exhaustive collection of equal-sized subsets is
exactly the full-menu average, because every representative has equal
inclusion count. Thus folds assess how well a pooled decision transfers to
unseen representatives; they should not be searched or majority-voted into a
post-hoc rule. This refinement makes “selection-rule complete” more literal
and prevents a schema-level version of validation overtuning.

## 56. Menu-averaged pooled HPO survives at prediction level

The decision audit alone could not establish whether alternative partitions
preserve the quotient-risk repair. The existing full-refit paths cover every
Adult menu choice and all but configuration 2 for each Churn family. Because
the final training set, schema orbit, and model seed are unchanged across
validation splits, only those two missing 32-member configuration orbits need
new fits. No test labels enter any configuration or menu choice.

For each balanced development menu, choose one configuration, freeze it over
the orbit, and compare its test predictions with per-representative selection
on the complementary feature/category levels. Average the 36 menu contrasts
within split before inference. Pooled selection lowers mean held-out schema
risk on 7/7 splits in Adult CatBoost, Churn forest, and Churn CatBoost; both the
magnitude sign-flip and binomial sign tests equal `.015625` in every case.
Grand menu-average reductions are 29.2%, 37.5%, and 34.7%.

The action is distribution-robust, not pointwise. It lowers risk in 228/252,
180/252, and 198/252 individual menu/split cells. Mean Brier is worse under
pooling on 6/7, 5/7, and 5/7 splits; the corresponding exact-test pairs are
`.0469/.125`, `.25/.453`, and `.0781/.453`, so none passes both tests. In a
joint factorization of output predictions, all terms involving menu account
for 20.0%, 27.1%, and 40.0% of variance. This is a meaningful cost of sampled
schema menus, even though averaging their held-out risk contrasts gives a
consistent targeted reduction.

The result upgrades pooled HPO from a one-partition anecdote to a replicated
menu-distribution action inside the three selected cases. It does not estimate
population prevalence, guarantee each menu, improve accuracy, or replace the
prospective dataset/model transfer gate. Artifacts:
[`selection_menu_output_risk.py`](selection_menu_output_risk.py),
[`selection_menu_output_risk.json`](selection_menu_output_risk.json), and
[`selection_menu_config_predictions.npz`](selection_menu_config_predictions.npz).

## 57. The pooled action has a measurable dependence on `mu`

Uniform averaging is part of the quotient estimand, so the consistent mean
repair does not imply robustness to another admissible weighting of the same
menus. A finite density-ratio stress test maximizes the schema-risk contrast
over weights `w_i <= kappa/36`. For integer divisors here, `kappa=2` averages
the worst 18 menu contrasts and `kappa=4` averages the worst nine.

At `kappa=2`, the worst-weighted contrast remains negative on 7/7 Adult
CatBoost, 4/7 Churn-forest, and 5/7 Churn-CatBoost splits. At `kappa=4`, the
counts are 6/7, 2/7, and 4/7. The across-split mean worst contrast remains
negative in all cases at both caps (`-1.03e-4/-7.05e-5`,
`-2.49e-4/-3.88e-5`, and `-9.69e-5/-6.32e-5` for caps 2/4), but this does not
create a splitwise guarantee.

This is a valuable limitation: OrbitANOVA requires an explicit `mu` because
both measured risk and audit-guided action can depend on it. The paper should
show a small bounded-reweighting curve for promoted actions and avoid treating
uniform menu weights as a law of nature. Density-ratio robustness itself is
standard and is not a novelty claim.

## 52. A validation-margin certificate separates stable HistGB controls

For identity winner `h_0`, define `gamma` as its smallest validation-loss gap
and `delta` as the largest schema-induced change in any competitor-minus-
`h_0` gap. The elementary sufficient condition `delta<gamma` guarantees the
winner is fixed over the measured orbit. It is not claimed as a new theorem.

All 24 saved split orbits from Adult CatBoost, Churn forest, and Churn CatBoost
fail the certificate, with `delta/gamma` from 1.55 to 132; all are actually
selection-unstable. The certificate holds on stable Adult HistGB (`.564`) and
Churn HistGB (`.395`) controls. It fails on one stable Adult-forest control
(`5.00`) because the uniform absolute bound is conservative, although the
exact competitor gaps remain positive. Thus passing can cheaply certify a
measured-menu tuning path, while failure should trigger decision fANOVA rather
than be interpreted as instability. Control artifacts are in
[`selection_margin_controls`](selection_margin_controls/).

## 58. The prospective 3×3 decision panel is family-structured

The three untouched Otto cells complete the baseline-stable screen without
test labels or prediction refits. Across seven unseen validation splits, Otto
forest switches on 1/7, while Otto HistGB and CatBoost remain stable on 7/7.
The decision-only harness exactly reproduces an independently completed full
forest run at the overlapping seed (selection grid identical; maximum mean-
loss error `5.6e-17`).

Combining all original cells, every forest pipeline ever switches (Adult,
Churn, Otto: 3/3), Adult and Churn CatBoost switch while Otto CatBoost does not
(2/3), and no HistGB pipeline switches (0/3). Five of nine cells ever exhibit
selection sensitivity. The pattern is much stronger evidence for family-
structured heterogeneity than the earlier selected-case count, but it remains
a nine-cell pilot rather than a population prevalence estimate.

The sole Otto switch occurs at split `20260830`. Identity selects forest config
0, while one feature permutation selects config 3 for all four aligned class
IDs, so decision fANOVA is 100% feature main. The identity margin is
`1.34e-4`; the maximum schema shift in competitor gaps is 40.6 times larger.
The promoted output run shows schema risk `0.003212→0.007549` (2.35×), hard
flips `13.8%→16.9%`, and an orbit-mean Brier improvement of `0.005104`.
Conditional row-bootstrap intervals exclude zero for both changes. This
extends the stability–accuracy trade-off to a numeric multiclass task, but one
of seven splits cannot establish its frequency.

Artifacts:
[`selection_otto_prospective_decisions.py`](selection_otto_prospective_decisions.py),
[`selection_otto_prospective_decisions.json`](selection_otto_prospective_decisions.json),
and
[`selection_split_otto_screen`](selection_split_otto_screen/).

## 59. One-factor controls separate native category invariance from pipeline sensitivity

The Day 3 narrative initially grouped feature order, category IDs, class IDs,
and units under one informal notion of schema spelling. That invited a valid
objection: native categorical learners and rank-based trees should already
absorb some of these transformations. A prospective extension therefore
crossed Adult, Churn, and Otto with six pipelines and four factors, varying one
factor at a time. Each cell first selected on the identity view and froze that
configuration, then reran validation selection in every view. The design used
2,000 semantic training rows, 750 test rows, one identity plus up to three
nonidentity views, and a predeclared `1e-10` aligned-probability tolerance.

The objection is correct for category renaming. Native HistGB and CatBoost are
exactly invariant to every tested category-ID permutation on Adult and Churn.
Their maximum aligned probability deviation and category-induced validation
loss range are zero. The ordinal forests are positive controls rather than
native-category counterexamples: arbitrary codes enter numerical thresholds,
and their maximum probability changes are `0.425/0.230` for the sqrt-feature
forests and `0.402/0.404` for full-feature forests on Adult/Churn. One-hot
logistic is invariant to `7.1e-12` or better on category IDs.

Other factors remain pipeline-visible. Feature order changes every forest and
CatBoost frozen fit on all three datasets; it changes HistGB only on numeric
multiclass Otto (`max Δp=.214`). Binary class reversal changes HistGB
(`.216/.096`) and CatBoost (`.123/.105`) on Adult/Churn after output alignment,
while both are invariant on multiclass Otto. HistGB is exactly invariant to all
positive affine unit views. Raw forests move under those views on all datasets,
whereas standardized one-hot forests are exact. A float32 rank/equality audit
finds a direct rounding change on Churn but not Adult or Otto, so the remaining
raw-forest mechanism is not resolved and must be described as implementation
arithmetic/tie behavior rather than proven float32 rank collapse. CatBoost is
unit-invariant on Adult/Churn but not Otto (`max Δp=.073`, zero hard flips).

Validation switches in 12/66 applicable dataset×pipeline×factor cells on the
new split. One-hot logistic and HistGB never switch, and no unit view switches
any family. The switches concentrate in forest feature/category handling and
CatBoost feature/class handling. HistGB demonstrates why selection stability
is not prediction invariance: it has no decision switches despite large fixed-
fit changes in the binary class and Otto feature-order cells.

This extension narrows rather than broadens the claim. Day 3 should not say
that native categorical learners are sensitive to arbitrary category names in
these experiments. It should say that invariance is transformation- and
algorithm-specific, and that the complete fit/selection path can violate a
symmetry even when the abstract model class admits it. Artifacts:
[`invariance_matrix_config.json`](invariance_matrix_config.json),
[`invariance_matrix.py`](invariance_matrix.py),
[`invariance_matrix_results.json`](invariance_matrix_results.json),
[`analyze_invariance_matrix.py`](analyze_invariance_matrix.py),
[`invariance_matrix_summary.csv`](invariance_matrix_summary.csv), and
[`INVARIANCE_MATRIX_REPORT.md`](INVARIANCE_MATRIX_REPORT.md).
