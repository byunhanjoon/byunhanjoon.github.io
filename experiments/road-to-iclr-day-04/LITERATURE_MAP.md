# Day 4 literature map: what is already occupied

This is a novelty map, not a general tabular-DL bibliography.  Each row asks
which version of the Day 1--3 motivation is already claimed, and therefore what
Day 4 must *not* present as its primary contribution.

## Performance landscape

| Work | Venue | Main claim | Consequence for Day 4 |
| --- | --- | --- | --- |
| [Numerical feature embeddings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | NeurIPS 2022 | PLE and periodic embeddings make scalar numerical inputs substantially stronger across backbones. | "Embed numerical values" and generic multiresolution encodings are not new. PLE is the minimum baseline. |
| [TabR](https://openreview.net/forum?id=rhgIgTSSxW) | ICLR 2024 | Label-aware nearest-neighbor retrieval inside a feed-forward model. | A support-memory or local-residual module without a sharper structural distinction collides directly. |
| [RealMLP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html) | NeurIPS 2024 | Careful regularization, preprocessing, and training make an MLP highly competitive. | Small gains from another training recipe are insufficient. |
| [LoCalPFN](https://neurips.cc/virtual/2024/poster/96776) | NeurIPS 2024 | Retrieval and local fine-tuning adapt TabPFN to the query neighborhood. | "Use local examples when support is dense" is occupied for foundation models. |
| [TabM](https://openreview.net/forum?id=Sd4wYYOhmY) | ICLR 2025 | BatchEnsemble-like parameter-efficient MLP ensemble gives a strong accuracy/efficiency frontier. | Generic diversity or rank expansion is not a clean new axis; TabM must be a backbone, not an easy comparator. |
| [TabReD](https://openreview.net/forum?id=L14sqcrUC3) | ICLR 2025 | Eight feature-rich, temporally split industrial datasets; simple MLPs with embeddings and GBDTs transfer better than many complex methods. | Day 4 needs official temporal splits and evidence beyond tidy random-split datasets. |
| [ModernNCA](https://openreview.net/forum?id=JytL2MrlLT) | ICLR 2025 | Modernized neighborhood-component learning is a strong tabular baseline. | Generic learned neighborhoods are occupied beyond TabR. |
| [TabICL](https://proceedings.mlr.press/v267/qu25d.html) | ICML 2025 | Scales tabular in-context learning to large classification datasets. | A proposed mechanism should plausibly transfer to TFMs, but TFM pretraining is not required for the first falsification. |
| [TabDPT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html) | NeurIPS 2025 | Real-data pretraining plus retrieval scales tabular foundation models. | "Pretrain on real tables" is not a Day 4 novelty. |
| [TabArena](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2025 | Validation, time budgets, and hyperparameter/cross-model ensembling materially change rankings. | Report paired cells, fixed budgets, validation-only choices, and strong ensembles; do not rely on mean rank from one seed. |
| [TabPFN v2 analysis](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c57b3718381cb2bf9c1ccd63377f2448-Abstract-Conference.html) | NeurIPS 2025 | Dissects and extends the current tabular foundation model. | Any foundation-model story must identify an intervention at tokenization or synthetic-prior construction, not merely claim compatibility. |
| [Robustness is important](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520) | PNAS Nexus 2026 | Measures prediction sensitivity of general LLMs, TabPFN, and LimiX to task-irrelevant names, row/column order, precision, and format; contrasts this with conventional supervised methods. | The discovery that harmless table rewrites change predictions is occupied. OrbitANOVA must establish conventional full-pipeline effects and contribute aligned proper-risk value, product attribution, tuning-path coupling, and action. |
| [PREF](https://openreview.net/pdf?id=1JhhSxdBS1) | TMLR submission, 2026 | Audits single preprocessing knobs across boosted trees, MLP-style models, and TFMs with sensitivity/volatility indices. | Broad preprocessing robustness across model families is occupied. OrbitANOVA must keep every branch task-equivalent, analyze simultaneous interactions and aligned predictions, and demonstrate a schema-specific repair. |
| [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112) | 2024 preprint | Crosses expert feature engineering, HPO regimes, model choice, and test-time adaptation on ten Kaggle tasks; rankings change materially. | Preprocessing/HPO coupling and ranking sensitivity are occupied. Its branches deliberately add dataset-specific information; OrbitANOVA must isolate equivalent task spellings and audit representation-induced tuning decisions in aligned prediction space. |
| [EquiTabPFN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html) | NeurIPS 2025 | Defines a target-permutation equivariance gap and builds an equivariant PFN. | Neither the first group-equivariance gap nor symmetrization benefit is available as a claim. Target-ID order is one factor and architectural closure is a baseline. |
| [Bregman information](https://proceedings.mlr.press/v206/gruber23a.html) | AISTATS 2023 | Gives a general bias--variance decomposition for proper losses through Bregman information. | The label-free ensemble ambiguity identity is established mathematics. OrbitANOVA's contribution must be the schema-quotient estimand, attribution, evidence, and use. |
| [Functional ANOVA in Computer Models With Time Series Output](https://doi.org/10.1198/TECH.2010.10029) | Technometrics 2010 | Decomposes functional-output variance to attribute computer-model inputs. | Applying fANOVA to vector- or function-valued predictions is not itself new. OrbitANOVA needs its equivalence-restricted factors, proper-risk total, tuning decisions, and action transfer. |
| [Modeling the Machine Learning Multiverse](https://proceedings.neurips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html) | NeurIPS 2022 | Models conclusions across combinations of methods and hyperparameters. | Branch sensitivity and interaction-aware multiverses are occupied. OrbitANOVA is differentiated only as a quotient multiverse in which all branches denote the same semantic task and zero dispersion is meaningful. |
| [GGPL](https://www.lgresearch.ai/publication/view?seq=180) | KDD 2026 | GBDT-guided, trainable PLE breakpoint placement improves regression across many models. | Adaptive breakpoints and supervised bin allocation are occupied. |
| [Data uncertainty in tabular DL](https://arxiv.org/abs/2509.04430) | ICML 2026 | Numerical embeddings, retrieval, and ensembles help particularly in high-aleatoric-uncertainty regions; LRLR-triplet pretraining explicitly improves target consistency of whole-row neighborhoods. | "Route uncertain rows" and generic target-consistent embedding pretraining are occupied. FieldRiesz must distinguish its per-field declared geometry, out-of-fold anchor residual estimand, and false-geometry controls. |
| [A Mechanistic Study of Tabular Foundation Models](https://arxiv.org/abs/2605.21288) | 2026 preprint | Locates row/column/class permutation behavior in specific TFM components and shows that surgical edits can make approximate invariance exact without harming accuracy. | OrbitANOVA cannot claim that permutation auditing or audit-guided invariant surgery is new. Its remaining scope is the complete conventional pipeline, a product of multiple admissible schema actions, proper-risk-valued attribution, and selection-path coupling. |
| [Same Content, Different Representations](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4c638a7cf71c060b4bed15500da38800-Abstract-Conference.html) | ICLR 2026 | Holds table content fixed while changing structured versus semi-structured representation for Table QA. | Controlled representation studies are occupied even beyond predictive tables. OrbitANOVA must distinguish equivalent typed-schema actions, aligned probabilistic predictions, and supervised tabular pipelines rather than claim the generic controlled-study template. |
| [Improving Robustness of Tabular Retrieval via Representational Stability](https://arxiv.org/abs/2604.24040) | 2026 preprint | Measures instability across semantically equivalent serializations and learns a one-view approximation to the multi-serialization embedding centroid. | Equivalent-representation centroids and cheap amortized closure are occupied in table retrieval. OrbitCover is an implementation baseline; novelty must come from the quotient contract, risk attribution, tuning path, and held-out factor-specific action in conventional tabular learning. |
| [Metamorphic testing of supervised classifiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144/) | JSS 2011 | Checks classifier behavior under declared data/label/attribute transformations when a conventional test oracle is unavailable. | Equivalent-input testing and label-free inconsistency detection are established. OrbitANOVA must contribute a risk-valued product estimand, interactions, training/selection randomness, and action transfer—not the metamorphic relation itself. |
| [Design Choices That Matter](https://arxiv.org/abs/2608.04702) | Discovery Science 2026 (to appear) | Uses fANOVA across architecture, initialization, fine-tuning, learning choices, and interactions in a seven-dataset remote-sensing benchmark. | Factor-attributed benchmark choices and interactions are occupied. OrbitANOVA's distinction must be aligned prediction outputs over a semantics-equivalent schema product, exact proper-risk value, and a zero-invariance target. |
| [MAgg](https://proceedings.mlr.press/v235/wei24i.html) | ICML 2024 | Aggregates predictions across metamorphic relations at test time for combinatorial problems. | Metamorphic aggregation is occupied; OrbitCover cannot be sold as the first relation-aware aggregation method. |
| [Data Augmentation: A Fourier Analysis Perspective](https://proceedings.mlr.press/v336/tahmasebi26a.html) | COLT 2026 | Characterizes full versus partial finite-group augmentation and when exact invariance requires the full group. | Compute-limited partial group coverage has recent theory. OrbitCover's plausible delta is empirical factor-risk allocation over a mixed schema product, not generic partial augmentation theory. |
| [PRESTO: Mapping the Multiverse of Latent Representations](https://proceedings.mlr.press/v235/wayland24a.html) | ICML 2024 | Maps latent representations across methods, hyperparameters, and datasets for sensitivity analysis and search-space navigation. | Representation multiverses and hyperparameter sensitivity are occupied. OrbitANOVA must restrict every branch to one semantic quotient and value aligned output dispersion as removable proper risk. |
| [Predictive Multiplicity in Classification](https://proceedings.mlr.press/v119/marx20a.html) | ICML 2020 | Quantifies predictive disagreement among near-optimal classifiers. | Predictor disagreement is occupied. OrbitANOVA's distinction is that one complete pipeline generates a structured family through declared equivalent schema spellings, with nuisance attribution and an invariance target. |
| [RieszNet and ForestRiesz](https://proceedings.mlr.press/v162/chernozhukov22a.html) | ICML 2022 | Learns Riesz representers of linear functionals for automatic debiased machine learning. | Learning or naming a Riesz representer is not new; its causal estimand differs, but the mathematical vocabulary is occupied. |
| [Recursive estimation of conditional kernel mean embeddings](https://www.jmlr.org/papers/v25/23-0168.html) | JMLR 2024 | Develops conditional kernel mean/regression operators in RKHSs. | `M^dagger c` is classical conditional-mean/Galerkin regression algebra, not a new residual learner. |
| [Function Basis Encoding of Numerical Features in Factorization Machines](https://openreview.net/forum?id=M4222IBHsh) | OpenReview work | Encodes numerical fields through chosen function bases for factorization machines. | "A numerical feature is a function basis" is not by itself a novel framing; FieldRiesz needs its measured semantic operator and neural transport. |
| [Knowledge-Enriched Machine Learning for Tabular Data](https://proceedings.mlr.press/v288/kim25a.html) | NeSy 2025 | Encodes problem-specific deterministic knowledge, including column descriptions. | "Use schema knowledge" is not a novelty claim; the contribution must be the precise field operator, covariance, and falsifiable geometry protocol. |
| [OU-Net](https://www.sciencedirect.com/science/article/pii/S0020025526007462) | Information Sciences 2026 | Separates ordered and unordered features in a dual-stream architecture under partial monotonicity constraints. | Declaring that some fields are ordered is not new. FieldRiesz must distinguish a within-field smoothness form from monotonic prediction constraints and earn its chart-covariant residual/control package empirically. |
| [Deep Regression Representation Learning with Topology](https://proceedings.mlr.press/v235/zhang24z.html) | ICML 2024 | PH-Reg matches learned regression representations to target-space topology. | Topology matching for regression is occupied. FieldRiesz differs only if declared *input-field* topology, chart transport, and residual controls matter jointly. |
| [A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning](https://www.jmlr.org/papers/v26/24-1315.html) | JMLR 2025 | Uses Lie derivatives to enforce, discover, or softly promote symmetries in basis regression, neural networks, operators, and fields. | Function-space symmetry and symmetry-breaking penalties are occupied; schema-specific tabular fields and complete-pipeline evidence must carry the distinction. |
| [VectorAdam](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1a774f3555593986d7d95e4780d9e4f4-Abstract-Conference.html) | NeurIPS 2022 | Repairs Adam's rotation nonequivariance for vector-valued parameters. | A field-block rotation-equivariant optimizer is a baseline/intervention, not a generic optimizer novelty. |
| [Preconditioned Norms](https://arxiv.org/abs/2510.10777) | 2025 preprint | Unifies steepest-descent, quasi-Newton, and adaptive methods through preconditioned matrix norms and gives affine/scale-invariance conditions. | The transported field-metric step cannot claim generic affine-invariant optimization; only its declared tabular-field role and audit-guided use remain distinctive. |
| [Neural Additive Models](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html) | NeurIPS 2021 | Learns one neural shape function per input and sums their contributions. | Per-field nonlinear functions and additive interpretability are occupied; a marginal FieldRiesz feature is not a new model class. |
| [Scalable Interpretability via Polynomials](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ee81a23d6b83ac15fbeb5b7a30934e0b-Abstract-Conference.html) | NeurIPS 2022 | SPAM uses low-rank polynomial tensors to learn all higher-order feature interactions without enumerating them. | Efficient interaction modeling, tensor factorization, and interpretable higher-order terms are occupied. FieldRiesz cannot claim novelty from interactions alone. |
| [Scalable Higher-Order Tensor Product Spline Models](https://proceedings.mlr.press/v238/ruegamer24a.html) | AISTATS 2024 | Factorizes higher-order tensor-product splines and supplies a penalization scheme with main-effect-like computational cost. | Tensor-product spline surfaces and their smoothness penalties are directly occupied. The only plausible gap is a declared field group paired with empirical joint mass, anchor residuals, neural transport, and exact false-geometry controls. |
| [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html) | AISTATS 2020 | Makes learned interaction effects identifiable by removing variation representable by lower-order effects, with an exact purification algorithm for piecewise-constant functions. | Projecting a tensor chart off constant and marginal spaces is a correctness requirement, not novelty. Dependence makes marginal centering inadequate; the empirical joint measure must define the projection. |
| [The SKIM-FA Kernel](https://jmlr.org/beta/papers/v24/21-1403.html) | JMLR 2023 | Gives scalable sparse nonlinear interaction discovery through an orthogonal functional-ANOVA kernel decomposition. | Orthogonal interaction spaces, kernelized ANOVA, and scalable interaction discovery are occupied. The proposed field-group story must rest on declared geometry, residual estimands, controls, and backbone utility. |
| [Out-of-Sample Extensions for Spectral Methods](https://proceedings.neurips.cc/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html) | NeurIPS 2003 | Gives a common framework for extending sample-defined spectral embeddings to new points. | Empirical-null and out-of-support extension are classical issues. Reference-mass completion must be justified as the schema-covariant tabular solution, not as discovery of the problem. |
| [Manifold Regularization](https://jmlr.org/papers/v7/belkin06a.html) | JMLR 2006 | Combines an ambient RKHS norm with a data-distribution geometry penalty and obtains an out-of-sample representer theorem. | Mixing a full-support ambient notion with empirical geometric regularization is classical. FieldRiesz can claim only its chart-covariant mass/stiffness realization, anchor-residual target, and tabular controls. |
| [Sampling-based Nyström Approximation and Kernel Quadrature](https://proceedings.mlr.press/v202/hayakawa23a.html) | ICML 2023 | Studies kernel eigenspaces relative to a reference probability measure and connects non-i.i.d. landmarks to kernel quadrature. | A reference-measure Gram operator and quadrature completion are not new mathematics. The proposed empirical/reference mass mixture is an audit device specialized to declared tabular function spaces. |

## Adjacent mathematical claims that close tempting routes

- General representation or preprocessing sensitivity is already an empirical
  topic.  OrbitANOVA's factorial accounting is useful instrumentation, but
  "equivalent encodings change neural predictions" is not enough by itself.
- Generic functional-ANOVA interaction learning, learned pair selection, and
  ANOVA decompositions are occupied. Neural additive models already learn one
  nonlinear function per feature; SPAM and scalable tensor-product splines
  already cover efficient higher-order interactions. Day 4 should not market
  another interaction selector or product basis as the conceptual core.
- "Purifying" an interaction by removing lower-order components is explicitly
  occupied by functional-ANOVA work. For dependent fields, marginal centering
  is not enough; empirical joint-measure orthogonality is a required repair,
  not an extra contribution.
- Splines, smoothing-spline ANOVA, generalized additive mixed models, and
  random effects already combine smooth functions with level-specific effects.
  A mixed-measure method needs a tabular-neural contribution beyond merely
  writing down `smooth + lookup`.
- Generic local learning is covered by TabR, ModernNCA, LoCalPFN, and older
  neighborhood propagation.  Exact repeated values must be justified as the
  *atomic component of a numerical measure*, not described as nearest
  neighbors at distance zero.
- The ICML 2026 uncertainty analysis is a particularly close conceptual
  neighbor: its embedding is pretrained with a triplet loss so rows with closer
  targets are closer in latent space. FieldRiesz instead asks whether a
  *declared within-column topology* allocates a fixed generalized spectrum to
  the right field modes, and estimates only what a cross-fitted anchor misses.
  This is a genuine distinction, but also raises the required baseline:
  LRLR-triplet must be compared once an official implementation is available.
- Generic residual-layer boosting is crowded by classical boosting and modern
  residual random-feature methods.  "Boost TabM" is not a defensible headline.
- At `tau=0`, `g=M^dagger c` is the least-squares/Galerkin projection of a
  conditional residual mean into the chosen field space.  With `tau S`, it is
  classical smoothness-regularized regression.  The residual Riesz formula is
  a performance bridge and coordinate-clean implementation, not a standalone
  novelty claim.
- Reference-mass completion is adjacent to ambient/intrinsic manifold
  regularization, Nyström extension, and kernel quadrature. Its value here is
  diagnostic: it makes a rank-deficient semantic/control pencil comparable on
  the full declared function space. The mixture itself is not a theorem-level
  novelty, and its sign-changing performance sweep rules out presenting it as
  a finished method.
- Affine/rotation-covariant optimization is crowded by natural-gradient,
  VectorAdam, and recent preconditioned-norm frameworks. Likewise, topology
  matching and general symmetry promotion are established. FieldRiesz's only
  defensible novelty is the full tabular package; exact trajectory closure or
  a semantic stiffness matrix alone cannot headline a paper.
- Generic common-factor/PCA features are old.  The Day 4 linear
  common/innovation pilot is useful negative evidence, not a contribution.
- [Neural Feature Learning in Function Space](https://arxiv.org/abs/2309.10140)
  already develops a broad function-space geometry for statistical dependence
  and neural feature approximation.  It is not a tabular schema/operator
  method, but it closes any attempt to claim "feature learning in function
  space" as the novelty by itself.

## Internal prior: RAPLE is the performance baseline

The nearby `multifeature_ple_tabular` project already contains Residual-
Anchored PLE (RAPLE): an out-of-fold LightGBM anchor, smoothed marginal
target-response curves, relation pools, and residual-selected pair operators.
On four official TabReD temporal datasets, its checked-in report records raw
neural wins on 11/16 full-budget dataset–model means across MLP, ResNet, TabM,
and TabR; a validation-selected three-way hybrid wins 16/16. These are
three-seed engineering means, not significance claims, and the gated result is
an ensemble. This is stronger temporal and
architecture evidence than the current FieldRiesz pilot.

FieldRiesz is not a renaming of RAPLE.  RAPLE is supervised, cross-field, and
adds response/anchor features; FieldRiesz is a per-field function-space metric
estimated from covariates plus declared schema geometry.  The constructive
synthesis is a cross-fitted **residual Riesz representer**:

```text
c_j = E[phi_j(X_j) r_oof],
h_j(x) = c_j^T (M_j + tau_j S_j)^(-1) phi_j(x).
```

This is the minimum-Riesz-norm field function aligned with the anchor residual
and is invariant to invertible changes of field chart.  It turns RAPLE's
existing empirical success into a sharper mathematical hypothesis.  In the
complete fixed-pilot three-seed panel over five datasets and MLP, ResNet, and
TabM, correct Riesz beats raw RAPLE in only 17/45 paired cells (-0.15% mean),
anchor-only in 23/45 (-0.11%), and one node-permuted geometry in 33/45 (+0.57%).
The result supports geometry sensitivity more than performance superiority.
Five permutation controls on the positive California/Weather MLP/ResNet subset
give 52/60 control wins (+1.05%), or 11/12 after within-cell averaging; five exact generalized-spectrum-
preserving rotations across MLP, ResNet, and TabM give 80/90 (+0.90%).
The same 18 semantic fits are reused across rotations; after averaging controls
within cell, all 18/18 cells favor the semantic operator. Strength probes show
that this control gap weakens at `tau=3`. The two datasets were selected after
inspection, so 18/18 is mechanism stress evidence, not a confirmatory test;
independent datasets are the replication unit. Recovered official
configurations and the validation-gated full RAPLE hybrid remain mandatory
controls.

## The gap that remains plausible

Numerical-embedding papers choose coordinates for a scalar field; retrieval
papers use neighborhoods of whole rows.  Neither line usually treats each
field as a finite function space equipped with two separately declared forms:

```text
empirical mass M_j                 semantic stiffness S_j
how functions are measured        which functions are rough
```

That separation yields a narrow candidate contribution: build train-only
support elements, normalize their empirical function mass, and add only the
geometry licensed by field metadata.  The resulting Riesz operator
`K_j=M_j+tau_j S_j` makes the first-layer prior, penalty, and metric update
covariant to invertible changes of chart within a field.  It also creates a
mechanism-specific test: correct stiffness must beat mass alone and a
permuted-geometry control without changing the represented functions.

The refined control works in `M_j`-whitened coordinates and rotates the
semantic operator while preserving its generalized eigenvalues exactly.  The
corresponding residual energy
`E_j(tau)=sum_k q_jk^2/(1+tau lambda_jk)` is a completely monotone spectral-
retention curve.  Neither isospectral rotation nor complete monotonicity is new
mathematics; their possible value is a stricter schema-semantic mechanism
protocol that separates eigenvalue shrinkage from assignment to field modes.

The ingredients—finite elements, mass and stiffness matrices, generalized
eigenproblems, spline smoothness, and graph Laplacians—are classical.  The
plausible gap is their tabular-specific composition with declared field
semantics, support estimation, chart covariance, and transport across modern
tabular backbones.  Focused searches found no paper making this exact
composition, but that is **not a novelty certificate**.  The current positive
evidence is concentrated in Adult/Black Friday mass effects and California
spatial stiffness.  Three unconditional TabReD temporal checks are negative;
the later residual-Riesz Weather pilot is small but positive beyond its shared
anchor.
