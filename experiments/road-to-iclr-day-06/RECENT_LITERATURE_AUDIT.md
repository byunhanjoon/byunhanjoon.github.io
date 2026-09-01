# Day-6 recent-literature audit

Audit date: 2026-08-28; H5/H9 extensions audited 2026-08-29.  This file records claim boundaries, not proof of
novelty.  Searches covered tabular permutation invariance, floating-point
non-associativity, deterministic GPU reductions, numerical training
instability, equivariant stochastic paths, and precision-dependent training.

## Closest collisions

### Bit-level perturbations can cause macroscopic neural-training differences

Sun, Lao, Yezzi, and Sundaramoorthi, *A PDE-based Explanation of Extreme
Numerical Sensitivities and Edge of Stability in Training Neural Networks*
(JMLR 2024), explicitly compare mathematically equivalent optimizer updates
that differ at the last floating-point bit, fix initialization/batch randomness
and nondeterminism, and observe amplified path and test-accuracy differences.
They also study learning-rate stability boundaries and float64.

Boundary: Day 6 cannot claim discovery that roundoff is amplified, that exact
real-arithmetic identities can diverge in finite precision, or that a common
random tape isolates this phenomenon.  Its defensible distinction is an exact
*semantic schema conjugacy* in a supervised tabular pipeline, prediction-orbit
measurement across neural architectures, and elimination by precision applied
only at the schema-facing affine interface.

Source: https://www.jmlr.org/papers/v25/23-0137.html

### Floating-point non-associativity and deterministic GPU execution

Shanmugavelu et al., *Impacts of Floating-Point Non-Associativity on
Reproducibility for HPC and Deep Learning Applications* (2024), analyze GPU
reductions, deterministic alternatives, PyTorch nondeterministic operations,
and performance costs.  Yashwanth, *On the Structure of Floating-Point Noise
in Batch-Invariant GPU Matrix Multiplication* (2025), shows that GPU matmul
error is structured rather than well modeled as IID static and emphasizes
kernel/reduction-order effects.

Boundary: Day 6 cannot claim discovery of GPU reduction-order error,
deterministic execution, or structured matmul noise.  H2's failed nominal-
precision law is consistent with the warning that effective numerical noise is
kernel-structured.  H1's experiment is deliberately deterministic: different
schema coordinate orders alter a single reduction while the kernel and random
tape are fixed.

Sources: https://arxiv.org/abs/2408.05148 and
https://arxiv.org/abs/2511.00025

### Higher-precision accumulation and selective precision are established

Sakr et al., *Accumulation Bit-Width Scaling for Ultra-Low Precision Training
of Deep Networks* (ICLR 2019), analyze precision requirements for partial-sum
accumulation during training.  El Arar et al., *Mixed precision accumulation
for neural network inference guided by componentwise forward error analysis*
(2025), derive componentwise error/conditioning criteria and selectively
recompute sensitive output components in higher precision.  Standard mixed-
precision training already uses wider accumulators and master weights.

Boundary: IEA64 cannot claim invention of wider inner-product accumulation,
layer/component-selective precision, or using precision to improve numerical
stability.  Its remaining distinction is the *choice of interface from a
known semantic group action* and the exact schema-orbit estimand: it changes
only the schema-facing affine accumulator to test whether two algebraically
conjugate training paths commute.  H9 can add a post-breach consequence on its
untouched panel, not a generic mixed-precision contribution.

Sources: https://openreview.net/pdf?id=BklMjsRqY7 and
https://arxiv.org/abs/2503.15568

### Permutation invariance is established tabular structure

HyTrel (NeurIPS 2023) proves a maximal-invariance result for hypergraph table
representations.  EquiTabPFN (NeurIPS 2025) enforces target-permutation
equivariance and removes an equivariance gap.  TabPFN's established inference
pipeline also ensembles feature and label permutations to approximate missing
invariances.

Boundary: Day 6 cannot claim discovery that tables admit row/column/label
symmetries, that invariant/equivariant architectures are desirable, or that
permutation ensembling helps.  Day 6 studies a different object: pathwise
commutation of a trained dense pipeline after parameters are conjugated so the
initial real-valued function is already identical.  It is not proposing a new
set architecture.

Sources:
https://proceedings.neurips.cc/paper_files/paper/2023/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html,
https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html,
and https://doi.org/10.1038/s41586-024-08328-6

### Learned feature ordering is now an explicit tabular method

DynaTab (Habib, Doretto, and Adjeroh, PMLR 2026 workshop proceedings) treats
feature order as a learnable, performance-relevant design choice.  It uses
dynamic feature reordering, positional embeddings, gating, and masked
attention, and reports a broad 36-dataset comparison.

Boundary: Day 6 cannot claim that feature order is newly recognized as
important for tabular neural networks, nor can a three-dataset numerical study
compete with DynaTab's predictive breadth.  DynaTab changes the learned
representation and objective; Day 6 instead compares exactly conjugated
parameters whose initial real-valued functions match and asks whether the
training path commutes in finite precision.  The distinction is causal, not a
feature-order accuracy claim.

Source: https://proceedings.mlr.press/v308/habib26a.html

### Attribute permutations and conjugate training dynamics are established diagnostics

Xie et al., *Testing and Validating Machine Learning Classifiers by Metamorphic
Testing* (JSS 2011), explicitly use attribute-order and class-label
permutations as metamorphic relations for supervised classifiers.  Redman et
al., *Identifying Equivalent Training Dynamics* (NeurIPS 2024), use topological
conjugacy and Koopman spectra to identify equivalent and nonequivalent neural
training dynamics.

Boundary: Day 6 cannot claim invention of feature permutation as a supervised
metamorphic test, nor of conjugacy as a language for neural training-dynamics
equivalence.  Its narrower construction fixes a known algebraic parameter
conjugacy so the two tabular predictors are already the same real-valued
function, then treats finite-precision *failure* of that training-path
commutation as the object of study.  H4/H5 add only a prospectively tested
forecasting use of the resulting numerical shadow.  These close conceptual
collisions reinforce the frozen novelty cap even if a successor gate passes.

Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144/ and
https://arxiv.org/abs/2302.09160

### Equivariant stochastic trajectories exist outside training

Work on group-equivariant diffusion models observes that stochastic sampling
paths require transformed/equivariant noise, not merely an equivariant drift.
Probabilistic-symmetry and emergent-equivariance literature more broadly
characterizes stochastic equivariance and ensemble averaging.

Boundary: Day 6 cannot claim the abstract idea that randomness must transform
under a group action.  Its stochastic tape is a control rather than the method;
H1's remaining mismatch is deterministic interface arithmetic after the tape
is matched.

Representative source: https://openreview.net/pdf?id=65XylEuDLB

### Symmetry-orbit alignment during training is an active diagnostic target

Amarel et al., *Loss Landscape Geometry of Partial Differential Equation
Emulators: Or, Symmetry Learning via Gradient Alignment* (PMLR 2026), measure
gradient alignment between symmetry-related examples and connect coherent
updates across physical symmetry orbits to learned equivariance.

Boundary: Day 6 cannot claim that comparing optimization behavior across a
symmetry orbit is itself new.  The adjacent work studies learned physical
equivariance and gradient geometry; Day 6 begins from an exactly conjugated
real-valued tabular function and isolates finite-precision failure of pathwise
commutation.  H8's possible distinction remains its prospective numerical
schema-orbit screen, not the general use of orbit diagnostics.

Source: https://proceedings.mlr.press/v334/amarel26a.html

### Early dynamics and prediction multiplicity are established targets

Hu et al., *Latent State Models of Training Dynamics* (TMLR 2023), model
multi-seed training trajectories and show that seed instability depends on
hyperparameters and can manifest early.  Frankle et al., *The Butterfly
Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial
Conditions* (2025), use controlled parent-child perturbations to localize
training sensitivity.  Hamman et al., *Quantifying Prediction Consistency
Under Fine-tuning Multiplicity in Tabular LLMs* (ICML 2025), use local
embedding-space stability to anticipate prediction consistency across
fine-tuned tabular language models without retraining the entire model family.
Repeated-seed prediction instability on clinical tabular data is also directly
documented by Lopez-Martinez et al. (ML4H 2022).

Boundary: H5 cannot claim discovery that early dynamics forecast later
training behavior, that controlled perturbations probe sensitivity, or that
tabular prediction multiplicity can be estimated cheaply.  Its narrower
candidate distinction is *cross-source* transfer: an exactly
function-matched schema-coordinate perturbation isolates deterministic
interface arithmetic and is tested as a two-epoch, label-free configuration
probe for later independent-seed prediction variance.  This distinction is
only meaningful if the prospectively frozen H5 gates pass.

Sources: https://arxiv.org/abs/2308.09543,
https://openreview.net/pdf?id=L1Bm396P0X,
https://openreview.net/forum?id=AXJnqocQpm, and
https://proceedings.mlr.press/v193/lopez-martinez22a.html

### Lyapunov and perturbation-growth diagnostics are not new

Finite-time Lyapunov analysis is established for layer/input dynamics, RNN
state dynamics, reinforcement learning, and neural ODE robustness.  Work on
network trajectories along training also explicitly measures separation of
nearby networks and connects positive exponents to training regimes.  The 2026
*Leveraging chaotic transients in the training of artificial neural networks*
uses maximum network Lyapunov exponents to locate sensitive optimization
regimes.  Sun et al.'s numerical-sensitivity work already establishes that
optimizer instability can be localized in time and parameter space.

Boundary: H6 is not an estimator of a formal tangent-space Lyapunov spectrum,
and it cannot claim novelty for exponential perturbation growth, finite-time
exponents, or early instability detection.  It deliberately fits a simple
finite-difference prediction-orbit slope and calls it a *screen*.  Its only
possible contribution is prospective evidence that an exact, semantically
defined tabular schema orbit predicts delayed 200-epoch arithmetic divergence
better than its 20-epoch level.  Even a pass would be a practical consequence
of H1, not standalone dynamical-systems novelty.

Frankle et al.'s *The Butterfly Effect* further reports that separation rates
depend strongly on task/model and need not behave like a simple dynamical
system.  That observation directly collides with a broad exponential-screen
claim and makes H6's FreMTPL/MLP false positive a mechanism failure rather
than mere tuning noise.  H8's modal-mixture proposition is therefore only a
restricted motivation; a gate pass would validate its fixed two-branch rule
on this panel, not establish universal log-convex optimizer dynamics.

Sources: https://www.frontiersin.org/journals/complex-systems/articles/10.3389/fcpxs.2024.1367957/full,
https://research.chalmers.se/publication/539884/file/539884_Fulltext.pdf,
https://doi.org/10.1103/t5p9-kv5w, and
https://www.jmlr.org/papers/v25/23-0137.html

### Probabilistic inner-product roundoff bounds are established

Ipsen and Zhou, *Probabilistic Error Analysis for Inner Products* (SIMAX
2020), derive nonasymptotic probabilistic forward-error bounds for sequential
inner products using bounded zero-mean roundoff variables, conditional mean
assumptions, martingales, and Azuma's inequality.  Higham and Mary develop a
broader probabilistic rounding-error framework, while stochastic-rounding work
provides variance and concentration bounds for inner products and other
kernels.  Rigorous tools also propagate distributions of probabilistic inputs
through floating-point expressions.

Boundary: H7 cannot claim invention of probabilistic dot-product analysis,
roundoff martingales, `sqrt(n)u`-type improvements, stochastic rounding, or the
general idea of bounding boundary-crossing probability.  Proposition 9 is an
elementary sufficient bounded-phase bridge tailored to two deterministic
coordinate orders; it is not presented as a numerical-analysis advance.  H7's
only possible novelty is the exact tabular schema conjugacy, survival framing
over learned optimizer trajectories, and its prospectively split long-horizon
boundary map.

H9 likewise cannot claim the generic fact that reducing arithmetic error or
raising precision can reduce downstream numerical discrepancy.  Its only
testable distinction is the paired *post-breach* boundary: after an exact
schema-conjugate training path has already separated, does interface-only
float64 still attenuate final semantic orbit error on 25 untouched bundles?
This is a successor clause inside H1/H7, not a numerical-analysis novelty.

A targeted H9 search on 2026-08-29 also checked low-precision training and
floating-point-network theory.  Gupta et al. (ICML 2015) establish that the
rounding scheme materially changes low-precision training; Ozkara et al.
(AISTATS 2025) analyze stochastic rounding for stable low-precision training;
and Hwang et al. (ICML 2025) study which discrete functions floating-point
networks can represent.  These strengthen the generic collision above.  The
search found no paper directly evaluating H9's paired intervention after an
exact schema-conjugate orbit has already breached, but absence from this
targeted search is not evidence of novelty.  H9 therefore retains only the
narrow empirical distinction and receives no novelty credit for higher
precision, rounding control, or floating-point-aware neural analysis itself.

Sources: https://epubs.siam.org/doi/10.1137/19M1270434,
https://epubs.siam.org/doi/10.1137/18M1226312,
https://epubs.siam.org/doi/10.1137/22M1510819, and
https://arxiv.org/abs/2105.13217.  Additional H9 audit sources:
https://proceedings.mlr.press/v37/gupta15.html,
https://proceedings.mlr.press/v258/ozkara25b.html, and
https://proceedings.mlr.press/v267/hwang25b.html

## Pre-H3 defensible claim (historical freeze)

The narrow candidate claim after H1/H2, before the long matrix matured, was:

> For exact tabular schema conjugacies, real-arithmetic functional matching and
> a shared stochastic tape do not imply pathwise semantic commutation in finite
> precision.  In a dense-stem FT-Transformer, coordinate-order roundoff at the
> first affine interface alone is amplified into macroscopic prediction-orbit
> error; changing only that accumulator to float64 closes the entire paired
> training path.  The first prospective full-scale result further shows that a
> ResNet can undergo delayed symmetry breaking after appearing stable for 20
> epochs.

At that stage, the strongest novelty was the causal localization and exact closure, not the
underlying numerical-instability fact.  H1/H2 evidence is limited to 2,048
training rows and 20–30 epochs; H3 has only two completed all-row cells and is
not adjudicated.  Predictive accuracy is not improved on average by theorem or
then-current evidence.  H3 and its successors supersede this partial snapshot;
the final report must use their complete summaries rather than this paragraph.

## ICLR risk assessment before H3

- mathematical/causal clarity: high;
- empirical effect magnitude in FT-Transformer: high;
- breadth across datasets: preliminary but positive;
- architecture breadth: the mechanism is deliberately non-universal;
- predictive utility: weak;
- novelty after subtracting JMLR 2024 and numerical-reproducibility work:
  moderate;
- likely framing if H3 passes: reproducibility/robustness and numerical
  semantics, not tabular predictive SOTA;
- biggest rejection risk: “a valuable tabular case study and engineering fix,
  but the central instability is already known.”

H1 should therefore not automatically displace OrbitCover as the strongest
paper direction.  A further iteration needs either a new theorem with a useful
decision consequence, a low-overhead general compiler-like intervention, or a
downstream reliability result reviewers care about.
