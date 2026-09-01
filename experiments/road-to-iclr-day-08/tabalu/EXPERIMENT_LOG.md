# Experiment Log

## Phase A implementation and frozen protocol — 2026-08-31

**Hypothesis.** Separating learned program selection from exact arithmetic
execution produces flatter magnitude-extrapolation error than an MLP while
retaining competitive IID prediction.

**Setup.** Ground-truth numerical DAGs use 1–3 nodes and the frozen primitives
add, subtract, multiply, protected divide, abs, and square. The smoke panel is a
pipeline check. The decision panel contains 100 independently generated tasks,
five training seeds, independent train/validation/test draws, and OOD shells at
2x, 4x, and 8x. Baselines are linear regression, random forest, sparse degree-3
polynomial regression, MLP, and an EQL-style arithmetic network.

**Preregistered gate.** See `README.md` and `configs/phase_a_pilot.json`.

**Result.** Smoke v1 failed the gate. Random straight-through discovery collapsed
to single-feature shortcuts; see `NEGATIVE_RESULTS.md`. This is a pipeline and
optimizer diagnostic, not the frozen 100-task decision panel.

**Interpretation.** Exact execution has not yet been tested cleanly because the
induction optimizer did not recover the known computations. A constrained
complete chain search is now used to warm-start selectors, explicitly separating
the execution question from general-DAG search.

**Possible confound.** The first panel uses a restricted primitive library and
short programs. Passing is necessary but not sufficient for the broader claim.
The sparse polynomial baseline is not a substitute for the later PySR/SRBench
comparison.

**Next experiment.** Rerun the smoke panel with regenerated nondegenerate tasks
and the disclosed chain-search warm start. Run the frozen pilot only if compiled
programs retain their soft performance.

## Phase A decision panel — 2026-08-31

**Hypothesis.** Same as above.

**Setup.** The frozen 100-task/five-seed protocol was executed without omitted
cells. Confidence intervals were recomputed by resampling independent tasks,
with seed results averaged within task.

**Result.** GO. Compiled TabALU achieved mean 8× NRMSE 2.06e-20 versus 2.301 for
the MLP and 1.577 for random forest. The maximum compiled NRMSE over all shifts
was 1.67e-17. Feature F1 was 1.0, operator accuracy 0.948, and exact syntactic
graph recovery 0.79. All 16,000 planned metric cells are present and finite.

**Interpretation.** The exact executor preserves computation under magnitude
shift when complete constrained chain search identifies a functionally
equivalent program. The result does not rescue random differentiable induction.

**Possible confound.** Search and generation share the same short chain family.
The polynomial control is not a competitive general symbolic-regression system.
Five seeds are identical for the strongly initialized selector, so inference is
clustered by task and claims are restricted accordingly.

**Next experiment.** Phase B: add nuisance features, correlated alternatives,
target noise, and measurement noise; separate functional from syntactic recovery
and test whether the constrained search remains stable.

## Phase B structural stress — 2026-08-31

**Hypothesis.** Program search and compilation remain functionally stable under
moderate target noise, measurement noise, nuisance features, and correlated
alternatives.

**Setup.** Twenty tasks; five corruption seeds; clean, 10% target noise, 10%
measurement noise, four irrelevant features, and four correlated alternatives;
8× functional evaluation.

**Result.** Partial pass. Clean, target-noise, irrelevant-feature, and correlated
conditions passed. Measurement-noise mean NRMSE was 0.697 versus the frozen 0.25
maximum. Median was 0.061 and maximum 34.60. Feature F1 remained 0.952, showing
that selecting approximately the right variables was insufficient.

**Interpretation.** Protected division amplifies small operand errors and noisy
training can select the wrong equivalent-looking structure. Raw execution is not
measurement robust.

**Possible confound.** Search and executor share the chain family; correlated
features make literal graph identity non-identifiable. The measurement noise is
i.i.d. with independent latent operands, limiting how much any denoiser can
recover from a single row.

**Next experiment.** Phase C operand-ablation sweep with the exact graph fixed,
so denoising is evaluated separately from program-search mistakes.

## Phase C operand inference — 2026-08-31

**Hypothesis.** Conservative operand correction improves measurement-noise
robustness without becoming an unrestricted predictor or damaging clean data.

**Setup.** Twelve tasks, five seeds, fixed exact graphs, four required operand
variants, train noise 0.10, and test sweep 0–0.40 on IID and 8× inputs.

**Result.** NO-GO. Best conservative/raw NRMSE ratio at noise 0.20 was 0.930
(required ≤0.85). Clean conservative NRMSE was 0.113 (required ≤0.05). The
unrestricted encoder was not better.

**Interpretation.** Independent single noisy measurements are weakly
identifiable; learned corrections trade small noisy-IID gains for clean and OOD
bias.

**Possible confound.** No repeated, temporal, or correlated measurements were
available to identify latent operands. This is deliberate for the initial H2
falsification but narrows the negative claim.

**Next experiment.** Exclude the operand estimator and test H3 regime routing
independently against a conventional neural MoE.

## Phase D categorical regimes — 2026-08-31

**Hypothesis.** Routing to sparse executable experts improves shifted-regime
arithmetic prediction relative to one program and a conventional neural MoE.

**Setup.** Twelve tasks, five seeds, two distinct programs, noisy but separable
categorical context, 50/50 training and 20/80 8× testing.

**Result.** GO. Hard program-MoE OOD NRMSE was 4.01e-12, neural-MoE 0.954, and
single-program 1.700. Regime accuracy was 1.0 and operator recovery 0.933.

**Interpretation.** Correct routing preserves exact arithmetic extrapolation;
generic neural experts do not.

**Possible confound.** The context makes regimes nearly trivial to cluster.
This is a categorical mechanism pilot, not latent or temporal discovery.

**Next experiment.** Temporal coefficient shift with shared graph versus global,
independent, context-conditioned, and neural-MoE variants.

## Phase E temporal structure/parameters — 2026-08-31

**Hypothesis.** One invariant graph with regime constants is more stable and
sample-efficient than a global graph, separate graphs, context coefficients, or
neural MoE.

**Setup.** Sixteen tasks, five seeds, known labels for the structural controls,
~64 post-change rows, 5% target noise, future-only 8× evaluation.

**Result.** Partial. Shared future NRMSE was 0.00639, global 0.761, independent
0.00591, context-conditioned 0.1385, and neural MoE 4.248. Operator recovery
0.65 missed the 0.70 gate. Shared did not beat independent programs.

**Interpretation.** Discrete coefficients work; continuous time-conditioned
coefficients do not. Predictive stability is clear, but sample-efficiency and
literal structure claims are not.

**Possible confound.** Known regimes are supplied to symbolic controls, and
functional equivalence depresses syntactic operator accuracy.

**Next experiment.** Direct exact-versus-neural primitive execution ablation,
then heterogeneous typed execution.

## Direct exact-execution ablation — 2026-08-31

**Hypothesis.** Exact arithmetic nodes extrapolate more reliably than learned
neural approximations even when graph structure and operands are oracle-given.

**Setup.** Thirty Phase-A tasks, five seeds, fixed true graphs, exact protected
nodes versus one independently trained 64×64 MLP per node, plus the original
whole-function MLP. Evaluation covers 1×, 2×, 4×, and 8× magnitudes.

**Result.** GO. Exact execution had zero numerical error at all multipliers.
Neural primitives moved from 0.130 mean NRMSE at 1× to 6.475 at 8×, a 49.8×
increase. The whole MLP reached 3.461. All 1,800 records are finite and all
three frozen checks passed.

**Interpretation.** Learned arithmetic fails outside its training shell even
when discovery is removed as a confound. Exact execution is retained as a core
component.

**Possible confound.** The neural node budget and optimization are finite, and
the test uses the same short synthetic operator family as Phase A. Exact nodes
receive the true protected semantics by construction.

**Next experiment.** Phase F typed categorical, ordinal, and datetime operators
against embedding and manual-preprocessing controls, including family ablations.

## Phase F heterogeneous types — 2026-08-31

**Hypothesis.** Exact typed operators provide useful inductive biases beyond
learned embeddings or manual preprocessing on heterogeneous computations.

**Setup.** Sixteen tasks, five seeds, continuous/category/ordinal/time inputs,
1% fitting noise, and a joint future plus 4× magnitude shift. Compared a sparse
typed program, three operator-family removals, manual features + MLP, and learned
embeddings.

**Result.** GO on the matched synthetic library. Full typed NRMSE was 0.00088
IID and 0.00102 future. Manual-feature MLP future NRMSE was 0.559 and learned
embeddings 0.711. Removing each typed family increased future error by at least
163×. All 960 records were finite and all frozen checks passed.

**Interpretation.** Explicit type semantics are retained for real-data testing.
Every categorical, ordinal, and time family contributed materially in the
constructed setting.

**Possible confound.** Data generation and search use the same typed library;
the selector is bounded-cardinality OMP, categories are low-cardinality, and
neural budgets are finite. This is not evidence for arbitrary typed discovery.

**Next experiment.** Phase G penalized residual sweep on targets with increasing
non-symbolic fractions, then real temporal and general tabular panels.

## Phase G neural escape hatch — 2026-08-31

**Hypothesis.** A strongly penalized residual stays off for symbolic targets and
turns on smoothly as the true non-symbolic component grows.

**Setup.** Ten tasks, five seeds, five α values, pure program/MLP and three
residual variants, 2% fitting noise, clean IID and 4× tests.

**Result.** The frozen IID gate passed. Scalar residual usage rose from 0.000036
at α=0 to 0.688 at α=1 with 0.989 correlation, and reduced α=1 IID error to
20.7% of the pure program. However, the secondary OOD panel failed: at α=1,
pure-program/adaptive/scalar/unpenalized NRMSE was
0.287/1.587/2.356/3.851. All 2,500 records were finite.

**Interpretation.** Penalization preserves the exact branch in distribution but
does not make neural extrapolation safe. The residual is excluded from the
extrapolating core and may be reported only as a guarded fallback.

**Possible confound.** The non-symbolic term is bounded while symbolic programs
can grow at 4×, favoring omission under normalized error. That is nevertheless
the intended safety test: the learned residual should not catastrophically
overrule a safer base.

**Next experiment.** Real temporal evaluation with the unguarded residual shown
as a separate ablation, not silently folded into TabALU.

## UCI Bike Sharing temporal pilot — 2026-08-31

**Hypothesis.** A typed season-routed executable model degrades more gracefully
than global and conventional models from a 2011 IID holdout to all of 2012.

**Setup.** Official UCI archive with pinned checksum; five seeds; leakage
columns excluded; 2011 70/15/15 split and untouched 2012 future test; CatBoost,
XGBoost, Random Forest, typed-feature MLP, global program, and season router.

**Result.** NO-GO. Router IID/future NRMSE was 0.590/9.779. Global future was
1.734, XGBoost 0.602, CatBoost 0.605, and Random Forest 0.626. All 60 records
were finite, but every scientific gate failed.

**Interpretation.** Per-season elapsed-time polynomials fit local historical
trends and extrapolate catastrophically. Exact arithmetic does not make an
empirical time trend stable.

**Possible confound.** This is one public dataset, a hand-designed typed basis,
and a known season router rather than learned regimes. It refutes the current
mechanism, not every possible temporally regularized executable model.

**Next experiment.** Post-hoc bounded-time diagnostic, then independent temporal
datasets if accessible without unaudited credentials.

### Post-hoc bounded-time diagnosis

Removing unbounded elapsed-time terms reduced season-router future NRMSE from
9.779 to 0.782 and bounded-global NRMSE to 0.802. This supports the failure
mechanism but does not pass the original gate; the bounded router still trails
XGBoost by 30% and the intervention was chosen after test inspection.

## General numeric pilot — 2026-08-31

**Hypothesis.** A sparse exact numeric library remains broadly competitive and
does not catastrophically fail on ordinary regression and classification.

**Setup.** Diabetes, Wisconsin breast cancer, and binary wine; five seeds;
60/20/20 splits; six models; NRMSE or log loss as primary errors.

**Result.** Modest gate passed. Sparse-exact/best-baseline error ratios were
1.011, 1.158, and 1.462. Linear or logistic regression was best on all three.
All 90 records were finite.

**Interpretation.** The exact library is a compact non-catastrophic model here,
not a winner. Its additional interactions show no aggregate advantage.

**Possible confound.** Only three small, all-numeric datasets are included; no
formal across-dataset inference is valid.

**Next experiment.** Depth and regime-count scaling ablations, then consolidate
the surviving and rejected components.

## Depth and regime scaling — 2026-08-31

**Depth result.** NO-GO for discovery. Oracle execution remained exact, but
beam 4× NRMSE grew from 0.00044 at depth 2 to 1.027 at depth 8 and functional
recovery fell to 6.7%. All 720 records were finite.

**Regime-count result.** Partial. With known categorical labels, hard routing
reached 0.00375 OOD NRMSE at eight regimes versus 1.009 for neural MoE, but only
20% of task-seed mixtures met the strict 10⁻³ functional threshold. Operator
accuracy was 0.969. Four of five gates passed; all 960 records were finite.

**Interpretation.** Exact execution and labeled routing scale computationally;
program discovery and exact mixture recovery do not. Neither result supports
latent or temporal regime discovery.
