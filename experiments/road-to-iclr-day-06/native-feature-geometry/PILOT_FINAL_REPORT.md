# Day 6 Native Feature Geometry final report

Status: **FINAL PILOT VERDICT**

## Executive verdict

Retain a narrowed mechanism candidate: **Metric Chart Transport**.

Discard the stronger story that a neural lookup table visibly recovers its
feature's native Gram geometry.  The more precise result is that a declared
metric can act as a kernel for transporting a task-trained embedding chart from
observed to unseen feature values.  Correct transport is specific (observed
predictions are bitwise unchanged), beats mean/random/shuffled controls, and
degrades perfectly monotonically as the metric is corrupted.

This is not yet an ICLR lead.  The evidence is synthetic, full-rank, and close
to established kernel, knowledge-enriched tabular, cold-start, and zero-shot
transfer methods.

## Hypothesis lifecycle

| ID | Frozen verdict | Scientific decision |
|---|---|---|
| H1 Native Gram Equivariance | pass after logged analyzer precision correction | keep as construction theorem, not evidence |
| H2 Native Geometry Emergence | fail | discard global CKA recovery story |
| H3 Geometry–Schema Risk Coupling | frozen gate passes | change/demote: pooled interface confound |
| H4 Native Geometry Mitigation | fail | discard universal fixed-interface benefit |
| H5 Native Chart Transport | pass | keep as main causal mechanism |
| H6 Metric-Corruption Dose Response | pass | keep as strongest prospective consequence |

## Main result in plain language

Suppose categories have a meaningful metric—for example, adjacent hours on a
clock, neighboring ordinal levels, or nearby leaves of a hierarchy.  A neural
model learns embedding rows only for categories it sees.  We fit one affine map
from the metric coordinates of seen categories to those learned rows, then use
the map to fill unseen rows.  This is kernel ridge extension written in the
model's own learned embedding chart.

The intervention changes no trained weight except unseen lookup rows.  Hence
all seen-category predictions remain exactly unchanged.  If the metric is
correct, unseen predictions improve; if semantic rows are shuffled, they
worsen; if the metric is gradually corrupted, error rises gradually.

## Frozen H1–H5 evidence

The pilot contains 24 write-once bundles and 720 trained paths: four domains ×
two regimes × three seeds, with six interfaces and five schema charts in every
bundle.

### H1 — construction passes

The float64 native Gram compiler has maximum chart-aligned error 0 under the
`1e-10` gate.  Casting the training table to float32 produces maximum Gram
error `7.15e-7`, reported separately.  The first analyzer incorrectly applied
the float64 threshold to that float32 copy; `ANALYSIS_CORRECTION_LOG.md`
documents the correction without changing a gate or artifact.

### H2 — spontaneous geometry recovery fails

Unconstrained learned tables increase native CKA in every structured cell, but
the pooled median gain is only `.0357`, below `.10`; domain medians are `.0365`
(cycle), `.0426` (ordinal), and `.0290` (tree).  Final native CKA beats corrupt
CKA in 73.3% of paths, below 80%.  A useful task embedding need not reconstruct
the declared global Gram geometry.

### H3 — frozen pass, reviewer demotion

The frozen pooled correlations are strong: CKA versus held MSE `rho=-.822` and
CKA versus orbit damage `rho=-.764`, with all three domain correlations below
`-.40`.  But post-hoc stratification reveals `rho=-.121` within unconstrained
learned tables and `-.450` within native-tuned tables.  The pooled result mostly
distinguishes interface families.  H3 therefore cannot carry the mechanism.

### H4 — fixed native mitigation fails its boundary

On structured tasks, native-fixed beats label, learned, and random-fixed in
9/9 domain × seed cells, with 66.6% median held-MSE reduction against the best
baseline; it is also 4.4% better than the best learned/random interface in
interpolation and has essentially zero chart variance by construction.

However, it also improves nominal/random held MSE by 25.8%, violating the
frozen <5% negative-control boundary.  The universal mitigation claim is
falsified.  Correct native fixed geometry does beat corrupted geometry in 9/9
structured cells with 81.2% median reduction, retained only as a diagnostic.

### H5 — causal transport passes

For the primary native-tuned table, correct transport beats original, mean,
random, and shuffled patches in 9/9 structured cells.  Median held-MSE
reduction versus the best control is 76.4%; shuffled transport has median
`-435.5%` “reduction” versus original (severe worsening).  Maximum seen-
prediction change is exactly zero.  The nominal control wins 0/3 and worsens by
13.0% at the median.

The declared secondary unconstrained learned table is even stronger: correct
transport wins 9/9, reduces held MSE 91.2% versus the best control and 95.3%
versus its original unseen rows.  This matters because it does not require a
native initialization.

## H6 — prospective within-model dose response

H6 deterministically replayed 120 trained paths and applied 600 interventions.
For both unconstrained learned and native-tuned tables:

- correct-versus-shuffled endpoint wins: 9/9 cells;
- dose Spearman gate: 9/9 cells, every `rho` numerically 1.0;
- median correct-versus-shuffled reduction: 94.9% learned, 97.2% native-tuned;
- replay error: exactly zero;
- maximum seen-prediction change: exactly zero.

The nominal equality kernel is permutation invariant and its maximum relative
MSE range over dose is `4.35e-9`, passing the `1e-5` negative-control gate.
Endpoint reproduction differs from the original spectral basis by at most
`3.58e-7`, while the kernel-defined dose result is unchanged.

## Why the result is interesting

The motivating language-model papers link semantic algebra or hierarchy to a
representation and then intervene causally.  Here the analogous object is not
an activation circle or simplex but a chart transition:

`semantic metric -> kernel coordinates -> learned embedding chart -> unseen row`.

H2's failure and H5/H6's success sharpen the theory.  Global isometry is not
necessary.  A task network may warp semantic coordinates heavily; the metric
still supplies the correct interpolation relation for moving its learned chart
to a missing value.

## Novelty boundary

None of Fourier encodings, spectral kernels, category embeddings, kernel ridge
extension, cold-start prediction, or semantic zero-shot transfer is novel.
The closest tabular collisions include Gorishniy et al.'s numerical embeddings,
Kim et al.'s concept-kernel framework, TabTransformer geometry, and
Contemporary Continuous Aggregation for unseen category extrapolation.  The
residual candidate is the narrow composition of typed *value* geometry,
transport of a task-trained neural lookup chart, exact intervention
specificity, schema-risk accounting, and a metric-corruption dose response.

That residual is plausible but not established as publication-level novelty.

## Evidence and audit

- 36 complete bundles;
- 840 trained paths (720 pilot + 120 deterministic H6 replays);
- 600 H6 interventions;
- 624.0 summed fit-seconds;
- artifact hashes, menus, shapes, and finiteness pass with zero errors;
- five analysis outputs reproduce byte-for-byte;
- 5/5 construction and intervention tests pass.

## Decisive next experiments

1. Test true alternative schemas: flat leaf ID versus hierarchical path
   columns, cyclic cut choices, ordinal bin refinements, and redundant
   factorizations—not only label permutations.
2. Use real feature metrics frozen without target access: time cycles,
   geographic/administrative trees, medical or product taxonomies, and ordinal
   stages, under temporal unseen-value splits.
3. Compare directly with CCA, concept-kernel/KE-TALENT methods, kernel ridge on
   the target, graph/Laplacian regularization, target encoding, and text-derived
   category embeddings.
4. Test reduced ranks, missing/noisy/wrong metrics, uncertainty, classification,
   multiple neural backbones, and more than three source units.
5. Require benefit beyond a standalone kernel predictor; otherwise inserting
   transported rows into a neural embedding table is unnecessary machinery.
