# IDEA TRIAGE — HELICES, FEATURE GEOMETRY, AND MODULAR RETRIEVAL

## Bottom line

None of the Day-8 ideas is ICLR-ready. The strongest new empirical observation
is a **post-hoc** real-panel gain from aggregate conditional-mismatch weighting,
but it is not the originally hypothesized candidate-reliability mechanism and
does not transfer cleanly to ModernNCA on the primary synthetic control.

## Hypotheses and outcomes

| Idea | Falsifiable hypothesis | Evidence | Decision |
|---|---|---|---|
| Addition helix and long training | longer convergence reveals a stable arithmetic manifold that is not present under short training | 2025 work already identifies and causally intervenes on a generalized helix; 2026 work already studies carry fibers, layer transitions, and convergence-dependent sharpening | do not reproduce as an ICLR novelty project |
| Helix for physical-law extrapolation | a periodic embedding alone improves out-of-range law extrapolation | not directly tested; a helix supplies phase/periodic coordinates, not conservation, equivariance, dimensional consistency, or a guarantee outside the training support | unsupported; would require a separately frozen physics benchmark and symmetry-aware baselines |
| Feature-specific geometries | nonlinear per-feature maps improve retrieval beyond added representation capacity | capacity-matched LocalWarp retrieval gave score changes `-0.0002` (TabR) and `-0.0035` (ModernNCA); PLE/PLR screens were mixed; shallow/standard/deep key interactions were not monotone | reject a generic feature-geometry claim |
| Modular compatibility × reliability | a candidate-only OOF reliability term complements symmetric learned distance | true reliability lowered the diagnostic but did not beat a permutation control; score wins were only 7/12 and 4/12 | reject the candidate-wise reranker |
| Full aggregate-risk weighting | preserving signed mismatch and squared noise weights improves prediction | post-hoc real gains `+0.01829` (TabR, 10/12) and `+0.01826` (ModernNCA, 9/12), but mismatch-only matched full and ModernNCA failed the S3 transfer gate | interesting post-hoc signal, stop under frozen rule |

## Addition and helix novelty boundary

[Language Models Use Trigonometry to Do Addition](https://arxiv.org/abs/2502.00873)
already reports a generalized helix for arithmetic and supports the geometry
with causal interventions. [The Shape of Addition](https://arxiv.org/abs/2606.03645)
goes further into carry-fiber structure, layerwise transitions, and the
difference between under-converged and converged representations. Therefore
“train longer and map the addition helix” is useful replication or pedagogy,
but not a defensible standalone ICLR contribution in 2026.

A helix is a reasonable coordinate system for a quantity with linear and
periodic components. That does not make it a physical law. Extrapolation needs
the relevant inductive structure—such as a conserved quantity, group
equivariance, a differential constraint, or dimensional consistency—and an
out-of-distribution split that rules out interpolation. No Day-8 result supports
a physics extrapolation claim.

## What “feature-specific geometry” meant here

LocalWarp learned eight monotone increments per numerical feature while keeping
the emitted representation dimension, branch width, and key capacity fixed.
PLE and PLR were also screened but were not treated as capacity-matched proof.
The tests separated prediction-only from retrieval-only changes and varied key
depth. This addresses per-feature shape and depth redundancy, but not every
possible choice of feature-specific embedding dimension.

The outcome argues against spending a dimension budget independently for every
feature without a stronger structural prior. Such a design multiplies capacity,
confounds geometry with parameter count, and is especially prone to fitting
sparse feature regions. A credible future test would need a fixed total
dimension budget, group sparsity or MDL-style selection, and held-out ranges per
feature. The present evidence does not justify that expansion.

## Epochs, model size, and data size

The 216-cell prospective panel used validation early stopping with a 48-epoch
cap. TabR stopped at mean 18.04 epochs (median 18, max 44); ModernNCA at mean
25.48 (median 24.5, max 48). Some ModernNCA cells therefore remain compatible
with a longer-training effect, but most models converged well before the cap
and the primary failure replicated across three split seeds and three model
seeds. This is evidence against a one-run undertraining artifact, not evidence
about frontier-scale models.

Training sets were capped at 8,192 rows in the prospective panel and 4,096 in
the synthetic follow-up. No data-scaling law, very-large-model claim, or
out-of-range physical extrapolation claim is made. The frozen stop rule was
reached before a scale sweep, so launching one after seeing these outcomes
would turn a failed screen into an unconstrained search.

## ICLR assessment

- Addition/helix atlas: novelty too low because the central geometry,
  convergence sharpening, and layer transitions are already occupied.
- Generic feature-specific embedding geometry: crowded and empirically mixed.
- Candidate-wise reliability retrieval: prospectively falsified.
- Aggregate-risk QP: real post-hoc signal, but standard bias-variance/QP
  ingredients, a changed mechanism, and incomplete cross-model transfer.

Status: **useful negative mechanism study; no current ICLR method claim.**
