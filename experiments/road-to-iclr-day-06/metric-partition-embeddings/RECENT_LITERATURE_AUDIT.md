# Literature and collision audit

## Inspiration translated into the tabular question

- [Language Models Use Trigonometry to Do Addition](https://arxiv.org/abs/2502.00873)
  finds a causally relevant helix/clock-like representation for arithmetic.
  The transferable lesson is that the right coordinates can reflect the
  algebra of a variable rather than its printed scalar code.
- [Not All Language Model Features Are Linear](https://arxiv.org/abs/2405.14860)
  finds irreducible circular features for weekdays and months.  The tabular
  analogue is direct: an hour or weekday should not be forced onto an interval
  with two unrelated endpoints.
- [AI Engram](https://arxiv.org/abs/2606.14997) insists on causal criteria such
  as sufficiency and necessity rather than attractive geometry alone.  Here the
  corresponding test is a controlled metric intervention: correct and
  corrupted metrics share capacity but produce different unseen-state risk.
- [The Geometry of Categorical and Hierarchical Concepts in Large Language
  Models](https://arxiv.org/abs/2406.01506) relates categorical/hierarchical
  semantics to geometric structure.  The tabular translation is to compile a
  declared category metric or ontology into the feature tokenizer.

## Direct collisions

1. [On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)
   already establishes PLE and trainable periodic encodings.  This track cannot
   claim that nonlinear numerical embeddings, local bases, or trigonometric
   features are new.
2. [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
   establishes explicit feature maps for radial kernels.  Gaussian landmark
   weights are classical kernel machinery.
3. [Function Basis Encoding of Numerical Features in Factorization
   Machines](https://openreview.net/forum?id=M4222IBHsh) systematizes
   user-chosen function bases for numerical features.  A “choose a new basis”
   claim is too broad.
4. Graph positional/structural encodings already use shortest-path, heat,
   diffusion, and Laplacian geometry.  Day 4 FieldRiesz and Day 6 Native Feature
   Geometry in this repository already explored spectral versions.
5. Day 2 already tested local information-equivalent PLE coordinates; Day 3
   tested mixed-measure PLE; Day 4 tested support-aware, spectral/heat,
   universal-rank, and multi-chart variants.  None may be renamed as new.

## Residual novelty, if any

MPE's ingredients are not novel.  Its narrow possible contribution is a
tabular interface and evaluation principle:

```text
raw typed value + declared metric
  -> normalized landmark partition
  -> learned landmark token
  -> exact invariance to equivalent code schemas
  -> corrupted-metric causal control
  -> safe interval/nominal boundaries.
```

The current search did not find this exact combination as a modern per-field
tabular tokenizer, but absence from a search is not novelty proof.  Because
fixed Fourier features beat MPE on the real cyclic diagnostic, the credible
paper claim would be metric-space interpolation for typed discrete fields—not
better cyclic encoding in general and not a universal PLE replacement.
