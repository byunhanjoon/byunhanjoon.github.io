# Native Feature Geometry — closest-work and novelty subtraction

Status: **INITIAL AUDIT FROZEN BEFORE PILOT OUTCOMES**

## Four motivating papers

1. Kantamneni and Tegmark, [Language Models Use Trigonometry to Do
   Addition](https://arxiv.org/abs/2502.00873), infer a generalized helical
   number representation, propose a concrete Clock algorithm, and test it with
   component-level causal interventions.  Lesson used here: derive geometry
   from an algebra and require causal computation evidence.
2. Engels et al., [Not All Language Model Features Are
   Linear](https://arxiv.org/abs/2405.14860), formalize irreducible
   multidimensional features, discover circular features, and intervene on the
   circle rather than relying on visualization.  Lesson used here: a feature
   can be a multidimensional object, and representation claims need a targeted
   patch.
3. Kwon et al., [AI Engram: In Search of Memory Traces in Artificial
   Intelligence](https://arxiv.org/abs/2606.14997), translate specificity,
   reactivation, sufficiency, and necessity into a constrained geometric
   estimator and connect it to natural-gradient geometry.  Lesson used here:
   define an intervention by causal criteria; H5 patches only unseen rows, so
   specificity is exact and sufficiency remains empirical.
4. Park et al., [The Geometry of Categorical and Hierarchical Concepts in
   Large Language Models](https://arxiv.org/abs/2406.01506), derive simplices,
   hierarchical orthogonality, and direct sums under a causal inner product.
   Lesson used here: nominal, hierarchical, ordinal, and cyclic domains should
   not automatically share one linear chart.

## Direct collisions that must be subtracted

- Gorishniy et al., [On Embeddings for Numerical Features in Tabular Deep
  Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html), already establish
  piecewise-linear and periodic numerical embeddings as strong tabular
  interfaces.  Therefore Fourier or periodic encoding is not novel.
- Kim, Squires, and Ravikumar, [Knowledge-Enriched Machine Learning for
  Tabular Data](https://proceedings.mlr.press/v288/kim25a.html), already define
  concept kernels, spectral transformations, kernel-enriched algorithms, and a
  benchmark with category/column metadata.  Therefore “give a tabular model a
  semantic kernel” is not novel.
- Huang et al., [TabTransformer](https://arxiv.org/abs/2012.06678), already
  study learned contextual categorical-embedding geometry and robustness.
  Therefore learned category geometry or a CKA/t-SNE picture is not novel.
- Spectral embeddings, graph heat kernels, multidimensional scaling, kernel
  ridge regression, tree metrics, regular simplices, entity embeddings,
  Procrustes alignment, and permutation-equivariant lookup are established
  mathematical or engineering tools.
- The ICLR-2025 submission [Contemporary Continuous Aggregation: A Robust
  Categorical Encoding for Zero-Shot Transfer Learning on Tabular
  Data](https://openreview.net/pdf?id=kg4A2nGyY2) directly targets unseen
  categorical values without retraining, using cross-column empirical pair
  patterns to construct an unsupervised encoding.  Therefore zero-shot
  categorical extrapolation in tables is not novel.  NFG differs only in the
  source and target of transport: declared value geometry transports a task-
  trained neural lookup chart, rather than constructing an encoding from
  observed cross-column distributions.
- Zero-shot learning has long transferred predictors through semantic
  attributes and kernels; for example Elhoseiny et al.'s [Tell and
  Predict](https://arxiv.org/abs/1506.08529) predicts visual classifiers for
  unseen classes from a semantic text kernel.  Therefore semantic-kernel
  transfer to unseen entities is not novel in general.

## Residual candidate contribution

The only potentially distinct claim is the following composition:

> Treat information-equivalent tabular schemas as coordinate charts of a typed
> value metric space; measure chart-induced prediction risk; compile the same
> intrinsic geometry across flat or permuted schemas; and test whether that
> geometry supports a specific causal transport of learned embeddings to
> semantically related categories absent from training.

Even this is a novelty hypothesis, not a novelty fact.  A broader search must
still examine metric-aware categorical encoders, graph-regularized entity
embeddings, hierarchical/hyperbolic category encodings, cold-start
recommendation, and manifold regularization before any paper claim.

After H5/H6, the residual distinction narrows further: a typed metric over the
*values of one tabular feature* is used as a kernel to transport the task-
specific rows of an already-trained neural embedding table, with exact
observed-row specificity and a prospective metric-corruption dose response.
This is plausibly a useful tabular specialization of established kernel and
zero-shot transfer ideas, but current evidence does not justify a “first”
claim.

## Claims forbidden after the pilot

- “first metric-aware tabular embedding”;
- “first use of Fourier/tree/spectral features for tables”;
- “schema invariant” without distinguishing exact construction from empirical
  training stability;
- “causal geometry” based only on correlation or visualization;
- practical generality from synthetic data;
- novelty based on combining named existing components without a distinct
  theorem or consequence.
