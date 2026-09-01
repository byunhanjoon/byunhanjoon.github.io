# Literature and novelty-subtraction audit

Audit date: 2026-08-29. The search covered publications and public preprints
available through August 2026. Queries covered piecewise-linear numerical
embeddings, similarity encoding, entity/categorical embeddings, unseen-category
generalization, hierarchy and ontology encodings, Laplacian and random-walk
graph embeddings, Nyström and RBF features, metric interpolation, Shepard and
partition-of-unity methods, cold start, and tabular structural encodings. This
is a collision audit, not proof that no closer unpublished or unindexed work
exists.

## Closest work matrix

| Work | Method | Input structure | Externally declared metric? | Unseen states evaluated? | Tabular tokenizer? | Arbitrary metrics? | Closest overlap with MPE | Remaining distinction |
|---|---|---|---|---|---|---|---|---|
| [Cerda, Varoquaux & Kégl, *Similarity encoding for learning with dirty categorical variables* (2018)](https://doi.org/10.1007/s10994-018-5724-2) | Similarities to all categories or selected prototypes | String categories | No; similarity comes from strings | Yes, including categories absent from training | Fixed categorical feature map | Similarity can be changed, but experiments are string based | Prototype similarities directly anticipate MPE's landmark coordinates and inductive handling of new values | MPE accepts an external metric and learns a low-dimensional token mixture; this is meaningful only if it improves on the identical similarity vector |
| [Mumtaz & Giese, *Hierarchy-based semantic embeddings for single-valued & multi-valued categorical variables* (2022)](https://doi.org/10.1007/s10844-021-00693-2) | Hierarchy similarities and semantic embeddings | Ontology/tree categories, including ICD-9 | Yes; an is-a hierarchy supplies structure | The method explicitly addresses new hierarchical values and evaluates MIMIC-III | Yes, hierarchy-specific | No | Direct prior work for ontology-aware categorical encoding, LCH/path/Wu-Palmer-style similarities, and MIMIC hierarchy prediction | MPE's only distinction is one unchanged tokenizer across non-hierarchy metrics; it is not the first hierarchy-aware unseen-category encoder |
| [Gorishniy, Rubachev & Babenko, *On Embeddings for Numerical Features in Tabular Deep Learning* (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | PLE and periodic numerical embeddings | Ordered real features | Intrinsic line/cycle charts, not an arbitrary supplied metric | Numerical interpolation, not disjoint semantic states | Yes | No | Triangular interval MPE reduces to piecewise-linear interpolation | MPE cannot claim local numerical bases or PLE; Proposition 7 is a relationship/boundary result |
| [Guo & Berkhahn, *Entity Embeddings of Categorical Variables* (2016)](https://arxiv.org/abs/1604.06737) | Supervised category lookup embeddings | Categorical IDs | No | Sparse categories, but a lookup has no semantic rule for a genuinely new ID | Yes | No | Learned per-category vectors are the standard token baseline and motivate MPE's trainable landmark tokens | MPE computes an unseen state's token from metric weights instead of requiring its own learned lookup row |
| [Kang et al., *Learning to Embed Categorical Features without Embedding Tables for Recommendation* (2020)](https://arxiv.org/abs/2010.10784) | Deep hash embedding | High-cardinality IDs | No; hash features are code-derived | Yes | Recommender feature encoder | No | Generates embeddings for unseen IDs without a lookup table | It supplies no semantic geometry, so its generalization is collision/code based rather than metric interpolation |
| [Williams & Seeger, *Using the Nyström Method to Speed Up Kernel Machines* (2001)](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf) | Landmark approximation of a kernel Gram matrix | Any valid kernel domain | A metric may define the kernel | Yes, by out-of-sample kernel evaluation | No | Kernel-dependent | Same landmarks and Gaussian kernel make Nyström a direct classical collision | MPE needs better accuracy, dimension, or efficiency under the same metric to justify the tokenizer |
| [Rahimi & Recht, *Random Features for Large-Scale Kernel Machines* (2007)](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) | Explicit randomized kernel features | Shift-invariant Euclidean kernels | Usually an analytic coordinate metric | Yes | No | No | Fixed RBF/Fourier feature maps predate MPE | MPE covers finite non-Euclidean metric spaces, but Gaussian/RBF ingredients are not novel |
| [Jung et al., *Efficient Partition-of-Unity RBF Interpolation for Coupled Problems* (2025)](https://doi.org/10.1137/24M1663843) and the [2026 RBF-PU review](https://www.sciencedirect.com/science/article/pii/S0378475426002910) | Shepard-normalized local RBF interpolation | Metric/Euclidean point clouds | Geometry supplies distances | Out-of-sample interpolation is central | No | Broad metric kernels in principle | Normalized nonnegative landmark weights and support-radius error logic are classical approximation machinery | The possible contribution is a supervised per-field tabular interface, not partition of unity, Shepard weighting, or RBF interpolation |
| [Trask et al., *Hierarchical partition of unity networks* (2022)](https://proceedings.mlr.press/v190/trask22a.html) | Learned partitions with local polynomial experts | Continuous approximation domains | Geometry is implicit in the domain | Approximation outside local samples | No | No | Neural partitions of unity already combine learned components | MPE uses a fixed declared metric and landmark-token mixture for a tabular field; it has a much narrower architecture |
| [Belkin & Niyogi, *Laplacian Eigenmaps* (2001)](https://papers.nips.cc/paper_files/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html) | Laplacian eigenvectors | Manifold/neighborhood graph | Yes, via graph/neighborhood structure | Transductive nodes can be embedded | No | Graphs/manifolds | A standard structure-only low-dimensional coordinate system for hierarchy/network states | MPE is inductive through distances to training landmarks; it must still beat spectral coordinates empirically |
| [Perozzi, Al-Rfou & Skiena, *DeepWalk* (2014)](https://research.google/pubs/deepwalk-online-learning-of-social-representations/) and [Grover & Leskovec, *node2vec* (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5108654/) | Random-walk node embeddings | Graph topology | Yes, topology only | Usually transductive known nodes | No | Graphs | Strong target-independent graph representations | MPE can query a new known-schema node by metric distance, but cannot claim graph embedding novelty |
| [Gokhale et al., *Compact Geometric Representations of Hierarchies* (COLT 2026)](https://proceedings.mlr.press/v336/gokhale26a.html) | Low-dimensional reachability embeddings with dimension bounds | Trees and bounded-treewidth DAGs | Hierarchy is externally supplied | Known hierarchy nodes, including retrieval queries | No | Hierarchies/DAGs | Shows hierarchy structure can have compact specialist embeddings and gives structural dimension guarantees | MPE offers a generic metric interface, not a stronger hierarchy representation theorem |
| [Franz et al., *Universal Embeddings of Tabular Data* (2025)](https://arxiv.org/abs/2507.05904) | Graph autoencoder entity embeddings aggregated to rows | Table-induced entity graph | No; structure is inferred from the table | Unseen samples composed of known/similar entities | Yes, task-independent row/entity representation | No | Target-independent geometry/embedding for heterogeneous tabular entities | MPE consumes a declared per-field metric and targets unseen field states, rather than learning a table-wide co-occurrence graph |
| [Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (2021)](https://papers.nips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | ResNet and FT-Transformer baselines | Ordinary mixed tables | No | No special cold-state protocol | Yes, backbone/tokenization | No | Establishes the MLP/ResNet/FT comparison standard | These backbones contribute no MPE novelty and are controls only |
| [Gorishniy, Kotelnikov & Babenko, *TabM* (ICLR 2025)](https://openreview.net/forum?id=Sd4wYYOhmY) | Parameter-efficient MLP ensembling | Ordinary mixed tables | No | No special cold-state protocol | Backbone | No | A particularly strong modern neural baseline | TabM is evaluation infrastructure, not part of MPE's contribution |
| [Hollmann et al., *Accurate predictions on small data with a tabular foundation model* (2025)](https://doi.org/10.1038/s41586-024-08328-6) | Prior-data fitted in-context predictor | Small/medium tables | No externally declared field metric | Distributional generalization, not schema-state interpolation | Whole-table model | No | Raises the contemporary tabular performance bar | It does not test the declared-metric field abstraction, but prevents broad tabular-SOTA claims |
| [STRABLE, *Benchmarking Tabular Machine Learning with Strings* (2026)](https://soda-inria.github.io/strable/) | 445 modular/end-to-end string pipelines over 108 tables | Raw string columns plus ordinary features | Surface strings/text encoders | Generalization across a broad public corpus | Yes | No | Shows lightweight string encoders plus strong learners are difficult baselines and that string type matters | MPE's three string tasks are only a secondary collision panel, not evidence for a broad string-learning claim |
| [Nobani, *Categorical variable encoding methods for tabular data: a benchmarking study* (2026)](https://doi.org/10.1007/s41060-025-00886-w) | Benchmark of 26 encoders, 13 datasets, 7 learners | General categorical columns | Varies by encoder | Not primarily state-disjoint | Yes | No | Broad encoder benchmarking reduces novelty from adding one more unqualified categorical encoder | MPE's defensible question is specifically externally declared geometry and state-disjoint induction |

## Novelty subtraction

The following are established and receive no novelty credit:

- Gaussian/RBF kernels, landmarks, Nyström approximations, normalized Shepard
  weights, partitions of unity, and support-radius interpolation arguments;
- piecewise-linear numerical encoding and periodic/Fourier feature bases;
- learned entity/category lookup embeddings;
- prototype similarity encoding of unseen dirty categories;
- hierarchy semantic similarity, ancestor/path encodings, and ontology-aware
  prediction;
- Laplacian, DeepWalk, and node2vec graph coordinates;
- MLP, ResNet, FT-Transformer, TabM, CatBoost, and LightGBM backbones.

The narrow residual candidate is therefore:

> A single per-field tabular interface that consumes a target-independent,
> externally declared metric, forms a training-landmark partition, learns a
> landmark-token mixture, remains exactly invariant under transported codebook
> relabelings, and can be evaluated inductively on unseen semantic states.

That wording is a candidate distinction, not yet a contribution. In the
primary case `m = D = 32`, normalized Similarity Encoding followed by an
unconstrained first linear layer can realize the same linear composition as
`wV`. MPE adds a factorization/optimization choice, not additional metric
information. Consequently, if MPE does not improve on the same-metric
Similarity Encoding or Nyström/RBF controls across several independent public
sources, the tokenizer itself is not empirically justified even if declared
geometry is useful.

## Claims that are forbidden by this audit

The project must not claim to be the first metric embedding, semantic
categorical embedding, hierarchy-aware unseen-category method, landmark/RBF
feature map, partition-of-unity interpolator, graph embedding, or categorical
encoder. It must also avoid a universal tabular-SOTA claim. The only possible
paper-level claim is the generic declared-metric field abstraction, and that
claim survives only if the frozen real-data gates support the MPE architecture
rather than merely supporting the use of geometry.
