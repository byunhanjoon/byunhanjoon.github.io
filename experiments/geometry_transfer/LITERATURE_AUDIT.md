# Literature audit — Geometry Transfer Law

Search completed through 2026-08-29. Queries covered graph signal
reconstruction/denoising, kernel alignment, spectral smoothing, harmonic
extension, cold start, unseen categories, hierarchical/random-effects models,
group CV, spatial prediction, kriging, and similarity encoding. The subtraction
below is intentionally conservative: generic bias–variance, spectral SNR,
kernel alignment, graph interpolation, kriging, and group CV are not novel.

Legend: `Z` = arbitrary remaining tabular covariates; `external` = geometry is
not learned from the prediction target; `cold` = complete semantic states are
unobserved in training; `residual` = conditional state signal after `Z` is made
explicit; `law` = exact comparison to the zero-residual fallback including
state-effect estimation covariance.

| Work | Setting / main result | Z | External | Cold | Residual | Exact help–harm law | Heterogeneous field geometries |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [Zhu, Ghahramani & Lafferty (2003)](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf) | Gaussian fields; unlabeled-node predictions are harmonic functions | No | Usually | Yes, nodes | No | No | No |
| [Belkin, Niyogi & Sindhwani (2006)](https://jmlr.org/papers/v7/belkin06a.html) | Manifold regularization combines ambient and graph RKHS penalties | Limited | Often data-built | Transductive | No | No | No |
| [Smola & Kondor (2003)](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_12) | Diffusion/regularization kernels on graphs | No | Yes | Nodes | No | No | No |
| [Chen & Liu (2017)](https://arxiv.org/abs/1706.00544) | Explicit graph-Laplacian bias–variance and spectral SNR scaling | No | Yes | Sampled graph signal | No | Closely related smoother risk, not tabular fallback formulation | No |
| [Sadhanala, Wang & Tibshirani (2016)](https://proceedings.mlr.press/v51/sadhanala16.html) | Statistical/computational behavior of Laplacian smoothing | No | Yes | Denoising | No | No cold-state fallback law | No |
| [Chen et al. (2015)](https://arxiv.org/abs/1503.05432) | Perfect/robust recovery of bandlimited graph signals under sampling conditions | No | Yes | Sampled nodes | No | No | No |
| [Puy & Pérez (2018)](https://academic.oup.com/imaiai/article/7/4/657/5122685) | Stable structured sampling/reconstruction; graph cumulative coherence controls sample count | No | Yes | Sampled nodes | No | No | No |
| [Cristianini et al. (2001)](https://papers.neurips.cc/paper/1946-on-kernel-target-alignment.pdf) | Kernel–target alignment and concentration | No | Kernel candidate | Transductive | No | No estimation-noise threshold | Kernels, not fields |
| [Rasmussen & Williams (2006)](https://gaussianprocess.org/gpml/chapters/) | GP regression/equivalent kernels; predictive mean and covariance | Yes via mean functions | Kernel supplied/learned | Locations | Sometimes | GP risk machinery subsumes special cases | Primarily Euclidean/kernel inputs |
| [Cressie-style universal spatial prediction](https://academic.oup.com/jrsssb/article/54/3/813/7035745) | Minimum-MSE linear interpolation for spatial processes with correlated errors | Yes | Spatial covariance | Locations | Regression residuals | Operator-specific optimum, not cross-geometry decision law | Spatial only |
| [Cerda, Varoquaux & Kégl (2018)](https://doi.org/10.1007/s10994-018-5724-2) | Similarity coordinates for dirty high-cardinality categories | Yes | String similarity | Partly | No | No | Strings |
| [How to Learn Item Representation for Cold-Start Multimedia Recommendation? (2020)](https://doi.org/10.1145/3394171.3413628) | Learns side-information representations for cold items | User context | Learned content | Yes, items | No | No | Recommender only |
| [Cold-start HIN with side information (2023)](https://doi.org/10.1016/j.future.2023.07.003) | Heterogeneous graph/meta-path embeddings for cold items | User/item | Partly learned | Yes | No | No | Recommender graphs |
| [Qiu et al. (2025)](https://doi.org/10.1177/09622802251345486) | Cluster-aware criterion approximates leave-one-cluster-out deviance | Yes | N/A | Clusters | N/A | No geometry | No |
| [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) | Cross-validation strategies for spatial/hierarchical dependence | Yes | Spatial grouping | Blocks/groups | No | No | Spatial/ecological |

## What the spectral corollary does not claim

Chen and Liu already give a graph-Laplacian bias–variance/SNR analysis, and
graph sampling theory already characterizes stable recovery from observed
nodes. Theorem 3 here is a transparent corollary and pedagogical bridge, not a
new graph-signal theorem. Likewise, the trace variance in Theorem 1 is standard
linear-estimator risk algebra, kernel alignment already formalizes target–kernel
agreement, and GP/kriging theory already combines interpolation bias and
uncertainty for specified covariance models.

## Narrowest defensible residual

No close work found in this audit combined all of the following in one tested
formulation:

```text
arbitrary ordinary tabular covariates Z
+ a target-independent geometry attached to one semantic field
+ complete cold states
+ residualization to isolate conditional state contribution
+ an operator-agnostic exact zero-fallback risk decomposition
+ a no-metric-only sign theorem
+ prospective tests across geographic, graph, hierarchy, and string fields.
```

The algebra alone is not a defensible novelty claim. The possible contribution
is the conditional cold-state tabular formulation, the explicit
transfer-versus-state-estimation accounting, the impossibility result, and the
cross-geometry empirical program. Even this must be presented as a synthesis
and problem formulation built from established graph smoothing, kernel,
spatial, hierarchical, and validation ideas.

## Search limitations

The space spans several communities and terminology is inconsistent. This was
a targeted audit rather than a formal systematic review. The 2026 cutoff was
searched explicitly; no paper found invalidates the narrow distinction above,
but absence from this audit is not proof of novelty. A submission should add a
domain-expert citation pass, especially for empirical-Bayes small-area
estimation, multi-task relatedness, and recent strict cold-start recommender
systems.
