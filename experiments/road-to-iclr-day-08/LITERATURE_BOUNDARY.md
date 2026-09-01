# LITERATURE BOUNDARY — RETRIEVAL RISK GEOMETRY

Audit date: 2026-08-31 (Asia/Seoul). Scope: work available through August
2026. This is a direction-search subtraction, not a claim of exhaustive
systematic review. The table distinguishes what is already occupied from the
narrow hypothesis tested on Day 8.

Legend: **A** = retrieval-risk law, **B** = nonlinear/local feature metric,
**C** = Transformer geometry, **D** = OrbitCover successor.

| Work | Learned geometry | Global or input-dependent? | Retrieval? | Nonlinear numerical warp? | Local risk theorem? | Neighbor noise in theory? | Retrieval vs prediction representation distinguished? | Closest overlap |
|---|---|---|---:|---:|---:|---:|---:|---|
| [NCA (Goldberger et al., 2004)](https://papers.nips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf) | supervised Mahalanobis distance optimized for stochastic leave-one-out kNN classification | global linear | yes | no | no conditional candidate-risk identity | no | no | A/B: directly learning label-useful neighbors is classical |
| [LMNN (Weinberger & Saul, 2009)](https://jmlr.org/papers/v10/weinberger09a.html) | PSD Mahalanobis metric trained by neighbor/push constraints | global, with cluster-local variants discussed | yes | no | margin objective, not the Day-8 local regression-risk law | no | no | A/B: supervised/global and multiple local metrics are occupied |
| [Parametric Local Metric Learning (Wang et al., 2012)](https://arxiv.org/abs/1209.3056) | a smoothly varying PSD metric field from anchor basis metrics | explicitly input-dependent | yes | effectively yes | metric-approximation bound, not target-mismatch plus candidate-noise risk | no | no | B/A5: smooth local metric fields long predate this project |
| [Risk-based adaptive metric learning (Zhang et al., 2015)](https://doi.org/10.1016/j.neucom.2015.01.009) | local feature weights from dimensionwise local LOOCV risk | input-dependent | yes | no | empirical-risk construction, not an exact conditional retrieval MSE decomposition | no | no | A/B: even “risk-based local neighbor metric” language is occupied |
| [TabR (Gorishniy et al., ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) | learned row encoder and key-only squared-L2 candidate similarity; retrieved labels/features enter a correction-like value module | global Mahalanobis for TabR-S's linear encoder; input-dependent for nonlinear encoder | yes | optional PLE/PLR input module | no | no explicit candidate aleatoric-noise term | partly: encoder/key retrieval module and prediction/value module are separate, but numerical embeddings normally feed the shared encoder | A/B: closest architectural baseline; deep retrieval distance is not new |
| [ModernNCA (Ye et al., ICLR 2025)](https://openreview.net/forum?id=JytL2MrlLT) | a deep embedding trained by a differentiable neighbor-weighted prediction objective | input-dependent with deep backbone | yes | supported through deep numerical embeddings | no | no explicit candidate-noise penalty | prediction is the neighbor-weighted output of the embedding; no Day-8 branch separation | A/B: independent retrieval paradigm and essential transfer baseline |
| [On Embeddings for Numerical Features (Gorishniy et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf) | per-feature PLE and periodic/PLR embeddings | nonlinear but not query-adaptive after encoding; induces local input metrics through the embedding Jacobian | no in the core paper | yes | no | no | no retrieval branch | B/C: nonlinear numerical geometry and PLE/PLR are occupied; gains also occur in MLPs |
| [Function Basis Encoding (Shtoff et al., TMLR 2024)](https://openreview.net/forum?id=M4222IBHsh) | numerical values mapped to chosen bases, especially B-splines | nonlinear per feature | no | yes | approximation/segment-function analysis, not retrieval risk | no | no | B: spline/function-basis encodings are occupied |
| [From Uniform to Learned Knots (Kumar et al., 2026 preprint)](https://arxiv.org/abs/2604.05635) | B-, M-, and I-spline encodings with uniform, quantile, target-aware, and end-to-end learned knots | nonlinear per feature | no | yes | no | no | no | B/C: learnable knots and spline comparisons across MLP/ResNet/FT-Transformer are occupied |
| [GGPL (Suh et al., KDD 2026)](https://www.lgresearch.ai/publication/view?seq=180) | GBDT-split-guided piecewise-linear breakpoints, differentiably fine-tuned and stochastically regularized | nonlinear per feature | evaluated as a plug-in to multiple tabular backbones, not defined by retrieval | yes | no | no | no explicit retrieval/prediction split | B: target-guided optimized breakpoints and practical piecewise-linear gains are occupied |
| [Unveiling the Role of Data Uncertainty (Kartashev et al., ICLR 2026 submission)](https://openreview.net/forum?id=AE98vmT1HJ) | analyzes numerical embeddings, ModernNCA, and ensembles through aleatoric/data-uncertainty regions; proposes LRLR embeddings | varies by analyzed model | includes ModernNCA | yes | no exact candidate-level retrieval-risk identity | uncertainty explains query-region gains, but there is no explicit retrieved-candidate noise-cost term | no | A/B: nearest prior to the uncertainty story; Day 8 must test candidate reliability, not merely “methods help in uncertain regions” |
| [AWARE retrieval-aligned tabular foundation models (Pham et al., 2026 preprint)](https://arxiv.org/abs/2604.01841) | supervised task-aligned embeddings and adapters for retrieving EHR examples for tabular in-context learning | learned nonlinear/global encoder, hence locally varying pullback metric | yes | not primarily a scalar warp method | no Day-8 conditional candidate-risk theorem | imbalance/heterogeneity motivate retrieval but no explicit additive candidate-noise term | yes at system level: retriever and inference model alignment are explicit | A: target/task-consistent retrieval and retrieval-inference alignment are already claimed |
| [Tab-PET (Leng et al., AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39445) | feature-graph Laplacian positional encodings concatenated to tabular feature tokens | dataset-global feature structure | no row retrieval | no | effective-rank theory, not neighbor risk | no | not applicable | C: generic structural/geometry-aware tabular attention is occupied |
| [Classical local bandwidth/kernel regression theory](https://doi.org/10.24546/E0003671) | locally adaptive smoothing bandwidth selected from bias/variance structure | input-dependent | kernel-neighbor smoothing | possibly through bandwidth/anisotropy | yes, asymptotic bias/variance theory | heteroscedastic variance is classical | no neural branch split | A/B: bias–variance and local smoothing are classical; novelty cannot rest on decomposition algebra alone |
| [Cross-validation with antithetic Gaussian randomization (Liu et al., JRSS-B 2026)](https://doi.org/10.1093/jrsssb/qkag073) | negatively dependent randomized folds for lower risk-estimation variance | not a data metric | no | no | risk-estimation theory | covariance enters directly | not applicable | D: generic covariance reduction by coupling is occupied |
| [Optimality of antithetic randomization for CV (Chattopadhyay et al., 2026)](https://arxiv.org/abs/2608.08089) | equicorrelated joint randomization; minimax/necessity results in stated regimes | not a data metric | no | no | yes | covariance is the central object | not applicable | D: broad covariance-optimal coupling language is unsafe |
| [Controlling antithetic variates (Kawai, 2026)](https://doi.org/10.1016/j.ejor.2025.08.027) | jointly optimized weights and control matrix for covariance-controlled antithetic estimators | not a data metric | no | no | closed-form variance optimization | covariance is explicit | not applicable | D: optimal covariance/weight control is already classical-modern Monte Carlo territory |
| [Randomized quasi-Monte Carlo integration (Owen, 2026 survey)](https://arxiv.org/abs/2608.17143) | randomized collectively space-filling integration designs | global design | no | no | rates and variance theory | not neighbor noise | not applicable | D: structured randomized integration is comprehensively occupied |

## Literature subtraction

The following are rejected novelty claims:

- nonlinear embedding before kNN is new;
- deep retrieval metric is new;
- target-consistent or task-aligned embeddings are new;
- geometry-aware tabular attention is generically new;
- learnable knots, monotone splines, B-splines, or guided breakpoints are new;
- local Mahalanobis/Riemannian metric fields are new;
- bias–variance decomposition for local smoothing is new;
- covariance-optimal or antithetic coupling is new.

## Narrow novelty that remains testable

The defensible candidate is a tabular-retrieval analysis, not a new metric
family:

1. state the finite-candidate conditional risk of a weighted retrieval
   predictor as the squared aggregate conditional-mean mismatch plus propagated
   candidate noise (and covariance when noises are dependent);
2. derive the oracle simplex weights and use the one-neighbor specialization as
   an interpretable target for neighborhood audits;
3. interpret a trained nonlinear TabR/ModernNCA encoder as a pullback metric
   field and test whether its distances rank the theoretical candidate risk;
4. separate prediction representation from retrieval representation, so any
   gain can be attributed to neighborhood geometry rather than ordinary
   nonlinear feature capacity;
5. test whether candidate reliability adds explanatory value beyond the 2026
   query-uncertainty analysis.

Even this remaining contribution is high prior-art risk. A viable paper needs
the mechanism to transfer across TabR and ModernNCA and to predict real-data
performance; the elementary identity alone is **not** enough.

## OrbitCover successor boundary

The requested same-target covariance-optimal reframing is not promoted.
Antithetic CV now has direct optimality results, control-variate work optimizes
covariance and weights, and RQMC/OA integration already supplies the general
structured-integration frame. OrbitCover remains differentiated only by its
declared finite semantic pipeline product and interaction-specific coupling.
That is the mature paper's existing boundary, not a stronger Day-8 successor.
