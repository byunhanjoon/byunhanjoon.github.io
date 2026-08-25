# Day 3 related-work boundary

This note records what the project may and may not claim. It was written before
the broad benchmark outcomes were inspected.

| Line of work | Established result | What Day 3 must not claim | Remaining empirical question |
|---|---|---|---|
| Natural gradient | Approximate invariance to smooth invertible parameterizations | Invariant optimization is new | Does the ideal property survive finite tabular training? |
| K-FAC | Idealized K-FAC is invariant to arbitrary affine input and hidden-activity transformations | First-layer covariance preconditioning or affine invariance is new | How do damping, initialization, minibatches, and partial K-FAC affect the result? |
| VectorAdam / isometric optimizers | Coordinate-wise Adam is not rotation equivariant; vector-wise or isometric updates can remove particular rotation/linear-coordinate biases | Discovering that Adam has coordinate-system bias is new | Does this bias create a systematic schema-equivalence problem in supervised tabular prediction? |
| PRONG / whitening | Reparameterizing networks by whitening activations can improve conditioning | Whitening neural inputs or layers is new | Does whitening remove scale only, or also arbitrary basis sensitivity in practice? |
| Shampoo / SOAP | Full-matrix/Kronecker preconditioning and Adam in a Shampoo eigenbasis are established | Full-matrix adaptive optimization is new | Are these practical optimizers invariant enough on tabular representation orbits, and at what cost? |
| Canonicalization | Canonicalization is a general route to invariance, with known continuity limitations | Canonicalization as a general idea is new | Is a train-row anchor construction a useful exact diagnostic or deployable tabular transform under rank changes and shift? |
| Numerical embeddings | PLE and periodic numerical embeddings can substantially improve tabular DL | PLE or numerical embeddings are new | Can two information-equivalent encodings learn differently solely because of their basis? |
| Tabular orientation / rotation | Grinsztajn et al. test feature rotations and argue that tabular learners should preserve meaningful feature orientation | The broad observation that feature rotations or orientation matter in tabular learning is new | Can conditioning be isolated causally within exactly equivalent numerical, nominal, ordinal, and cyclic representation orbits, and can optimizer invariance close the resulting gaps? |
| Stretch transformation | Supervised and unsupervised transformations can change feature geometry to make target functions smoother | Feature transformation for tabular learning is new | Unlike stretch, do target-independent, exactly equivalent recodings expose optimizer artifacts without changing target geometry? |
| Rotation Forest | Full PCA rotations of random feature subsets can create accurate and diverse tree ensemble members while retaining all components | Transforming full-information feature coordinates per ensemble member is new | Do semantic, exactly equivalent tabular bases improve a parameter-efficient neural ensemble where arbitrary rotations fail? |
| TabPFN transformation ensembles | Feature-order shifts and alternative preprocessing configurations are ensembled to improve prediction and approximate missing symmetries | Ensembling tabular input transformations or permutations is new | Can basis-orbit views provide training-time diversity in a shared-weight model, rather than inference configurations for a pretrained foundation model? |
| TabM | Shared-weight parameter-efficient ensembling is a strong tabular DL baseline | Parameter-efficient tabular ensembling is new | Can exact schema-basis charts serve as a useful structured diversity axis inside TabM without adding trainable parameters? |
| Multi-view tabular learning | Feature subsets and alternative table views are established augmentation and representation-learning tools | Multi-view tabular learning is new | Can multiple full-information numerical charts specialize shared-weight supervised ensemble members? |
| Function-basis numerical encoding | Numerical features can be expanded in general function bases, including piecewise-linear and spline families | Function-basis expansion or splines are new | Does separating empirical atoms from a conditional continuous measure prevent quantile-resolution collapse and provide useful ensemble diversity? |

## Primary sources

- ICLR 2026 reviewer guide:
  https://iclr.cc/Conferences/2026/ReviewerGuide
- Martens and Grosse, *Optimizing Neural Networks with Kronecker-factored
  Approximate Curvature*, ICML 2015:
  https://proceedings.mlr.press/v37/martens15.html
- Desjardins et al., *Natural Neural Networks*, NeurIPS 2015:
  https://proceedings.neurips.cc/paper/2015/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html
- Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR
  2020: https://www.jmlr.org/papers/v21/17-678.html
- Ling, Sharp, and Jacobson, *VectorAdam for Rotation Equivariant Geometry
  Optimization*, NeurIPS 2022:
  https://openreview.net/forum?id=df1g_KeEjQ
- Jackson, *An Isometric Stochastic Optimizer*, 2023:
  https://arxiv.org/abs/2307.12979
- Gorishniy et al., *On Embeddings for Numerical Features in Tabular Deep
  Learning*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html
- Grinsztajn, Oyallon, and Varoquaux, *Why do tree-based models still
  outperform deep learning on typical tabular data?*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html
- Dym, Lawrence, and Siegel, *Equivariant Frames and the Impossibility of
  Continuous Canonicalization*, ICML 2024:
  https://proceedings.mlr.press/v235/dym24a.html
- Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, ICLR 2025:
  https://openreview.net/forum?id=IDxZhXrpNf
- Ye et al., *Stretch Transformation for Tabular Data*, ICLR 2026 submission:
  https://openreview.net/forum?id=TkerLrovDn
- Rodríguez, Kuncheva, and Alonso, *Rotation Forest: A New Classifier
  Ensemble Method*, TPAMI 2006:
  https://doi.org/10.1109/TPAMI.2006.211
- Hollmann et al., *Accurate predictions on small data with a tabular
  foundation model*, Nature 2025:
  https://www.nature.com/articles/s41586-024-08328-6
- Gorishniy et al., *TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling*, ICLR 2025:
  https://openreview.net/forum?id=Sd4wYYOhmY
- Ucar, Hajiramezanali, and Edwards, *SubTab: Subsetting Features of Tabular
  Data for Self-Supervised Representation Learning*, NeurIPS 2021:
  https://openreview.net/forum?id=vrhNQ7aYSdr
- Shtoff et al., *Function Basis Encoding of Numerical Features in
  Factorization Machines*, TMLR 2024:
  https://openreview.net/forum?id=M4222IBHsh

## Claim that remains defensible only if the broad gates pass

The candidate contribution is not a new invariant optimizer. It is a systematic
demonstration that modern tabular learners can assign materially different
performance to information-equivalent schema representations, together with a
causal orbit benchmark that separates representational expressivity from
optimization geometry and measures how well practical remedies recover the
known ideal invariance.

It is also not the first tabular feature-rotation study. Relative to Grinsztajn
et al., the narrower possible advance is the determinant/energy-controlled
condition-number intervention, exact cross-schema equivalence verification,
paired closure by canonicalization, and comparison with affine-invariance prior
art. If those details are removed, the empirical thesis is too close to known
tabular-orientation evidence.

That is potentially useful new empirical knowledge under the ICLR 2026 review
standard, which explicitly does not require state-of-the-art performance. It is
not sufficient if the effect is narrow, synthetic-only, small relative to seed
noise, absent from modern backbones, or removed by ordinary tuning.
