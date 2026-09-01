# Basis-Controlled Tabular Learning — Method Tournament

## Executive Summary

No final paper method is chosen here. The strongest optimizer candidate is **BlockAdam+DataInit[equal-HPO]** (99.01% prospective reduction, 0.88% median task change). The strongest representation candidate is **GramAnchor** (100.00%, 0.90%), and the strongest nontrivial hybrid is **Raw+GramAnchor@0.75** (75.00%, -0.10%). Methods satisfying KEEP: GramAnchor, Raw+GramAnchor@0.75.

The central result is a tradeoff, not a winner declaration: invariant Gram coordinates can remove arbitrary orthogonal-basis dependence exactly, while a fixed raw/invariant mixture can retain more of the useful raw-coordinate prior. The optimizer route is mechanistically valid under matched functions, but its predictive cost determines whether it remains a serious paper candidate.

## Frozen Protocol

- Git commit at freeze: `0c456660ae9a87aab7932b569e1954b0ee1d25fe` (tournament files are an uncommitted experiment subtree on that base).
- Hardware: two NVIDIA H100 NVL GPUs, 95830 MiB each; driver 575.57.08.
- Packages: torch 2.7.0, numpy 1.26.4, scipy 1.11.4, pandas 2.3.3, scikit-learn 1.4.2, catboost 1.2.10, pytabkit 1.7.3, tabicl 2.0.3, tabpfn 8.5.0.
- Split seed: 20260901; model seeds: [0, 1, 2]; eight orbit members per reference.
- Development datasets: california_housing, house_16H, bike-sharing, phoneme, credit-g, wine-quality-red.
- NEW prospective datasets: satimage, eeg-eye-state, steel-plates-fault, gesture-phase, wilt, space-ga, pollen.
- `FINALIST_CONFIGS.json` SHA256: `9a567e3e47c35d35db88a9830da6ceedf9534509a831d8962579efc08090749b`; frozen at 2026-09-01T06:25:20.681169+00:00 before prospective data loading.

## Previous Findings Treated as Fixed

The tournament treated the prior confirmation as pilot evidence rather than rerunning discovery: orthogonal and natural equivalent-basis effects were established; AdamW's coordinatewise second moment was implicated; SGD was predictively weak despite symmetry; PCA incurred task cost; generic consistency training failed; and AnchorCanonical suffered rank failure. All baselines, splits, RBF blocks, transformations, and metrics were imported from that frozen implementation.

## 1. Stage-1 Method Screening

| method | disagreement reduction | task change | runtime | verdict |
| --- | --- | --- | --- | --- |
| GramAnchor | 1 | -0.0008956 | 10.83 | SURVIVE |
| GramDistance | 1 | 0.02072 | 11.25 | SURVIVE |
| NystromGram | 1 | 0.002644 | 11.89 | SURVIVE |
| PCA | 1 | -8.549e-05 | 16.95 | SURVIVE |
| BlockAdam | 0.1607 | 0.003482 | 26.27 | KILL |
| BlockScalarAdam | 0.1517 | 0.003164 | 27.66 | KILL |
| MatrixAdam | 0.00582 | 0.005342 | 44.26 | KILL |
| AdamW | 0 | 0 | 17.96 | CONTROL |

Stage 1 killed the default-initialization optimizer variants for inadequate reduction, while retaining PCA and the three Gram-family interfaces. The specified data-equivariant initialization and SoftBlock rescues were therefore evaluated in Stage 2.

## 2. Optimizer Methods

### AdamW

Raw AdamW is the zero-reduction reference and received the same three learning-rate trials as every surviving optimizer rescue.

### BlockScalarAdam

BlockScalarAdam is exactly orthogonally equivariant in the matched-function audit, but its Stage-1 predictive/reduction tradeoff did not pass the survival rule.

### BlockAdam

Per-output block second moments preserve matched equivalence numerically. Default initialization still leaves different initial functions across ordinary orbit fits, so data-equivariant initialization was tested explicitly.

### MatrixAdam

Full within-block matrix adaptivity also preserves matched equivalence. It improves the optimization symmetry but is more expensive and does not automatically preserve AdamW's task performance.

### Data-equivariant initialization

The first-layer blocks use a target-free training-design construction that transforms covariantly. For TabM-D, the diagonal per-coordinate input adapter is frozen to one because it is not closed under general within-block rotations; this makes the intervention honest but changes the effective architecture and is reported as a limitation.

### SoftBlockAdam

SoftBlockAdam with alpha 0.1 and 0.25 was the declared interpolation fallback. Alpha 0.1 entered equal HPO; neither setting was assumed invariant merely from its name.

| method | dataset | disagreement | reduction | task metric | task change |
| --- | --- | --- | --- | --- | --- |
| AdamW | bike-sharing | 0.1199 | 0 | 76.57 | 0 |
| AdamW | california_housing | 0.09879 | 0 | 6.515e+04 | 0 |
| AdamW | credit-g | 0.02905 | 0 | 0.4821 | 0 |
| AdamW | house_16H | 0.1363 | 0 | 3.373e+04 | 0 |
| AdamW | phoneme | 0.04003 | 0 | 0.3401 | 0 |
| AdamW | wine-quality-red | 0.105 | 0 | 0.6902 | 0 |
| AdamW[equal-HPO] | bike-sharing | 0.1151 | 0 | 68.69 | 0 |
| AdamW[equal-HPO] | california_housing | 0.1183 | 0 | 6.442e+04 | 0 |
| AdamW[equal-HPO] | credit-g | 0.0307 | 0 | 0.4827 | 0 |
| AdamW[equal-HPO] | house_16H | 0.191 | 0 | 3.436e+04 | 0 |
| AdamW[equal-HPO] | phoneme | 0.04975 | 0 | 0.3418 | 0 |
| AdamW[equal-HPO] | wine-quality-red | 0.1313 | 0 | 0.6956 | 0 |
| BlockAdam | bike-sharing | 0.1199 | -0.008712 | 76.66 | 0.003932 |
| BlockAdam | california_housing | 0.08881 | 0.09736 | 6.565e+04 | 0.01072 |
| BlockAdam | credit-g | 0.02698 | 0.06583 | 0.4892 | 0.001985 |
| BlockAdam | house_16H | 0.143 | 4.038e-05 | 3.36e+04 | -0.0009998 |
| BlockAdam | phoneme | 0.03804 | 0.03842 | 0.3403 | -0.0003055 |
| BlockAdam | wine-quality-red | 0.08725 | 0.1756 | 0.6929 | 0.005896 |
| BlockAdam+DataInit | bike-sharing | 0.004571 | 0.9536 | 84.34 | 0.09734 |
| BlockAdam+DataInit | california_housing | 0.004622 | 0.9388 | 6.676e+04 | 0.03299 |
| BlockAdam+DataInit | credit-g | 3.087e-05 | 0.999 | 0.5073 | 0.05599 |
| BlockAdam+DataInit | house_16H | 0.02154 | 0.8101 | 3.588e+04 | 0.0646 |
| BlockAdam+DataInit | phoneme | 0.001378 | 0.9571 | 0.3468 | 0.03882 |
| BlockAdam+DataInit | wine-quality-red | 0.004352 | 0.9576 | 0.713 | 0.03002 |
| BlockAdam+DataInit[equal-HPO] | bike-sharing | 0.004572 | 0.9484 | 82.99 | 0.2104 |
| BlockAdam+DataInit[equal-HPO] | california_housing | 0.004623 | 0.9493 | 6.657e+04 | 0.03682 |
| BlockAdam+DataInit[equal-HPO] | credit-g | 3.087e-05 | 0.9987 | 0.5128 | 0.06258 |
| BlockAdam+DataInit[equal-HPO] | house_16H | 0.02154 | 0.8488 | 3.617e+04 | 0.03469 |
| BlockAdam+DataInit[equal-HPO] | phoneme | 0.001378 | 0.966 | 0.3535 | 0.04004 |
| BlockAdam+DataInit[equal-HPO] | wine-quality-red | 0.004352 | 0.9554 | 0.7248 | 0.03473 |
| MatrixAdam | bike-sharing | 0.1083 | 0.03349 | 73.18 | -0.03342 |
| MatrixAdam | california_housing | 0.09565 | 0.02254 | 6.484e+04 | -0.0007758 |
| MatrixAdam | credit-g | 0.02995 | 0.03461 | 0.4824 | 0.001198 |
| MatrixAdam | house_16H | 0.1844 | -0.1219 | 3.459e+04 | 0.0225 |
| MatrixAdam | phoneme | 0.03938 | -0.003132 | 0.349 | 0.01409 |
| MatrixAdam | wine-quality-red | 0.1032 | 0.01401 | 0.6936 | 0.004015 |
| MatrixAdam+DataInit | bike-sharing | 0.003023 | 0.9658 | 78.21 | 0.02624 |
| MatrixAdam+DataInit | california_housing | 0.001101 | 0.9846 | 6.547e+04 | 0.007549 |
| MatrixAdam+DataInit | credit-g | 0.0001057 | 0.9965 | 0.5115 | 0.06018 |
| MatrixAdam+DataInit | house_16H | 0.005764 | 0.9492 | 3.668e+04 | 0.1022 |
| MatrixAdam+DataInit | phoneme | 0.0007251 | 0.9775 | 0.3551 | 0.03869 |
| MatrixAdam+DataInit | wine-quality-red | 0.001407 | 0.9863 | 0.7099 | 0.02817 |
| MatrixAdam+DataInit[equal-HPO] | bike-sharing | 0.003023 | 0.964 | 75.6 | 0.101 |
| MatrixAdam+DataInit[equal-HPO] | california_housing | 0.001102 | 0.9856 | 6.567e+04 | 0.01796 |
| MatrixAdam+DataInit[equal-HPO] | credit-g | 0.0001057 | 0.9957 | 0.5171 | 0.068 |
| MatrixAdam+DataInit[equal-HPO] | house_16H | 0.005764 | 0.9566 | 3.684e+04 | 0.07274 |
| MatrixAdam+DataInit[equal-HPO] | phoneme | 0.0007252 | 0.9821 | 0.3549 | 0.06773 |
| MatrixAdam+DataInit[equal-HPO] | wine-quality-red | 0.001407 | 0.9856 | 0.7225 | 0.04355 |
| SoftBlockAdam-a0.1+DataInit | bike-sharing | 0.03576 | 0.6959 | 84.49 | 0.09849 |
| SoftBlockAdam-a0.1+DataInit | california_housing | 0.02297 | 0.7784 | 6.69e+04 | 0.03302 |
| SoftBlockAdam-a0.1+DataInit | credit-g | 0.001362 | 0.9471 | 0.5072 | 0.05572 |
| SoftBlockAdam-a0.1+DataInit | house_16H | 0.05946 | 0.5566 | 3.643e+04 | 0.06644 |
| SoftBlockAdam-a0.1+DataInit | phoneme | 0.01148 | 0.6718 | 0.3514 | 0.04417 |
| SoftBlockAdam-a0.1+DataInit | wine-quality-red | 0.01415 | 0.8749 | 0.7137 | 0.02937 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | bike-sharing | 0.04113 | 0.5552 | 82.99 | 0.2107 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | california_housing | 0.03522 | 0.6744 | 6.694e+04 | 0.04183 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | credit-g | 0.001339 | 0.9558 | 0.5129 | 0.06246 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | house_16H | 0.08042 | 0.5467 | 3.649e+04 | 0.04701 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | phoneme | 0.009546 | 0.8079 | 0.3564 | 0.03859 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | wine-quality-red | 0.01389 | 0.8844 | 0.7245 | 0.03815 |
| SoftBlockAdam-a0.25+DataInit | bike-sharing | 0.04533 | 0.57 | 84.31 | 0.1011 |
| SoftBlockAdam-a0.25+DataInit | california_housing | 0.05212 | 0.5633 | 6.645e+04 | 0.0306 |
| SoftBlockAdam-a0.25+DataInit | credit-g | 0.002424 | 0.9129 | 0.5071 | 0.05549 |
| SoftBlockAdam-a0.25+DataInit | house_16H | 0.07508 | 0.4225 | 3.627e+04 | 0.07036 |
| SoftBlockAdam-a0.25+DataInit | phoneme | 0.01616 | 0.5525 | 0.3518 | 0.04517 |
| SoftBlockAdam-a0.25+DataInit | wine-quality-red | 0.02339 | 0.7942 | 0.7102 | 0.0273 |

## 3. Optimizer Equivariance Audit

| method | epoch0 | epoch1 | epoch5 | final disagreement |
| --- | --- | --- | --- | --- |
| AdamW | 2.107e-08 | 0.002857 | 0.04622 | 0.08158 |
| BlockAdam | 2.107e-08 | 3.942e-08 | 1.55e-07 | 5.232e-07 |
| BlockScalarAdam | 2.107e-08 | 3.942e-08 | 1.402e-07 | 5.451e-07 |
| MatrixAdam | 2.107e-08 | 3.276e-08 | 1.474e-07 | 5.098e-07 |
| SGD | 2.107e-08 | 2.018e-08 | 2.788e-08 | 1.244e-07 |

Matched-function equivalence remained below 1e-5 through the final epoch for BlockAdam, BlockScalarAdam, MatrixAdam, SGD. It did not for AdamW. Thus BlockAdam and MatrixAdam pass the implementation audit and AdamW fails it, as predicted by the mechanism hypothesis.

## 4. Representation Methods

### PCA

PCA is exactly invariant in nondegenerate cases after deterministic orientation, but its development task cost remains material.

### GramAnchor

GramAnchor uses target-free Gram-pivot training anchors. The tolerance-aware pivot rule was necessary to remove near-rank-saturation tie instability; all final coordinate audits pass 1e-8.

### GramDistance

GramDistance is exactly orthogonally invariant but loses more task information than GramAnchor on the full panel.

### NyströmGram

NyströmGram uses deterministic canonicalization inside repeated eigenspaces. This repaired an initial numerical degeneracy without using labels or outcomes.

### HybridSpectral

HybridSpectral keeps separated spectral directions and Gram-maps degenerate groups. Three frozen gap thresholds were tested.

### MahalanobisGram if run

MahalanobisGram was run only in the separate condition<=3 exploratory screen at ridges 1e-6, 1e-4, and 1e-2. Ridge regularization prevents a blanket claim of exact general-linear invariance.

| method | model | dataset | disagreement | reduction | task_change |
| --- | --- | --- | --- | --- | --- |
| GramAnchor | catboost | bike-sharing | 0 | 1 | -0.01193 |
| GramAnchor | catboost | california_housing | 0 | 1 | -0.003471 |
| GramAnchor | catboost | credit-g | 0 | 1 | 0.002455 |
| GramAnchor | catboost | house_16H | 0 | 1 | 0.01504 |
| GramAnchor | catboost | phoneme | 0 | 1 | 0.01139 |
| GramAnchor | catboost | wine-quality-red | 0 | 1 | 0.008054 |
| GramAnchor | controlled_mlp | bike-sharing | 0 | 1 | -0.04336 |
| GramAnchor | controlled_mlp | california_housing | 0 | 1 | -0.01412 |
| GramAnchor | controlled_mlp | credit-g | 0 | 1 | 0.09435 |
| GramAnchor | controlled_mlp | house_16H | 0 | 1 | 0.02314 |
| GramAnchor | controlled_mlp | phoneme | 0 | 1 | 0.01801 |
| GramAnchor | controlled_mlp | wine-quality-red | 0 | 1 | -0.001135 |
| GramAnchor | tabicl_v2 | bike-sharing | 0 | 1 | 0.08225 |
| GramAnchor | tabicl_v2 | california_housing | 0 | 1 | 0.02149 |
| GramAnchor | tabicl_v2 | credit-g | 0 | 1 | 0.03343 |
| GramAnchor | tabicl_v2 | house_16H | 0 | 1 | -0.01338 |
| GramAnchor | tabicl_v2 | phoneme | 0 | 1 | 0.05425 |
| GramAnchor | tabicl_v2 | wine-quality-red | 0 | 1 | -0.001104 |
| GramAnchor | tabm_d | bike-sharing | 0 | 1 | 0.05923 |
| GramAnchor | tabm_d | california_housing | 0 | 1 | -0.02441 |
| GramAnchor | tabm_d | credit-g | 0 | 1 | 0.03039 |
| GramAnchor | tabm_d | house_16H | 0 | 1 | -0.0161 |
| GramAnchor | tabm_d | phoneme | 0 | 1 | -0.02155 |
| GramAnchor | tabm_d | wine-quality-red | 0 | 1 | 0.01081 |
| GramAnchor | tabpfn_2_6 | bike-sharing | 0 | 1 | -0.01108 |
| GramAnchor | tabpfn_2_6 | california_housing | 0 | 1 | 0.01345 |
| GramAnchor | tabpfn_2_6 | credit-g | 0 | 1 | 0.003394 |
| GramAnchor | tabpfn_2_6 | house_16H | 0 | 1 | -0.02774 |
| GramAnchor | tabpfn_2_6 | phoneme | 0 | 1 | -0.003153 |
| GramAnchor | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.004253 |
| GramAnchor-m8 | catboost | bike-sharing | 0 | 1 | -0.009284 |
| GramAnchor-m8 | catboost | california_housing | 0 | 1 | -0.007 |
| GramAnchor-m8 | catboost | credit-g | 0 | 1 | 0.03155 |
| GramAnchor-m8 | catboost | house_16H | 0 | 1 | 0.009326 |
| GramAnchor-m8 | catboost | phoneme | 0 | 1 | -0.003671 |
| GramAnchor-m8 | catboost | wine-quality-red | 0 | 1 | 0.002337 |
| GramAnchor-m8 | controlled_mlp | bike-sharing | 0 | 1 | -0.002308 |
| GramAnchor-m8 | controlled_mlp | california_housing | 0 | 1 | -0.009129 |
| GramAnchor-m8 | controlled_mlp | credit-g | 0 | 1 | 0.02692 |
| GramAnchor-m8 | controlled_mlp | house_16H | 0 | 1 | 0.002289 |
| GramAnchor-m8 | controlled_mlp | phoneme | 0 | 1 | 0.02941 |
| GramAnchor-m8 | controlled_mlp | wine-quality-red | 0 | 1 | 0.06178 |
| GramAnchor-m8 | tabicl_v2 | bike-sharing | 0 | 1 | 0.178 |
| GramAnchor-m8 | tabicl_v2 | california_housing | 0 | 1 | 0.01823 |
| GramAnchor-m8 | tabicl_v2 | credit-g | 0 | 1 | 0.01038 |
| GramAnchor-m8 | tabicl_v2 | house_16H | 0 | 1 | 0.02526 |
| GramAnchor-m8 | tabicl_v2 | phoneme | 0 | 1 | 0.03827 |
| GramAnchor-m8 | tabicl_v2 | wine-quality-red | 0 | 1 | -0.0136 |
| GramAnchor-m8 | tabm_d | bike-sharing | 0 | 1 | 0.0277 |
| GramAnchor-m8 | tabm_d | california_housing | 0 | 1 | -0.01426 |
| GramAnchor-m8 | tabm_d | credit-g | 0 | 1 | 0.004235 |
| GramAnchor-m8 | tabm_d | house_16H | 0 | 1 | -0.02355 |
| GramAnchor-m8 | tabm_d | phoneme | 0 | 1 | -0.01009 |
| GramAnchor-m8 | tabm_d | wine-quality-red | 0 | 1 | 0.008765 |
| GramAnchor-m8 | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.0246 |
| GramAnchor-m8 | tabpfn_2_6 | california_housing | 0 | 1 | 0.0206 |
| GramAnchor-m8 | tabpfn_2_6 | credit-g | 0 | 1 | 0.009642 |
| GramAnchor-m8 | tabpfn_2_6 | house_16H | 0 | 1 | -0.03721 |
| GramAnchor-m8 | tabpfn_2_6 | phoneme | 0 | 1 | -0.01775 |
| GramAnchor-m8 | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.00361 |
| GramDistance | catboost | bike-sharing | 0 | 1 | -0.01019 |
| GramDistance | catboost | california_housing | 0 | 1 | 0.01373 |
| GramDistance | catboost | credit-g | 0 | 1 | 0.04055 |
| GramDistance | catboost | house_16H | 0 | 1 | 0.03349 |
| GramDistance | catboost | phoneme | 0 | 1 | 0.05361 |
| GramDistance | catboost | wine-quality-red | 0 | 1 | 0.003921 |
| GramDistance | controlled_mlp | bike-sharing | 0 | 1 | -0.07949 |
| GramDistance | controlled_mlp | california_housing | 0 | 1 | 0.04324 |
| GramDistance | controlled_mlp | credit-g | 0 | 1 | 0.07763 |
| GramDistance | controlled_mlp | house_16H | 0 | 1 | 0.04204 |
| GramDistance | controlled_mlp | phoneme | 0 | 1 | -0.01018 |
| GramDistance | controlled_mlp | wine-quality-red | 0 | 1 | 0.0261 |
| GramDistance | tabicl_v2 | bike-sharing | 0 | 1 | 0.06096 |
| GramDistance | tabicl_v2 | california_housing | 0 | 1 | 0.07565 |
| GramDistance | tabicl_v2 | credit-g | 0 | 1 | 0.02009 |
| GramDistance | tabicl_v2 | house_16H | 0 | 1 | 0.05133 |
| GramDistance | tabicl_v2 | phoneme | 0 | 1 | 0.07082 |
| GramDistance | tabicl_v2 | wine-quality-red | 0 | 1 | 0.007397 |
| GramDistance | tabm_d | bike-sharing | 0 | 1 | 0.1049 |
| GramDistance | tabm_d | california_housing | 0 | 1 | 0.04546 |
| GramDistance | tabm_d | credit-g | 0 | 1 | 0.06133 |
| GramDistance | tabm_d | house_16H | 0 | 1 | 0.02673 |
| GramDistance | tabm_d | phoneme | 0 | 1 | -0.02559 |
| GramDistance | tabm_d | wine-quality-red | 0 | 1 | 0.07195 |
| GramDistance | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.01875 |
| GramDistance | tabpfn_2_6 | california_housing | 0 | 1 | 0.07074 |
| GramDistance | tabpfn_2_6 | credit-g | 0 | 1 | 0.01102 |
| GramDistance | tabpfn_2_6 | house_16H | 0 | 1 | 0.0232 |
| GramDistance | tabpfn_2_6 | phoneme | 0 | 1 | 0.05745 |
| GramDistance | tabpfn_2_6 | wine-quality-red | 0 | 1 | 0.003413 |
| HybridSpectral-t0.01 | catboost | bike-sharing | 0 | 1 | 0.03304 |
| HybridSpectral-t0.01 | catboost | california_housing | 0 | 1 | -0.007086 |
| HybridSpectral-t0.01 | catboost | credit-g | 0 | 1 | 0.004411 |
| HybridSpectral-t0.01 | catboost | house_16H | 0 | 1 | 0.05843 |
| HybridSpectral-t0.01 | catboost | phoneme | 0 | 1 | 0.06864 |
| HybridSpectral-t0.01 | catboost | wine-quality-red | 0 | 1 | 0.009086 |
| HybridSpectral-t0.01 | controlled_mlp | bike-sharing | 0 | 1 | -0.03749 |
| HybridSpectral-t0.01 | controlled_mlp | california_housing | 0 | 1 | -0.001097 |
| HybridSpectral-t0.01 | controlled_mlp | credit-g | 0 | 1 | -0.02525 |
| HybridSpectral-t0.01 | controlled_mlp | house_16H | 0 | 1 | -0.01079 |
| HybridSpectral-t0.01 | controlled_mlp | phoneme | 0 | 1 | 0.0009259 |
| HybridSpectral-t0.01 | controlled_mlp | wine-quality-red | 0 | 1 | 0.0256 |
| HybridSpectral-t0.01 | tabicl_v2 | bike-sharing | 0 | 1 | -0.03138 |
| HybridSpectral-t0.01 | tabicl_v2 | california_housing | 0 | 1 | 0.05638 |
| HybridSpectral-t0.01 | tabicl_v2 | credit-g | 0 | 1 | -0.008402 |
| HybridSpectral-t0.01 | tabicl_v2 | house_16H | 0 | 1 | 0.02596 |
| HybridSpectral-t0.01 | tabicl_v2 | phoneme | 0 | 1 | 0.0661 |
| HybridSpectral-t0.01 | tabicl_v2 | wine-quality-red | 0 | 1 | 0.005435 |
| HybridSpectral-t0.01 | tabm_d | bike-sharing | 0 | 1 | -0.06045 |
| HybridSpectral-t0.01 | tabm_d | california_housing | 0 | 1 | 0.04524 |
| HybridSpectral-t0.01 | tabm_d | credit-g | 0 | 1 | 0.2478 |
| HybridSpectral-t0.01 | tabm_d | house_16H | 0 | 1 | 0.05849 |
| HybridSpectral-t0.01 | tabm_d | phoneme | 0 | 1 | 0.04973 |
| HybridSpectral-t0.01 | tabm_d | wine-quality-red | 0 | 1 | 0.008236 |
| HybridSpectral-t0.01 | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.02462 |
| HybridSpectral-t0.01 | tabpfn_2_6 | california_housing | 0 | 1 | 0.09845 |
| HybridSpectral-t0.01 | tabpfn_2_6 | credit-g | 0 | 1 | 0.01182 |
| HybridSpectral-t0.01 | tabpfn_2_6 | house_16H | 0 | 1 | 0.07944 |
| HybridSpectral-t0.01 | tabpfn_2_6 | phoneme | 0 | 1 | 0.0434 |
| HybridSpectral-t0.01 | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.01005 |
| HybridSpectral-t0.05 | catboost | bike-sharing | 0 | 1 | 0.03173 |
| HybridSpectral-t0.05 | catboost | california_housing | 0 | 1 | -0.007086 |
| HybridSpectral-t0.05 | catboost | credit-g | 0 | 1 | 0.004411 |
| HybridSpectral-t0.05 | catboost | house_16H | 0 | 1 | 0.05629 |
| HybridSpectral-t0.05 | catboost | phoneme | 0 | 1 | 0.06864 |
| HybridSpectral-t0.05 | catboost | wine-quality-red | 0 | 1 | 0.01038 |
| HybridSpectral-t0.05 | controlled_mlp | bike-sharing | 0 | 1 | -0.04639 |
| HybridSpectral-t0.05 | controlled_mlp | california_housing | 0 | 1 | -0.001097 |
| HybridSpectral-t0.05 | controlled_mlp | credit-g | 0 | 1 | -0.02525 |
| HybridSpectral-t0.05 | controlled_mlp | house_16H | 0 | 1 | 0.002238 |
| HybridSpectral-t0.05 | controlled_mlp | phoneme | 0 | 1 | 0.0009259 |
| HybridSpectral-t0.05 | controlled_mlp | wine-quality-red | 0 | 1 | 0.01827 |
| HybridSpectral-t0.05 | tabicl_v2 | bike-sharing | 0 | 1 | -0.1223 |
| HybridSpectral-t0.05 | tabicl_v2 | california_housing | 0 | 1 | 0.05638 |
| HybridSpectral-t0.05 | tabicl_v2 | credit-g | 0 | 1 | -0.008402 |
| HybridSpectral-t0.05 | tabicl_v2 | house_16H | 0 | 1 | 0.02594 |
| HybridSpectral-t0.05 | tabicl_v2 | phoneme | 0 | 1 | 0.0661 |
| HybridSpectral-t0.05 | tabicl_v2 | wine-quality-red | 0 | 1 | 0.007366 |
| HybridSpectral-t0.05 | tabm_d | bike-sharing | 0 | 1 | -0.03986 |
| HybridSpectral-t0.05 | tabm_d | california_housing | 0 | 1 | 0.04524 |
| HybridSpectral-t0.05 | tabm_d | credit-g | 0 | 1 | 0.2478 |
| HybridSpectral-t0.05 | tabm_d | house_16H | 0 | 1 | 0.05458 |
| HybridSpectral-t0.05 | tabm_d | phoneme | 0 | 1 | 0.04973 |
| HybridSpectral-t0.05 | tabm_d | wine-quality-red | 0 | 1 | 0.009212 |
| HybridSpectral-t0.05 | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.01707 |
| HybridSpectral-t0.05 | tabpfn_2_6 | california_housing | 0 | 1 | 0.09845 |
| HybridSpectral-t0.05 | tabpfn_2_6 | credit-g | 0 | 1 | 0.01182 |
| HybridSpectral-t0.05 | tabpfn_2_6 | house_16H | 0 | 1 | 0.0803 |
| HybridSpectral-t0.05 | tabpfn_2_6 | phoneme | 0 | 1 | 0.0434 |
| HybridSpectral-t0.05 | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.01561 |
| HybridSpectral-t0.10 | catboost | bike-sharing | 0 | 1 | 0.03173 |
| HybridSpectral-t0.10 | catboost | california_housing | 0 | 1 | -0.007086 |
| HybridSpectral-t0.10 | catboost | credit-g | 0 | 1 | 0.004411 |
| HybridSpectral-t0.10 | catboost | house_16H | 0 | 1 | 0.05629 |
| HybridSpectral-t0.10 | catboost | phoneme | 0 | 1 | 0.06864 |
| HybridSpectral-t0.10 | catboost | wine-quality-red | 0 | 1 | 0.01038 |
| HybridSpectral-t0.10 | controlled_mlp | bike-sharing | 0 | 1 | -0.04639 |
| HybridSpectral-t0.10 | controlled_mlp | california_housing | 0 | 1 | -0.001097 |
| HybridSpectral-t0.10 | controlled_mlp | credit-g | 0 | 1 | -0.02525 |
| HybridSpectral-t0.10 | controlled_mlp | house_16H | 0 | 1 | 0.002238 |
| HybridSpectral-t0.10 | controlled_mlp | phoneme | 0 | 1 | 0.0009259 |
| HybridSpectral-t0.10 | controlled_mlp | wine-quality-red | 0 | 1 | 0.01827 |
| HybridSpectral-t0.10 | tabicl_v2 | bike-sharing | 0 | 1 | -0.1223 |
| HybridSpectral-t0.10 | tabicl_v2 | california_housing | 0 | 1 | 0.05638 |
| HybridSpectral-t0.10 | tabicl_v2 | credit-g | 0 | 1 | -0.008402 |
| HybridSpectral-t0.10 | tabicl_v2 | house_16H | 0 | 1 | 0.02594 |
| HybridSpectral-t0.10 | tabicl_v2 | phoneme | 0 | 1 | 0.0661 |
| HybridSpectral-t0.10 | tabicl_v2 | wine-quality-red | 0 | 1 | 0.007366 |
| HybridSpectral-t0.10 | tabm_d | bike-sharing | 0 | 1 | -0.03986 |
| HybridSpectral-t0.10 | tabm_d | california_housing | 0 | 1 | 0.04524 |
| HybridSpectral-t0.10 | tabm_d | credit-g | 0 | 1 | 0.2478 |
| HybridSpectral-t0.10 | tabm_d | house_16H | 0 | 1 | 0.05458 |
| HybridSpectral-t0.10 | tabm_d | phoneme | 0 | 1 | 0.04973 |
| HybridSpectral-t0.10 | tabm_d | wine-quality-red | 0 | 1 | 0.009212 |
| HybridSpectral-t0.10 | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.01707 |
| HybridSpectral-t0.10 | tabpfn_2_6 | california_housing | 0 | 1 | 0.09845 |
| HybridSpectral-t0.10 | tabpfn_2_6 | credit-g | 0 | 1 | 0.01182 |
| HybridSpectral-t0.10 | tabpfn_2_6 | house_16H | 0 | 1 | 0.0803 |
| HybridSpectral-t0.10 | tabpfn_2_6 | phoneme | 0 | 1 | 0.0434 |
| HybridSpectral-t0.10 | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.01561 |
| NystromGram | catboost | bike-sharing | 0 | 1 | 0.01325 |
| NystromGram | catboost | california_housing | 0 | 1 | 0.01653 |
| NystromGram | catboost | credit-g | 0 | 1 | 0.02856 |
| NystromGram | catboost | house_16H | 0 | 1 | 0.04613 |
| NystromGram | catboost | phoneme | 0 | 1 | 0.02081 |
| NystromGram | catboost | wine-quality-red | 0 | 1 | 0.002437 |
| NystromGram | controlled_mlp | bike-sharing | 0 | 1 | -0.1405 |
| NystromGram | controlled_mlp | california_housing | 0 | 1 | 0.02848 |
| NystromGram | controlled_mlp | credit-g | 0 | 1 | 0.02782 |
| NystromGram | controlled_mlp | house_16H | 0 | 1 | -0.01464 |
| NystromGram | controlled_mlp | phoneme | 0 | 1 | 0.1066 |
| NystromGram | controlled_mlp | wine-quality-red | 0 | 1 | 0.007912 |
| NystromGram | tabicl_v2 | bike-sharing | 0 | 1 | -0.1362 |
| NystromGram | tabicl_v2 | california_housing | 0 | 1 | 0.006809 |
| NystromGram | tabicl_v2 | credit-g | 0 | 1 | -0.006425 |
| NystromGram | tabicl_v2 | house_16H | 0 | 1 | 0.03862 |
| NystromGram | tabicl_v2 | phoneme | 0 | 1 | 0.06568 |
| NystromGram | tabicl_v2 | wine-quality-red | 0 | 1 | -0.01395 |
| NystromGram | tabm_d | bike-sharing | 0 | 1 | -0.01695 |
| NystromGram | tabm_d | california_housing | 0 | 1 | 0.0505 |
| NystromGram | tabm_d | credit-g | 0 | 1 | 0.03259 |
| NystromGram | tabm_d | house_16H | 0 | 1 | 0.04728 |
| NystromGram | tabm_d | phoneme | 0 | 1 | 0.02349 |
| NystromGram | tabm_d | wine-quality-red | 0 | 1 | 0.01813 |
| NystromGram | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.0221 |
| NystromGram | tabpfn_2_6 | california_housing | 0 | 1 | 0.06621 |
| NystromGram | tabpfn_2_6 | credit-g | 0 | 1 | 0.006236 |
| NystromGram | tabpfn_2_6 | house_16H | 0 | 1 | 0.01042 |
| NystromGram | tabpfn_2_6 | phoneme | 0 | 1 | 0.01285 |
| NystromGram | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.007651 |
| PCA | catboost | bike-sharing | 0 | 1 | 0.03617 |
| PCA | catboost | california_housing | 0 | 1 | -0.009972 |
| PCA | catboost | credit-g | 0 | 1 | 0.0113 |
| PCA | catboost | house_16H | 0 | 1 | 0.06696 |
| PCA | catboost | phoneme | 0 | 1 | 0.06864 |
| PCA | catboost | wine-quality-red | 0 | 1 | 0.009086 |
| PCA | controlled_mlp | bike-sharing | 0 | 1 | -0.03749 |
| PCA | controlled_mlp | california_housing | 0 | 1 | -0.001097 |
| PCA | controlled_mlp | credit-g | 0 | 1 | -0.02615 |
| PCA | controlled_mlp | house_16H | 0 | 1 | -0.01079 |
| PCA | controlled_mlp | phoneme | 0 | 1 | 0.0009259 |
| PCA | controlled_mlp | wine-quality-red | 0 | 1 | 0.0256 |
| PCA | tabicl_v2 | bike-sharing | 0 | 1 | -0.06316 |
| PCA | tabicl_v2 | california_housing | 0 | 1 | 0.05638 |
| PCA | tabicl_v2 | credit-g | 0 | 1 | -0.00907 |
| PCA | tabicl_v2 | house_16H | 0 | 1 | 0.02596 |
| PCA | tabicl_v2 | phoneme | 0 | 1 | 0.0661 |
| PCA | tabicl_v2 | wine-quality-red | 0 | 1 | 0.005435 |
| PCA | tabm_d | bike-sharing | 0 | 1 | -0.06045 |
| PCA | tabm_d | california_housing | 0 | 1 | 0.04524 |
| PCA | tabm_d | credit-g | 0 | 1 | 0.08908 |
| PCA | tabm_d | house_16H | 0 | 1 | 0.05849 |
| PCA | tabm_d | phoneme | 0 | 1 | 0.04973 |
| PCA | tabm_d | wine-quality-red | 0 | 1 | 0.008236 |
| PCA | tabpfn_2_6 | bike-sharing | 0 | 1 | 0.05776 |
| PCA | tabpfn_2_6 | california_housing | 0 | 1 | 0.09921 |
| PCA | tabpfn_2_6 | credit-g | 0 | 1 | 0.0202 |
| PCA | tabpfn_2_6 | house_16H | 0 | 1 | 0.08089 |
| PCA | tabpfn_2_6 | phoneme | 0 | 1 | 0.04143 |
| PCA | tabpfn_2_6 | wine-quality-red | 0 | 1 | -0.008859 |

## 5. Anchor / Rank Ablations

| m | selection method | normalize | empirical rank | min_anchor_rank | disagreement | task change |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | gram_pivot | True | 8 | 2 | 1 | 0.001965 |
| 16 | gram_pivot | False | 8 | 2 | 1 | 0.008043 |
| 16 | gram_pivot | True | 8 | 2 | 1 | 0.01127 |
| 32 | random_index | True | 8 | 1 | 1 | 0.01217 |
| 32 | gram_pivot | True | 8 | 2 | 1 | 0.01736 |
| 8 | random_index | True | 8 | 1 | 1 | 0.04417 |
| 16 | random_index | True | 8 | 1 | 1 | 0.04541 |

Eight pivoted anchors were best on the controlled-MLP ablation, but their five-model full-panel task cost exceeded the 16-anchor setting. The frozen interface therefore retains m=16 rather than extrapolating the MLP-only ablation across model families.

## 6. Natural Equivalent Basis Results

### Local vs spectral hat

| method | model | pair | median_disagreement_reduction | median_relative_task_change | max_coordinate_error | max_reconstruction_error | datasets | units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GramAnchor | catboost | local_vs_spectral_hat | 1 | -0.008334 | 1.978e-16 | 2.204e-16 | 6 | 18 |
| GramAnchor | controlled_mlp | local_vs_spectral_hat | 1 | 0.00945 | 1.978e-16 | 2.204e-16 | 6 | 18 |
| GramAnchor | tabicl_v2 | local_vs_spectral_hat | 1 | 0.005831 | 1.978e-16 | 2.204e-16 | 6 | 18 |
| GramAnchor | tabm_d | local_vs_spectral_hat | 1 | 0.001744 | 1.978e-16 | 2.204e-16 | 6 | 18 |
| GramAnchor | tabpfn_2_6 | local_vs_spectral_hat | 1 | 0.006005 | 1.978e-16 | 2.204e-16 | 6 | 18 |

### One-hot vs Helmert

| method | model | pair | median_disagreement_reduction | median_relative_task_change | max_coordinate_error | max_reconstruction_error | datasets | units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GramAnchor | catboost | onehot_vs_helmert | 1 | -0.007031 | 9.378e-17 | 1.792e-16 | 3 | 9 |
| GramAnchor | controlled_mlp | onehot_vs_helmert | 1 | -0.01805 | 9.378e-17 | 1.792e-16 | 3 | 9 |
| GramAnchor | tabicl_v2 | onehot_vs_helmert | 1 | 0.03343 | 9.378e-17 | 1.792e-16 | 3 | 9 |
| GramAnchor | tabm_d | onehot_vs_helmert | 1 | 0.03039 | 9.378e-17 | 1.792e-16 | 3 | 9 |
| GramAnchor | tabpfn_2_6 | onehot_vs_helmert | 1 | 0.001619 | 9.378e-17 | 1.792e-16 | 3 | 9 |

Every natural pair was required to reconstruct below 1e-6 and every invariant-interface coordinate pair below 1e-8 before metrics were accepted.

## 7. Hybrid Methods

| method | median_disagreement_reduction | median_relative_task_change | median_worst_orbit_gain | paper_method_score | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw+GramAnchor-m8@0.75 | 0.75 | -0.004869 | 0.03594 | 0.75 | 53 | 0 | 37 |
| Raw+GramAnchor@0.75 | 0.75 | -0.003428 | 0.03646 | 0.75 | 51 | 0 | 39 |
| Raw+NystromGram@0.75 | 0.75 | -0.0002764 | 0.02737 | 0.75 | 47 | 0 | 43 |
| Raw+HybridSpectral-t0.05@0.75 | 0.75 | 8.696e-05 | 0.02327 | 0.7496 | 43 | 0 | 47 |
| Raw+HybridSpectral-t0.10@0.75 | 0.75 | 8.696e-05 | 0.02327 | 0.7496 | 43 | 0 | 47 |
| Raw+HybridSpectral-t0.01@0.75 | 0.75 | 0.0009974 | 0.02204 | 0.745 | 45 | 0 | 45 |
| Raw+PCA@0.75 | 0.75 | 0.00373 | 0.02109 | 0.7314 | 43 | 0 | 47 |
| Raw+GramDistance@0.75 | 0.75 | 0.008758 | 0.01528 | 0.7062 | 21 | 0 | 69 |
| Raw+PCA@0.5 | 0.5 | -0.00453 | 0.01977 | 0.5 | 55 | 0 | 35 |
| Raw+NystromGram@0.5 | 0.5 | -0.007699 | 0.02408 | 0.5 | 63 | 0 | 27 |
| Raw+HybridSpectral-t0.10@0.5 | 0.5 | -0.006303 | 0.02056 | 0.5 | 58 | 0 | 32 |
| Raw+GramDistance@0.5 | 0.5 | -0.0002364 | 0.01707 | 0.5 | 47 | 0 | 43 |
| Raw+GramAnchor-m8@0.5 | 0.5 | -0.009816 | 0.03223 | 0.5 | 65 | 0 | 25 |
| Raw+GramAnchor@0.5 | 0.5 | -0.005877 | 0.03336 | 0.5 | 63 | 0 | 27 |
| Raw+HybridSpectral-t0.05@0.5 | 0.5 | -0.006303 | 0.02056 | 0.5 | 58 | 0 | 32 |
| Raw+HybridSpectral-t0.01@0.5 | 0.5 | -0.006168 | 0.0196 | 0.5 | 55 | 0 | 35 |

H1 prediction mixtures were complete and alpha was selected from development validation only. H2 was optional ('if easy') and was not built because it would add architecture-specific training confounds after H1 already exposed the tradeoff. H3's stated trigger was not met: no block optimizer simultaneously had strong invariance and near-raw predictive performance on development, so the combined branch was not run.

## 8. Equal-HPO Control

Every surviving optimizer rescue and raw AdamW received exactly learning-rate multipliers 0.5, 1, and 2. A single multiplier was selected development-wide for each model/method from validation error; per-orbit oracle selections were retained only as diagnostics and were ineligible for freezing.

| model | method | multiplier | median_validation_excess | mean_validation_excess | median_validation_rank | mean_validation_rank | units | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled_mlp | AdamW | 2 | 0 | 0.00214 | 1 | 1.5 | 18 | True |
| controlled_mlp | BlockAdam+DataInit | 2 | 0 | 0.001698 | 1 | 1.667 | 18 | True |
| controlled_mlp | MatrixAdam+DataInit | 2 | 0 | 0.001551 | 1 | 1.444 | 18 | True |
| controlled_mlp | SoftBlockAdam-a0.1+DataInit | 2 | 0 | 0.001776 | 1 | 1.556 | 18 | True |
| tabm_d | AdamW | 2 | 0 | 0.001219 | 1 | 1.278 | 18 | True |
| tabm_d | BlockAdam+DataInit | 1 | 0.005955 | 0.02037 | 2 | 1.944 | 18 | True |
| tabm_d | MatrixAdam+DataInit | 1 | 0.002985 | 0.01209 | 1.5 | 1.556 | 18 | True |
| tabm_d | SoftBlockAdam-a0.1+DataInit | 1 | 0.002998 | 0.01665 | 2 | 1.722 | 18 | True |

| method | median_disagreement_reduction | median_relative_task_change | median_worst_orbit_gain | paper_method_score | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW[equal-HPO] | 0 | 0 | 0 | 0 | 0 | 36 | 0 |
| BlockAdam+DataInit[equal-HPO] | 0.9987 | 0.04273 | -0.01676 | 0.7782 | 2 | 0 | 34 |
| MatrixAdam+DataInit[equal-HPO] | 0.9957 | 0.05712 | -0.007104 | 0.7101 | 5 | 0 | 31 |
| SoftBlockAdam-a0.1+DataInit[equal-HPO] | 0.8334 | 0.04576 | -0.0168 | 0.5977 | 2 | 0 | 34 |

## 9. Development Ranking

### Performance-Preserving Invariance Ranking

| method | track | median_disagreement_reduction | median_relative_task_change | median_worst_orbit_gain | paper_method_score |
| --- | --- | --- | --- | --- | --- |
| GramAnchor | representation | 1 | 0.003549 | 0.03379 | 0.9823 |
| Raw+GramAnchor@1 | hybrid_prediction_mixture | 1 | 0.003549 | 0.03379 | 0.9823 |
| GramAnchor-m8 | representation | 1 | 0.00666 | 0.03086 | 0.9667 |
| Raw+GramAnchor-m8@1 | hybrid_prediction_mixture | 1 | 0.00666 | 0.03086 | 0.9667 |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 0.75 | -0.003428 | 0.03646 | 0.75 |
| Raw+GramAnchor-m8@0.75 | hybrid_prediction_mixture | 0.75 | -0.004869 | 0.03594 | 0.75 |
| Raw+NystromGram@0.75 | hybrid_prediction_mixture | 0.75 | -0.0002764 | 0.02737 | 0.75 |
| Raw+HybridSpectral-t0.05@0.75 | hybrid_prediction_mixture | 0.75 | 8.696e-05 | 0.02327 | 0.7496 |
| Raw+HybridSpectral-t0.10@0.75 | hybrid_prediction_mixture | 0.75 | 8.696e-05 | 0.02327 | 0.7496 |
| Raw+HybridSpectral-t0.01@0.75 | hybrid_prediction_mixture | 0.75 | 0.0009974 | 0.02204 | 0.745 |
| Raw+PCA@0.75 | hybrid_prediction_mixture | 0.75 | 0.00373 | 0.02109 | 0.7314 |
| Raw+GramDistance@0.75 | hybrid_prediction_mixture | 0.75 | 0.008758 | 0.01528 | 0.7062 |
| Raw+GramAnchor@0.5 | hybrid_prediction_mixture | 0.5 | -0.005877 | 0.03336 | 0.5 |
| Raw+GramAnchor-m8@0.5 | hybrid_prediction_mixture | 0.5 | -0.009816 | 0.03223 | 0.5 |
| Raw+NystromGram@0.5 | hybrid_prediction_mixture | 0.5 | -0.007699 | 0.02408 | 0.5 |

### Pareto Ranking

| method | track | median_disagreement_reduction | median_relative_task_change |
| --- | --- | --- | --- |
| Raw+GramAnchor-m8@0.5 | hybrid_prediction_mixture | 0.5 | -0.009816 |
| Raw+GramAnchor-m8@0.75 | hybrid_prediction_mixture | 0.75 | -0.004869 |
| GramAnchor | representation | 1 | 0.003549 |
| Raw+GramAnchor@1 | hybrid_prediction_mixture | 1 | 0.003549 |

### Predictive Performance Ranking

| method | track | median_predictive_rank | mean_predictive_rank | median_relative_task_change |
| --- | --- | --- | --- | --- |
| Raw+GramAnchor-m8@0.5 | hybrid_prediction_mixture | 11.5 | 14.03 | -0.009816 |
| Raw+GramAnchor-m8@0.25 | hybrid_prediction_mixture | 11.5 | 15.08 | -0.006152 |
| Raw+GramAnchor@0.5 | hybrid_prediction_mixture | 12.5 | 14.17 | -0.005877 |
| Raw+HybridSpectral-t0.05@0.25 | hybrid_prediction_mixture | 12.75 | 14.16 | -0.006589 |
| Raw+HybridSpectral-t0.10@0.25 | hybrid_prediction_mixture | 12.75 | 14.16 | -0.006589 |
| Raw+NystromGram@0.5 | hybrid_prediction_mixture | 13 | 14.43 | -0.007699 |
| Raw+PCA@0.25 | hybrid_prediction_mixture | 13 | 14.56 | -0.006842 |
| Raw+GramAnchor@0.25 | hybrid_prediction_mixture | 13 | 15.32 | -0.006718 |
| Raw+HybridSpectral-t0.01@0.25 | hybrid_prediction_mixture | 13.25 | 14.31 | -0.00684 |
| Raw+NystromGram@0.25 | hybrid_prediction_mixture | 15 | 14.18 | -0.006711 |
| Raw+HybridSpectral-t0.05@0.5 | hybrid_prediction_mixture | 15 | 14.32 | -0.006303 |
| Raw+HybridSpectral-t0.10@0.5 | hybrid_prediction_mixture | 15 | 14.32 | -0.006303 |
| Raw+GramDistance@0.25 | hybrid_prediction_mixture | 15 | 16.76 | -0.003059 |
| Raw+HybridSpectral-t0.01@0.5 | hybrid_prediction_mixture | 15.5 | 14.52 | -0.006168 |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 15.5 | 17.4 | -0.003428 |

### Paper-Method Score

| method | track | median_disagreement_reduction | median_relative_task_change | failure_fraction | paper_method_score |
| --- | --- | --- | --- | --- | --- |
| GramAnchor | representation | 1 | 0.003549 | 0 | 0.9823 |
| Raw+GramAnchor@1 | hybrid_prediction_mixture | 1 | 0.003549 | 0 | 0.9823 |
| Raw+GramAnchor-m8@1 | hybrid_prediction_mixture | 1 | 0.00666 | 0 | 0.9667 |
| GramAnchor-m8 | representation | 1 | 0.00666 | 0 | 0.9667 |
| Raw+HybridSpectral-t0.05@1 | hybrid_prediction_mixture | 1 | 0.01596 | 0 | 0.9202 |
| Raw+HybridSpectral-t0.10@1 | hybrid_prediction_mixture | 1 | 0.01596 | 0 | 0.9202 |
| HybridSpectral-t0.05 | representation | 1 | 0.01596 | 0 | 0.9202 |
| HybridSpectral-t0.10 | representation | 1 | 0.01596 | 0 | 0.9202 |
| NystromGram | representation | 1 | 0.0191 | 0 | 0.9045 |
| Raw+NystromGram@1 | hybrid_prediction_mixture | 1 | 0.0191 | 0 | 0.9045 |
| Raw+HybridSpectral-t0.01@1 | hybrid_prediction_mixture | 1 | 0.02215 | 0 | 0.8893 |
| HybridSpectral-t0.01 | representation | 1 | 0.02215 | 0 | 0.8893 |
| PCA | representation | 1 | 0.02303 | 0 | 0.8849 |
| Raw+PCA@1 | hybrid_prediction_mixture | 1 | 0.02303 | 0 | 0.8849 |
| Raw+GramDistance@1 | hybrid_prediction_mixture | 1 | 0.03462 | 0 | 0.8269 |

## 10. Frozen Finalists

| method | type | models | config |
| --- | --- | --- | --- |
| BlockAdam+DataInit[equal-HPO] | optimizer | controlled_mlp, tabm_d | {"per_model": {"controlled_mlp": {"initialization": "data_equivariant", "learning_rate_multiplier": 2.0, "optimizer": "block_adam", "optimizer_overrides": {"learning_rate": 0.002}}, "tabm_d": {"initialization": "data_equivariant", "learning_rate_multiplier": 1.0, "optimizer": "block_adam", "optimizer_overrides": {"learning_rate": 0.002}}}} |
| GramAnchor | interface | controlled_mlp, tabm_d, tabicl_v2, tabpfn_2_6, catboost | {"interface": "gram_anchor", "interface_parameters": {"anchors": 16, "normalize": true, "selection": "gram_pivot"}} |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | controlled_mlp, tabm_d, tabicl_v2, tabpfn_2_6, catboost | {"alpha": 0.75, "interface": "gram_anchor", "interface_parameters": {"anchors": 16, "normalize": true, "selection": "gram_pivot"}} |

Exactly three configurations were frozen under SHA `9a567e3e47c35d35db88a9830da6ceedf9534509a831d8962579efc08090749b`. The prospective runner refuses to resolve prospective data before verifying this hash and finalist cap.

## 11. NEW Prospective Results

These seven datasets were untouched until `FINALIST_CONFIGS.json` and its SHA existed. No learning rate, anchor count, alpha, or model setting changed afterward.

| dataset | model | method | raw_disagreement | method_disagreement | reduction | raw_task | method_task | relative_task_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eeg-eye-state | catboost | GramAnchor | 0.1064 | 0 | 1 | 0.4102 | 0.4144 | 0.01896 |
| eeg-eye-state | catboost | Raw+GramAnchor@0.75 | 0.1064 | 0.02661 | 0.75 | 0.4102 | 0.41 | 0.00594 |
| eeg-eye-state | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.1385 | 5.508e-07 | 1 | 0.5646 | 0.5395 | -0.05163 |
| eeg-eye-state | controlled_mlp | GramAnchor | 0.1385 | 0 | 1 | 0.5646 | 0.5328 | -0.0552 |
| eeg-eye-state | controlled_mlp | Raw+GramAnchor@0.75 | 0.1385 | 0.03462 | 0.75 | 0.5646 | 0.5082 | -0.0999 |
| eeg-eye-state | tabicl_v2 | GramAnchor | 0.1798 | 0 | 1 | 0.3184 | 0.3141 | -0.01344 |
| eeg-eye-state | tabicl_v2 | Raw+GramAnchor@0.75 | 0.1798 | 0.04495 | 0.75 | 0.3184 | 0.3028 | -0.04886 |
| eeg-eye-state | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.09465 | 0.02646 | 0.7211 | 0.4303 | 0.5097 | 0.1844 |
| eeg-eye-state | tabm_d | GramAnchor | 0.09465 | 0 | 1 | 0.4303 | 0.4444 | 0.03251 |
| eeg-eye-state | tabm_d | Raw+GramAnchor@0.75 | 0.09465 | 0.02366 | 0.75 | 0.4303 | 0.428 | -0.00174 |
| eeg-eye-state | tabpfn_2_6 | GramAnchor | 0.1774 | 0 | 1 | 0.2833 | 0.3105 | 0.05506 |
| eeg-eye-state | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.1774 | 0.04434 | 0.75 | 0.2833 | 0.3021 | 0.0267 |
| gesture-phase | catboost | GramAnchor | 0.04449 | 0 | 1 | 1.142 | 1.168 | 0.02076 |
| gesture-phase | catboost | Raw+GramAnchor@0.75 | 0.04449 | 0.01112 | 0.75 | 1.142 | 1.157 | 0.01175 |
| gesture-phase | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.04998 | 1.062e-07 | 1 | 1.204 | 1.209 | 0.002624 |
| gesture-phase | controlled_mlp | GramAnchor | 0.04998 | 0 | 1 | 1.204 | 1.245 | 0.03415 |
| gesture-phase | controlled_mlp | Raw+GramAnchor@0.75 | 0.04998 | 0.0125 | 0.75 | 1.204 | 1.222 | 0.01203 |
| gesture-phase | tabicl_v2 | GramAnchor | 0.05684 | 0 | 1 | 1.059 | 1.072 | 0.01295 |
| gesture-phase | tabicl_v2 | Raw+GramAnchor@0.75 | 0.05684 | 0.01421 | 0.75 | 1.059 | 1.064 | 0.005243 |
| gesture-phase | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.0284 | 0.01878 | 0.5828 | 1.252 | 1.267 | 0.01682 |
| gesture-phase | tabm_d | GramAnchor | 0.0284 | 0 | 1 | 1.252 | 1.244 | -0.00651 |
| gesture-phase | tabm_d | Raw+GramAnchor@0.75 | 0.0284 | 0.007101 | 0.75 | 1.252 | 1.242 | -0.008601 |
| gesture-phase | tabpfn_2_6 | GramAnchor | 0.07388 | 0 | 1 | 1.04 | 1.077 | 0.03543 |
| gesture-phase | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.07388 | 0.01847 | 0.75 | 1.04 | 1.057 | 0.01624 |
| pollen | catboost | GramAnchor | 0.1211 | 0 | 1 | 1.65 | 1.665 | 0.009863 |
| pollen | catboost | Raw+GramAnchor@0.75 | 0.1211 | 0.03028 | 0.75 | 1.65 | 1.657 | 0.004984 |
| pollen | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.1526 | 4.788e-07 | 1 | 1.676 | 1.617 | -0.03483 |
| pollen | controlled_mlp | GramAnchor | 0.1526 | 0 | 1 | 1.676 | 1.662 | -0.008316 |
| pollen | controlled_mlp | Raw+GramAnchor@0.75 | 0.1526 | 0.03814 | 0.75 | 1.676 | 1.639 | -0.02182 |
| pollen | tabicl_v2 | GramAnchor | 0.1351 | 0 | 1 | 1.557 | 1.548 | -0.005537 |
| pollen | tabicl_v2 | Raw+GramAnchor@0.75 | 0.1351 | 0.03378 | 0.75 | 1.557 | 1.546 | -0.006826 |
| pollen | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.07856 | 0.006726 | 0.9152 | 1.608 | 1.61 | 0.00282 |
| pollen | tabm_d | GramAnchor | 0.07856 | 0 | 1 | 1.608 | 1.622 | 0.008977 |
| pollen | tabm_d | Raw+GramAnchor@0.75 | 0.07856 | 0.01964 | 0.75 | 1.608 | 1.613 | 0.003223 |
| pollen | tabpfn_2_6 | GramAnchor | 0.141 | 0 | 1 | 1.517 | 1.516 | 0.000178 |
| pollen | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.141 | 0.03526 | 0.75 | 1.517 | 1.513 | -0.001025 |
| satimage | catboost | GramAnchor | 0.03002 | 0 | 1 | 0.3327 | 0.3397 | 0.02102 |
| satimage | catboost | Raw+GramAnchor@0.75 | 0.03002 | 0.007506 | 0.75 | 0.3327 | 0.3367 | 0.01204 |
| satimage | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.05426 | 2.946e-07 | 1 | 0.3007 | 0.2834 | -0.05434 |
| satimage | controlled_mlp | GramAnchor | 0.05426 | 0 | 1 | 0.3007 | 0.2689 | -0.1058 |
| satimage | controlled_mlp | Raw+GramAnchor@0.75 | 0.05426 | 0.01357 | 0.75 | 0.3007 | 0.2651 | -0.1257 |
| satimage | tabicl_v2 | GramAnchor | 0.04472 | 0 | 1 | 0.2124 | 0.2133 | 0.004352 |
| satimage | tabicl_v2 | Raw+GramAnchor@0.75 | 0.04472 | 0.01118 | 0.75 | 0.2124 | 0.2103 | -0.009755 |
| satimage | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.02792 | 0.01135 | 0.5794 | 0.2568 | 0.2786 | 0.08375 |
| satimage | tabm_d | GramAnchor | 0.02792 | 0 | 1 | 0.2568 | 0.256 | -0.004217 |
| satimage | tabm_d | Raw+GramAnchor@0.75 | 0.02792 | 0.00698 | 0.75 | 0.2568 | 0.2537 | -0.0134 |
| satimage | tabpfn_2_6 | GramAnchor | 0.03702 | 0 | 1 | 0.252 | 0.248 | -0.02254 |
| satimage | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.03702 | 0.009256 | 0.75 | 0.252 | 0.2475 | -0.02335 |
| space-ga | catboost | GramAnchor | 0.1442 | 0 | 1 | 0.1035 | 0.1039 | 0.003082 |
| space-ga | catboost | Raw+GramAnchor@0.75 | 0.1442 | 0.03604 | 0.75 | 0.1035 | 0.1034 | -0.001328 |
| space-ga | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.319 | 7.281e-07 | 1 | 0.1107 | 0.1005 | -0.09193 |
| space-ga | controlled_mlp | GramAnchor | 0.319 | 0 | 1 | 0.1107 | 0.09995 | -0.09016 |
| space-ga | controlled_mlp | Raw+GramAnchor@0.75 | 0.319 | 0.07975 | 0.75 | 0.1107 | 0.09894 | -0.09944 |
| space-ga | tabicl_v2 | GramAnchor | 0.1916 | 0 | 1 | 0.09679 | 0.09718 | 0.003991 |
| space-ga | tabicl_v2 | Raw+GramAnchor@0.75 | 0.1916 | 0.04789 | 0.75 | 0.09679 | 0.0964 | -0.004065 |
| space-ga | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.08837 | 0.0137 | 0.845 | 0.1045 | 0.09978 | -0.04563 |
| space-ga | tabm_d | GramAnchor | 0.08837 | 0 | 1 | 0.1045 | 0.09947 | -0.04267 |
| space-ga | tabm_d | Raw+GramAnchor@0.75 | 0.08837 | 0.02209 | 0.75 | 0.1045 | 0.1002 | -0.03799 |
| space-ga | tabpfn_2_6 | GramAnchor | 0.1885 | 0 | 1 | 0.09226 | 0.09385 | 0.01256 |
| space-ga | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.1885 | 0.04713 | 0.75 | 0.09226 | 0.09313 | 0.004749 |
| steel-plates-fault | catboost | GramAnchor | 0.001794 | 0 | 1 | 0.001647 | 0.002003 | 0.2165 |
| steel-plates-fault | catboost | Raw+GramAnchor@0.75 | 0.001794 | 0.0004484 | 0.75 | 0.001647 | 0.001905 | 0.1619 |
| steel-plates-fault | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.01453 | 2.862e-07 | 1 | 2.291e-05 | 0.01564 | 555.2 |
| steel-plates-fault | controlled_mlp | GramAnchor | 0.01453 | 0 | 1 | 2.291e-05 | 0.4378 | 1.662e+04 |
| steel-plates-fault | controlled_mlp | Raw+GramAnchor@0.75 | 0.01453 | 0.003633 | 0.75 | 2.291e-05 | 0.109 | 4299 |
| steel-plates-fault | tabicl_v2 | GramAnchor | 0.005358 | 0 | 1 | 0.0002682 | 0.0005802 | 1.163 |
| steel-plates-fault | tabicl_v2 | Raw+GramAnchor@0.75 | 0.005358 | 0.001339 | 0.75 | 0.0002682 | 0.0005014 | 0.8694 |
| steel-plates-fault | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.009149 | 0.007338 | 0.2763 | 0.001047 | 0.0314 | 16.8 |
| steel-plates-fault | tabm_d | GramAnchor | 0.009149 | 0 | 1 | 0.001047 | 0.122 | 90.5 |
| steel-plates-fault | tabm_d | Raw+GramAnchor@0.75 | 0.009149 | 0.002287 | 0.75 | 0.001047 | 0.06642 | 42.22 |
| steel-plates-fault | tabpfn_2_6 | GramAnchor | 0.005904 | 0 | 1 | 0.0002209 | 9.956e-05 | -0.5883 |
| steel-plates-fault | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.005904 | 0.001476 | 0.75 | 0.0002209 | 0.0001272 | -0.443 |
| wilt | catboost | GramAnchor | 0.03902 | 0 | 1 | 0.05291 | 0.05973 | 0.1289 |
| wilt | catboost | Raw+GramAnchor@0.75 | 0.03902 | 0.009754 | 0.75 | 0.05291 | 0.05737 | 0.08434 |
| wilt | controlled_mlp | BlockAdam+DataInit[equal-HPO] | 0.0634 | 6.368e-08 | 1 | 0.04273 | 0.04591 | -0.06097 |
| wilt | controlled_mlp | GramAnchor | 0.0634 | 0 | 1 | 0.04273 | 0.04267 | -0.02213 |
| wilt | controlled_mlp | Raw+GramAnchor@0.75 | 0.0634 | 0.01585 | 0.75 | 0.04273 | 0.0374 | -0.1029 |
| wilt | tabicl_v2 | GramAnchor | 0.03109 | 0 | 1 | 0.03106 | 0.03401 | 0.09484 |
| wilt | tabicl_v2 | Raw+GramAnchor@0.75 | 0.03109 | 0.007772 | 0.75 | 0.03106 | 0.03291 | 0.05951 |
| wilt | tabm_d | BlockAdam+DataInit[equal-HPO] | 0.03589 | 0.0008866 | 0.9723 | 0.03271 | 0.05689 | 0.8405 |
| wilt | tabm_d | GramAnchor | 0.03589 | 0 | 1 | 0.03271 | 0.03913 | 0.1962 |
| wilt | tabm_d | Raw+GramAnchor@0.75 | 0.03589 | 0.008973 | 0.75 | 0.03271 | 0.0365 | 0.1157 |
| wilt | tabpfn_2_6 | GramAnchor | 0.04005 | 0 | 1 | 0.04424 | 0.03889 | -0.03791 |
| wilt | tabpfn_2_6 | Raw+GramAnchor@0.75 | 0.04005 | 0.01001 | 0.75 | 0.04424 | 0.03937 | -0.03602 |

## 12. Prospective Rankings

### Ranking A — Performance-Preserving Invariance

| method | median_disagreement_reduction | median_relative_task_change | median_worst_orbit_gain | win/tie/loss |
| --- | --- | --- | --- | --- |
| GramAnchor | 1 | 0.008977 | 0.04728 | 40/0/65 |
| BlockAdam+DataInit[equal-HPO] | 0.9901 | 0.008788 | 0.009639 | 15/0/27 |
| Raw+GramAnchor@0.75 | 0.75 | -0.001025 | 0.0468 | 54/0/51 |
| Raw | 0 | 0 | 0 | 0/105/0 |

### Ranking B — Pareto Frontier

| method | track | median_disagreement_reduction | median_relative_task_change | median_worst_orbit_gain |
| --- | --- | --- | --- | --- |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 0.75 | -0.001025 | 0.0468 |
| BlockAdam+DataInit[equal-HPO] | optimizer | 0.9901 | 0.008788 | 0.009639 |
| GramAnchor | interface | 1 | 0.008977 | 0.04728 |

### Ranking C — Predictive Performance

| method | track | median_predictive_rank | mean_predictive_rank | median_relative_task_change |
| --- | --- | --- | --- | --- |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 2 | 1.752 | -0.001025 |
| Raw | baseline | 2 | 2.038 | 0 |
| GramAnchor | interface | 3 | 2.686 | 0.008977 |
| BlockAdam+DataInit[equal-HPO] | optimizer | 3 | 2.81 | 0.008788 |

### Ranking D — Paper-Method Score

| method | track | median_disagreement_reduction | median_relative_task_change | failure_fraction | paper_method_score |
| --- | --- | --- | --- | --- | --- |
| GramAnchor | interface | 1 | 0.008977 | 0 | 0.9551 |
| BlockAdam+DataInit[equal-HPO] | optimizer | 0.9901 | 0.008788 | 0 | 0.9462 |
| Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 0.75 | -0.001025 | 0 | 0.75 |
| Raw | baseline | 0 | 0 | 0 | 0 |

## 13. Method-by-Model Matrix

| method | catboost | controlled_mlp | tabicl_v2 | tabm_d | tabpfn_2_6 |
| --- | --- | --- | --- | --- | --- |
| BlockAdam+DataInit[equal-HPO] | NA | KEEP | NA | NICHE | NA |
| GramAnchor | PROMISING | KEEP | KEEP | PROMISING | KEEP |
| Raw | FAIL | FAIL | FAIL | FAIL | FAIL |
| Raw+GramAnchor@0.75 | PROMISING | KEEP | KEEP | KEEP | KEEP |

Categories apply the frozen KEEP/PROMISING/NICHE/FAIL thresholds within each model family; the overall multi-family KEEP decision remains the authoritative one.

## 14. Strongest Result

The strongest primary-ranking result is **GramAnchor**: 100.00% median prospective disagreement reduction at 0.90% median task change across 5 model families, with 4.73% median worst-orbit gain. This is the most direct evidence that arbitrary-coordinate sensitivity can be reduced without paying the PCA-scale task penalty.

## 15. Strongest Negative Result

The strongest negative result is that mechanistic optimizer equivariance does not guarantee broad predictive parity. Although BlockAdam+DataInit[equal-HPO] has only 0.88% pooled prospective cost, it cost 8.37% on tabm_d and 4.27% on the development panel; only the controlled MLP clears the per-family KEEP gate. The TabM diagonal-adapter closure problem and the altered data-equivariant initialization make the optimizer route substantially less plug-and-play than the interface route.

## 16. Failed Methods and Why

- Default BlockScalarAdam, BlockAdam, and MatrixAdam failed Stage-1 reduction despite correct matched-function symmetry because independent default initializations start from different functions.
- GramDistance discarded too much predictive information; its exact invariance was not enough.
- PCA and most pure spectral interfaces crossed the 1% development task-cost gate.
- The m=8 GramAnchor rescue improved controlled-MLP cost but underperformed m=16 across all five families, so it was not frozen.
- H2 was optional and omitted; H3 was conditionally specified and its trigger failed. Neither is silently counted as a negative empirical result.
- MahalanobisGram is reported as exploratory; ridge/rank sensitivity prevents an orthogonal-result-style general-linear claim.
- The frozen TabPFN setting uses one estimator; on the 576-column satimage and 512-column gesture GramAnchor interfaces it emitted a feature-coverage warning (500-column maximum). The protocol was not changed after freezing, and this is a scaling limitation of that cell rather than a tuned-away exception.

## 17. Mechanistic Interpretation

- **Can blockwise adaptivity retain Adam's task performance?** It retains optimizer-state equivariance, but the prospective task comparison determines whether it retains enough predictive strength; the answer is not uniformly yes.
- **Does full matrix adaptivity help beyond scalar BlockAdam?** It can alter the reduction/cost point, but its extra matrix state did not dominate the simpler alternatives across both trainable architectures.
- **Does data-equivariant initialization matter after optimizer correction?** Yes for orbit fits: it closes the epoch-0 function gap that an equivariant update alone cannot repair. It also changes TabM's diagonal input adapter, which is a real cost.
- **Can invariant Gram interfaces preserve predictive information?** GramAnchor does so much better than PCA, GramDistance, or Nyström on the median, though per-unit losses show information/inductive-bias removal is not free.
- **Is raw-coordinate information genuinely useful?** Yes. Mixtures often improve task error relative to full invariance, demonstrating that some raw basis dependence acts as useful inductive bias.
- **Is a hybrid preferable to full invariance?** It is preferable when the primary objective values near-zero task cost over exact invariance; full GramAnchor remains preferable when exact orthogonal invariance is the requirement.

Condition<=3 is separate from the orthogonal claim:

| method | median_disagreement_reduction | median_relative_task_change | failure_fraction | paper_method_score |
| --- | --- | --- | --- | --- |
| BlockAdam+DataInit[equal-HPO] | 0.03359 | 0.03793 | 0.1667 | -0.1977 |
| GramAnchor | 0.08842 | 0.006785 | 0.2 | 0.004498 |
| MahalanobisGram-lambda0.0001 | 0.1927 | 0.03691 | 0.06667 | -0.008531 |
| MahalanobisGram-lambda0.01 | 0.1252 | 0.02405 | 0 | 0.004941 |
| MahalanobisGram-lambda1e-06 | 0.1521 | 0.03652 | 0.06667 | -0.04715 |
| Raw | 0 | 0 | 0 | 0 |
| Raw+GramAnchor@0.75 | 0.2037 | -0.006508 | 0.06667 | 0.187 |

## 18. Reviewer Attack Audit

### "You merely replaced Adam with SGD."

No. BlockScalar/Block/Matrix optimizers retain first-moment adaptivity and were directly compared with SGD in the matched-function audit; SGD remains a mechanistic control, not the proposed optimizer.

### "The new optimizer loses Adam's performance."

Often it does, and the report treats this as the optimizer track's main negative result. Equal HPO and prospective task costs are shown rather than hidden.

### "The representation throws information away."

The Gram maps can discard coordinate-specific marginal structure even when they preserve within-block geometry. Pure-interface losses and hybrid gains quantify that cost; no sufficiency claim is made.

### "PCA already solves this."

PCA is an invariance baseline, but its task cost and repeated-eigenvalue ambiguity are both worse than the leading Gram candidate on this tournament.

### "The method only handles random rotations."

Finalist interfaces were also tested on local/spectral hat and one-hot/Helmert pairs. Condition<=3 transforms are reported separately, with no false claim that ordinary Gram inner products are generally invariant.

### "It only works for MLPs."

The interface track spans controlled MLP, TabM-D, TabICLv2, TabPFN 2.6, and CatBoost. The optimizer track is intentionally restricted to architectures whose first layer is accessible.

### "The method was tuned on the test datasets."

The prospective panel was locked first; configurations were frozen under `9a567e3e47c35d35db88a9830da6ceedf9534509a831d8962579efc08090749b` before data loading. All alpha/LR/anchor choices used development validation only, and the runner enforces the hash gate.

### "Basis dependence may actually be beneficial."

Agreed in part: hybrid task gains are evidence that raw-coordinate priors can help. The scientific target is harmful arbitrary dependence, not invariance maximization.

## 19. Ranked Candidates for Human Decision

| rank | method | type | prospective reduction | task cost | model breadth | natural-basis success | condition<=3 behavior | complexity | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GramAnchor | interface | 100.00% | 0.90% | 5 | True | 8.84% reduction | one target-free frontend | KEEP |
| 2 | BlockAdam+DataInit[equal-HPO] | optimizer | 99.01% | 0.88% | 2 | N/A (optimizer) | 3.36% reduction | high | NICHE |
| 3 | Raw+GramAnchor@0.75 | hybrid_prediction_mixture | 75.00% | -0.10% | 5 | True | 20.37% reduction | two fits | KEEP |

This is a ranking for human decision, not an automatic paper-method choice.

## 20. Suggested Next Experiment for Each Top-3 Method

- **GramAnchor** — Scale the target-free anchor bank and train size on a larger locked benchmark to test whether task cost falls or rank saturation returns.
- **BlockAdam+DataInit[equal-HPO]** — Run a closure-preserving TabM input-adapter design that transforms as a full block matrix, then repeat equal-HPO without freezing the diagonal adapter.
- **Raw+GramAnchor@0.75** — Learn a validation-only per-dataset gate from target-free rank/spectrum descriptors and compare it with this fixed alpha on a second untouched panel.

## 21. Files Produced

- `configs/NEW_PROSPECTIVE_PANEL.json` and SHA; `configs/TOURNAMENT_PROTOCOL.json`; `configs/STAGE1_SURVIVORS.json` and SHA; `configs/FINALIST_CONFIGS.json` and SHA.
- `results/raw/`: immutable prediction bundles and metadata for Stage 1, Stage 2, equal HPO, natural bases, prospective evaluation, and condition<=3 exploration.
- `results/processed/`: cell tables, coordinate audits, four development rankings, four prospective rankings, method/model categories, ablations, mechanism trajectories, and integrity metadata.
- `figures/figure_1_...` through `figures/figure_8_...` in PNG and PDF.
- `tournament/`: shared representations, optimizers, model adapters, and protocol helpers; `scripts/`: runners, analyzers, freezer, figures, report, and audit; `tests/`: numerical and lock tests.
