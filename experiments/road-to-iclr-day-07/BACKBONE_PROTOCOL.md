# Frozen protocol — architecture transfer of the optional-bias certificate

Status: **FROZEN BEFORE RESNET, FT-TRANSFORMER, OR TABM OUTCOMES**

Freeze date: 2026-08-30 (Asia/Seoul).

## Question

Is the Day-7 residual geometry/certificate signal specific to the first MLP,
or does it persist across materially different neural tabular backbones?

## Scope and label

This is a development architecture-transfer experiment. It reuses the same
four previously studied sources and two sealed complete-state splits. It is
not fresh-source confirmation. The new outcomes are ResNet, FT-Transformer,
and TabM residual bases; the already sealed MLP is retained unchanged.

The ordinary covariates, state-balanced loss, three-fold row-OOF residuals,
50,000-row fit cap, 40-epoch budget, nine target-independent operators, and
five-fold state-held-out selectors are identical to `NEURAL_PROTOCOL.md`.
Backbone choice changes only the residual base. Seeds include the backbone
name, and every backbone writes to an isolated result directory.

## Fixed backbone definitions

- ResNet: width 128, two pre-normalized residual blocks;
- FT-Transformer: dense stem to eight 32-dimensional tokens, two blocks and
  four attention heads;
- TabM: dense 64-dimensional stem, width 128, two blocks, eight members;
- all use AdamW with learning rate `1e-3`, weight decay `1e-4`, batch size
  2,048, and no outcome-dependent tuning.

TabM is trained by averaging the per-member squared losses and predicts by
averaging members. The semantic state and all derivatives remain excluded
from every residual base.

## Frozen gates

1. At least two of the three new backbones have positive source-balanced gain
   under the pessimistic selector.
2. No new backbone has a pessimistic source mean below `-0.002` standardized
   MSE.
3. The pessimistic selector causes no more harmful task-split cells than the
   positive-mean selector on each backbone.
4. Pooled source-by-backbone-by-operator Spearman correlation is at least
   `0.60` and sign accuracy is at least `70%`.
5. OOF completeness, finite outputs, and fixed-base exclusion pass for all
   cells.

Failure demotes the method to a backbone-conditional observation. Passing is
still only architecture robustness on development sources; a paper-level
claim requires fresh source families and source-level inference.
