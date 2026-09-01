# Optional-bias certificate across neural backbones — consolidated result

Status: **COMPLETE — SAFETY SIGNAL, PAPER-LEVEL ACTIONABILITY FAILED**

## Sealed architecture transfer

The original five-fold selector was replayed with MLP, ResNet,
FT-Transformer, and TabM residual bases over four datasets and two complete
state splits. All four backbones had positive average pessimistically selected
gain, but the frozen joint calibration gate failed:

- pooled source-by-backbone-by-operator Spearman: `0.514`;
- pooled gain-sign accuracy: `61.8%`;
- FT-Transformer airline split 1 harm: `-0.01203` standardized MSE;
- ResNet also had two negative selected cells.

Thus the five-fold Gaussian rule is not a transferable safety certificate.

## State-level and specification-robust audits

Twenty state folds improve pooled operator correlation to `0.675`. A
Bonferroni family-wise selector removes three of four harms but still selects
the harmful FT-Transformer airline cell. Averaging evidence across backbones is
worse: it selects seven source-splits and produces five harmful backbone
cells, three beyond `-0.002`.

A post-hoc worst-backbone rule jointly corrected over `4 × 9` comparisons is
safe on the replay: zero harms and `+0.00387` mean gain. It selects only both
ACS splits.

## Clean disjoint-state replay

The final development run uses construction states for base/operator fitting,
disjoint validation states for decisions, and untouched test states for
evaluation.

| Rule | Mean test gain | Harmful cells | Practical harms | Selected sources |
|---|---:|---:|---:|---:|
| positive validation mean | +0.01617 | 3 | 2 | 4 |
| per-backbone corrected LCB | +0.00489 | 0 | 0 | 2 |
| worst-backbone joint LCB | +0.00400 | 0 | 0 | **1** |

The worst-backbone rule again selects only ACS. It passes safety and positive-
gain gates but fails the frozen requirement to act on at least two sources.

## Decision

**Do not keep this as the standalone lead.** The exact residual-value identity,
zero-label impossibility, and zero fallback remain useful theory, and the
per-backbone LCB is a meaningful safety baseline. Uniform specification
robustness is too conservative at available semantic-state counts and has only
one-source support.

The result instead motivates the promoted latent-prior direction: make bias
relevance a task variable inferred by a pretrained model, while retaining the
explicit certificate as a conservative comparator and deployment audit.
