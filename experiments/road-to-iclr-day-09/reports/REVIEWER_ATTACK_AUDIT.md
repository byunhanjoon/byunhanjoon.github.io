# Reviewer attack audit

## Claim boundary after the continuation

The defensible package is a target-alignment benchmark/theory result with scoped
mixture-performance evidence. It is not a state-of-the-art tabular method paper in its
current form.

The final literature recheck strengthens that boundary. The 2026
[TFM ensembling study](https://arxiv.org/abs/2605.18696) already documents an
accuracy/AUC-versus-log-loss calibration trap across 153 OpenML tasks; recent
[DES-AS](https://doi.org/10.1016/j.patcog.2024.110899) and
[DES-bADE](https://doi.org/10.1016/j.asoc.2026.115425) continue to develop dynamic
competence and weighting criteria. Neither soft weighting nor calibration failure is a
novel standalone observation. The narrower opening is the exact information-controlled
separation of routing targets and its task-dependent tail manifestation.

| Attack | Evidence / mitigation | Residual severity |
|---|---|---|
| “Competence routing and stacking are old.” | Explicitly conceded; META-DES, Super Learner, MixturePFN, and TFM ensembling are in the novelty ledger. Novelty is the controlled four-target separation and exact information dial. | high if framed as method novelty; low under benchmark framing |
| “The original method failed.” | M6 G3 failure and unauthorized E4–E10 are preserved. The loss router is a separate fallback opportunity using frozen experts. | low if chronology stays explicit |
| “Real preprocessing used more than 96 rows.” | Original protocols used outer-training-fold feature scaling and regression target normalization. A fresh context-rescaled audit passed classification but not regression. | medium/high for strict few-shot regression; claim must retain outer-fold normalization |
| “Repeated query samples overlap.” | Dataset identity is the inferential unit. Classification confirmation remains positive under a dataset-means-only bootstrap and all leave-one-dataset-out checks. | low for that result; moderate for per-dataset episode CIs |
| “The real lambda was selected after outcomes.” | Explicitly labeled real-development-tuned; lambda=0.1 was frozen before a deterministic five-dataset CC18 confirmation and fresh context-rescaled run. | low for scoped confirmation; fatal to a synthetic-only claim |
| “Five datasets and a tiny effect are fragile.” | +0.000600 log loss, 3/5 positive, dataset-only CI positive; fresh context-rescaled +0.000732, 5/5 positive. Absolute effect remains small. | medium; breadth expansion required |
| “Regression is driven by two outliers.” | 14/16 dataset signs positive, positive median/trimmed mean, every leave-one-out aggregate positive. Worst-decile effect is also 14/16 positive. | low under original normalization |
| “Regression robustness disappears with proper scaling.” | Context-rescaled point gain remains +0.09290 with 4/5 positives but CI crosses zero. Reported as a failed robustness gate. | medium; needs broader rerun |
| “Classification failure is just worse ranking.” | AUC change is near zero; loss harm localizes to worst-decile NLL and high-loss event frequency. | low for diagnostic statement |
| “Tail analyses were selected post hoc.” | Every tail/weight diagnostic is labeled retrospective; total log loss/MSE remain endpoints. | medium; needs prospective tail hypotheses |
| “Only numeric inputs are tested.” | All real claims explicitly say numeric-only; categorical and multiclass tasks are excluded. | high for broad tabular relevance |
| “The expert roster is weak.” | Six lightweight frozen experts make controlled analysis possible, but no modern GBDT/TFM performance comparison exists. | high for SOTA relevance |
| “No external shift benchmark.” | No temporal/grouped/OOD confirmation is claimed; BeyondArena/TabReD remained unauthorized for killed M6. | high for broad external validity |
| “Float storage changed results.” | Parent reconstructions agree within declared 1e-5 float32 tolerances; newest classification result matches within 7.93e-9. | low |
| “Multiple analyses inflate discovery.” | Only separately frozen gates are treated as confirmation; all-panel syntheses and mechanism probes are labeled retrospective. | medium; a single prospective replication is still needed |

## Paper-safe statements

- Nearly perfect generator-family identification can coexist with harmful matched routing.
- Correct predictive-loss assignment and soft aggregation matter independently of hard
  expert identification on PriorDial and the evaluated real panels.
- Under outer-fold normalization, full competence improves numeric regression on two
  disjoint frozen panels and remains sensitivity-positive over 16 identities.
- A real-development-tuned 10% competence step gives a small independent numeric-binary
  improvement and survives context-rescaled affine preprocessing.
- The real task asymmetry is concentrated in opposite-sign pointwise loss tails.

## Statements to avoid

- “new dynamic ensemble selection algorithm”;
- “state-of-the-art tabular prediction”;
- “synthetic-only classification transfer”;
- “strict context-only regression confirmation”;
- “categorical, multiclass, OOD, or broad TabArena coverage”;
- causal language for the retrospective tail and weight-shift associations.

## Minimum next evidence for an ICLR method claim

1. Freeze lambda and all tail hypotheses before a larger identity panel.
2. Fit imputation, scaling, and target normalization strictly inside context.
3. Add native categorical handling and multiclass proper losses.
4. Replace or augment the lightweight experts with modern GBDTs and at least one open TFM.
5. Evaluate dataset-level effects on an untouched IID plus temporal/grouped panel.
6. Compare compute-adjusted performance against fixed ensembles and standard stacking.
