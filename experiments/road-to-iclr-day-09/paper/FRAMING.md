# Paper framing and figure plan

## One-sentence thesis

In tabular expert systems, learning which generator produced a dataset is not equivalent
to learning which expert to select or how to aggregate experts; an exact information dial
makes this separation measurable, and the resulting misalignment appears as
task-dependent tail risk.

## Contribution order

1. **Information:** quotienting has an exact conditional-mutual-information log-risk
   cost, and PriorDial varies mechanism–warp information without changing marginals.
2. **Counterexample:** perfectly identified, label-informative metadata can still make a
   nominally matched expert worse than a fixed mixture.
3. **Controlled separation:** hard selection, correct expert assignment, and calibrated
   soft aggregation give different answers on the same frozen predictions.
4. **External behavior:** full adaptation transfers under the frozen regression
   normalization but creates classification tail failures; light real-tuned shrinkage
   gives a small independent binary improvement.

## Causal chain to visualize

```text
fixed-marginal information dial
          │
          ├── generator family identification (99.2%) ──X── predictive safety
          │
context predictive losses
          │
          ├── hard argmin ──X── classification performance
          └── correct loss→expert assignment ──> soft mixture ──> held-out gain
                                                        │
                              regression tail suppressed / classification tail amplified
```

## Four-figure main paper

1. PriorDial information calibration plus mechanism identification and harmful matched
   classification routing (`fallback_dial_information_performance_v1.png`).
2. Soft-versus-hard and cyclic assignment controls, emphasizing the four targets.
3. Real dataset forest (`real_data_synthesis_v1.png`) plus opposite-sign tail inset.
4. Synthetic-to-real shrinkage curve (`shrinkage_transfer_v1.png`) and the independent
   classification confirmation forest (`classification_shrinkage_confirmation_v1.png`).

## What belongs in the appendix

The killed M6 sequence, axis/family decompositions, cross-fitted real calibration,
context scaling, CV-fold budget, full per-dataset tables, preprocessing robustness, and
the complete reviewer attack audit.

## Submission decision rule

Treat the current work as a strong direction memo, not a complete method submission,
until a broader strict-context panel and modern expert roster pass. The present novelty
is sufficient for a benchmark/analysis paper pitch; the performance evidence is not yet
sufficient for a state-of-the-art method pitch.
