# Adversarial reviewer audit

## Bottom line

OrbitCover is now a credible ICLR project, but not yet a safe submission. Its
best claim is a new *composition and estimand* for finite discrete pipeline
nuisances, not a new orthogonal array, antithetic sampler, U-statistic, or
general optimum. The experiments strongly establish nuisance-risk efficiency;
they do not establish predictive SOTA or reliable held-out model selection.

## Likely attacks and current answers

| Reviewer attack | Current evidence | Honest answer / remaining action |
|---|---|---|
| “This is just randomized OAs plus antithetic CV.” | Propositions 1--35; exact finite/Gaussian operator comparison; discrete mixed-level complete-pipeline tensors; independent outer cross-score. | The ingredients are classical. Novelty is the finite nuisance-quotient estimand and the OA/fANOVA/packed-cross-score composition. Add a direct theorem/notation comparison to Liu--Panigrahi--Soloff and Chattopadhyay--Liu--Panigrahi in the main paper. |
| “The significance treats correlated tensors as IID.” | Source-level aggregation only; 15 unique sources, 15/15 favorable; bootstrap and sign tests at source level; deletion audit. | Do not use cell-level p-values as external inference. Run a preregistered 20--30-source repeated-split benchmark for the submission. |
| “Better validation estimates should improve test selection.” | Original source C is 4/4, but the frozen alternate split is 2/4; earlier repeats are 28/32; Proposition 34 and scale audit isolate partition shift. | Explicitly reject that implication. Claim accurate quotient-risk estimation and validation-side selection; make repeated data partitions or set-valued selection a co-primary remedy. |
| “The theorem only covers squared loss.” | Exact Brier/MSE identities; clipped/smoothed log and AUC repeats are favorable; accuracy is mixed; Taylor/jackknife results are approximate. | Keep quadratic proper risk primary. Present nonlinear scores as bounded/interior sensitivity, never as exact generalization. |
| “Fit count is not compute.” | Timed 128-fit refits and source-C medians around 78 seconds locally; action generation is cheap. | No portable wall-clock or energy claim. Benchmark parallel schedules and compare matched latency before submission. |
| “Strength-2 cannot be universally better.” | Exact pure triple/four-way counterexamples; late source-B failure; interaction phase and strength hierarchy. | State the interaction-spectrum condition and use an independent pilot to select strength. Do not use prediction-dependent optional stopping. |
| “The new panel was chosen after seeing earlier sources.” | Source C was frozen before download/outcomes and passed every gate; its alternate split was separately frozen after original outcomes. | Treat source C as prospective conditional evidence, not a community benchmark. Publish selection criteria and expand prospectively. |

## Submission threshold

Before submission, require: (1) a preregistered 20--30-source panel with at
least three data splits; (2) main-text comparison against the two 2026
antithetic-CV papers; (3) a small reusable implementation; (4) matched latency
and memory reporting; and (5) wording that makes validation/test partition
failure as prominent as the 15/15 nuisance-risk result.
