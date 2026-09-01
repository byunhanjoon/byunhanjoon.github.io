# Day-09 four-hour continuation: consolidated outcome

## Bottom line

The continuation found both a defensible novelty direction and measurable performance,
but not the original M6 method. The strongest framing is:

> **Identifiable is not selectable:** tabular generator metadata, hard expert choice,
> correct loss-to-expert assignment, and calibrated mixture prediction are distinct
> targets; their mismatch appears as opposite-sign tail risk in classification and
> regression.

This is a benchmark/theory contribution around an existing competence-weighting tool,
not a claim that CV stacking or mixture routing is new. The frozen pool contains six
lightweight mechanism-oriented experts—linear, additive, threshold, interaction,
partition, and periodic—not frontier tabular foundation models.

## Performance ledger

| Result | Scope | Gain over fixed | Evidence status |
|---|---|---:|---|
| full competence, PriorDial classification | 9,600 untouched synthetic episodes | +0.005465 log loss [0.004797, 0.006129] | untouched synthetic pass |
| full competence, PriorDial regression | same | +0.254185 MSE [0.246516, 0.262313] | untouched synthetic pass |
| full competence, real regression breadth | 7 unseen identities | +0.29103 [0.02525, 0.94647] | frozen outer-fold-normalized pass |
| full competence, real regression confirmation | 5 deterministic identities | +0.10045 [0.00173, 0.25055] | independent outer-fold-normalized confirmation |
| full competence, all real regression synthesis | 16 identities | +0.217965 [0.041682, 0.459988] | retrospective outer-fold sensitivity |
| full competence, real binary classification | 9 initial identities | -0.000266 [-0.002976, 0.003167] | rejected; 3/9 positive |
| 10% competence, real binary confirmation | 5 fresh CC18 identities | +0.000600 [0.000038, 0.001471] | independent scoped confirmation |

The regression synthesis has 14/16 positive datasets, positive median and trimmed mean,
and every leave-one-dataset-out mean above zero. The light-classification result is small
(0.134% relative log-loss reduction) but survives a dataset-only bootstrap and every
leave-one-dataset-out check. Its lambda was chosen on earlier real panels, so it is not
synthetic-only transfer.

A final preprocessing audit refit affine scaling inside each context. The light
classification rule strengthened to +0.000732 [0.000142, 0.001680] with 5/5 positives.
Regression retained +0.09290 and 4/5 positives but its interval crossed zero, so the
confirmed regression statement must retain its outer-training-fold normalization scope.

## Novel mechanism evidence

1. **Metadata is not predictive competence.** At rho=1, mechanism identity is recovered
   on 99.2% of classification episodes, yet matched-family routing is harmful.
2. **Selection is not aggregation.** Soft competence beats hard CV choice by 0.023797
   classification log loss and 0.024655 regression MSE; hard classification selection is
   worse than fixed.
3. **Assignment matters.** Cyclically mapping the same weight spectrum to the wrong
   experts harms both tasks on synthetic and real panels.
4. **The real asymmetry lives in tails.** Worst-decile regression squared error improves
   by 1.88738 [0.22577, 5.03685], while unseen-classification worst-decile NLL worsens by
   0.02512 [0.00850, 0.04525] without detectable aggregate AUC change.
5. **Adaptation strength does not transfer equally.** Synthetic and real regression both
   prefer full routing. Larger weight shifts consistently worsen classification tails;
   the real classification curve needs shrinkage and remains dataset-heterogeneous.

Together with the exact PriorDial mutual-information calibration and T5 counterexample,
these controls form a coherent four-target separation rather than a collection of
ensemble ablations.

## Preserved negatives

- The original adaptive raw/rank M6 route failed G3 and remains killed; E4–E10 were never
  authorized.
- Full synthetic-tuned classification transfer failed on unseen real identities.
- Real regression gain did not scale detectably with context size from 32 to 192.
- Two-fold CV remained better than fixed but failed its 0.01 noninferiority margin versus
  three-fold; three-fold stays default.
- Real leave-one-dataset-out calibration did not beat the original regression rule.
- Black Friday and Auction Verification remain explicit regression negatives.

## Best next experiment

Freeze the 10% classification shrinkage before touching a substantially broader binary
panel, then evaluate it once with dataset-level inference and a categorical-capable expert
pipeline. In parallel, test a query-label-free tail proxy computed only from context CV;
the present results do not authorize choosing a threshold from real query tails. A paper
claim should require both broader identity coverage and a stronger modern expert roster.

## Reproducibility state

- The registered loss-alignment and real-transfer runs executed 18,260 episodes and
  446,640 expert fits; immutable-prediction diagnostics added no hidden refits.
- 150 manifest records; 49 unique referenced artifacts; zero missing paths, config hash
  mismatches, or non-finite numeric NPZ arrays.
- 25 tests pass; every runner and analyzer compiles.
- Final workspace size: approximately 358 MiB.
- Primary handoff reports: `LOSS_ALIGNED_ROUTING_RESULTS.md`,
  `OPENML_COMPETENCE_RESULTS.md`, `REAL_DATA_SYNTHESIS_RESULTS.md`,
  `TAIL_RISK_CONTRAST_RESULTS.md`, `SHRINKAGE_TRANSFER_RESULTS.md`, and
  `CLASSIFICATION_SHRINKAGE_CONFIRMATION_RESULTS.md`. Final scope challenges and the
  preprocessing robustness result are in `REVIEWER_ATTACK_AUDIT.md` and
  `CONTEXT_RESCALED_CONFIRMATION_RESULTS.md`.
