# Frozen protocol — can a small PFN learn when field geometry transfers?

Status: **FROZEN BEFORE ANY LEARNED-PFN OUTCOME**

Freeze date: 2026-08-30 (Asia/Seoul).

## Hypothesis

A transformer pretrained on a mixture of smooth and unstructured semantic
state tasks can amortize the Bayes rule identified in `PFN_PRIOR_RESULTS.md`:
infer from context whether the declared geometry generated the residual task,
then softly route the geometry-conditioned prediction. This is stronger than
always injecting metadata and different from a hand-coded source-only
certificate.

## Task prior

The generator is the analytic 32-state cycle experiment:

- 20 randomly observed context states and 12 query states per task;
- smooth heat-kernel or independent Gaussian state effects with equal
  pretraining probability;
- heat scales `{0.3, 1.0, 3.0}` and observation noise
  `{0.1, 0.3, 1.0}`, sampled uniformly;
- scale and noise are supplied as task descriptors;
- context values are noisy; query effects never enter the input or routing.

## Models and compute

- `structured`: three-layer, width-64, four-head set transformer whose tokens
  receive fixed cycle Fourier coordinates;
- `set`: parameter-matched transformer without coordinates, so it observes
  values, masks, scale, and noise but not state relations;
- three seeds per model, 4,000 AdamW steps, batch size 512;
- no positional embeddings: token order alone cannot leak the cycle;
- query-state MSE is the sole training objective.

Analytic comparators are zero, always-smooth Gaussian conditioning, hard
posterior routing, Bayes posterior mixture, and the regime oracle. Evaluation
uses 4,096 new tasks in every prior × scale × noise cell for true deployment
smooth-task rates `{0.1, 0.5, 0.9}`.

## Frozen gates

At the matched 0.5 task mixture, after averaging seeds:

1. structured beats the geometry-free set transformer in at least 7/9 phase
   cells and by at least `0.02` MSE on average;
2. structured mean regret to the analytic Bayes mixture is at most `0.05`;
3. its implicit trust coefficient has correlation at least `0.70` with the
   analytic posterior and regime AUROC at least `0.75`;
4. structured beats both always-zero and always-smooth in at least 7/9 cells;
5. all three seeds are finite and the structured advantage over set has the
   same sign in at least two seeds.

Also report prior-shift regret at true rates 0.1 and 0.9. Passing establishes
learnability of the theorem-defined task family, not novelty or real-table
utility. Failure kills the current small-transformer realization before any
expensive pretraining scale-up.
