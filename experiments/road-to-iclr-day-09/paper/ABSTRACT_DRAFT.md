# Abstract draft

Tabular predictors often trade invariance to monotone coordinate changes against access
to informative feature marginals, but observing which data-generating family produced a
table need not reveal how to combine predictors for minimum loss. We formalize the Bayes
log-risk cost of quotienting as conditional mutual information and introduce PriorDial,
a fixed-marginal synthetic construction with an exact mechanism–warp information dial.
On this construction, a classifier identifies the generator family on 99.2% of episodes
yet family-matched expert routing increases log loss. Using the same six frozen experts,
context-only cross-validated loss weighting improves an untouched synthetic test by
0.00547 log loss and 0.25419 standardized MSE, while hard selection and cyclic
loss-to-expert assignments fail. Across real numeric data, full weighting improves MSE
on two frozen regression panels and a 16-dataset sensitivity synthesis, under
outer-training-fold normalization. Full binary-classification transfer fails through
rare high-NLL errors rather than an aggregate AUC loss. A 10% adaptation step selected
on real development data subsequently improves log loss by 0.00060 on a deterministic
five-dataset confirmation panel and by 0.00073 with affine scaling refit inside each
context. These results separate four targets—generator identification, individual expert
selection, loss-to-expert assignment, and calibrated aggregation—and show that their
misalignment appears as task-dependent tail risk. The contribution is an
information-controlled benchmark and diagnosis, not a new ensemble algorithm.

## Candidate titles

1. **Identifiable Is Not Selectable: Target Alignment in Tabular Expert Routing**
2. **When Tabular Metadata Misroutes: Information Dials, Soft Aggregation, and Tail Risk**
3. **Four Targets of Tabular Routing: From Generator Identity to Calibrated Mixtures**

## Required qualification in the main paper

The regression interval under fresh context-only affine rescaling crosses zero; real
regression confirmation must therefore retain its predeclared outer-fold-normalization
scope until a broader strict-context rerun passes.
