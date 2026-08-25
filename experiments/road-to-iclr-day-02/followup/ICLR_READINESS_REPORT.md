# ICLR readiness decision

## Verdict

**Exact-state augmentation is not a standalone ICLR contribution under the
frozen evidence. It is a strong mechanism and case study for the broader
information-equivalent schema-sensitivity paper.**

This decision follows the predeclared stop rule rather than aesthetic judgment:
the method needed nonempty selections on at least 2/6 untouched datasets and
obtained 1/6.

## The paper-level synthesis

The stronger thesis is:

> Tabular learners are not invariant to information-equivalent numerical
> schemas. This is not only a benchmarking nuisance: equivalent bases change
> optimization and regularization geometry. We measure the sensitivity,
> explain concrete failures through basis geometry, and reduce it with a
> compute-matched multi-view intervention.

The exact-state follow-up fills two missing pieces in that paper:

1. **Mechanism.** On finite support, singleton PLE and identity are
   information-equivalent and rank-equivalent, yet Adult's useful columns are
   much worse conditioned in PLE coordinates.
2. **Targeted intervention.** A training-only residual audit finds small basis
   additions that consistently improve Adult and a prospectively discovered
   Miami pair, while abstaining on most untouched datasets.

It does not fill the prevalence or broad method requirements. Those belong to
the 51-dataset schema-invariance protocol.

## Claims that are currently safe

- Exact-state residual structure exists and is architecture-stable on Adult
  and Black Friday.
- Nested selection wins 30/30 outer folds over those two development datasets.
- Adult's support-gated view wins 9/9 MLP, ResNet, and TabM runs.
- Black Friday's view gives a modest 0.13% mean RMSE reduction with 7/9 wins;
  this is supporting, not headline, evidence.
- One prospectively selected Miami interaction wins five nested folds and both
  downstream neural backbones.
- The conservative selector makes 0/40 shuffled-label discoveries and abstains
  on all unpermuted smooth synthetic runs.
- Singleton identity can help through basis geometry without adding information
  or finite-support expressivity.

## Claims that are not safe

- The encoder improves tabular deep learning broadly.
- Exact states are generally missing from PLE's function class.
- The current residual score predicts neural gains across a representative
  dataset population.
- The method is competitive with tuned foundation models or current tabular
  ensembles.
- The selected interactions are semantically causal.

## Submission-critical next work

1. Complete the already-frozen 51-dataset schema-sensitivity prevalence gate.
2. Add LightGBM, XGBoost, TabM, and TabPFN external controls only after that
   gate passes.
3. Evaluate uniform multi-view prediction ensembles and stochastic
   consistency training against seed ensembles and longer training at matched
   compute.
4. Require the intervention to halve median schema sensitivity, lose at most
   0.25% mean accuracy, and win on at least 60% of datasets.
5. Use exact-state augmentation as an interpretable ablation: singleton views
   test reparameterization geometry; pure pairs test added interaction space.
6. Hold out a final dataset block from all intervention design and threshold
   choices.

If the 51-dataset gate fails, the honest paper outcome is a well-powered
negative result or no ICLR submission from this line. More tuning of Adult,
Black Friday, or Miami cannot repair a prevalence failure.
