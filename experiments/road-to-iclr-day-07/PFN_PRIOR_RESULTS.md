# Latent structured-field prior — results

Status: **COMPLETE — ANALYTIC TARGET VALIDATED**

## Main result

At the matched 50/50 task mixture, posterior soft routing has lower mean MSE
than always using geometry and always ignoring it in all 9 heat-scale × noise
cells. Mean advantages are `0.25529` and `0.20718`, respectively. It beats hard
posterior routing in 8/9 cells by `0.01045` MSE on average; the lone loss is
`0.000057`, far below the roughly `0.0078` per-method Monte Carlo standard
errors.

The posterior can identify whether geometry generated the task: mean regime
AUROC is `0.871`. Difficulty has the expected phase structure. AUROC ranges
from roughly `0.63` for weakly structured, high-noise tasks to `1.0` for the
strongest smooth prior.

## Prior shift

The predictor deliberately keeps an assumed smooth-task probability of 0.5.
Its average posterior trust changes from `0.279` when the true deployment rate
is 0.1, to `0.500` at 0.5, to `0.721` at 0.9: context evidence partially but not
completely overrides the pretrained mixture.

The mismatched soft rule still beats each single fixed rule on average at all
three deployment mixtures. However, an outcome-informed cell-wise oracle that
chooses the better simple rule is lower by `0.00240` MSE at true prior 0.1 and
`0.00526` at true prior 0.9. Thus Bayes optimality under the pretraining prior
does not provide a deployment-shift guarantee.

## Scientific consequence

This gives a precise target for a geometry-conditioned PFN:

`posterior trust × conditional geometry transfer`.

The network should not be rewarded merely for using a metric. It should be
tested against the analytic posterior across signal/noise/structure phases,
then stressed under a changed task mixture. A small transformer that cannot
learn this phase boundary is not worth scaling; a model that learns it only
in-prior is not deployment-safe.

The simulation is a theorem/implementation check, not evidence about real
tables or transformer learnability. It raises the geometry-conditioned PFN's
raw novelty upside but does not displace the explicit certificate as the
evidence-weighted lead.

