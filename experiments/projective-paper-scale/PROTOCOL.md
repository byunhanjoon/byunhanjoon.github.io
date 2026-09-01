# Closest-baseline pilot for projective temporal queries

## Frozen question

Does the four-component projective Gaussian mixture retain useful predictive quality on arbitrary linear queries when compared with (i) MOSES, the closest marginalization-consistent non-Gaussian model, and (ii) TACTiS-2, a flexible joint copula forecaster?

This is a potential screen, not an ICLR-grade final benchmark. In particular, the pre-existing projective-mixture checkpoints saw larger training batches than the official baselines in this compute-limited run.

## Data and repetitions

- JenaWeather, Electricity, and Traffic arrays already frozen under `phasecover-confirmation/raw/data`.
- 32 historical steps, 8 channels, 4 forecast steps (32 future scalars).
- 16,384 deterministic training windows and 4,096 deterministic test windows.
- Seeds: 20261301, 20261302, and 20261303.
- Evaluation subset: the first 1,024 test windows.

## Models

- `projective_mixture4`: existing capacity-matched checkpoint; 136,580 parameters; four diagonal-Gaussian components define one joint future law. It was trained for 3,000 updates with batch 512 on broad scalar-query NLL.
- `direct_mixture4_matched`: existing 137,211-parameter query-conditional control, trained with the same budget as the projective mixture. It does not define a joint future law.
- `moses`: official MOSES code at commit `302aa7dd6a017ebb8390dcbcd2649264b92930e9`; 4 components, latent width 64, 2 flow layers, 1 encoder layer, 2 heads; 129,600 parameters. Train 3,000 updates, batch 32, joint NLL.
- `tactis2`: official TACTiS code at commit `19df68b20b574f662fb1b2e1bf022f4116027f90`; official random-walk demo configuration adapted only from 10 to 8 series; 193,282 total parameters after copula initialization. Train marginal stage for 1,500 updates and copula stage for 1,500 updates, batch 32.

AdamW uses learning rate 4e-4, weight decay 1e-5, and gradient clipping at 1.0. A cell fails rather than silently changing its frozen budget.

## Query families and scoring

For every test context, fix four out-of-sample query families:

1. a signed point query;
2. a two-coordinate difference;
3. a dense unit-norm Gaussian projection;
4. the same kind of dense projection scaled uniformly from 0.3 to 2.7.

Generate 256 predictive samples per context. Joint models generate one future sample and reuse it for every linear projection; the direct model samples each scalar query separately. Score:

- ensemble CRPS for each family and its macro average;
- empirical 50% and 90% central-interval coverage and mean absolute coverage error;
- per-context latency for producing predictions for all four queries;
- train time and parameter count.

The same sample count and ensemble CRPS estimator are used for all models, including the analytically tractable projective mixture.

## Frozen potential gate

Call the result promising only if all hold across seed-averaged dataset results:

1. projective mixture CRPS is no more than 2% worse than MOSES on at least two of three datasets;
2. projective mixture CRPS is no more than 2% worse than TACTiS-2 on at least two of three datasets;
3. its average coverage error is no more than 3 percentage points worse than the better joint baseline;
4. no NaNs or failed cells occur.

Separately report whether it is at least 5x faster for the four-query workload. Speed cannot rescue failure of the predictive-quality gate, but competitive quality plus a large analytic-query speed advantage justifies a full paper-scale study.

## Interpretation guardrail

A pass means “continue”; it does not establish novelty or oral-level strength. A projective-mixture win is conservative evidence only for the baselines, because their compute-limited batches expose them to fewer training examples. Any submission-grade comparison must equalize optimization opportunity, tune on validation data, expand datasets/horizons/query families, and report uncertainty.
