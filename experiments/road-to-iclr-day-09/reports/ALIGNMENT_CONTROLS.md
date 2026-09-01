# Routing alignment controls

## Soft aggregation versus hard selection

On the untouched synthetic test, soft competence weighting beats hard CV selection by
0.023797 log loss [0.022615, 0.024993] and 0.024655 MSE
[0.022748, 0.026604]. Hard selection is worse than fixed in classification. The parent
gain is therefore a calibrated aggregation effect, not reliable single-expert recovery.

## Correct versus cyclically broken assignment

The cyclic control preserves every episode's six CV losses, temperature, entropy, and
weight concentration while rotating which expert receives each loss.

| Domain | Task | Correct-assignment gain over five-shift null (95% CI) |
|---|---|---:|
| PriorDial untouched test | Classification | 0.025738 [0.024851, 0.026649] |
| PriorDial untouched test | Regression | 0.555535 [0.540464, 0.570512] |
| 9 real datasets | Classification | 0.009831 [0.002776, 0.020556] |
| 16 real datasets | Regression | 0.566421 [0.096062, 1.251801] |

All four intervals are positive. Episode-varying concentration alone cannot explain the
result; the context loss must be attached to the correct expert. Parent aligned losses
recompute within `1.322e-6`, inside the frozen float32 bound.

## Context scaling

On the five-dataset confirmation panel, competence improves over fixed already at 32
context rows (+0.082574 [0.008910, 0.210109]). The paired gain slope per context doubling
is +0.006651 [-0.024557, 0.046181], so there is no supported scaling trend from 32 to
192 rows. The result is a robust level effect in this range, not a demonstrated learning
curve.

## Combined interpretation

The controls separate four targets:

1. identifying the synthetic mechanism family;
2. selecting the query-best individual expert;
3. attaching noisy predictive evidence to the correct expert;
4. producing a calibrated mixture prediction.

Family identification can be nearly perfect yet harmful; hard expert selection can fail
while soft weighting succeeds; cyclic misassignment fails even with the same weight
spectrum. This target hierarchy is the substantive benchmark insight. Exponential
weighting, dynamic selection, and task-library calibration remain prior art.
