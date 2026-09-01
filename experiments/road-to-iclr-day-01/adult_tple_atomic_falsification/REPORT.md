# Adult T-PLE atomic falsification

Adult test labels were already inspected on Day 1, so this is a mechanism falsification rather than a confirmatory benchmark.

Validation selected `b32_leaf32_gain0` with 145 total numerical bins (mean validation log loss 0.287262). All neural networks were matched to the original Q-PLE ResNet parameter budget.

## Four-seed ensemble results

| System | Test accuracy | Test AUC | Test log loss |
| --- | ---: | ---: | ---: |
| Q-PLE | 0.8607 | 0.9155 | 0.2981 |
| Q-PLE + full identity | 0.8724 | 0.9279 | 0.2745 |
| T-PLE | 0.8659 | 0.9195 | 0.2905 |
| T-PLE + full identity | 0.8737 | 0.9278 | 0.2748 |
| Q-PLE + atom-only indicators | 0.8613 | 0.9156 | 0.2976 |
| Atom-bracketed Q-PLE | 0.8638 | 0.9157 | 0.2973 |

## Predeclared contrasts

Negative log-loss deltas and positive AUC deltas favor the candidate.

| Contrast | Part | Mean-member log-loss delta | Mean-member AUC delta | Ensemble AUC delta |
| --- | --- | ---: | ---: | ---: |
| full identity over Q-PLE | val | -0.020436 | +1.1637 pp | +1.0967 pp |
| full identity over Q-PLE | test | -0.022418 | +1.2600 pp | +1.2350 pp |
| T-PLE over Q-PLE | val | -0.009402 | +0.5669 pp | +0.5093 pp |
| T-PLE over Q-PLE | test | -0.007647 | +0.4383 pp | +0.3956 pp |
| full identity over T-PLE | val | -0.012744 | +0.6846 pp | +0.6361 pp |
| full identity over T-PLE | test | -0.016815 | +0.8594 pp | +0.8308 pp |
| atom-only indicators over Q-PLE | val | -0.000104 | +0.0337 pp | +0.0341 pp |
| atom-only indicators over Q-PLE | test | -0.000038 | -0.0054 pp | +0.0117 pp |
| atom-bracketing over Q-PLE | val | +0.000186 | +0.0270 pp | +0.0668 pp |
| atom-bracketing over Q-PLE | test | +0.000150 | -0.0365 pp | +0.0221 pp |

## Fixed-test-row uncertainty

These paired bootstrap intervals condition on the already-used Adult test split and do not measure training-seed or dataset uncertainty.

| Contrast | Accuracy delta (95% CI) | AUC delta (95% CI) |
| --- | ---: | ---: |
| T-PLE over Q-PLE | +0.5159 [+0.2640, +0.7432] pp | +0.3956 [+0.2715, +0.5083] pp |
| full identity over T-PLE | +0.7801 [+0.5158, +1.0257] pp | +0.8308 [+0.6781, +0.9829] pp |
| atom-only indicators over Q-PLE | +0.0614 [-0.1106, +0.2211] pp | +0.0117 [-0.0461, +0.0718] pp |

## Decision

- Broader-transfer gate: **PASS**.
- Directional atom-specific gate: **PASS**, but the observed atom-only effect is practically negligible.

The full-identity view encodes every observed value in numerical columns 3, 4, and 5. The atom-only view encodes only values occurring at least `ceil(n_train / 128)` times. Atom-bracketed Q-PLE adds train-support midpoints around those atoms but no equality indicator. T-PLE uses one target-aware one-dimensional decision tree per numerical feature, exactly the construction used by the official PLE implementation.
