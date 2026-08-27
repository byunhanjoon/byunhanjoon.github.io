# Exact-support residual-token transfer

## Question

Does the exact-value identity signal that survived tuned T-PLE on Adult become
a general numerical encoder across MLP, ResNet, FT-Transformer, and temporal
TabReD datasets?

The proposed encoder retains each field's PLE token and adds a gated learned
embedding when its target-free training cardinality is between 2 and 128.
Unseen levels receive a zero residual. The strongest control has exactly the
same support tables, gates, backbone shape, and parameter count, but indexes
the tables with compressed Q-PLE bin codes rather than exact values. T-PLE
uses one training-label-aware one-dimensional regression tree per numerical
field with 16 leaves, minimum leaf size 32, and no impurity cutoff.

All preprocessing uses the 50,000-row training subsample only. Model selection
uses the 15,000-row validation subsample. The Weather and Cooking Time test
partitions have appeared in earlier Day 4 work, so test columns below are
developmental diagnostics rather than confirmation.

## Frozen results

Positive gains favor exact support.

| Dataset | Model | Q-support val gain | Exact vs bin-control val gain | T-support val gain | Q-support test gain | T-support test gain | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cooking Time | FT-Transformer | +0.246% | -0.342% | +0.368% | +0.003% | +0.279% | FAIL |
| Cooking Time | MLP | +0.069% | -0.089% | -0.071% | +0.143% | +0.198% | FAIL |
| Cooking Time | ResNet | +0.017% | -0.066% | +0.237% | +0.241% | -0.000% | FAIL |
| Weather | FT-Transformer | +0.718% | +0.230% | -0.435% | +0.148% | -0.006% | FAIL |
| Weather | MLP | +0.079% | -0.100% | -0.770% | +0.842% | -0.074% | FAIL |
| Weather | ResNet | -0.217% | +0.027% | -0.425% | -0.585% | -0.277% | FAIL |

No cell satisfies the preregistered conjunction: exact-support Q-PLE must beat
both ordinary Q-PLE and the equal-parameter bin control, and exact-support
T-PLE must beat T-PLE. The architecture gate required two passing cells on at
least one development dataset. It therefore fails 0/6, and Delivery ETA was
not run.

## Interpretation

The Adult result does not transfer under a target-free low-cardinality rule and
an additive residual-token interface. Exact support sometimes improves Q-PLE,
but the gain is usually reproduced by generic discrete capacity, and it is not
consistently complementary to T-PLE. Learned mean gates remain close to their
initial value, providing no evidence that the model discovered a sharp
field-selection mechanism.

This narrows rather than completely erases the Day 1 observation. Adult used
supervised residual diagnostics to select only three fields and appended full
identity coordinates as a separate view. The present test deliberately asked
whether a simpler target-free, model-native rule generalized. It did not. Any
continuation must isolate those two differences—supervised field selection and
separate rather than additive support channels—on validation data, with the
same bin-capacity control. It should not proceed by loosening this transfer
gate or reporting Delivery ETA test performance.

## Reproduction

```bash
PYTHONPATH=experiments/road-to-iclr-day-04 \
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/support_identity_transfer_pilot.py
```

The immutable choices are in `support_identity_transfer_config.json`. Raw
fits, prediction arrays, cell comparisons, and the machine-readable decision
are under `results/support_identity_transfer*`.
