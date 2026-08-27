# Cross-fitted, zero-start support routing

## Motivation

Applying exact identity to every low-cardinality TabReD numerical field failed
0/6 architecture–dataset cells. This follow-up implements the proposed
"activate only with enough signal" rule without LightGBM or validation labels.

For each candidate field, five training-only folds compare a smoothed exact-
level predictor with a smoothed Q-PLE-bin predictor. A field is enabled only if
exact levels win at least four folds and improve mean loss by at least 0.1%.
The neural exact route uses a signed gate initialized to exactly zero, so the
initial function is the PLE baseline. The bin control retains identical tables,
gates, backbone, and parameter count.

The selector enables:

- Weather: columns `46,98,99`;
- Cooking Time: columns `0,170`.

## Frozen result

| Dataset | Model | Q-support val gain | Exact vs bin val gain | T-support val gain | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Cooking Time | MLP | +0.059% | +0.000% | -0.045% | FAIL |
| Cooking Time | ResNet | +0.319% | +0.002% | -0.085% | FAIL |
| Cooking Time | FT-Transformer | -0.079% | +0.001% | +0.273% | FAIL |
| Weather | MLP | -0.349% | -0.001% | +0.212% | FAIL |
| Weather | ResNet | +0.322% | -0.038% | -0.077% | FAIL |
| Weather | FT-Transformer | +0.418% | +0.025% | +0.264% | **PASS** |

Only 1/6 cells passes; the frozen architecture gate requires 2/3 on at least
one dataset. Delivery ETA transfer is therefore not authorized.

The zero-start gate is useful as a safety mechanism: irrelevant routes often
stay near zero and avoid the larger always-on failures. But cross-fitted
marginal exact-level signal is not sufficient to predict conditional neural
improvement. This selector is not the general method.

## Hybrid attention–MLP extension

A parallel attention predictor was added to the MLP through a scalar residual
gate initialized at zero. It gives the best Q-PLE validation RMSE among the four
pilot backbones on Cooking Time (`0.46703`) but loses on Weather. Exact support
still fails the control hierarchy on both datasets. The hybrid is therefore a
dataset-dependent architecture candidate, not a universal repair.

## Next branch

The next representation applies one automatic pipeline to every column:

1. empirical-rank PLE coordinates;
2. frequent exact levels plus hashed rare levels;
3. empirical frequency/self-information positional coordinates;
4. zero-start identity and frequency gates;
5. optional zero-start attention residual over the resulting field tokens.

This directly tests high-mass and rare values, new positional encoding,
attention–MLP structure, and the requirement that practitioners need not label
features as numerical or categorical. It must retain equal-capacity bin/hash
and wide-MLP controls.
