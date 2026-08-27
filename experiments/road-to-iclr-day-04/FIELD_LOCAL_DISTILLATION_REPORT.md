# Day 4 continuation: field-local chart distillation

## Decision first

Stop the field-local contrastive branch. It clears only **3/6** validation
cells, below the predeclared **5/6** promotion gate. Mean validation RMSE gain
over PLE is **+0.004%**, and correct cyclic geometry is **-0.009%** worse than
the permuted-geometry control on average. Weather/Cooking test metrics were not
computed, and the method was not transferred to Delivery ETA.

This is the gated follow-up proposed after full-row semantic multi-view
alignment failed. It repairs the granularity problem successfully at the
software level—only declared fields change and every model begins as exact
PLE—but it does not repair the empirical problem.

## Frozen method and controls

The deployed model has one token stream. For each declared cyclic field `j`,
the tokenizer is

```text
t_j(x) = PLE_j(x) + tanh(g_j) R_j(x),       g_j initialized at 0.
```

Thus every method is exactly the paired PLE model at initialization. Ordinary
ordered fields, nominal fields, and binary fields are untouched. The residual
families are:

1. `ple`: no residual;
2. `ple_adapter`: parameter-matched residual rendered from a second PLE map;
3. `semantic_noalign`: Fourier residual, supervised loss only;
4. `semantic_local`: Fourier residual plus field-local stop-gradient
   distillation toward the paired PLE token;
5. `semantic_wrong_local`: the same local loss after permuting cyclic phase
   cells.

The PLE teacher is detached in the local objective. This prevents the
auxiliary loss from moving the baseline representation merely to make
alignment easier. Focused tests verify exact PLE initialization, locality,
teacher gradient isolation, and the wrong-geometry chart.

## Predeclared gate

In each Weather/Cooking × MLP/ResNet/FT-Transformer cell,
`semantic_local` must have lower validation RMSE than all three of:

- PLE;
- the parameter-matched PLE adapter; and
- the aligned wrong-geometry residual.

Promotion required at least five of six cells. Only validation was evaluated.
Had the gate passed, the next stage would have used three seeds and transferred
the unchanged method to Delivery ETA.

## Validation results

Lower is better. A check marks a cell that beats all three gate comparators.

| Dataset | Backbone | PLE | PLE adapter | Semantic, no alignment | Semantic local | Wrong local | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Weather | MLP | 1.679714 | 1.678531 | 1.679088 | **1.675061** | 1.679471 | ✓ |
| Weather | ResNet | 1.664537 | 1.664351 | 1.664328 | **1.663896** | 1.663950 | ✓ |
| Weather | FT-Transformer | 1.661938 | 1.662522 | **1.661257** | 1.663485 | 1.661730 | — |
| Cooking Time | MLP | 0.466932 | **0.466926** | 0.466940 | 0.466963 | 0.466968 | — |
| Cooking Time | ResNet | 0.467766 | 0.467772 | 0.467807 | **0.467704** | 0.467792 | ✓ |
| Cooking Time | FT-Transformer | 0.469476 | 0.469796 | 0.469340 | 0.470436 | **0.469339** | — |

Across the panel, `semantic_local` wins 3/6 against PLE and 4/6 against wrong
geometry. The apparent Weather ResNet advantage over wrong geometry is only
+0.0033%. Cooking FT-Transformer is more diagnostic: the unaligned semantic
residual improves PLE by +0.029%, and wrong local does the same, while correct
local alignment loses -0.204%. The auxiliary objective, not insufficient
adapter capacity, causes the failure there.

## Interpretation

The zero gates solve the safety-at-initialization issue. Field-locality solves
the dilution of three to five meaningful fields among 103–192 total fields.
Neither is enough to make cross-chart alignment useful across architectures.

The MLP Weather cell is directionally interesting: semantic local is +0.277%
over PLE and +0.263% over wrong geometry on validation. Under the frozen gate,
however, it is one development cell—not permission to inspect its test result
or tune the loss around it. ResNet gains are much smaller, and both
FT-Transformer cells reject local distillation.

The appropriate conclusion is not “try a different alignment weight.” That
would select on the same validation panel after seeing the outcome. A future
independent project could test a single-view, identity-initialized spectral
filter without a contrastive objective, but it would need new development
datasets and a new frozen protocol. For this Day 4 branch, the contrastive
hypothesis has received both its global and local tests and should stop.

## Reproduction

- Runner: [`field_local_distillation_pilot.py`](field_local_distillation_pilot.py)
- Gate: [`analyze_field_local_distillation.py`](analyze_field_local_distillation.py)
- Paired validation table:
  [`results/field_local_distillation_comparisons.csv`](results/field_local_distillation_comparisons.csv)
- Machine-readable decision:
  [`results/field_local_distillation_summary.json`](results/field_local_distillation_summary.json)

Every raw row contains `test_evaluated=false`; both metadata companions state
`test_metrics_computed=false`.
