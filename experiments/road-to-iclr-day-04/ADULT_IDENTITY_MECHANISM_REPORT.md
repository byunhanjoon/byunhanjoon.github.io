# What actually helped Adult?

## Frozen 2×2 audit

The Adult identity effect was decomposed along two axes:

- target-free selection of every nonconstant numerical field with training
  cardinality at most 128 (`0,2,3,4,5`) versus the frozen Day 1 supervised
  residual selection (`3,4,5`);
- adding the exact-level embedding to its PLE field token versus preserving it
  as a separate field token.

Every exact representation was paired with an identical-parameter control that
replaced exact-level codes by compressed Q-PLE bin codes. Q-PLE and one-tree-
per-field T-PLE were included. All systems used the same TabPack split, seed,
training budget, and closely matched parameter budget.

## Result

All 12 exact-support Q-PLE cells pass the preregistered validation gate across
MLP, ResNet, and FT-Transformer: exact support beats both plain Q-PLE and its
matched bin-code control.

| Selection | Interface | Architectures passed | Mean validation log-loss advantage over bin control |
| --- | --- | ---: | ---: |
| Target-free | Additive | 3/3 | 0.03628 |
| Target-free | Separate | 3/3 | 0.03410 |
| Supervised residual | Additive | 3/3 | **0.03729** |
| Supervised residual | Separate | 3/3 | 0.03382 |

The supervised additive route is numerically best, but the differences among
the four exact-support mechanisms are small compared with their common gap over
the bin controls. Thus neither supervised selection nor a separate identity
channel is necessary for Adult. Exact discrete support itself is the stable
mechanism.

The selected exact route also improves T-PLE in all architectures:

| Architecture | Validation log-loss delta | Validation AUC delta |
| --- | ---: | ---: |
| MLP | -0.01285 | +0.5280 pp |
| ResNet | -0.01238 | +0.8132 pp |
| FT-Transformer | -0.01148 | +0.6250 pp |

This strengthens the Day 1 conclusion while narrowing its scope: Adult has
unusually valuable exact levels that target-aware interval binning does not
absorb. It does not establish that low-cardinality identity is generally useful.

## Reproduction

```bash
PYTHONPATH=experiments/road-to-iclr-day-04 \
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/adult_identity_mechanism_pilot.py
```

The frozen protocol is `adult_identity_mechanism_config.json`; all 36 fits and
derived mechanism tables are under `results/adult_identity_mechanism*`.
