# TriChart multi-representation report

## Capacity-control correction

The evidence below accurately shows that AnchorTriChart improves its frozen
T-PLE anchor. A later equal-capacity audit shows that this is not sufficient to
attribute the gain to Q-PLE or midrank charts. Replacing the chart residual
with an independently trained T-PLE residual wins 5/11 cells against the
cascade; equivalently, the chart residual wins 6/11 with only +0.055% mean
validation gain. A per-field gated Q/rank residual wins 3/11 (-0.003%).

AnchorTriChart should therefore be read as a successful safe two-model cascade,
not as confirmed chart-specific evidence. The equal-compute follow-up and
Adult atom-controlled result are in
[`HETEROPLE_REPORT.md`](HETEROPLE_REPORT.md).

## Historical result: AnchorTriChart

**AnchorTriChart** first trains a parameter-matched T-PLE predictor,
restores its best validation checkpoint, freezes it in evaluation mode, and
learns a separate residual predictor from

`0.5 * ((Q_token - T_token) + (midrank_token - T_token))`.

A scalar residual gate starts at exactly zero. Epoch zero is retained as an
early-stopping candidate, so the selected model can return the untouched T-PLE
anchor when the extra charts are unhelpful. This uses two backbones for a
corrected prediction; the claim is safety and performance, not compute or
parameter efficiency.

### Three-seed regression confirmation

The frozen regression panel contains 33 dataset/model/seed cells:

| Dataset | Cells | Validation-safe | Strict wins | Mean validation gain | Test wins | Mean test gain |
|---|---:|---:|---:|---:|---:|---:|
| Cooking Time | 9 | 9 | 9 | +0.224% | 5 | +0.134% |
| Delivery ETA | 9 | 9 | 3 | +0.015% | 3 | +0.197% |
| Maps Routing | 6 | 6 | 5 | +0.285% | 5 | +0.264% |
| Weather | 9 | 9 | 8 | +0.653% | 8 | +0.752% |
| **Overall** | **33** | **33** | **25** | **+0.295%** | **21** | **+0.343%** |

Eight cells select the exact epoch-zero fallback. Unlike the earlier shared
model, every dataset has positive mean descriptive test gain and Maps Routing
is positive. Validation non-degradation is structural; the positive test means
are the substantive generalization evidence, although these test partitions
remain developmental because they had already appeared earlier in Day 4.

### Three-seed binary-classification confirmation

The first classification pilot accidentally inherited the shorter regression
schedule. Before multi-seed confirmation, a documented protocol correction
restored the established Adult schedule (batch 256, 40 epochs, patience 6).
The corrected results are:

| Dataset | Cells | Validation-safe | Substantive wins | Mean validation gain | Test wins | Mean test gain |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 9 | 9 | 9 | +0.319% | 9 | +0.227% |
| Churn | 9 | 9 | 9 | +1.133% | 9 | +1.275% |
| Higgs-small | 9 | 9 | 8 | +0.478% | 8 | +0.370% |
| **Overall** | **27** | **27** | **26** | **+0.643%** | **26** | **+0.624%** |

Here gain is relative reduction in binary log-loss. The one non-substantive
Higgs/FT validation cell and its test result differ from the anchor by about
`1e-8`, so it is an effective fallback. The classification gate passes.

### Adult atoms remain a separate mechanism

AnchorTriChart improves its internal T-PLE anchor on Adult, but it does not beat
the earlier exact-support atom intervention. In the seed-27 comparison, the
best validation-selected exact-support method has validation log-loss
0.271/0.276/0.271 for MLP/ResNet/FT, versus 0.283/0.287/0.282 for AnchorTriChart.
The mean gap is +0.0116 log-loss in favor of exact support, and exact support
wins all three architectures.

This resolves the Day 1 question without overclaiming: Q-PLE and T-PLE do encode
empirical atoms through repeated quantile boundaries or supervised splits, but
that does not necessarily give each level its own stable identity parameter.
Adult benefits from such exact identity. AnchorTriChart addresses a different,
more transferable failure mode—complementary coordinate charts—and should be
combined with, rather than presented as replacing, an atom-aware encoder when
that structure is known.

The claim supported before the later capacity audit was:

> A frozen T-PLE anchor plus a zero-start residual is validation-safe across
> the tested temporal regression and binary-classification panels. The later
> control does not establish that Q/midrank charts are the cause. Adult exact
> support remains the stronger specialized atom mechanism.

## Earlier shared-backbone method

### Method

TriChart renders each row through three deterministic, lossless charts:

1. typed Q-PLE;
2. typed T-PLE; and
3. type-agnostic empirical-midrank PLE.

Each chart has its own tokenizer and all three token streams use one shared MLP,
ResNet, or FT-Transformer backbone. Training minimizes the mean supervised MSE
of the three predictions. The selected variant adds `0.1` times their within-row
prediction variance and averages the predictions at inference.

This is representation-level augmentation: the views change the coordinate
system, not the row or label. It is the appropriate analogue of multi-view
augmentation here; arbitrary GAN rows would introduce an additional synthetic-
distribution question before testing the representation hypothesis.

### Why test it

No individual chart survived the broad panel. Their errors were complementary:
a fixed equal ensemble of three independently trained models beat the better of
Q-PLE/T-PLE in 30/33 validation cells (+0.387% mean) and 26/33 descriptive test
cells (+0.315% mean). This justified testing whether a shared backbone could
retain the benefit with fewer model parameters.

### Frozen results

On the six Weather/Cooking development cells:

| Variant | Wins versus both Q/T | Mean gain vs Q | Mean gain vs T |
|---|---:|---:|---:|
| Shared, no alignment | 4/6 | +0.225% | +0.329% |
| Shared + consistency | 6/6 | +0.517% | +0.620% |

The consistency variant was frozen and evaluated over three seeds. The full
validation panel is:

| Dataset | Cells | Wins vs Q | Wins vs T | Wins vs both | Mean vs Q | Mean vs T |
|---|---:|---:|---:|---:|---:|---:|
| Cooking Time | 9 | 9 | 9 | 9 | +0.231% | +0.177% |
| Delivery ETA | 9 | 7 | 9 | 7 | +0.164% | +0.261% |
| Maps Routing | 6 | 5 | 3 | 3 | +0.419% | -0.108% |
| Weather | 9 | 8 | 7 | 7 | +0.519% | +0.693% |
| **Overall** | **33** | **29** | **28** | **26** | **+0.325%** | **+0.289%** |

TriChart also improves over standalone midpoint rank by +0.364% on average.
The selected method shares 40% fewer parameters than the three independent
models on average over the 11 seed-27 cells (about 52–56% fewer for MLP/ResNet
and 4–7% fewer for the small FT configurations). It still evaluates the shared
backbone three times, so this is a parameter-efficiency claim, not a single-view
FLOP or latency claim.

### Frozen verdict

The strict confirmation gate does **not** pass. It required positive mean gain
against both baselines on every dataset; Maps Routing is -0.108% versus T-PLE.
The method nevertheless clears the aggregate win and mean-gain clauses by a
wide margin. The correct claim is:

> Shared multi-chart consistency is broadly useful across these temporal tables
> and backbones, but supervised T-PLE remains better for high-dimensional Maps
> Routing ResNet cells.

Descriptive test results are weaker than validation: TriChart beats the better
typed baseline in 18/33 cells, with +0.114% mean gain. Weather is strongly
positive, while Cooking, Delivery, and Maps have slightly negative per-dataset
means. Since every test partition had appeared earlier in Day 4, validation is
the frozen endpoint and neither result is confirmatory benchmark evidence.

### Relation to the Adult finding

Adult's robust signal is exact level identity, not the original selector or
token interface. That mechanism does not become a universal residual branch.
TriChart generalizes a different lesson: when defensible representations win on
different cells, keep them as parallel charts and regularize their predictions
instead of forcing one global feature ontology.

This also answers the Q/T-PLE atom concern. They often handle atoms adequately,
and exact identity is special on Adult. The broader value comes from chart
complementarity, not from claiming that PLE universally mishandles atoms.

## Novelty and next falsification

The potentially novel unit is not ensembling by itself. It is the combination
of supervised Q/T/midrank charts, a shared architecture-agnostic backbone, and
explicit prediction consistency. Before a paper claim, the decisive next tests
are:

- an untouched dataset family and untouched test partitions;
- comparison to three independent models at matched training compute, not only
  shared parameter count;
- an atom-aware frozen anchor to test whether exact support and chart residuals
  are additive on Adult without sacrificing the target-free method;
- an untouched classification and regression family with test partitions that
  have never been inspected during Day 4; and
- comparison to recent multi-view and additive/multiplicative tabular attention
  methods under the same tuning budget.

The POMO/Pointerformer analogy should remain narrow: their augmentations exploit
known routing symmetries. TriChart similarly uses label-preserving coordinate
views; it does not justify arbitrary feature permutations or synthetic GAN rows.

Reproduce the analysis with:

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/analyze_trichart.py

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/analyze_trichart_frozen_anchor.py
```

Key artifacts are
[`results/trichart_frozen_anchor_decision.json`](results/trichart_frozen_anchor_decision.json),
[`results/trichart_frozen_anchor_confirmation.csv`](results/trichart_frozen_anchor_confirmation.csv),
[`results/trichart_frozen_anchor_classification_confirmation.csv`](results/trichart_frozen_anchor_classification_confirmation.csv),
[`results/trichart_frozen_anchor_adult_exact_comparison.csv`](results/trichart_frozen_anchor_adult_exact_comparison.csv),
[`results/trichart_decision.json`](results/trichart_decision.json),
[`results/trichart_confirmation.csv`](results/trichart_confirmation.csv),
[`results/trichart_summary_by_dataset.csv`](results/trichart_summary_by_dataset.csv),
and [`results/trichart_independent_ensemble.csv`](results/trichart_independent_ensemble.csv).
