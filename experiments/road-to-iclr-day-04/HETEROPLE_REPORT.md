# HeteroPLE and exact-atom additivity report

## Bottom line

The clean surviving general method is **HeteroBag-3**:

> Start with a three-member T-PLE bag and replace one member by a fixed
> alternate representation: Q-PLE for classification and empirical-midrank
> PLE for regression. On a fresh frozen OpenML panel, this beats T+T+T in
> 10/12 test cells (+0.843% mean), is positive in both task families, and has
> positive mean gain on all four datasets. Model count, active parameters,
> seeds, and training budget match exactly.

The complementary mechanism result is **atom-aware HeteroPLE**. On Adult, an
equal ensemble of atom-aware T-PLE and atom-aware Q-PLE beats an equal-compute
ensemble of two atom-aware T-PLE models in 2/3 frozen MLP/ResNet/FT validation
cells (+0.424% mean). The same comparison is positive in a post-gate
attention-plus-MLP hybrid (+0.293%). This shows that the broad HeteroBag signal
is not merely recovering atoms missed by T-PLE.

## Why Q-PLE and T-PLE do not settle the atom question

Q-PLE can place a quantile knot at a repeated value. T-PLE can isolate a value
when a supervised split, leaf-size constraint, and bin budget permit it. Both
therefore react to atoms. Neither construction guarantees a dedicated learned
identity vector for each repeated numerical level:

- duplicate quantiles can collapse and are padded to obtain a valid strict
  basis;
- a value at a knot is still represented through the ordered piecewise-linear
  coordinate system; and
- T-PLE isolates only thresholds selected by its target-aware tree.

The exact-support residual instead adds a learned lookup vector for a training
level and uses zero for unseen levels. The earlier Adult mechanism audit showed
that exact indexing, rather than the selector or additive/separate interface,
was the stable source of the gain. Thus an atom may be visible to PLE without
being represented as an independent identity.

## Broad heterogeneous-view experiment

The basic representation-level augmentation is deliberately simple. Train two
models of the same architecture and active parameter count with two coordinate
charts of the same table, then average their predictions with a fixed 50/50
weight. The control trains two T-PLE models with the same seeds and compute.

The development panel selected T+Q or T+midrank per validation cell. It won
11/12 validation cells (+1.384% mean) and 12/12 descriptive test cells
(+1.607%). A new selection panel won 10/12 test cells (+1.990%), but that
procedure trained three candidate members and therefore was not the final
equal-search-compute claim.

A policy was then frozen before a third OpenML panel:

- classification: T-PLE + Q-PLE;
- regression: T-PLE + empirical-midrank PLE.

This fixed policy trained and deployed exactly two models. It won 10/12 test
cells and had positive per-dataset mean gains on 3/4 datasets. It nevertheless
failed the predeclared gate. One Banknote/ResNet cell changed log loss from
0.000975 for T+T to 0.001823 for T+Q, an -87.0% relative result. Consequently,
the unweighted mean test gain was -3.307% overall and -11.513% for
classification, while regression was positive in 6/6 cells (+4.900% mean).
No post-hoc robust aggregation replaces that frozen decision.

This is promising evidence for heterogeneous error structure, especially in
regression, but it did not promote the two-member policy as a broad default.

## Prospective equal-compute result: HeteroBag-3

The two-member failure suggested a simple robustness correction that also
permits an exact compute control. Keep two ordinary T-PLE members and replace
only the third:

```text
classification candidate = T(seed A) + T(seed B) + Q(seed C)
regression candidate     = T(seed A) + T(seed B) + rank(seed C)
control                  = T(seed A) + T(seed B) + T(seed C)
```

All predictions are averaged with weight 1/3. The panel and policy were frozen
from OpenML metadata before downloading values or labels: Ionosphere (59),
Diabetes (37), CPU Small (562), and House-16H (574). The gate required at least
8/12 test wins, positive mean gain overall and in both task families, and
positive dataset means on at least 3/4 datasets.

| Partition | Test wins | Mean relative gain |
|---|---:|---:|
| Classification | 4/6 | +1.084% log loss |
| Regression | 6/6 | +0.602% RMSE |
| **Overall** | **10/12** | **+0.843%** |

Dataset mean gains are +1.360% on Ionosphere, +0.809% on Diabetes, +0.316% on
CPU Small, and +0.888% on House-16H. Median cell gain is +0.768%. Every active-
parameter count matches exactly between the replaced member and its T-PLE
control. The full predeclared gate passes.

This is the first Day-4 result that isolates representation value from both a
single wider network and an equal-size homogeneous ensemble on untouched
datasets. It requires no practitioner field typing: OpenML dtypes determine
the ordinary numeric/categorical split, and the alternate numerical chart is
fixed from task family rather than selected per dataset or feature.

## Atom-controlled additivity experiment

The sharper Adult experiment holds the validated atom mechanism constant in
both arms. Every member uses additive exact embeddings on the frozen Day-1
residual fields 3, 4, and 5. It compares:

```text
candidate = 0.5 * (T-PLE + exact atoms) + 0.5 * (Q-PLE + exact atoms)
control   = 0.5 * (T-PLE + exact atoms) + 0.5 * (T-PLE + exact atoms)
```

The members have exactly equal parameter counts within each architecture and
use the same two seeds, training schedule, optimizer, and early stopping rule.
The gate was fixed before the new second-member seed: at least 2/3 validation
wins and positive mean validation gain.

| Backbone | Validation gain | Descriptive test gain | Parameters/member |
|---|---:|---:|---:|
| MLP | +0.372% | +0.078% | 65,258 |
| ResNet | +0.916% | +1.023% | 181,346 |
| FT-Transformer | -0.015% | -0.087% | 16,256 |
| Attention+MLP hybrid | +0.293% | +0.274% | 60,246 |

The frozen three-architecture gate passes, and the post-gate hybrid check is
positive. FT-Transformer is effectively a tie and prevents an architecture-
universal claim.

## What is genuinely new and what is not

Ordinary T+Q prediction averaging is an ensemble baseline. HeteroBag-3 makes
the stronger controlled claim that deliberate representation heterogeneity is
better than spending the same third-model budget on another seed of the
established representation. The useful research hypothesis is the
factorization:

```text
field representation = ordered chart + explicit atomic identity
model diversity       = multiple defensible charts of that same field
```

The controlled result says the second term adds after the first; it is not just
recovering atoms that T-PLE missed. A general method would need a train-only,
target-free or cross-fitted rule for when to create the atomic channel. The
current automatic low-cardinality and signal-gated rules already failed broad
Weather/Cooking transfer, so the Adult selector cannot simply be declared
universal.

The next decisive experiments are a second prospective HeteroBag-3 panel with
multiple seed triplets, and a multi-dataset atom panel with an automatic atom
rule and T-exact+T-exact as the primary equal-compute control. HeteroBag-3 is a
validated candidate general method on the present panel; atom-aware HeteroPLE
remains a one-dataset mechanism extension.

## Reproduction

The broad audit is generated by `analyze_multiview_equal_compute.py`. The Adult
comparison is run by `adult_exact_multiview_pilot.py` with
`adult_exact_multiview_config.json`; the hybrid transport uses
`adult_exact_multiview_hybrid_config.json`. Frozen decisions are in
`results/multiview_equal_compute_decision.json` and
`results/adult_exact_multiview_decision.json`. HeteroBag-3 is run by
`heterobag_three_member_pilot.py`, audited by
`analyze_heterobag_three_member.py`, and frozen in
`results/heterobag_three_member_decision.json`.
