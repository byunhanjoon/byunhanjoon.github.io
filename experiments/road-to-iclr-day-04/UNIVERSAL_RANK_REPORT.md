# Universal rank and geometry report

## Question

Can one representation be applied to every scalar column—without a practitioner
declaring it numerical, ordinal, cyclic, or nominal—and improve MLP, ResNet, and
FT-Transformer over Q-PLE and T-PLE?

The main candidate maps every training support value to its empirical midpoint
rank and applies a 16-bin PLE to that coordinate. The same operation is used for
numerical, binary, and categorical columns. This is LightGBM-free and uses only
the training marginal distribution.

## Result

The first seed was encouraging: midpoint-rank PLE beat Q-PLE in 9/11
dataset/architecture cells and T-PLE in 8/11. Delivery ETA also passed its
predeclared three-seed transfer gate (6/9 versus Q-PLE and 8/9 versus T-PLE).

The balanced 33-cell confirmation reverses the broad claim:

| Comparison | Validation wins | Mean gain |
|---|---:|---:|
| Midrank versus Q-PLE | 18/33 | -0.040% |
| Midrank versus T-PLE | 18/33 | -0.076% |

Only Cooking Time and Delivery ETA have positive mean gain against both typed
baselines. Weather is seed-sensitive and Maps Routing favors T-PLE. Midrank is
therefore a useful complementary chart, not a general standalone replacement.

## Atom question

Q-PLE and T-PLE can isolate an atom through repeated quantiles or a supervised
split, but that does not make them equivalent to an exact-level token. The Adult
audit shows that distinction: exact support beats both PLE baselines and an
equal-parameter compressed-bin control across MLP, ResNet, and FT-Transformer.
However, the same exact-level mechanism does not transfer broadly.

An explicit atom-mass construction was also tested. For each value `x`, interval
PLE averages every PLE basis function over

`[P_train(X < x), P_train(X <= x)]`.

It adds no learned parameters; continuous observations nearly collapse to their
midrank, while an atom occupies an interval proportional to its empirical mass.
It passed the Weather/Cooking development gate (4/6 wins, +0.052% mean), then
lost to midpoint rank in all 5 Delivery/Maps transfer cells (-0.074% mean).
Representing atomic mass more faithfully is therefore not sufficient.

## Falsified universal branches

| Branch | Frozen result | Decision |
|---|---|---|
| Exact/hash identity plus rarity Fourier PE on all fields | 0/6 method-gate cells | Stop |
| Cross-fitted exact-over-bin selector with zero-start gates | 1/6 cells | Stop |
| Integer rank-circle Fourier residual | 1/6 cells; -0.611% versus midrank | Stop |
| Atom-interval PLE | 4/6 development, 0/5 transfer | Stop as general method |
| Midpoint-rank PLE alone | 18/33 versus each typed baseline; negative mean | Retain only as a complementary view |

The cyclic experiment included an equal-parameter noninteger-frequency control.
The integer-harmonic branch averaged -0.170% relative to that control, rejecting
the idea that treating every ranked feature as a circle can discover useful
topology automatically.

## Mechanism boundary

The evidence separates two phenomena:

1. Adult contains exact discrete support whose identity is genuinely useful.
2. Across temporal TabReD tables, no single identity, mass, rank, or cyclic chart
   is consistently best.

This motivates retaining several defensible charts and learning a shared
predictor, rather than trying to infer one universal feature type. That follow-up
is TriChart in [`TRICHART_REPORT.md`](TRICHART_REPORT.md).

Machine-readable evidence is in
[`results/universal_rank_decision.json`](results/universal_rank_decision.json),
[`results/universal_rank_confirmation.csv`](results/universal_rank_confirmation.csv),
[`results/universal_interval_panel.csv`](results/universal_interval_panel.csv),
and [`results/universal_cycle_cells.csv`](results/universal_cycle_cells.csv).
