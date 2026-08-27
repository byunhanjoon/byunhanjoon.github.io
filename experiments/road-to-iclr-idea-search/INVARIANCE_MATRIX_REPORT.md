# Day 3 invariance matrix report

This extension varies one declared nuisance at a time, first with the
identity-selected configuration frozen and then with validation selection
rerun in every view. The preregistered numerical tolerance is
`1e-10`; `1e-6` is reported separately as a practical scale, not
as a replacement threshold.

## Frozen-fit maximum aligned probability deviations

Each entry is the maximum absolute change from the identity view across
evaluation rows and aligned output classes. A dash means the factor is not
applicable (Otto has no categorical fields).

### Adult

| pipeline | feature order | category IDs | class IDs | numeric units |
|---|---:|---:|---:|---:|
| one-hot logistic | 4.69e-12 | 7.03e-12 | 9.21e-15 | 2.44e-12 |
| ordinal forest (sqrt) | 0.183 | 0.425 | 0 | 0.0583 |
| ordinal forest (all) | 0.0331 | 0.402 | 2.22e-16 | 0.0422 |
| one-hot forest (all) | 0.0125 | 0.0102 | 2.22e-16 | 2.22e-16 |
| native HistGB | 0 | 0 | 0.216 | 0 |
| native CatBoost | 0.166 | 0 | 0.123 | 0 |

### Churn

| pipeline | feature order | category IDs | class IDs | numeric units |
|---|---:|---:|---:|---:|
| one-hot logistic | 2.22e-16 | 2.22e-16 | 2.78e-16 | 4.44e-16 |
| ordinal forest (sqrt) | 0.148 | 0.23 | 2.22e-16 | 0.0169 |
| ordinal forest (all) | 0.0114 | 0.404 | 3.33e-16 | 0.0142 |
| one-hot forest (all) | 0.0114 | 0.0107 | 3.33e-16 | 2.22e-16 |
| native HistGB | 0 | 0 | 0.0957 | 0 |
| native CatBoost | 0.159 | 0 | 0.105 | 0 |

### Otto

| pipeline | feature order | category IDs | class IDs | numeric units |
|---|---:|---:|---:|---:|
| one-hot logistic | 5.4e-07 | — | 4.85e-07 | 5.84e-07 |
| ordinal forest (sqrt) | 0.208 | — | 0 | 0.05 |
| ordinal forest (all) | 0.158 | — | 0 | 0.125 |
| one-hot forest (all) | 0.158 | — | 0 | 0 |
| native HistGB | 0.214 | — | 4.44e-16 | 0 |
| native CatBoost | 0.297 | — | 1.05e-15 | 0.0734 |

## Selection switches

| dataset | pipeline | nuisance | fraction of views differing from identity choice |
|---|---|---|---:|
| Adult | ordinal forest (sqrt) | category IDs | 75% |
| Adult | ordinal forest (all) | category IDs | 50% |
| Adult | native CatBoost | feature order | 25% |
| Adult | native CatBoost | class IDs | 50% |
| Churn | ordinal forest (sqrt) | feature order | 25% |
| Churn | ordinal forest (sqrt) | category IDs | 75% |
| Churn | ordinal forest (all) | category IDs | 25% |
| Churn | native CatBoost | feature order | 50% |
| Churn | native CatBoost | class IDs | 50% |
| Otto | ordinal forest (all) | feature order | 25% |
| Otto | one-hot forest (all) | feature order | 25% |
| Otto | native CatBoost | feature order | 50% |

## Main reading

- Native CatBoost and HistGB are exactly invariant to the tested category-ID permutations on Adult and Churn. The ordinal-code forests are not.
- One-hot logistic is invariant to machine precision on the binary tasks and within `6e-7` on multiclass Otto; it never changes its selected regularization.
- Feature order changes seeded forests and CatBoost, and changes multiclass HistGB on Otto even when the chosen configuration remains fixed.
- Binary class-ID reversal changes HistGB and CatBoost fits after output alignment, while the forests remain invariant. Multiclass Otto is invariant for both boosting pipelines in this menu.
- HistGB is exactly invariant to every tested positive affine unit change. Standardized one-hot forests are also exact; raw forests are not, despite the tree hypothesis class admitting the same partitions.

## Float32 unit diagnostic

One possible mechanism for the raw-forest unit result is finite-precision
split generation. The following diagnostic asks only whether casting the
affine rewrite to float32 changes a training equality boundary or a
test-to-training rank interval.

| dataset | changed float32 training boundary | changed test rank interval |
|---|---:|---:|
| Adult | false | false |
| Churn | true | true |
| Otto | false | false |

The diagnostic finds direct float32 rank changes only on Churn. It does
not explain the Adult or Otto changes, so threshold arithmetic, tied split
selection, and other implementation details remain hypotheses rather than
established mechanisms.

## Boundary of the claim

The matrix is a controlled three-dataset audit, not an estimate of how
often model families fail these invariances in the wild. The correct
claim is transformation-specific: native categorical handling absorbed
category renaming here, while other schema choices still reached fitting
and selection.
