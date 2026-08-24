# Road to ICLR — Day 2 experiments

This directory tests the narrow mechanism left by Day 1: numerical columns may
contain a smooth background plus a few important exact-value atoms.

No Day 2 blog post is generated here. The experiment artifacts are intended to
determine that post's eventual claim.

## Adult point-mass suite

`adult_point_masses.py` freezes the Day 1 Adult PLE + ResNet configuration and
runs two controlled stages.

1. **Attribution on the released split.** Compare the PLE baseline with identity
   views for capital gain, capital loss, and hours per week individually; leave
   each one out; add all three; and replace full identity encoding with three
   train-only mode indicators.
2. **Split stability.** Construct five new stratified splits. Inside each split,
   recompute the out-of-fold residual diagnostic using training data only, then
   compare its selected identity views with parameter-matched mode indicators.

All augmented models are width-adjusted to the parameter budget of the frozen
PLE baseline. Test scores are never used for feature or model selection.

Run the full suite with the same environment used for Day 1:

```bash
python adult_point_masses.py --device cuda
```

The command is resumable unless `--force` is supplied. It writes:

- `results/adult_residual_profile.csv`: exact-value counts, target rates, and
  grouped out-of-fold residuals;
- `results/adult_attribution.csv`: per-seed attribution and leave-one-out runs;
- `results/adult_split_stability.csv`: new-split diagnostic and control runs;
- `results/summary.json`: paired aggregate results and diagnostic selection
  frequencies.

The three Day 1 selected columns correspond to the standard Adult numerical
ordering: capital gain (column 3), capital loss (column 4), and hours per week
(column 5). Their dominant values in the released arrays are 0, 0, and 40.

## Initial results

The attribution stage uses five paired model seeds on the released split.

| Representation | Accuracy | Paired change vs PLE |
|---|---:|---:|
| PLE baseline | 0.8598 ± 0.0012 | — |
| Identity: capital gain | 0.8678 ± 0.0015 | +0.80 pp |
| Identity: capital loss | 0.8621 ± 0.0010 | +0.23 pp |
| Identity: hours per week | 0.8587 ± 0.0010 | −0.11 pp |
| Identity: gain + hours | 0.8688 ± 0.0010 | +0.90 pp |
| Identity: loss + hours | 0.8608 ± 0.0011 | +0.10 pp |
| Identity: gain + loss | **0.8727 ± 0.0008** | **+1.29 pp** |
| Identity: all three | 0.8714 ± 0.0014 | +1.16 pp |
| Mode indicators: 0, 0, 40 | 0.8598 ± 0.0017 | +0.00 pp |

Capital gain is the strongest individual contributor. Capital gain plus capital
loss outperforms the original three-column route, while an identity view for
hours per week is harmful on its own. The three mode indicators do not explain
the gain, so the useful identity signal is not merely `x == mode`.

Across five fresh stratified splits and two model seeds per split, the residual
diagnostic selected columns `3;4;5` in all 10 runs. Diagnostic identity improved
accuracy by 1.01 ± 0.21 points and won all 10 paired comparisons. Matched mode
indicators changed accuracy by −0.03 ± 0.13 points and won 5 of 10.

These results support the existence and split stability of an exact-value
effect, but expose an important weakness in the current selector: residual
between-group variance consistently selects hours per week even though its
marginal identity view does not improve prediction. The next experiment should
compare residual effect size with cross-fitted marginal predictive value and a
per-column regularized gate.

## Cross-dataset and cross-backbone suite

`cross_dataset_models.py` implements that next experiment. It tests five
TabPack datasets, three paired model seeds, and three backbones: an MLP, a
residual network, and the official TabM parameter-efficient ensemble with 16
members. Every augmented model is width-adjusted to the parameter count of its
PLE baseline.

The five representations are:

- `baseline_ple`: schema features plus 128-bin numerical PLE;
- `variance_identity`: the original residual between-group variance selector;
- `utility_identity`: the new cross-fitted predictive-utility selector;
- `utility_top8`: indicators for only the eight most frequent values of each
  utility-selected column;
- `all_identity`: identity views for every numerical column with no more than
  128 training values.

The new selector first obtains numerical-only PLE out-of-fold predictions. In a
second five-fold split, it learns smoothed exact-value corrections to those OOF
residuals and retains a column only when the corrections reduce aggregate
held-out loss and win on at least three folds. Thus feature discovery sees
training labels but no validation or test labels.

The full matrix contains 225 fits:

```bash
python cross_dataset_models.py \
  --datasets churn adult diamond black-friday california \
  --models mlp resnet tabm \
  --seeds 0 1 2

python analyze_cross_dataset.py
```

The table reports paired accuracy-point gains for classification and relative
RMSE reductions for regression. Positive values are better.

| Dataset | Utility selection | MLP | ResNet | TabM |
|---|---|---:|---:|---:|
| Adult | `3;4` | +1.276 | +1.208 | +1.075 |
| Black Friday | `1` | +0.031% | −0.029% | +0.033% |
| Churn | abstain | 0.000 | 0.000 | 0.000 |
| Diamond | abstain | 0.000% | 0.000% | 0.000% |
| California | abstain | 0.000% | 0.000% | 0.000% |

Adult is architecture-independent: utility-selected identity wins all nine
MLP, ResNet, and TabM comparisons. The new selector also fixes the hours/week
false positive, selecting only capital gain and capital loss in all three
diagnostic seeds. Black Friday is much weaker: the chosen 12-level column has a
small and model-dependent effect.

The abstentions matter. Adding every eligible identity view is not generally
safe: on California it changes RMSE by −0.66%, −2.01%, and −0.48% for MLP,
ResNet, and TabM, respectively. It helps Black Friday MLP and TabM by 0.20% and
0.35%, but slightly hurts its ResNet. The original variance selector similarly
produces unstable effects on Churn and Diamond. Residual structure is therefore
not the same as usable exact-value structure.

Artifacts:

- `results/cross_dataset_all.csv`: all 225 per-fit results;
- `results/cross_dataset_paired.csv`: paired changes from PLE;
- `results/cross_dataset_summary.csv`: dataset/model means;
- `results/selector_diagnostics.csv`: train-only per-column utility evidence;
- `results/cross_dataset_summary.json`: compact experiment manifest.

## How many exact values matter?

`adult_value_sparsity.py` varies the number of most frequent retained values for
the two selected Adult columns. The 81-fit sweep uses the same three backbones
and seeds as the main matrix.

| Values per column | MLP gain | ResNet gain | TabM gain |
|---:|---:|---:|---:|
| 1 | +0.037 | +0.023 | −0.055 |
| 2 | +0.135 | +0.027 | −0.055 |
| 4 | +0.178 | −0.023 | +0.010 |
| 8 | +0.645 | +0.594 | +0.657 |
| 16 | +0.913 | +0.864 | +0.803 |
| 32 | +1.063 | +0.917 | +0.821 |
| 64 | +1.163 | +1.110 | +1.056 |
| Full identity | **+1.276** | **+1.208** | **+1.075** |

One mode or a handful of common values is insufficient. The improvement appears
only after roughly eight values per column and continues growing through dozens
of levels. This is evidence for a distributed set of exceptions, not a single
point mass.

```bash
python adult_value_sparsity.py --models mlp resnet tabm --seeds 0 1 2
python analyze_value_sparsity.py --inputs \
  results/adult_value_sparsity_mlp_resnet.csv \
  results/adult_value_sparsity_tabm.csv
```

## Seen versus unseen values

`adult_frequency_generalization.py` reruns paired PLE and full selected identity
models and groups test rows by the minimum training frequency of their capital
gain/loss values. The following numbers average the descriptive paired results
over the three backbones; model seeds share the same test rows and are not
independent samples.

| Training frequency | Test rows | Accuracy gain | Log-loss reduction |
|---:|---:|---:|---:|
| Unseen | 20 | +0.00 | −77.00% |
| 1–9 | 318 | +8.98 | +60.46% |
| 10–99 | 992 | +15.80 | +79.67% |
| 100–999 | 756 | +0.73 | +51.69% |
| 1,000+ | 14,195 | +0.02 | −0.07% |

The aggregate gain is concentrated in rare-but-seen exact values. Very common
values—including the dominant zeros—contribute almost nothing, while the tiny
unseen group receives no accuracy benefit and has unstable, worse log loss.
Identity is therefore a supported-level correction rather than an extrapolating
representation; PLE remains necessary as its fallback.

```bash
python adult_frequency_generalization.py --models mlp resnet tabm
python analyze_frequency_generalization.py \
  --inputs results/adult_frequency_generalization_mlp_resnet.csv \
  results/adult_frequency_generalization_tabm.csv
```

## Can a compact residual map replace identity?

`residual_map_benchmark.py` replaces a full one-hot identity view with one
cross-fitted scalar per selected column. Each scalar is the mean residual left
by a numerical-only PLE model for that exact value. The map is learned in a
second set of training folds, shrunk toward zero by value frequency, and set to
zero for values absent from the fitting fold. Thus PLE remains the fallback and
the extra feature expresses only supported exceptions.

On Adult, the frequency-shrunk map (`alpha=5`) is more accurate than full
identity while adding only two features rather than 204 indicators.

| Backbone | Full identity gain | Residual-map gain | Map wins |
|---|---:|---:|---:|
| MLP | +1.276 pp | **+1.398 pp** | 3/3 |
| ResNet | +1.208 pp | **+1.308 pp** | 3/3 |
| TabM | +1.075 pp | **+1.267 pp** | 3/3 |

This does not solve Black Friday. A map for its one marginally selected column
slightly worsens average RMSE for all three backbones. The useful structure
there is therefore unlikely to be an additive single-column correction.

The dataset panel was expanded with Higgs-small and House from the same TabPack
release. Both the marginal selector and the full downstream run abstain on all
three seeds. A fixed 100k/25k/25k train/validation/test sample of Microsoft,
using 16 PLE bins for scalability, also abstains marginally in its first seed.
These are useful negative controls: the method does not manufacture a feature
merely because a dataset contains low-cardinality numbers.

```bash
python residual_map_benchmark.py \
  --datasets higgs-small house --models mlp resnet tabm

python residual_map_benchmark.py \
  --datasets microsoft --models mlp resnet tabm --seeds 0 --alphas 5 \
  --bins 16 --max-train-rows 100000 --max-eval-rows 25000
```

## The missing signal is sometimes a pair

The marginal diagnostic assumes that exact-value effects add column by column.
`interaction_utility_probe.py` applies the same held-out residual test to every
eligible pair instead, using a smoothed lookup table for the joint state.

| Dataset | Best pair | Joint states | Residual-loss gain | Fold wins |
|---|---:|---:|---:|---:|
| Adult | `3;4` | 203 | +4.600% | 5/5 in 3 seeds |
| Black Friday | `2;3` | 70 | +1.843% | 5/5 in 3 seeds |
| Microsoft sample | `13;120` | 320 | +0.136% | 5/5 in 1 seed |
| Churn | `2;4` | 43 | −4.138% | 0/5 on average |
| Higgs-small | varies | 9 | negative | at most 1/5 |

Diamond, California, and House contain fewer than two eligible numerical
columns, so the pair selector abstains structurally. The stable Black Friday
pair is the first strong evidence that its earlier all-identity result was not
just extra model capacity: two columns that fail marginally produce a large
cross-fitted gain jointly.

`interaction_view_benchmark.py` then compares additive identities, one-hot
joint states, additive residual maps, and one crossed residual map, always at
the PLE baseline's parameter budget. Adult prefers the additive residual map,
which is consistent with separate capital-gain and capital-loss exceptions.
The Black Friday and Microsoft confirmations test whether exposing the joint
state transfers that pair diagnostic to the three neural backbones.

| Dataset and fixed representation | MLP | ResNet | TabM | Paired wins |
|---|---:|---:|---:|---:|
| Adult, additive residual map | +1.398 pp | +1.308 pp | +1.267 pp | 9/9 |
| Black Friday, crossed identity | +0.068% | +0.164% | +0.172% | 7/9 |
| Black Friday, crossed residual map | +0.224% | +0.098% | +0.092% | 8/9 |
| Microsoft sample, crossed identity | +0.274% | +0.119% | +0.031% | 3/3 |

Positive regression entries are relative RMSE reductions. On Black Friday,
both additive pair variants worsen mean RMSE for all three backbones, whereas
both crossed variants improve mean RMSE for all three. The train-only pair
diagnostic therefore transfers directionally, although its 1.84% linear-model
loss gain overstates the smaller neural test gain.

Microsoft is deliberately labeled preliminary: it is one seed on a fixed
100k-row training sample. Crossed identity transfers to all three backbones,
but crossed residual compression is mixed. Together with Black Friday, this
suggests that pair discovery is more mature than the choice of pair encoder.

```bash
python interaction_utility_probe.py \
  --datasets black-friday adult churn diamond california higgs-small house

python interaction_view_benchmark.py \
  --datasets adult black-friday --models mlp resnet tabm --seeds 0 1 2

python analyze_interaction_views.py --inputs \
  results/interaction_view_adult.csv \
  results/interaction_view_black_friday.csv \
  results/interaction_view_microsoft_sample.csv
```

## Current conclusion

Across the initial suite and the new marginal, residual-map, expanded-dataset,
and interaction suites, Day 2 now contains 660 main downstream model fits. The
result is more specific than the original point-mass hypothesis:

> PLE can miss supported exact-value structure in two forms: additive
> per-column exceptions and pair-specific joint states. A train-only residual
> utility test can distinguish them and abstain when neither is supported.

Adult is the clean additive case: two shrunk scalar maps beat hundreds of
identity indicators across all backbones and seeds. Black Friday is the clean
diagnostic interaction case: neither pair member helps marginally, additive
views hurt on average, and crossed views improve every backbone on average.
Higgs-small and Churn reject pair corrections, while Diamond, California, and
House abstain structurally. Microsoft offers a promising second interaction
dataset, but only on one sampled seed.

This is still not an ICLR-level method. Pair selection searches many candidates
with the same cross-validation predictions, Microsoft needs full multi-seed
confirmation, and no single encoder dominates: crossed identity is robust but
wide, while a one-dimensional residual map is compact but can mismatch the
downstream model. The next decisive experiment is nested pair selection on a
larger dataset panel, followed by a backbone-aligned residual correction and a
cardinality-matched random-pair control. That creates a clear Day 3 question:
can joint-state discovery be turned into a compact encoder without losing its
cross-backbone reliability?
