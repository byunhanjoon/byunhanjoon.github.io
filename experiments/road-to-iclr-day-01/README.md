# Road to ICLR — Day 1

These experiments accompany [Day 1: When “Numerical” and “Categorical” Aren’t Types](https://byunhanjoon.github.io/blogposts/road-to-iclr-day-01.html).

## Synthetic sanity check

It compares three preprocessing routes with the same ridge readout:

- `schema_only`: raw standardized numerical columns and one-hot categorical columns;
- `semantic_oracle`: the right route for each synthetic feature’s known behavior;
- `multi_view`: raw, piecewise-linear, and one-hot views for every discrete feature.

This experiment is intentionally small. It tests whether representation choice alone can recover signal that a binary schema hides; it is not a neural-model benchmark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python feature_semantics.py
```

The command writes the aggregate results to `results.csv`.

## Focused real-data benchmark

The real-data experiment uses the public, preprocessed arrays and exact default
train/validation/test splits released with
[TabPack](https://github.com/yandex-research/tabpack). The four primary datasets
were selected before inspecting Day 1 results:

| Dataset | Role in the panel |
| --- | --- |
| Adult | Mixed numerical, binary, nominal, and plausibly ordinal features |
| Diamond | Regression with three explicitly ordinal categorical features |
| Black Friday | Larger regression task with several low-cardinality numerical codes |
| California Housing | All-numerical negative control |

Churn is available as a quick end-to-end smoke test, but it is not part of the
primary aggregate. The benchmark compares one fixed MLP backbone under three
feature front ends:

- `schema`: quantile-normalized numerical values, binary values, and one-hot
  categorical identities;
- `schema_ple`: `schema` plus piecewise-linear numerical views, providing the
  strong numerical-embedding baseline;
- `multi_view`: `schema_ple` plus identity views for numerical columns with at
  most 32 distinct training values, and smoothed five-fold out-of-fold target
  views (raw and PLE) for categorical columns.

Only training data are used to fit preprocessing. Validation controls early
stopping, and test data are touched once per seed for final reporting. Run five
seeds and report the mean and standard deviation, parameter counts, and runtime.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_tabpack_data.py
python real_data_benchmark.py
```

The second command writes per-seed results incrementally to `real_results.csv`.
For a quick pipeline check:

```bash
python real_data_benchmark.py \
  --datasets churn \
  --seeds 0 \
  --max-epochs 30 \
  --patience 5 \
  --output smoke_results.csv
```

The primary comparison is `multi_view` versus `schema_ple`, not versus the weak
`schema` baseline. This distinction prevents a known benefit of numerical PLE
from being misattributed to the broader feature-semantics hypothesis.

### Five-seed result

| Dataset | Metric | Schema | Schema + PLE | Multi-view | Multi-view vs PLE |
| --- | --- | ---: | ---: | ---: | ---: |
| Adult | Accuracy ↑ | 0.8552 ± 0.0022 | **0.8561 ± 0.0010** | 0.8553 ± 0.0012 | −0.09% |
| Diamond | RMSE ↓ | 0.1423 ± 0.0010 | **0.1395 ± 0.0021** | 0.1436 ± 0.0013 | −2.93% |
| Black Friday | RMSE ↓ | 0.7012 ± 0.0005 | 0.6946 ± 0.0011 | **0.6930 ± 0.0019** | +0.23% |
| California | RMSE ↓ | 0.5029 ± 0.0040 | **0.4950 ± 0.0056** | **0.4950 ± 0.0056** | 0.00% |

The macro-average relative change of multi-view versus schema + PLE is −0.70%.
Thus, this first real-data implementation does not support the broad multi-view
hypothesis. Numerical PLE is the reliable improvement; the added low-cardinality
identity and cross-fitted categorical target views help only Black Friday and
hurt Diamond in all five paired seeds. `real_results.csv` contains every run.

## Expanded ablation and routing sweep

The benchmark now exposes every proposed improvement separately:

- `numeric_identity`: schema + PLE plus identity views for coded numbers;
- `cat_target`: raw cross-fitted categorical target views;
- `cat_residual`: categorical views computed from numerical-model residuals;
- `multi_view_residual`: numeric identity plus residual categorical views;
- `diagnostic_identity`: residual diagnostics select coded numerical columns,
  but only their target-free identity views are added;
- `diagnostic_residual`: training-only residual diagnostics decide which coded
  numerical columns receive identity views, with residual categorical views;
- `gated_ple`: an architecture-matched control that projects schema and PLE
  separately before mixing them;
- `sparse_gate`: the same gate with diagnostic identity and residual categorical
  views available;
- `late_fusion`: validation-greedy ensembling of schema + PLE, numeric identity,
  and residual categorical models.

All single models are matched to the parameter count of schema + PLE. Binary
classification now early-stops on validation log loss while still reporting
accuracy. PLE supports 8, 16, 32, 64, and 128 bins:

```bash
python real_data_benchmark.py \
  --datasets diamond black-friday \
  --bins 8 16 32 64 128 \
  --seeds 0 \
  --output expanded_results.csv
```

The validation-selected configurations were then fixed and run for five seeds:

| Dataset | Bins | Schema + PLE | Gated PLE | Sparse gate | Late fusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diamond | 64 | 0.13709 ± 0.00157 | 0.13602 ± 0.00118 | **0.13588 ± 0.00068** | 0.13642 ± 0.00131 |
| Black Friday | 128 | 0.69257 ± 0.00111 | 0.69371 ± 0.00149 | 0.69298 ± 0.00132 | **0.68864 ± 0.00031** |

On Diamond, sparse gating improves over plain PLE by 0.88%, but gated PLE already
accounts for 0.77%; sparse gating beats the architecture-matched control by only
0.10% on average and in two of five paired seeds. On Black Friday, late fusion
improves over PLE by 0.57% in all five seeds. It is not parameter matched because
it retains multiple models. `expanded_results.csv` contains the full 150-row
screening and confirmation record, including selected ensemble members and gate
weights.

## Broad signal search

`signal_search.py` expands the search along four axes:

- PLE bins: 16, 32, 64, and 128;
- backbone: MLP or residual network, width, depth, activation, and dropout;
- optimization: learning rate and weight decay;
- representation: numerical identity thresholds, residual diagnostics,
  categorical target/residual views, sparse gates, and smoothing/entropy values.

For each dataset, 24 schema + PLE backbones are screened on seed 0. Semantic
routes are evaluated around the two best validation backbones. The best fixed
baseline, tuned backbone, semantic candidate, and validation-greedy ensemble are
then frozen and confirmed on seeds 0--4. Screening is ranked only by validation
log loss or MSE; screening CSVs intentionally omit test scores.

```bash
python signal_search.py \
  --datasets adult diamond black-friday california \
  --base-candidates 24 \
  --top-backbones 2 \
  --confirm-seeds 0 1 2 3 4
```

The Adult identity-only control for the selected backbone is reproducible with:

```bash
python real_data_benchmark.py \
  --datasets adult \
  --methods diagnostic_identity \
  --bins 128 \
  --low-cardinality 128 \
  --identity-effect-threshold 0.001 \
  --model resnet \
  --activation gelu \
  --width 384 \
  --depth 2 \
  --dropout 0 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --max-epochs 120 \
  --patience 12 \
  --output signal_search/adult_diagnostic_identity_control.csv
```

The completed protocol contains 397 model fits: 292 screening fits, 100
additional confirmation fits, and five fits for the Adult identity-only control.

| Dataset | Fixed baseline | Tuned backbone | Best semantic candidate | HP ensemble |
| --- | ---: | ---: | ---: | ---: |
| Adult accuracy ↑ | 0.85623 ± 0.00103 | 0.85948 ± 0.00169 | **0.87259 ± 0.00108** | **0.87299 ± 0.00072** |
| Diamond RMSE ↓ | 0.14078 ± 0.00076 | 0.13567 ± 0.00123 | 0.13525 ± 0.00079† | **0.13229 ± 0.00043** |
| Black Friday RMSE ↓ | 0.69436 ± 0.00025 | 0.69267 ± 0.00064 | **0.69119 ± 0.00051** | **0.68515 ± 0.00027** |
| California RMSE ↓ | 0.49501 ± 0.00499 | 0.47926 ± 0.00524 | 0.47804 ± 0.00399† | **0.45618 ± 0.00150** |

† The validation-selected `numeric_identity` candidate added no feature at its
cardinality threshold, so it is an architecture-identical repeat rather than a
semantic improvement. The small difference is treated as a null result.

Adult provides the clearest representation signal. Residual diagnostics select
numerical columns 3, 4, and 5 for identity views; the full route beats the tuned
backbone by 1.31 accuracy points in all five paired seeds. The identity-only
control reaches 0.87143 ± 0.00139, showing that most of the gain comes from those
numerical identity views rather than categorical residual encoding. Black Friday
also gains from identity views for all four numerical columns, reducing RMSE by
0.21% versus the tuned backbone in all five seeds.

Diamond and California provide no semantic signal under this search. Their
ensemble gains are nevertheless large (2.49% and 4.82% lower RMSE versus the
tuned single model), as are the Black Friday ensemble gains. These are evidence
for hyperparameter diversity, not for feature semantics. The per-dataset screen,
confirmation, and selection manifests are in `signal_search/`, with the combined
comparison in `signal_search_summary.csv`.

## Matched Adult identity-versus-diversity follow-up

A subsequent prediction-retaining control puts the tuned Day 1 Adult backbone,
selected identity, four ordinary seeds, and four information-equivalent schemas
on the same TabPack split and parameter budget. The schema models keep 128-bin
PLE fixed and replace only the normalized numerical coordinates with PAD,
DUPLICATE, POSNEG, or SIGNMAG. All ensemble comparisons cost four fits.

| System | Accuracy | AUC | Accuracy gain vs one PLE model |
| --- | ---: | ---: | ---: |
| Single PLE | 0.8617 | 0.9143 | — |
| Single PLE + selected identity | 0.8693 | 0.9250 | +0.7616 pp |
| Four PLE seeds | 0.8607 | 0.9155 | -0.0983 pp |
| Four PLE schemas | 0.8635 | 0.9152 | +0.1781 pp |
| Four selected-identity seeds | **0.8724** | **0.9279** | **+1.0749 pp** |

A single identity model beats both four-fit PLE ensembles. The schema ensemble
has lower pairwise prediction distance than the ordinary seed ensemble (0.0464
versus 0.0561), likely because PLE already supplies most of the numerical basis
and the schema intervention changes only six raw coordinates. The large Adult
identity gain is therefore not an ordinary ensembling artifact. Full predictions,
member-level metrics, and the frozen protocol are in
`adult_identity_schema_matched/` and
`configs/adult_identity_schema_matched.json`.
