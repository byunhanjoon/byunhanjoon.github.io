# Information-equivalent basis ensembles on all TabPack datasets

## The transferable idea

Adult's identity view suggested that the useful object was not necessarily a
new feature. Identity and cumulative PLE can carry the same information while
presenting very different coordinates to an optimizer. Exact-value identity,
however, applies only to low-cardinality numerical columns.

The local PLE experiment extends that geometry to every numerical column. If
`p_j(x)` is the usual cumulative PLE ramp over interval `j`, define

```text
h_0(x) = 1 - p_0(x)
h_j(x) = p_{j-1}(x) - p_j(x)
```

and drop the final redundant hat coordinate. With a model intercept, `p` and
`h` have the same dimension and exactly the same affine span. The cumulative
basis is dense; the local basis activates at most two coordinates per column.
The experiment trains one model with each basis and chooses their convex logit
or prediction blend from a 21-point grid using validation loss only.

This is a **basis ensemble**, not a new-information encoder.

## Protocol

- all nine real datasets in the TabPack release, including Otto multiclass;
- official train/validation/test splits;
- Microsoft capped at 100k/25k/25k rows, as in the preceding experiments;
- MLP and ResNet backbones;
- four model seeds;
- 64 and 128 PLE intervals;
- cumulative and local members have equal feature dimensions and parameter
  counts;
- validation chooses the local member's weight; test targets are used only for
  final reporting;
- compute-matched control: a validation-tuned blend of two cumulative-PLE
  seeds.

For regression, “proper loss” is MSE on the standardized target. For binary and
multiclass classification it is log loss. “Score improvement” is relative
accuracy increase or relative RMSE reduction, so positive is always better.

## Confirmation results

| Dataset | Runs | Loss wins | Mean loss reduction | Mean score improvement | Mean local weight | Basis minus seed ensemble |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | 16 | 16 | 0.5920% | 0.0974% | 0.4250 | +0.1819% |
| Black Friday | 16 | 16 | 1.0398% | 0.5214% | 0.5625 | +0.2512% |
| California | 16 | 16 | 5.3984% | 2.7404% | 0.3406 | +1.7060% |
| Churn | 16 | 16 | 1.7636% | 0.3881% | 0.3250 | +0.6143% |
| Diamond | 16 | 16 | 3.1447% | 1.6009% | 0.4094 | +0.9102% |
| Higgs Small | 16 | 16 | 0.4578% | 0.2085% | 0.2031 | -0.6143% |
| House | 16 | 13 | 0.9245% | 0.4649% | 0.1531 | -3.9395% |
| Microsoft | 16 | 16 | 0.2209% | 0.1106% | 0.2312 | -0.3545% |
| Otto | 16 | 16 | 2.3184% | 0.4185% | 0.3438 | -2.1571% |

Across all 144 paired runs, the basis ensemble improves proper test loss in
141 and the official score in 130, with two score ties. Mean proper-loss
reduction is 1.7622%; a dataset bootstrap gives a 95% interval of
[0.8664%, 2.8738%]. Every dataset has a positive mean result. The complete
sweep contains 1,008 audited result rows.

The two-seed ensemble is stronger on average: 2.1083% versus 1.7622% mean loss
reduction. Basis diversity is better on average for Adult, Black Friday,
California, Churn, and Diamond; seed diversity is better for Higgs, House,
Microsoft, and Otto. Thus the evidence supports complementary representation
diversity, not replacement of ordinary seed ensembling.

## What did not work

- The local model alone is not broadly better. It often performs much worse,
  especially on House, Higgs, and Otto.
- Concatenating both bases into one parameter-matched model failed the initial
  32-bin screen.
- Matching local and cumulative activation energy did not strengthen the local
  member and was rejected after a one-seed screen.
- Hard validation selection often falls back to cumulative PLE but gives away
  the ensemble benefit. Shrunk convex blending is the robust construction.

These negatives matter. The gain comes from averaging different fitted
solutions induced by equivalent coordinates, not extra features, extra width,
or a universally superior basis.

## What this does and does not establish

This is the first identity-inspired construction in this project with positive
mean performance on every TabPack dataset, across regression, binary
classification, and multiclass classification. It turns the Adult observation
into a broader hypothesis:

> Information-equivalent coordinate systems are useful ensemble
> hyperparameters because they induce complementary optimization paths.

It is not yet an ICLR-level result. The nine datasets were used to develop this
follow-up, and the basis ensemble uses two fits. The decisive next test is a
prospective TabArena-scale benchmark against equal-compute seed and ordinary
hyperparameter ensembles, followed by a mixed ensemble that asks whether basis
diversity adds value after seed diversity is already present.

## Reproduction

- `local_basis_benchmark.py`: data loading, equal-span bases, multiclass neural
  training, validation blending, and seed control
- `configs/local_basis_screen.json`: frozen screen and confirmation protocol
- `test_local_basis.py`: equal-span, equal-dimension, and local-support checks
- `analyze_local_basis.py`: sweep integrity checks, paired summaries, and
  bootstrap analysis
- `results/local_basis_summary.csv`: per-dataset confirmation table
- `results/local_basis_summary.json`: aggregate audit

Use `/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python`.
