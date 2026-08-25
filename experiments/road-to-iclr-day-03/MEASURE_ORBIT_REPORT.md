# Day 3 continuation — mixed-measure Measure-Orbit

> **External update:** the later seven-dataset globally untouched,
> exactly-update-matched test rejected the broad performance claim. Selective
> Measure-Orbit was 0.521% worse than a two-seed ordinary-TabM prediction
> ensemble, with a 95% dataset-bootstrap interval of [−0.831%, −0.195%], and it
> was neutral to worse than one baseline. See
> `EXTERNAL_MEASURE_ORBIT_REPORT.md`. The confirmation below remains part of
> the chronological record but must not be presented as external validation.

## Executive result

**A paper-level hypothesis finally validated.** Adult's identity gain is
largely caused by quantile PLE spending most of its knot budget on point
masses. Directly replacing PLE with a mixed discrete–continuous encoding
reproduces the Adult gain but does not transfer uniformly. Turning the failed
replacement into structured member diversity inside TabM, then using
validation proper loss to abstain to ordinary TabM, passes a prospectively
frozen 21-dataset confirmation gate.

Selective Measure-Orbit reduces proper loss by **1.103%** on average. The 95%
dataset-bootstrap interval is **[+0.333%, +2.154%]**. It has positive means on
17/21 datasets and positive paired changes in 39/63 dataset-seed cells. All
126 expected confirmation runs completed without failures and every arm is
parameter matched.

The strongest mechanistic result remains Adult: one-fit Measure-Orbit improves
accuracy by **1.013 percentage points** on average over three seeds and lowers
log loss by **4.931%**.

## What was falsified first

### Residual atlas

The earlier nested exact-state residual atlas explains Adult and finds a real
Miami interaction, but its frozen untouched breadth screen selected a
structure on only 1/6 datasets. It remains a mechanism, not the broad method.

### Mixed-measure PLE as a universal replacement

The 216-cell, eight-dataset, three-backbone screen cleanly reproduces Adult:

- mixed-measure PLE: +0.667 accuracy points, all three backbone means positive;
- tail-reallocated PLE: +0.749 accuracy points.

But mixed-measure PLE improves only 3/8 dataset means and averages −0.568%
proper-loss reduction. Tail reallocation wins only 2/8. The replacement
hypothesis is rejected.

### Measure-Orbit without abstention

Assigning four baseline, two tail, and two mixed-measure views to TabM members
raises Adult accuracy by +1.013 points and improves 6/8 dataset means. Its mean
proper-loss reduction is +1.019%, but its interval [−0.274%, +2.444%] crosses
zero. The frozen screen gate therefore rejects the unconditional claim.

## Prospective confirmation

The next hypothesis was fixed before confirmation: compare ordinary and
Measure-Orbit TabM using validation proper loss, then deploy the selected arm.

| Quantity | Result |
| --- | ---: |
| Confirmation datasets | 21 |
| Completed fits | 126/126 |
| Mean selected proper-loss reduction | **+1.103%** |
| 95% dataset-bootstrap interval | **[+0.333%, +2.154%]** |
| Positive dataset means | **17/21** |
| Positive paired cells | **39/63** |
| Measure-Orbit activations | 42/63 |
| Mean descriptive official-metric gain | +0.296 |
| Failures | 0 |

The official-metric number mixes classification accuracy-point changes and
relative RMSE reductions, so it is descriptive; proper loss is the registered
cross-task endpoint.

Largest selected proper-loss reductions include Polish Bankruptcy 3-year
(+9.42%), Polish Bankruptcy 2-year (+4.53%), Australian Credit Approval
(+3.22%), Facebook Comments (+1.69%), and Polish Bankruptcy 5-year (+1.23%).
Gesture, KDD17 return, Polish Bankruptcy 1-year, and Santander abstain to the
baseline on average; no dataset has a negative selected mean.

## Secondary controls

The unselected raw Measure-Orbit arm is also encouraging on confirmation:

- +0.891% mean proper-loss reduction;
- 95% dataset interval [+0.066%, +1.987%];
- 14/21 positive dataset means and 42/63 positive paired cells.

This was not the registered endpoint of the selective confirmation and should
be treated as a strong secondary result requiring a new external replication.

A validation selector between two ordinary TabM seeds, constructed from the
same completed baseline runs, improves proper loss by +0.449% on average. The
Measure-Orbit selector's +1.103% is larger, suggesting that representation
diversity adds value beyond hard seed selection. This is not yet the required
prediction-ensemble control.

Datasets with no applicable numerical transformation provide a useful identity
control: the two arms are numerically identical on Gesture and Santander.

## Mechanism

Adult's `capital-gain` and `capital-loss` columns contain 116 and 88 supported
values, but nominal 128-bin quantile PLE collapses to 12 and 7 knots because
zero dominates. Conditional tail PLE prevents that point mass from consuming
nearly the whole resolution budget. The one-fit result suggests that the
different views are most useful jointly: baseline members preserve global
quality while tail and atom members take shorter optimization paths to sparse
state-local structure.

This is not merely “more features.” Every member observes the same underlying
schema information, all member views have equal width, and the two arms have
identical trainable parameter counts. The intervention changes the structured
weight-sharing constraint across TabM members.

## Novelty boundary

Numerical PLE and target-aware/tree-derived bins are established by
[Gorishniy et al.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html),
and function-basis views of numerical features are broader prior art
([Shtoff et al.](https://openreview.net/forum?id=M4222IBHsh)). Parameter-efficient
tabular ensembling is the core contribution of
[TabM](https://openreview.net/forum?id=Sd4wYYOhmY). Multi-view tabular learning
also predates this work, for example
[SubTab](https://openreview.net/forum?id=vrhNQ7aYSdr).

The defensible candidate novelty is the conjunction:

1. diagnose quantile-resolution collapse on mixed atomic/continuous numerical
   measures;
2. construct fixed-budget baseline, conditional-tail, and atom-direct-sum
   charts without target-aware bin selection;
3. assign those charts to shared-weight TabM members as structured diversity;
4. use proper-loss validation as an explicit abstention rule;
5. connect an Adult-sized case result to prospective broad confirmation.

The literature search did not reveal this exact construction. That is a
plausible gap, not proof of novelty.

## What is still needed for submission

- a truly new external dataset panel; these confirmation datasets were new to
  this method but had appeared in earlier Day 3 studies;
- an equal-wall-clock two-seed TabM prediction ensemble, longer-training
  control, and mixed seed-plus-representation portfolio;
- calibrated classification metrics and official task metrics reported
  separately by task;
- ablations of member allocation, atom threshold, conditional-tail-only orbit,
  and optimized ordinary-TabM inference;
- comparison with target-aware PLE, CatBoost/LightGBM, learned splines, and the
  original raw numerical schema.

## Reproduction

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m pytest -q tests/test_mixed_measure_ple.py tests/test_orbit_ensemble.py

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.mixed_measure_ple --shard 0 --num-shards 2 --device cuda:0 --output results/day3/mixed_measure_ple/runs_shard0.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_mixed_measure_ple

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.measure_orbit --shard 0 --num-shards 2 --device cuda:0 --output results/day3/measure_orbit/runs_shard0.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_measure_orbit

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.selective_measure_orbit --shard 0 --num-shards 2 --device cuda:0 --output results/day3/selective_measure_orbit/runs_shard0.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_selective_measure_orbit
```

Machine-readable decisions are in:

- `results/day3/mixed_measure_ple/analysis_summary.json`;
- `results/day3/measure_orbit/analysis_summary.json`;
- `results/day3/selective_measure_orbit/analysis_summary.json`.
