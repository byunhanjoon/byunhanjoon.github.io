# Untouched external Selective Measure-Orbit result

## Verdict

**The external performance claim is rejected.** On seven datasets absent from
all previous Measure-Orbit development and confirmation, Selective
Measure-Orbit was 0.521% worse in proper loss than an exactly update-matched
two-seed ordinary-TabM prediction ensemble. The 95% dataset-bootstrap interval
was entirely negative: [−0.831%, −0.195%]. Only 2/7 dataset means and 7/21
paired dataset-seed cells were positive. Every performance clause in the
frozen primary gate failed.

It also failed the frozen preservation gate against one ordinary TabM fit:
−0.079% mean proper-loss reduction, 95% interval [−0.366%, +0.184%], with 3/7
positive dataset means and 7/21 positive cells.

This is a completed negative experiment, not an incomplete method result.

## Design and integrity

The panel, seeds, method, compute match, and gates were frozen before any
external outcome was generated. The panel contains four binary credit or
insurance datasets, Helena multiclass, Year Prediction regression, and a
date-purged Sberbank Housing regression window.

For each dataset and seed, the experiment trained:

1. an ordinary TabM anchor;
2. the locked Measure-Orbit 4/2/2 member assignment;
3. an independent ordinary TabM seedmate.

The selective portfolio consists of fits 1 and 2 and chooses the lower
validation-proper-loss arm. The control consists of fits 1 and 3 and averages
their predictions. The seedmate executed exactly the number of epochs and
minibatch updates used by fit 2. The result matrix has 63/63 successful fits,
63 prediction artifacts, equal parameter counts, and exact gradient-update
matching in all 21 portfolio comparisons.

Observed runtime also matched closely: seed-ensemble/selective time ratio was
0.993 on average and 0.997 at the median.

## Dataset results

Positive values favor Selective Measure-Orbit.

| Dataset | Task | Versus two-seed ensemble | Versus one baseline | Raw orbit versus baseline | Orbit activations |
| --- | --- | ---: | ---: | ---: | ---: |
| Coil2000 | Binary | +0.177% | +0.153% | +0.153% | 3/3 |
| Give Me Some Credit | Binary | +0.009% | +0.386% | +0.424% | 1/3 |
| Helena | Multiclass | −0.634% | −0.031% | −0.082% | 2/3 |
| OpenML Credit 43454 | Binary | −1.134% | −0.544% | −0.544% | 3/3 |
| Sberbank window 2 | Regression | −0.485% | −0.735% | +1.319% | 2/3 |
| Taiwanese Bankruptcy | Binary | −0.915% | +0.221% | −0.104% | 1/3 |
| Year Prediction | Regression | −0.662% | 0.000% | 0.000% | 0/3 |

The Year result is an intended null: no qualifying empirical atoms were found,
so all three mixed-measure views coincide and the orbit fit equals the paired
baseline fit. Ordinary seed diversity still improves Year, which directly
explains why the two-seed portfolio wins there.

## Why the internal confirmation did not transfer

The raw orbit itself averaged only +0.167% versus one baseline, with positive
means on 3/7 datasets and 9/21 cell wins. The validation selector did not make
that weak signal safe:

- it activated orbit in 12/21 cells;
- six activations helped test loss and six harmed it;
- it abstained in nine cells and missed three helpful orbit outcomes.

Sberbank is the sharpest example. Raw orbit improved 1.319% on average, but the
per-seed validation decision selected the wrong arm often enough that the
selective method ended 0.735% worse than baseline. On OpenML Credit 43454,
validation selected orbit in all three seeds although its mean test loss was
0.544% worse.

Meanwhile, ordinary two-seed ensembling improved 0.433% over a single baseline
on average and won 16/21 cells. The external result therefore supports a
simpler explanation: much of the earlier performance signal was ordinary
portfolio variance reduction, while the proposed validation selector was not
stable enough to identify measure-specific diversity out of distribution.

## Claim consequence

Do not claim Selective Measure-Orbit as a broadly improving method. The earlier
21-dataset result remains valid as prospective confirmation within a dataset
collection already touched by the preceding Orbit-TabM study, but it does not
survive a genuinely untouched external panel or the strongest equal-compute
control.

The basis-geometry paper can still use Measure-Orbit as a diagnostic showing
that coordinate choices induce distinct solutions. It should not use it as the
paper's performance remedy. No threshold, panel member, or selector was changed
after this result.

## Artifacts

- Frozen protocol: `EXTERNAL_MEASURE_ORBIT_PROTOCOL.md`
- Frozen configuration:
  `experiments/day3/configs/external_measure_orbit_preregistered.json`
- Per-fit results: `results/day3/external_measure_orbit/runs_shard*.csv`
- Stored predictions: `results/day3/external_measure_orbit/predictions/`
- Paired and dataset summaries:
  `results/day3/external_measure_orbit/{paired,dataset_summary}.csv`
- Machine-readable verdict:
  `results/day3/external_measure_orbit/analysis_summary.json`
- Completion audit: `EXTERNAL_MEASURE_ORBIT_COMPLETION_AUDIT.md` and
  `results/day3/external_measure_orbit/completion_audit.json`
