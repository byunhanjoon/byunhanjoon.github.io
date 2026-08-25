# Day 3 continuation — equivalent-basis orbit ensembles

## Executive result

**Orbit-TabM passed the frozen method gate, with a modest but broad proper-loss
gain.** The model gives TabM's eight internal members alternating cumulative
and local coordinates of the same block-whitened numerical/categorical state
spaces. It adds no trainable parameters and no feature information.

On the 21-dataset confirmation tier, Orbit-TabM reduced proper predictive loss
by **0.767%** on average relative to ordinary cumulative-basis TabM. The
dataset-cluster bootstrap interval was **[0.015%, 1.886%]**. It improved 48/63
paired seed cells and 15/21 dataset means. All five frozen gates passed.

Across the nine-dataset development screen plus confirmation, the mean
reduction was **0.718%** over 30 datasets, with interval **[0.173%, 1.492%]**
and positive means on 23/30 datasets.

This is the first Day 3 continuation that combines a distinct method idea with
a broad positive predictive result. It is not yet an identity-on-Adult-sized
score gain: most of the improvement is in probability/proper loss rather than
accuracy or RMSE, and the confirmation median is 0.252%.

## Method

Let `x` denote the cumulative/Helmert reference coordinates and let `B` be the
known invertible blockwise map to local/adjacent coordinates. Ordinary
dense-stem TabM sends the same `x` to every member. Orbit-TabM instead sends

`x, xB, x, xB, x, xB, x, xB`.

Each view retains exactly the same information. The views pass through one
shared learned dense stem and the ordinary TabM backbone. This creates fixed,
structured member-specific effective projections without learning eight dense
input matrices. Trainable parameter count is unchanged.

The intervention is best classified as a **structured ensemble inductive
bias**, not a whole-model exact reparameterization. Although every member input
is information-equivalent, imposing one shared learned stem across distinct
charts changes the joint weight-sharing constraint.

## Protocol and audit

The protocol was frozen in `ORBIT_ENSEMBLE_PROTOCOL.md` before the first orbit
result. The nine released TabPack datasets formed a development screen. The
remaining 16 broad-benchmark datasets and the five separately frozen extension
datasets formed method confirmation. These tables had been used by earlier Day
3 studies, so “confirmation” refers to outcomes for this new method, not wholly
untouched datasets.

The matrix contains 297 completed runs:

- screen: 9 datasets × 4 arms × 3 seeds = 108;
- confirmation: 21 datasets × 3 arms × 3 seeds = 189;
- zero failures;
- maximum cumulative-to-local relation error `3.42e-12`;
- every member map square and full rank;
- identical trainable parameter counts in every ordinary/orbit pair.

The primary outcome is test proper loss after validation-loss early stopping:
binary log loss, multiclass log loss, or standardized-target regression MSE.
All aggregates first average seeds within a dataset and then treat datasets as
the sampling units.

## Development screen

| Comparison | Mean proper-loss reduction | 95% dataset interval | Dataset wins |
| --- | ---: | ---: | ---: |
| Natural orbit vs cumulative TabM | +0.601% | [+0.188%, +1.082%] | 8/9 |
| Natural orbit vs validation-selected single basis | +0.448% | [+0.104%, +0.807%] | 7/9 |
| Random orthogonal orbit vs cumulative TabM | **−1.757%** | — | 3/9 |

Natural Orbit-TabM won 24/27 paired cells. Arbitrary rotations lowered member
correlation even further but hurt mean loss, especially on California,
Diamond, and Black Friday. The result is therefore not explained by maximizing
diversity alone; the sparse, semantic path/simplex charts matter.

## Confirmation

| Quantity | Result |
| --- | ---: |
| Mean reduction vs cumulative TabM | **+0.767%** |
| Dataset-cluster bootstrap interval | **[+0.015%, +1.886%]** |
| Median dataset reduction | +0.252% |
| Positive dataset means | 15/21 |
| Positive paired seeds | 48/63 |
| Reduction vs validation-selected single basis | **+0.929%** |
| Selected-basis dataset interval | [+0.078%, +1.994%] |
| Smallest leave-one-dataset-out mean | +0.303% |

All frozen gate clauses passed: positive clustered interval, at least 60%
dataset wins, at least 0.5% mean reduction, positive gain over
validation-selected cumulative/local TabM, and no excess failures.

### Per-dataset confirmation means

| Dataset | Relative proper-loss reduction |
| --- | ---: |
| Credit Card Fraud | +10.052% |
| Polish Bankruptcy 5-year | +2.835% |
| Australian Credit Approval | +1.400% |
| Polish Bankruptcy 2-year | +1.368% |
| Santander | +0.969% |
| Jannis | +0.806% |
| Gesture | +0.570% |
| Bank Marketing | +0.518% |
| LendingClub | +0.366% |
| HELOC | +0.295% |
| Credit Card Default | +0.252% |
| Covtype | +0.142% |
| FREM-TPL | +0.077% |
| KDD17 return | +0.058% |
| Compustat direction | +0.008% |
| KDD17 direction | −0.002% |
| Facebook Comments | −0.067% |
| Polish Bankruptcy 1-year | −0.201% |
| Polish Bankruptcy 3-year | −0.768% |
| German Credit | −1.012% |
| Polish Bankruptcy 4-year | −1.551% |

Credit Card Fraud is an influential relative-loss result because its baseline
log loss is very small. The conclusion does not depend on its sign: removing
any one dataset leaves a positive mean, with the smallest leave-one-out value
at +0.303%. It does affect whether the mean exceeds the predeclared 0.5%
practical threshold, so the median and full distribution must accompany the
headline mean.

## Is this “real performance”?

Yes for probabilistic prediction, but not yet at the magnitude of Adult's
exact-state accuracy jump.

- Confirmation classification accuracy improved by 0.141 percentage points on
  average across classification cells.
- Confirmation regression RMSE improved by only 0.011% on average.
- On Adult, proper loss improved 0.555%, but accuracy improved only 0.010
  percentage points.

Thus Orbit-TabM is evidence of broad performance value and a stronger novelty
direction than another canonicalizer, but it should not be sold as a large
across-the-board score improvement. Its current strength is better-calibrated
ensemble prediction at essentially unchanged training cost.

## Mechanism and efficiency

On confirmation, mean member prediction correlation fell from 0.9282 to 0.8956
(a 0.0326 reduction). The mean paired training-time ratio was 1.033 and the
peak-memory ratio was 1.078. Parameters matched exactly. The small runtime
increase is plausible because the fixed member transforms and shared first
stem are batched efficiently; this is parameter-matched and measured-compute,
not an exact FLOP match.

The random-orbit negative control is important. Orthogonal rotations preserve
all information, norm, covariance spectrum, and rank, but decreased screen
performance by 1.76% on average. Semantic locality appears to preserve member
quality while still increasing ensemble diversity.

## Novelty boundary

The broad idea “transform features differently for ensemble members” is prior
art. Rotation Forest retains all PCA components and rotates feature subsets to
promote tree diversity. TabPFN ensembles feature permutations and alternative
preprocessing configurations. TabM establishes parameter-efficient tabular
ensembling.

The narrower candidate contribution here is:

1. construct member views from exact numerical-path and categorical-simplex
   basis equivalence rather than random feature extraction or feature-order
   symmetry;
2. connect their diversity to measured initialization/optimizer
   non-equivariance;
3. turn one shared dense stem into structured member-specific effective
   projections without adding trainable parameters;
4. show that semantic exact charts help while arbitrary orthogonal charts hurt
   on the same benchmark.

That conjunction looks plausibly novel from the checked literature, but it
still needs a systematic scholarly search before submission.

## Verdict and next gate

**KEEP as the leading method extension to the Day 3 phenomenon paper.** It is
more promising than whitening, residualization, invariant weight decay, SOAP,
or random rotation ensembling because it both uses the discovered mechanism
and improves broad prediction.

The submission-critical next experiment should be a truly untouched dataset
panel with:

- tuned ordinary TabM;
- Orbit-TabM;
- a longer-trained/wider TabM compute control;
- a two-seed or two-checkpoint TabM ensemble at measured equal wall-clock
  budget;
- calibration metrics plus the official task metric;
- at least two independent dataset sources and temporal tasks.

The method should be dropped as a headline contribution if the untouched panel
does not retain a positive proper-loss interval or if equal-compute ordinary
ensembling explains the gain.

## Reproduction

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m pytest -q tests/test_orbit_ensemble.py

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m experiments.day3.orbit_ensemble --stage screen --shard 0 --num-shards 2 --device cuda:0

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m experiments.day3.orbit_ensemble --stage confirmation --shard 0 --num-shards 2 --device cuda:0

/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m experiments.day3.analyze_orbit_ensemble
```

Primary machine-readable artifacts are in `results/day3/orbit_ensemble/`:
`all_runs.csv`, `paired_results.csv`, `dataset_summary.csv`, and
`analysis_summary.json`.
