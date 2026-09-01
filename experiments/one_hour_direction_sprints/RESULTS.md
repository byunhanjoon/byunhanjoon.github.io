# Results: three one-hour-capped direction sprints

Date: 2026-08-31 (Asia/Seoul)

The one-hour limits were maximum budgets, not targets to consume.  Every panel
reached its predeclared endpoint early on the two local H100s or on already
audited cached predictions.  Unused time in the projective sprint was spent on
stronger direct comparators and a natural-label bridge rather than additional
hyperparameter search.

## Portfolio verdict

| Direction | Frozen primary gate | Additional check | Decision |
|---|---:|---:|---|
| Risk-gated neural geometry transfer | **FAIL** | Fresh split exposes one practical harm and ACS concentration | **Stop as standalone lead** |
| Static projective tabular law | **PASS** | 12-dataset natural-scale and covariance-mechanism gates pass | **Go; build on a strong TFM backbone** |
| OrbitCover as a performant predictor | **FAIL** | Test-oracle headroom is only 0.46% | **Do not sell as predictive improvement** |

The answer to “do all three have the desired performance potential?” is **no**.
Only static projectivity produced a new positive method-performance signal.
OrbitCover remains strong on its narrower quotient-estimation endpoint, while
geometry remains useful theory and a failure-mode example.

## 1. Fresh risk-gated neural geometry transfer

Protocol SHA-256:
`407609e697763db85b3d90cea6a6aefce2d0c268c3c504cefef474fcd3c886de`.

The prior neural development used state splits 0 and 1.  This sprint trained
MLP, ResNet, FT-Transformer, and TabM bases from scratch on split 2, retained
disjoint construction/validation/test states, and deployed the already-chosen
per-backbone Bonferroni LCB certificate without tuning.

### Primary outcome

- 16/16 backbone-task cells completed with finite out-of-fold residuals.
- 7/16 cells were selected, spanning all four backbones and three sources.
- Mean deployed standardized-MSE gain was `+0.002750`, below the `+0.005`
  gate; mean relative gain was `+0.670%`.
- One selected cell was practically harmful: MLP on TLC pickup zones,
  `-0.002087` absolute gain.
- All four ACS cells improved (`+0.007679` to `+0.014850`), but the other
  sources contributed only two tiny Medical gains, one TLC harm, and complete
  Airline abstention.

### Interpretation

The conditional-value identity remains correct, but the certificate does not
estimate state-shifted residual value reliably enough.  This agrees with the
earlier four-backbone development warning (`rho=0.514`, sign accuracy `61.8%`)
and now adds a genuinely fresh failure.  The apparent breadth is mostly ACS;
this is not an oral-level general method result.

Decision: retain the theorem, impossibility result, and ACS case study as
motivation or an appendix.  Do not allocate a standalone submission track
without new sources and a justified state-shift model.

Machine-readable result: `geometry/results/summary.json` and selected decisions
in `geometry/results/decisions.csv`.

## 2. Static projective tabular law

Primary protocol SHA-256:
`acb5a3675d0249289904b07b208a6a2568c4122021610c184be8bed45eb13081`.

The experiment trained a neural-process-style static tabular model that emits
one low-rank-plus-diagonal joint Gaussian over 12 query rows.  A larger direct
DeepSets comparator answered scalar linear queries without a joint-law
constraint.  Both saw identical episodic tasks and sparse training queries.

### Primary semisynthetic outcome

- Projective parameters: `53,130`; direct parameters: `138,850`.
- OOD dense/scaled-dense mean NLL advantage: `+1.774` nats.
- OOD paired wins: `120/120` across three seeds, four latent task families,
  and five covariate domains.
- Every held-out empirical feature distribution won: Breast Cancer `+2.238`,
  Diabetes `+1.479`, Digits `+1.805`, and Wine `+1.604` nats over the direct
  model, averaged across the two OOD query families.
- Point-query NLL also improved by `+0.144` nat, so aggregate performance was
  not purchased by sacrificing marginals.
- Maximum projective identity residual was `8.96e-8`; the direct model's mean
  violations ranged from `0.436` to `0.837`.

This passed every frozen gate.

### Strong direct-comparator stress test

Stress protocol SHA-256:
`327ed633bb90c065b9ba39b38698f583edab5f18fac621ca8888afaeabba32cd`.

The projective checkpoints were then compared with:

- the original direct model trained for 20,000 updates (4x longer);
- the direct model explicitly trained on dense and scaled-dense queries;
- a 196,610-parameter direct moment architecture with signed first-order and
  squared-coefficient summaries.

Against the per-cell best of these controls, projectivity retained:

| Query | Mean NLL advantage | Wins |
|---|---:|---:|
| Point | `+0.123` | 60/60 |
| Dense | `+0.138` | 60/60 |
| Scaled dense | `+0.147` | 60/60 |

The result is therefore not explained by the original direct baseline's size,
optimization budget, lack of moment features, or ignorance of the evaluation
query family.

### Zero-shot natural-label anchor

Natural-anchor protocol SHA-256:
`4d8d2537872cf7cc9668dd95876809a3f9c25c64c981081afafcee09d18ae398`.

Unchanged synthetic-prior checkpoints received 16 context rows from six real
regression datasets and predicted 12 held-out test targets across three
splits.  This diagnostic failed one breadth condition:

- Dense-query NLL advantage was `+0.0889`, with 41/54 wins.
- Projective point RMSE was within 25% of a fresh per-episode ridge fit on all
  6/6 datasets.
- Mean point RMSE was essentially tied with direct (`+0.00293` average
  direct-minus-projective), but projective won dataset means on only 2/6.
- Natural point NLL was `0.154` worse on average and point coverage error was
  8.4% versus the direct model's 3.1%, revealing prior/calibration mismatch.

Thus the new evidence establishes the *mechanism and inductive-bias
potential*, not competitive natural-table performance.

### Natural-scale escalation

That boundary was tested immediately in a frozen 12-dataset, three-split,
four-context benchmark with Bayesian linear and RBF-GP joints, TabPFN,
TabICLv2, and the direct network. The corrected 5,040-row run passed every
integrity audit.

- The original projective model beat direct aggregate NLL on 12/12 datasets,
  with `+0.3461` mean advantage and 87.3% matched-cell wins.
- Dense/scaled-dense advantage was `+0.5276`, with 89.93% wins.
- Point RMSE was within 25% of TabPFN on 11/12 datasets and essentially tied
  with the exact GP/Bayesian anchors in the pooled metric.
- Aggregate NLL was `1.6058` for projective, `1.7431` for RBF GP, and `1.9519`
  for direct. Removing the degenerate all-zero-context insurance dataset put
  the GP narrowly first (`1.5611` versus `1.5842`).

A separately frozen covariance ablation then held every mean and marginal
variance fixed. The full learned covariance improved NLL by `0.00400`, CRPS by
`0.00176`, won NLL on 11/12 dataset means, and beat a PSD correlation-shuffle
control. Thus both the broad projective training restriction and the learned
off-diagonal mechanism have natural-data support, although the latter effect is
modest.

A post-hoc TabPFN ensemble-view covariance improved pooled scores but won only
6/12 datasets, missing its 7/12 frozen gate. Ensemble views are not reliable
posterior function samples; this shortcut is retired.

Decision update: build a jointly pretrained projective process head on a strong
TabICL/TabDPT-style backbone. Do not spend the next block on another small
synthetic model or on post-hoc TFM correlations. Full results, theory framing,
recent-work positioning, and paper-scale gates are in
`projective_natural_scale/NATURAL_SCALE_RESULTS.md`.

### Connection to the existing repository evidence

This static result is consistent with the earlier temporal projective pilots:

- the non-Gaussian projective mixture beat a capacity-matched direct mixture
  in 9/9 cells and a matched Gaussian in 8/9 cells;
- at paper scale it beat a direct mixture by 21–23% CRPS on all three datasets;
- after validation-only calibration it was within 2% of TACTiS on JenaWeather,
  beat TACTiS on Traffic, and remained 13.95% worse on Electricity.

The new oral-level insight candidate is not merely “consistency is valid.”  It
is that representing the joint law restricts an infinite family of linear
query predictions to

`mean(a) = a^T mu` and `variance(a) = a^T Sigma a`,

turning projective coherence into a strong statistical regularizer and query
generalization advantage.  The direct stress controls make this interpretation
credible; the natural-label failure defines the next boundary.

Decision: this is the only direction that merits a focused performance sprint.
The next gate must use natural OpenML task pretraining/leave-dataset-out
evaluation and strong TabPFN/TabICL or ensemble marginals.  It should require
competitive point prediction *and* superior aggregate NLL/CRPS; another
semisynthetic win is not enough.

Machine-readable results are in `projective/results/summary.json`,
`projective/results/direct_controls/summary.json`, and
`projective/results/natural_anchor/summary.json`.

## 3. OrbitCover actual predictive utility

Protocol SHA-256:
`cc2d825ad05164142cd0ad34397d0b1a8d93650410c1bb7ea5872b106b7703d0`.

The existing final closure already establishes a 55.9% B=16 reduction in
method-relative quotient residual for coupled OC2.  This sprint instead audited
actual held-out predictive loss at the same 16-fit budget over all 144 neural
dataset/split/backbone cells and 512 estimator constructions.

### Outcome

- Equal-dataset predictive-loss improvement: `-0.471%` (a degradation).
- Dataset-clustered 95% interval: `[-1.297%, +0.412%]`.
- Cell wins: 44/144; dataset wins: 4/12.
- Architecture improvements: FT-Transformer `-0.787%`, MLP `-0.397%`,
  ResNet `-0.952%`, TabM `+1.180%`.
- Worst dataset: `-3.305%`.
- Independent OC2 was essentially neutral but still negative (`-0.024%`), as
  was SRS-joint (`-0.033%`).
- An invalid test-set oracle choosing the better of coupled OC2 and canonical
  per cell had only `0.456%` mean headroom, below the primary 0.5% target.

### Interpretation

OrbitCover efficiently estimates a different symmetrized target; reducing
Monte Carlo error to that target need not reduce loss to labels.  The new audit
shows that the distinction is practically decisive, not semantic wording.

Decision: OrbitCover can still be an ICLR theory/measurement paper under its
honest quotient-estimation thesis.  It should not occupy one of the portfolio's
“performant predictor” slots, and no accuracy claim should appear in its title
or abstract.

Machine-readable result: `orbitcover/results/summary.json` and all budget
comparisons in `orbitcover/results/comparisons.json`.

## Resource allocation

1. Put the next concentrated block into static projectivity, but precommit a
   natural-table gate before training.
2. Write OrbitCover only under the already-supported estimator-efficiency
   thesis; do not spend more compute seeking a raw-accuracy headline.
3. Stop standalone geometry-transfer method work.  Reuse its conditional-value
   theory as a motivating negative result.

## Reproduction

Geometry uses the repository's TabM environment:

```bash
LD_LIBRARY_PATH=/home/byunhanjoon/miniconda3/envs/tabred-repro/lib \
  /home/byunhanjoon/miniconda3/envs/tabred-repro/bin/python \
  geometry/run_geometry.py --analyze
```

The remaining analyses use the base Python 3.10 environment:

```bash
/home/byunhanjoon/miniconda3/bin/python projective/run_projective.py --analyze
/home/byunhanjoon/miniconda3/bin/python projective/run_direct_controls.py --analyze
/home/byunhanjoon/miniconda3/bin/python projective/run_natural_anchor.py --analyze
/home/byunhanjoon/miniconda3/bin/python projective_natural_scale/run_benchmark.py --analyze
/home/byunhanjoon/miniconda3/bin/python projective_natural_scale/run_covariance_ablation.py --analyze
/home/byunhanjoon/miniconda3/bin/python orbitcover/run_orbitcover.py
```
