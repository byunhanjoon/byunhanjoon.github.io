# RESULTS — GEOMETRY TRANSFER LAW

## 1. Executive verdict

**SUPPORTED WITH LIMITATIONS**

The squared-loss Geometry Transfer Identity is exact under its stated fixed-base, fixed-operator, unbiased-state-estimate assumptions. Synthetic Monte Carlo reproduces it at Pearson `0.9999`, with `98.97%` sign agreement. The same fixed metric helps an aligned target (`Delta=+0.136`) and hurts an anti-aligned target (`Delta=-0.409`), proving that geometry-only diagnostics cannot decide the sign. Across 405 retrospective cells, population plug-in Delta has Spearman `0.9909` and sign accuracy `96.30%` against actual gain. Every source family retains a positive within-source association, and all 79 harmful cells are explained by negative/insufficient transfer under the decomposition. The law decisively outranks support distance and smoothness heuristics in leave-one-source-out retrospective comparisons. This retrospective result is not independent prediction: using held-out residual means makes the realized-oracle MSE identity algebraic, while the population plug-in differs mainly by the estimated noise cost. The original prospective protocol was frozen and hashed before outcomes were evaluated. On NOAA, Beijing air quality, and Chicago Divvy, nested state-CV predictions achieve aggregate Spearman `0.9167` and `100%` direct sign accuracy over nine source/operator aggregates. The unavailable frozen BLS source was not replaced inside that protocol. A separately frozen, untouched official Census County Business Patterns addendum then tested 37,783 state-by-industry rows over 922 six-digit NAICS states. Its nine sealed split/operator cells achieve Spearman `0.8333`, `100%` sign accuracy, and MAE `0.00091`; all five predeclared gap criteria pass. Across all 12 source/operator aggregates from four runnable families, Spearman is `0.9091` and direct sign accuracy is `100%`. This closes the prospective hierarchy/breadth gap and supports committing to the theory paper. All prospective aggregates remain beneficial, so naturally occurring prospective harm detection is still untested and must remain an explicit limitation.

## 2. Core theoretical result

For `mu_hat_T = mu_T + epsilon`, `E epsilon=0`, `Cov(epsilon)=Sigma`, fixed
transfer operator `A`, and diagonal cold-state weights `Q`,

```text
Delta = R_fallback - E[R_geometry]
      = ||mu_U||_Q^2 - ||mu_U - A mu_T||_Q^2
        - tr(Q A Sigma A^T)
      = G_transfer - C_noise.
```

`G_transfer` is the transferable conditional residual state signal;
`C_noise` is the estimation noise introduced by transferring uncertain
training-state means. Geometry helps in expected squared risk exactly when
`G_transfer > C_noise`. For positive noise cost, `GTR=G_transfer/C_noise`; one
is the break-even point. Fresh test outcome noise cancels from the comparison.

## 3. Theorems and proof status

| Result | Status | Main statement | Empirical validation |
|---|---|---|---|
| Theorem 1 transfer identity | Proved | `Delta=G_transfer-C_noise` | 972-cell Monte Carlo; Pearson 0.99991 |
| Theorem 2 no metric-only decision | Proved | Same metric/operator can help or hurt different signals | aligned `+0.136`, anti-aligned `-0.409` |
| Theorem 3 spectral corollary | Proved, symmetric special case | mode gain `h(2-h)a²-sigma²h²`; threshold `a²/sigma²>h/(2-h)` | 100% spectral sign agreement |
| Theorem 4 state-held-out risk | Proved at `n-1`; K-fold consistency qualified | exchangeable-state LOSO estimates same-procedure new-state risk | nested prospective state CV |
| Theorem 5 finite library | Proved under bounded independent states | uniform error `B sqrt(log(2K/delta)/(2n_states))` | interpretive; not used as a gate |
| MPE factorization proposition | Proved | `(wV)W=w(VW)` | unit check below `1e-12`; prior neural result negative |

## 4. Synthetic exact validation

Across 972 configurations, Pearson is `0.999909`, Spearman `0.999606`, the
calibration slope is `0.9904`, and sign accuracy is `98.97%`. The maximum raw
Monte Carlo discrepancy is `0.461`, driven by high-gain kernel-ridge cells; the
largest standardized discrepancy is `2.94` Monte Carlo standard errors. The
2-D signal/noise phase grid has `99.61%` sign agreement and places the measured
zero boundary at `GTR=1` within Monte Carlo uncertainty.

## 5. Spectral experiment

Low-frequency signal becomes beneficial at sufficiently high SNR, ranging from
`-0.141` to `+0.358` in predicted mean gain. Middle-frequency signal crosses
later and reaches only `+0.108`; the selected highest mode is effectively
suppressed and remains around `-0.141` across the SNR sweep. Mixed signals
combine these modal contributions. Predicted and empirical signs agree in
`100%` of spectral cells, including the exact `h/(2-h)` thresholds.

## 6. Same metric, opposite target

The aligned and anti-aligned targets use the identical circle metric, 15 train
and 15 unseen states, operator, sample sizes, `Sigma`, nearest/median support
distance (`1`), cover radius (`1`), degree (`2`), diameter (`15`), coverage
(`0.5`), and metric dimension (`1`). Only `mu_U` changes. Geometry yields
`Delta=+0.13595` for the aligned target and `-0.40938` for the anti-aligned
target. This is the experimental no-free-lunch witness.

## 7. Retrospective dataset panel

| Source | Task | Metric field | Metric | Train states | Test states | Observed rows | Operators |
|---|---|---|---|---:|---:|---:|---:|
| ACS | occupation | occupation code | official hierarchy path | 345 | 88 | 246,957 | 9 |
| ACS | industry | industry code | official hierarchy path | 186 | 48 | 237,619 | 9 |
| NYC TLC | pickup zone | taxi zone | centroid haversine | 73 | 19 | 254,000 | 9 |
| NYC TLC | dropoff zone | taxi zone | centroid haversine | 133 | 34 | 245,946 | 9 |
| Citi Bike | start station | station | published-coordinate haversine | 1,204 | 301 | 338,331 | 9 |
| BTS | origin airport | airport | FAA-coordinate haversine | 172 | 43 | 231,345 | 9 |
| BTS | destination airport | airport | FAA-coordinate haversine | 169 | 43 | 231,437 | 9 |
| Employee Salaries | salary | job title | trigram Jaccard distance | 73 | 19 | 5,905 | 9 |
| Medical Charges | payment | DRG description | trigram Jaccard distance | 80 | 20 | 81,759 | 9 |

Counts show split 0; all five disjoint splits are retained.

## 8. Main retrospective law test

| Statistic | Estimate | Source-bootstrap 95% interval |
|---|---:|---:|
| Pearson | 0.9997 | [0.9543, 0.9999] |
| Spearman | 0.9909 | [0.9531, 0.9989] |
| R², identity prediction | 0.9993 | — |
| Calibration slope | 0.9993 | [0.7805, 1.0023] |
| MAE | 0.000917 | [0.000491, 0.001394] |
| Sign accuracy | 96.30% | [91.60%, 99.75%] |

The realized-oracle quadratic matches actual MSE gain with MAE `1.10e-17` and
R² `1.0`; this part is an arithmetic identity, not a predictive achievement.

## 9. Transferable signal vs noise

| Source | Mean transferable signal | Mean noise cost | Mean predicted Delta | Mean actual gain |
|---|---:|---:|---:|---:|
| ACS | 0.00382 | 0.000329 | 0.00349 | 0.00382 |
| BTS | -0.00137 | 0.001539 | -0.00291 | -0.00137 |
| Citi Bike | 0.00510 | 0.001470 | 0.00364 | 0.00510 |
| NYC TLC | 0.07113 | 0.001137 | 0.06999 | 0.07113 |
| String benchmark | 0.06051 | 0.000389 | 0.06012 | 0.06051 |

There are 326 beneficial and 79 harmful realized cells. Ninety-four cells have
noise cost exceeding transferable signal (including negative-transfer cells).

## 10. Why valid geometry sometimes hurts

- Employee Salaries split 2, 1-NN: misaligned transfer dominates
  (`G=-0.1149`, cost `0.00276`, actual `-0.1149`).
- TLC pickup split 2, 1-NN: local geography transfers in the wrong direction
  (`G=-0.0671`, cost `0.00801`).
- BTS origin split 0, 1-NN: weak/negative airport transfer and the panel's
  largest local estimation cost combine (`G=-0.0209`, cost `0.01166`).
- Employee Salaries split 0, broad RBF: oversmoothing yields `G=-0.0326` even
  though the trigram metric is semantically valid.

Every harmful cell satisfies negative/insufficient transfer or cost exceeding
gain; none requires a new post-hoc mechanism.

## 11. Comparison with heuristics

| Predictor | LO-source Spearman | Sign accuracy | MAE |
|---|---:|---:|---:|
| Support distance | 0.084 | 79.01% | 0.2761 |
| Cover radius | 0.088 | 77.04% | 0.0451 |
| Raw smoothness | -0.597 | 63.21% | 0.0881 |
| Conditional smoothness | -0.602 | 47.16% | 0.0556 |
| Dirichlet energy | 0.601 | 80.49% | 0.0407 |
| Geometry Transfer Law | **0.991** | **96.30%** | **0.000917** |

The comparison favors the law strongly, but its retrospective predictor is
oracle-assisted through `mu_U`; the prospective section is the fair deployment
comparison.

## 12. State-level analysis

Per-state `Delta_u` exactly identifies the local combination of signal
alignment and propagated uncertainty. The retained case-study table contains
near-equal-support state pairs with opposite gains within the same task and
operator. Support distance locates the state relative to training support; it
does not determine the sign of `mu_u (a_u^T mu_T)` or the local noise cost.

## 13. Correct vs corrupted metric

For ACS occupation RBF, corruption from 0% to 100% reduces transferable signal
from `0.000235` to `0.000136` while noise remains near `0.000005`. For TLC
pickup it reduces net predicted Delta from `0.01902` to `0.01408`, although the
10% perturbation briefly improves transfer (`0.01973`), rejecting strict
monotonic degradation. The decomposition agrees with the prior broad result
that correct geometry beats corruption without implying that correct geometry
always beats fallback or every specialist.

## 14. Sample-size phase transition

In the controlled synthetic RBF case, GTR rises from `0.389` at five rows per
state to `0.778` at ten and `1.556` at twenty. Predicted/empirical Delta changes
from `-0.00904/-0.00832` to `+0.00206/+0.00201`, directly validating the
threshold. On controlled real subsamples, noise cost falls with sample size on
both ACS and TLC. ACS predicts harm at five rows (`Delta=-0.000075`) and benefit
at ten, but its one realized test gain stays slightly positive; the expected
real sign transition is therefore not cleanly confirmed. Varying the number of
training states at fixed total rows shows the expected tradeoff: support
improves while per-state noise rises, and operator choice determines the net.

## 15. Conditioning on other tabular features

Yes—geometry usefulness is conditional on the rest of the table. For ACS
occupation, residual state signal falls from `0.280` under an intercept base to
`0.0635` with 30-tree CatBoost and `0.0427` with the strong base; transferable
signal falls from `0.00648` to `0.000235`. Medical Charges falls from residual
state signal `1.754` to `0.131` and transferable signal `0.0833` to `0.00334`.
TLC is mildly nonmonotone between medium and strong bases, demonstrating that
the residual signal can change shape, not only shrink.

## 16. Prospective protocol

The protocol/config were frozen at hashes
`f75dfede...569a7ba` and `36b2b069...04241b`. Three outer state splits, three
inner state folds, row-OOF CatBoost residualization, diagonal Sigma, 3-NN, RBF,
and one domain-specific operator were fixed. NOAA GHCN, Beijing air quality,
Chicago Divvy, and BLS OEWS were declared. BLS could not be acquired from the
execution environment and was not replaced.

After that program was complete, the single requested hierarchy gap was run as
a separately frozen addendum, not inserted retroactively into the original
protocol. Its protocol/config hashes are `36b20009...bb2e28` and
`3bdec9c0...68e7fd`. The source, official 2023 Census County Business Patterns,
target, NAICS prefix-tree metric, three splits (`8801`–`8803`), CatBoost base,
three operators, sealing procedure, and G1–G5 thresholds were fixed before the
11.1 MB outcome archive was downloaded. The retained panel has 37,783 rows and
922 six-digit NAICS industries; each outer split holds out 277 industries.

## 17. Prospective results

| Source | Operator | Predicted Delta | Actual Delta | Sign |
|---|---|---:|---:|:---:|
| Beijing | harmonic | 0.00042 | 0.00111 | correct |
| Beijing | 3-NN | 0.00140 | 0.00231 | correct |
| Beijing | RBF | 0.00047 | 0.00090 | correct |
| Divvy | graph harmonic | 0.00072 | 0.00131 | correct |
| Divvy | 3-NN | 0.04052 | 0.04642 | correct |
| Divvy | RBF | 0.02270 | 0.01825 | correct |
| NOAA | harmonic | 0.00335 | 0.00184 | correct |
| NOAA | 3-NN | 0.01765 | 0.02591 | correct |
| NOAA | RBF | 0.00350 | 0.00194 | correct |
| Census CBP | hierarchy kernel ridge | 0.02197 | 0.02219 | correct |
| Census CBP | 3-NN | 0.01967 | 0.02108 | correct |
| Census CBP | RBF | 0.00020 | 0.00020 | correct |

For the separately frozen Census gap, split-level Pearson is `0.9914`, Spearman
`0.8333`, MAE `0.00091`, calibration slope `1.037`, and sign accuracy `100%`.
Its three operator aggregates have Spearman `1.0` and all three signs correct;
G1–G5 pass. Across the combined 12 source/operator aggregates, Pearson is
`0.9783`, Spearman `0.9091`, MAE `0.00216`, calibration slope `1.099`, and
direct sign accuracy `100%`. The original three-source result remains
unchanged: aggregate Spearman `0.9167` and sign accuracy `100%`.

## 18. Prospective comparison with simple heuristics

Across the four-source combined panel, leave-one-source-out calibration gives
nested predicted Delta Spearman `0.888`; support distance and cover radius both
give `-0.367`, and raw smoothness gives `-0.410`. Nested Delta's LOSO sign
accuracy is `83.3%`. Raw smoothness obtains `100%` sign accuracy only because
all 12 aggregates are positive; it provides the wrong ranking. The law therefore
wins the untouched quantitative comparison, while the 12-point aggregate unit
still warrants uncertainty language.

## 19. MPE reinterpretation

MPE was solving the representation problem when the decisive question was
whether target residual signal should be transferred at all. Since MPE followed
by a linear stem is `w(VW)`, it adds no information beyond normalized
similarity. The prior 44/108 neural wins and 0/4 source wins are therefore
consistent with the proposition: correct geometry can matter while MPE adds no
advantage over similarity, kernels, graph methods, coordinates, or hierarchy
features exposing the same structure.

## 20. What is genuinely new after literature subtraction

Graph smoothing bias–variance, spectral SNR, graph sampling, harmonic
extension, kernel alignment, GP/kriging, similarity encoding, random effects,
and cold-state CV are established. The narrow residual is their synthesis for
one externally metricized tabular field after arbitrary covariates `Z` have
been residualized: an exact zero-fallback transfer/noise decomposition, a
no-metric-only sign theorem, and cross-geometry retrospective plus sealed
prospective testing. The scalar algebra alone is not novel.

## 21. Failed hypotheses

- Support distance predicts benefit: failed retrospectively and prospectively.
- Semantic correctness guarantees improvement: failed in 79 retrospective cells.
- Metric degradation is monotone in corruption fraction: failed at TLC 10%.
- A real sample-size sign transition would be clean in one realization: not confirmed on ACS.
- Prospective harmful-geometry detection: untested because all aggregates helped.
- The original BLS hierarchy acquisition: failed operationally and was not replaced inside that protocol; the separately frozen Census hierarchy gap subsequently passed.
- Primary conclusions were robustly repeated with an MLP/ResNet base: not run and not claimed.

## 22. Main useful insights

1. A correct feature metric can still hurt because target residual effects can be misaligned.
2. Geometry must be judged after the rest of the table is modeled, not on raw target smoothness.
3. Transferable signal must exceed propagated state-estimation noise in expected squared risk.
4. The same metric/operator can cross from harmful to useful as per-state precision increases.
5. Metric-only support and coverage cannot universally determine usefulness.
6. Nested state-held-out evaluation predicts gain across four prospective source families, including an official nonspatial hierarchy.

## 23. ICLR readiness

| Criterion | Score (1–5) |
|---|---:|
| conceptual novelty | 4 |
| theoretical novelty | 3 |
| theorem strength | 4 |
| synthetic validation | 5 |
| real-world explanation | 5 |
| prospective validation | 4 |
| dataset breadth | 5 |
| clarity | 4 |
| baseline/heuristic strength | 4 |
| reproducibility | 5 |

**CORE EMPIRICAL GAP CLOSED**

The untouched Census NAICS hierarchy passes every separately frozen gap
criterion. The program now supports committing to the paper. Naturally harmful
prospective behavior remains unobserved, so harm detection should be framed as
theoretically and retrospectively established rather than prospectively shown.

## 24. Reviewer simulation

1. **Objection:** The retrospective result is tautological. **For:** realized oracle R² is exactly 1. **Against:** population noise accounting changes signs and the prospective nested-CV result is sealed. **Weakness:** the headline retrospective scatter can mislead. **Response:** lead with the prospective prediction and label the oracle scatter an arithmetic audit.
2. **Objection:** The risk identity is elementary linear-estimator algebra. **For:** graph/GP bias–variance is established. **Against:** the conditional tabular formulation and no-metric-only theorem unify heterogeneous field evidence. **Weakness:** theoretical novelty is moderate. **Response:** claim synthesis/problem formulation, not invention of bias–variance.
3. **Objection:** Prospective evidence is uniformly positive. **For:** all 12 aggregates help, so harm detection is not prospectively tested. **Against:** 36 sealed split cells across four families now include a separately frozen official NAICS hierarchy; its split-level Spearman is `0.833` and all G1–G5 criteria pass. **Weakness:** the prospective panel cannot validate negative decisions. **Response:** claim prospective rank/sign calibration on observed positive cells, and reserve harm evidence for the theorem, synthetic experiment, and 79 retrospective cells.
4. **Objection:** State exchangeability is false for spatial extrapolation. **For:** stations and neighborhoods are dependent. **Against:** the theorem explicitly scopes CV to the declared split distribution. **Weakness:** performance outside that design is unknown. **Response:** avoid iid claims and report spatial split sensitivity.
5. **Objection:** Results depend on CatBoost residualization. **For:** no second neural base family. **Against:** weak/medium/strong conditioning behaves as theory predicts and the theorem is architecture-free. **Weakness:** empirical robustness is incomplete. **Response:** add one small cross-family check, not a neural matrix.

## 25. Final paper thesis

### Thesis A

Geometry helps unseen tabular states when transferable conditional
state signal exceeds estimation noise; an exact risk decomposition
predicts this help/harm boundary.

This thesis is supported across the theorem, synthetic boundary tests,
retrospective explanations, and four-family sealed prospective program, with
the prospective harm-observability caveat stated above.

## 26. Best paper titles

1. **When Does Feature Geometry Help? A Risk Decomposition for Unseen Tabular States**
2. **Valid Geometry, Wrong Prediction: When Structure Transfers to Unseen States**
3. **Geometry Is Not Enough: Signal, Noise, and Cold-State Tabular Prediction**
4. **Transfer or Fall Back? External Geometry for Unseen Tabular States**
5. **Conditional Geometry: Predicting Help and Harm Beyond Known Categories**

## 27. Final recommendation

**COMMIT TO THE THEORY PAPER**

Do not reopen MPE, expand the operator leaderboard, or continue source hunting.
Preserve the original protocol and the separately frozen Census addendum as
distinct evidence layers. Lead the paper with the conditional transfer/noise
law and sealed prospective calibration; describe prospective harm detection as
an untested boundary, not a demonstrated deployment capability.
