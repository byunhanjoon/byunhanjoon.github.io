# RESULTS — MPE ICLR VERDICT

Status: **DAY-6 CONSOLIDATED DECISION REPORT — 2026-08-29**

This report distinguishes the frozen MPE evidence, post-hoc mechanism work, and
the exploratory successor direction. The original MPE neural matrix was stopped
before completion, so incomplete cells are never presented as a completed
confirmatory program.

## 1. Executive verdict

**NOT SUPPORTED**

The completed evidence does not support MPE as a standalone ICLR method. The
frozen real-data program was stopped with 988 of 7,640 expected neural artifacts,
so its final completion audit is **FAIL**, not a protocol-complete verdict. The
complete ridge comparison nevertheless gives a consistent negative signal: MPE
beats the validation-selected strongest metric-aware baseline on 0/4 runnable
primary sources. Its source-balanced relative gain is -0.691%, with a 95%
source-bootstrap interval of [-1.525%, -0.086%]. On 108 currently paired neural
MPE-versus-normalized-similarity cells, MPE wins 44 (40.7%) and has a median
relative gain of -0.106%, so the learned token factorization has no observed
advantage. Correct geometry beats corrupted MPE in 35/38 task-split aggregates,
showing that the supplied metrics contain real predictive signal. That signal
does not establish that MPE is the best way to expose the metric. Synthetic
cycle/tree interpolation, exact chart invariance, and the formal interpolation
results remain valid mechanism evidence, but they do not transfer to a broad
real-world architecture win. The frozen Metric-Field Transport successor also
rejects an always-raw-distance replacement because a -38.68% string/medical
failure drives source-balanced neural performance to -4.82%. A training-state-
only Metric Trust Router avoids that failure and reaches +10.28%
source-balanced neural improvement, but this is post-outcome feasibility
evidence rather than confirmation. The broader thesis that tabular fields may
have useful external geometry survives. The narrower thesis that MPE is a
generic winning tokenizer does not. The best current direction is therefore to
pivot from representation invention to risk-controlled metric trust under
unseen-state shift.

## 2. Final dataset panel

State counts use frozen split 0. Unavailable sources remain in the panel.

| Panel | Source | Task | Metric field | Metric type | Rows | Train states | Val states | Test states | Median support gap |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| PRIMARY EXTERNAL-METRIC | ACS | Industry | industry code | official hierarchy shortest path | 299,300 | 140 | 46 | 48 | 4.000 |
| PRIMARY EXTERNAL-METRIC | ACS | Occupation | occupation code | official hierarchy shortest path | 297,300 | 259 | 86 | 88 | 6.000 |
| PRIMARY EXTERNAL-METRIC | BTS | Destination airport | airport | FAA-coordinate haversine | 293,600 | 127 | 42 | 43 | 120.851 |
| PRIMARY EXTERNAL-METRIC | BTS | Origin airport | airport | FAA-coordinate haversine | 293,800 | 129 | 43 | 43 | 128.802 |
| PRIMARY EXTERNAL-METRIC | Citi Bike | Start station | station | published-coordinate haversine | 430,100 | 903 | 301 | 301 | 0.243 |
| PRIMARY EXTERNAL-METRIC | NYC TLC | Dropoff zone | taxi zone | centroid haversine | 298,000 | 100 | 33 | 34 | 1.321 |
| PRIMARY EXTERNAL-METRIC | NYC TLC | Pickup zone | taxi zone | centroid haversine | 297,600 | 55 | 18 | 19 | 0.839 |
| PRIMARY EXTERNAL-METRIC | Amazon 2023 | Leaf category | product category | declared hierarchy | — | — | — | — | NOT RUN: frozen snapshot has no category paths |
| SECONDARY STRING-METRIC | Employee Salaries | Salary | position title | character-trigram Jaccard | 7,923 | 55 | 18 | 19 | 0.659 |
| SECONDARY STRING-METRIC | Medical Charges | Payment | DRG definition | character-trigram Jaccard | 100,000 | 60 | 20 | 20 | 0.372 |
| SECONDARY STRING-METRIC | Open Payments | Payment | product/payment descriptor | declared string metric | 73,558 | — | — | — | NOT RUN: mandatory amount field absent |
| OPTIONAL CONTROLLED-ACCESS | MIMIC-III | Diagnosis | ICD code | ICD hierarchy | — | — | — | — | NOT RUN: controlled access unavailable |

The nine runnable public tasks were evaluated under state-disjoint splits. The
two public-schema failures and MIMIC access failure were not replaced after
outcomes.

## 3. Main real-world result

Lower standardized MSE is better. The paired interval is for `best baseline −
MPE`; a negative interval favors the baseline. These are the available frozen
ridge comparisons, not a completed neural matrix.

| Source | MPE state | Best baseline state | Similarity | PLE | UNK | Mean corrupt MPE | MPE row | Best baseline row | Relative gain | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ACS | 0.6275 | 0.6153 | 0.6254 | 0.6202 | 0.6790 | 0.6760 | 0.6262 | 0.6077 | -1.98% | [-0.0197, -0.0062] |
| NYC TLC | 0.9510 | 0.9495 | 0.9465 | 1.0957 | 1.0901 | 1.1693 | 0.8408 | 0.8317 | -0.16% | [-0.0148, 0.0107] |
| Citi Bike | 1.0109 | 1.0047 | 1.0109 | 1.0141 | 1.0212 | 1.0212 | 0.9685 | 0.9634 | -0.61% | [-0.0108, -0.0015] |
| BTS | 1.1196 | 1.1195 | 1.1195 | 1.1218 | 1.1206 | 1.1202 | 1.0103 | 1.0102 | -0.01% | [-0.0003, 0.0001] |
| Amazon 2023 | — | — | — | — | — | — | — | — | — | NOT RUN |

MPE loses the source mean on all four runnable primary source families. Its best
case is an effective tie on BTS; its clearest loss is ACS.

## 4. Strongest-baseline showdown

| Source | MPE | Best baseline | Baseline family/families selected across cells | Relative gain | Winner |
|---|---:|---:|---|---:|---|
| ACS | 0.6275 | 0.6153 | metric kNN, Laplacian, node2vec, path-to-root | -1.98% | Baseline |
| NYC TLC | 0.9510 | 0.9495 | metric kNN, Nyström, coordinates, RBF | -0.16% | Baseline |
| Citi Bike | 1.0109 | 1.0047 | metric kNN, normalized RBF | -0.61% | Baseline |
| BTS | 1.1196 | 1.1195 | coordinates, lat/lon, normalized/raw RBF | -0.01% | Baseline |
| Amazon 2023 | — | — | unavailable hierarchy | — | — |

- Source wins: **0/4 runnable**; the frozen target was 4/5.
- Source-balanced mean relative gain: **-0.6906%**.
- Median relative gain: **-0.3871%**.
- 95% source-bootstrap interval: **[-1.5250%, -0.0862%]**.

The low number of independent sources gives the bootstrap low discrete
resolution, but its sign and the 0/4 win count both oppose the paper thesis.

## 5. Similarity Encoding vs MPE

**Answer: no.** Direct normalized Similarity Encoding exposes the same
landmark-weight vector `w(x)` used by MPE. Neural MPE computes `w(x)V`; when the
next stem is an unconstrained linear map, this is a bias-free linear
factorization rather than new information. Ridge results therefore give exactly
0.00% MPE gain over normalized same-metric similarity on every source. In the
currently completed neural pairs, MPE wins only 44/108 cells (40.7%), with
median gain -0.106% and mean gain -0.754%. Increasing the training ceiling from
300 to 600 epochs on the two frozen convergence cells does not help
(-0.154% and -1.395% validation changes). The learned landmark-token layer lacks
evidence of novelty or practical value.

## 6. Nyström/kernel comparison

**Answer: no.** MPE does not consistently outperform Nyström or RBF features
using the same metric, landmarks, and validation-selected bandwidth. At ridge,
both MPE and Nyström use 32 coordinates; neural MPE additionally learns a
32×32 token matrix. Nyström/RBF is selected among the strongest alternatives on
TLC, Citi Bike, and BTS, and is nearly indistinguishable from MPE on many cells.
At 10,000 states, the recorded dense MPE and Nyström representations use 1.28 MB
and 2.56 MB respectively with similar precompute time (0.323 s versus 0.324 s),
but that efficiency advantage is not accompanied by an accuracy advantage.

## 7. Hierarchy tasks

| Task | MPE | Ancestor multi-hot | Path-to-root | Laplacian | node2vec | Nyström | Best method |
|---|---:|---:|---:|---:|---:|---:|---|
| ACS industry | 0.6349 | 0.6227 | 0.6229 | 0.6271 | 0.6271 | 0.6334 | Ancestor multi-hot |
| ACS occupation | 0.6200 | 0.6024 | 0.6029 | 0.6196 | 0.6235 | 0.6181 | Ancestor multi-hot |
| Amazon leaf category | — | — | — | — | — | — | NOT RUN |

Hierarchy-specific encodings clearly beat generic MPE on ACS. MIMIC was
unavailable, and Amazon's frozen snapshot had no usable hierarchy. The evidence
therefore rejects a hierarchy-specific MPE thesis as well as the generic one.

## 8. Geographic/network tasks

| Task | MPE | Best specialist | Specialist | Relative interpretation |
|---|---:|---:|---|---|
| BTS destination airport | 0.8447 | 0.8444 | raw coordinates | effective tie, MPE loses |
| BTS origin airport | 1.3192 | 1.3190 | raw coordinates | effective tie, MPE loses |
| Citi Bike station | 1.0163 | 1.0162 | Nyström | effective tie, MPE loses |
| TLC dropoff | 0.6606 | 0.6481 | node2vec | MPE loses |
| TLC pickup | 0.9758 | 0.9988 | raw lat/lon | MPE wins this task mean |

Raw coordinates, lat/lon, Fourier coordinates, spatial RBF, graph spectral
features, node2vec, and Nyström were included where applicable. Generic metric
interpolation does not add a consistent benefit beyond these specialists: MPE
wins the TLC-pickup task mean, but node2vec wins TLC dropoff and the combined
TLC source still loses the validation-selected cellwise baseline. The
earlier Bike hour diagnostic is an additional boundary: MPE beat Q-PLE but lost
to Fourier features by 6.22% with an MLP and 8.20% with a ResNet.

## 9. Support-distance mechanism

| Source | Spearman(support distance, MPE advantage) | Near gain | Medium gain | Far gain |
|---|---:|---:|---:|---:|
| ACS | +0.083 | -0.0127 | -0.0205 | -0.0035 |
| BTS | +0.113 | -0.0002 | -0.00003 | -0.00007 |
| Citi Bike | +0.021 | -0.0047 | -0.0075 | -0.0062 |
| NYC TLC | -0.081 | +0.00005 | +0.0041 | -0.0077 |
| String benchmark | +0.028 | -0.1126 | -0.0068 | +0.0220 |

Four of five source correlations are positive, satisfying the weak frozen sign
gate, but the magnitudes are tiny and gains are usually negative. The far bin
can reverse, as on TLC. MPE does not become reliably more useful as states move
farther from observed support; beyond useful landmark coverage, extrapolation
can worsen.

## 10. Theoretical validation

| Result | Statement | Proof status | Numerical validation | Maximum violation/difference |
|---|---|---|---|---:|
| Theorem 1 | transported chart invariance | proved | 288 codebooks | 0 |
| Theorem 2 | partition-of-unity interpolation bound | proved | 324 cells | 0 |
| Theorem 3 | linear-head realizability | proved | 6 rank cases | 0 |
| Theorem 4 | equality-metric impossibility | proved | unseen-weight collapse | 0 |
| Theorem 5 | metric-perturbation stability | proved under positive normalizer | 27 cells | bound ratio 0.0271 |
| Theorem 6 | landmark coverage/metric complexity | proved | 567 cells | 0 |
| Proposition 7 | triangular interval special case | proved | interpolation identity | 2.998e-15 |

The formal results validate the construction and its boundaries. They do not
imply target smoothness, statistical consistency on arbitrary real fields, or
superiority over another representation containing the same distances.

## 11. Exact chart invariance

Across 72 real feature relabelings and 288 synthetic transported codebooks,
MPE and same-metric similarity features have maximum recorded difference 0.
Aligned lookup columns are equivariant but cannot distinguish new states. Q-PLE,
uniform PLE, and code-RBF remain sensitive to arbitrary storage codes. Exact
invariance is a valid property of the metric interface, but it is shared by the
strong metric-aware baselines and therefore is not an MPE-specific advantage.

## 12. Correct vs corrupted metric

| Source | Correct MPE | Mean corrupt MPE | Corrupt q10 / median / q90 | Wins | Comparable task-split cells |
|---|---:|---:|---|---:|---:|
| ACS | 0.6275 | 0.6760 | 0.4430 / 0.6632 / 0.9067 | 10 | 10 |
| BTS | 1.0969 | 1.0975 | 0.7301 / 1.0805 / 1.6410 | 7 | 9 |
| Citi Bike | 1.0185 | 1.0290 | 0.9572 / 1.0332 / 1.0681 | 3 | 3 |
| NYC TLC | 0.9304 | 1.1589 | 0.8651 / 1.1485 / 1.5305 | 6 | 6 |
| String benchmark | 1.0320 | 1.2056 | 0.5884 / 0.9440 / 2.2788 | 9 | 10 |

Correct geometry wins 35/38 aggregates (92.1%) and all five source means. Ten
independent capacity-preserving corruptions were retained for every completed
ridge cell. The win/mean columns use the comparable gate cells; the q10,
median, and q90 columns summarize all completed corrupt ridge rows for each
source. This is the cleanest evidence that external geometry contains
causal predictive information. It supports the problem formulation, not the
learned MPE factorization.

## 13. Nominal negative controls

| Field | Equality MPE | Best support-complete control | Relative gain | >2% MPE advantage? |
|---|---:|---:|---:|:---:|
| ACS class of worker | 1.5512 | 1.5395 | -0.76% | No |
| BTS reporting airline | 1.8934 | 1.8936 | +0.01% | No |
| TLC payment type | 1.1534 | 1.1504 | -0.26% | No |

Equality geometry gives every distinct unseen non-landmark state the same
partition weights. No nominal field shows a greater-than-2% MPE advantage, so
the negative control passes. The older 77.64% synthetic nominal gain was caused
by Q-PLE state collapse; support-complete uniform PLE reduced it to 0.094%.

## 14. Seen vs unseen states

The available ACS control gives -1.98% MPE gain on unseen states versus -15.35%
on IID seen-state rows, a +13.37-point relative shift. This suggests that metric
information is more relevant under cold-state shift than under ordinary row
splits. It does not turn MPE into a winner, and the seen-state comparison is not
complete across all source families. Treat it as a mechanism clue, not a broad
gate pass.

## 15. Target-smoothness diagnostic

The training-only diagnostic fails. Task-split Spearman correlation with MPE
gain is +0.258, while leave-one-source-out prediction has Spearman -0.900 and
mean absolute error 4.61 gain points across five sources. It should not be used
as a deployment rule or as evidence that the helpful fields can already be
identified reliably.

## 16. Landmark/metric complexity

Increasing the landmark budget materially helps the high-cardinality ACS task:
MSE falls from 0.6692 at `m=8` to 0.6200 at `m=32`, 0.6053 at `m=128`, and
0.6018 at `m=256`. Citi Bike improves more modestly, from 1.0213 at `m=8` to
1.0123 at `m=128`; the airport task is effectively flat. Larger landmark sets
also recover more raw distance geometry, but they do not fix the core
same-information equivalence between normalized weights and a learned linear
token map. Cover radius is more mechanistically relevant than raw cardinality,
yet neither predicts a universal performance advantage.

## 17. Ablations

The completed ablations cover Gaussian, Laplacian, triangular and inverse-
distance kernels; farthest-point, frequency, k-medoids and random landmarks;
partition, raw and softmax normalization; `D∈{16,32,64}`; bandwidth and metric
scale; landmark count; sparse neighborhoods; equality; partial/full metric
corruption; and secondary string metrics.

Main conclusions:

- Token dimension 16/32/64 is identical for the linear-head realizability test.
- More landmarks can improve approximation, especially on ACS, but this is a
  geometry-resolution effect rather than evidence for learned token mixing.
- Increasing metric corruption generally hurts ACS, Citi Bike, TLC, and
  medical strings, although the finite partial-corruption sequence is not
  monotone in every task; airport geometry is weak for the target.
- Kernel, bandwidth, normalization, and selection preferences are source-
  dependent; no frozen variant dominates.
- Multiscale MPE remains rejected from the earlier screen.
- No outcome-driven MPE-v2 is promoted.

## 18. Efficiency

At the primary budget MPE has 32 representation coordinates and, in neural
models, a 1,024-parameter tokenizer. At 10,000 states, dense MPE uses 1.28 MB,
sparse MPE 0.12 MB, Nyström 2.56 MB, full similarity 240 MB, and lookup roughly
0.12 MB in the recorded scaling test. Precompute times for dense MPE and
Nyström are nearly identical (0.323 s and 0.324 s). The available neural
telemetry is not a balanced full-matrix efficiency benchmark because the
neural program was stopped; it must not support a final speed claim. MPE's
compactness is useful engineering, but compactness without accuracy gain is
not an ICLR method result.

For the completed ridge scaling check, full-cap fitting takes 0.237 s for MPE,
0.492 s for Nyström, 0.391 s for raw similarity, and 0.042 s for the UNK
representation. Recorded per-query inference is 7.38, 11.03, 7.13, and 7.52
microseconds respectively. These measurements are diagnostic timings, not a
randomized system benchmark.

## 19. Failure cases

- **PLE:** MPE loses some real cells even to code-based PLE; the synthetic
  advantage does not transfer universally.
- **Similarity Encoding:** normalized similarity contains the same weights;
  MPE wins only 44/108 available paired neural cells.
- **Nyström/RBF:** classical same-metric representations match or beat MPE on
  many geography and hierarchy cells.
- **Hierarchy methods:** ancestor multi-hot and path-to-root clearly beat MPE
  on both ACS tasks.
- **Coordinates/Fourier:** raw coordinates edge out MPE on BTS, specialist
  spatial/graph encoders win TLC cells, and Fourier wins the observed Bike-hour
  diagnostic.
- **Corruption:** correct MPE fails to beat mean corrupt MPE in 3/38 aggregates,
  showing weak or unstable metric relevance in some cells.
- **Support distance:** correlations are weak; TLC is negative and far bins can
  reverse.
- **Target relevance:** a valid input metric may be irrelevant or actively
  harmful for a particular target; raw distance causes -38.68% neural transfer
  on the string/medical source.
- **Optimization:** random learned and frozen-orthogonal factorizations lose
  2.86% and 2.87% to direct weights; ReZero recovers only +0.24%.
- **Completion:** the original neural matrix has 988/7,640 artifacts, only
  261/3,600 required corrupt-control cells, and an audit result of 15/20
  integrity checks plus 0/38 unit tests in the current runtime; test collection
  is blocked by the missing `shapefile` package.

The audit's unit-test count reflects a dependency-blocked collection, not 38
demonstrated scientific test failures. The decisive scientific completion issue
is the missing stopped matrix; the later successor studies have separate
passing audits.

## 20. What the synthetic results did and did not prove

The synthetic cycle/tree targets were constructed to be smooth in the supplied
metric. They prove that MPE can interpolate such targets, remain invariant to
transported codebooks, and respond causally to metric corruption. They do not
prove that an independently declared hierarchy, coordinate metric, graph, or
string distance is smooth for a real target. The real panel confirms that
geometry sometimes matters, but it does not show that MPE uses geometry better
than similarity, kernels, graph methods, hierarchy encodings, or coordinates.

## 21. Final claim after novelty subtraction

After subtracting PLE, Similarity Encoding, Nyström/RBF, hierarchy encodings,
spectral/node2vec features, and Fourier/coordinate methods, the strongest MPE
claim is:

> Externally declared value geometry can improve cold-state tabular prediction
> and guarantee storage-code invariance, but the frozen evidence does not show
> that a learned metric-partition token layer improves over established ways to
> expose the same geometry.

The broader Day-6 portfolio sharpens the next question:

| Track | Scientific status | Main finding | Decision |
|---|---|---|---|
| OrbitCover incumbent | independently audited Day-5 evidence, re-ranked on Day 6 | coupled OC2 wins 144/144 neural cells with 55.9% equal-source residual reduction, but independent OC2 is 7.0% worse and target shift/selection utility remain concerns | Day-6 portfolio lead at 65/100; separate from the metric program |
| Semantic Arithmetic / IEA64 | prospective on 3 datasets | H1, H7, and H9 support narrow numerical reproducibility/attenuation; H2–H6 and H8 fail; no accuracy benefit | retain as separate narrow alternative |
| Native Feature Geometry | synthetic pilot + prospective corruption replay | spontaneous Gram recovery and universal fixed mitigation fail; chart transport and corruption dose response pass | mechanism candidate, not lead |
| MPE synthetic screen | frozen synthetic + one real cyclic diagnostic | strong unseen metric-smooth interpolation; loses to Fourier on observed hour | motivation only |
| MPE real program | stopped, incomplete audit | correct metric matters, MPE loses strongest source baseline 0/4 | abandon MPE as main method |
| Metric-Field Transport | frozen post-outcome development, audit 20/20 | raw distances help ACS/Citi/TLC but catastrophically hurt strings; E2 gate fails | reject always-raw transport |
| Metric Trust Router | exploratory post-outcome, audit 14/14 | +10.28% neural, 15/9/0 wins/ties/losses, no source degradation | prospectively confirm on new data |

## 22. ICLR contribution assessment

Scores assess the MPE paper as it stands, not the potential trust-routing paper.

| Criterion | Score (1–5) |
|---|---:|
| Conceptual novelty | 4 |
| Method novelty | 2 |
| Theoretical contribution | 4 |
| Synthetic mechanism evidence | 5 |
| Real-world evidence | 2 |
| Dataset breadth | 4 |
| Baseline strength | 5 |
| Unseen-state relevance | 5 |
| Statistical rigor | 3 |
| Reproducibility/completion | 3 |
| Story coherence | 3 |

The strongest assets are the problem formulation, frozen state-disjoint
benchmark, theory, corruption controls, and baseline breadth. The decisive
weaknesses are the negative method comparison and incomplete original neural
audit.

## 23. Reviewer simulation

1. **MPE is only a learned linear reparameterization of similarity weights.**
   Supporting evidence: `wV` plus a linear stem collapses algebraically, and the
   paired neural results are negative. Evidence against: factorization could
   alter optimization or regularization in deeper networks. Remaining weakness:
   E0 and the current neural pairs do not show such a benefit. Best response:
   concede the equivalence and stop claiming tokenizer novelty.
2. **The confirmatory program is incomplete.** Supporting evidence: 988/7,640
   neural artifacts and audit FAIL. Evidence against: ridge, theory, real-source
   mechanism tests, successor experiments, and their audits are complete.
   Remaining weakness: the frozen neural gate cannot be called completed. Best
   response: label the MPE verdict “not supported by available evidence” and do
   not claim protocol closure.
3. **The metric is often not target-relevant.** Supporting evidence: weak
   smoothness/support diagnostics and -38.68% raw-distance failure on medical
   strings. Evidence against: correct geometry wins 35/38 against corruption.
   Remaining weakness: geometry relevance varies by field and task. Best
   response: make transfer risk, not universal encoding, the research problem.
4. **Specialist representations are stronger.** Supporting evidence: hierarchy
   ancestors, graph/node2vec, coordinates, kernels, and Fourier win applicable
   tasks. Evidence against: one generic interface is simpler and exactly
   invariant. Remaining weakness: convenience is insufficient for ICLR. Best
   response: benchmark all specialists and seek calibrated fallback guarantees.
5. **The router is post-hoc and generic model selection.** Supporting evidence:
   its outer outcomes motivated the study, and five-fold group CV is standard.
   Evidence against: decisions use training states only, were written before
   outer joins, pass all exploratory gates, and directly prevent the observed
   failure. Remaining weakness: no prospective new-source evidence or risk
   theorem. Best response: freeze it on new cohorts and add an oracle/safety
   guarantee before submission.

## 24. ICLR decision

**PIVOT METHOD**

Do not write MPE as the main ICLR method. The evidence already isolates the
problem: externally supplied geometry can help greatly, can add no value, or
can harm severely depending on transferability to the target. The best next
paper is a problem + benchmark + risk-control paper about deciding when to
trust geometry under strict state-disjoint shift. The current router is only a
feasibility baseline; confirmation requires new source families or temporal
cohorts, frozen before outcomes.

## 25. Best final thesis

**Thesis C**

> Known geometry helps unseen categories, but existing similarity/kernel/graph
> methods are sufficient and MPE itself adds little.

For the successor paper, the hypothesis becomes: target-independent geometry
can be strongly helpful or harmful, and state-group cross-fitting can control
that transfer risk before new states arrive.

## 26. Best paper titles

1. **Trust, Then Encode: Risk-Controlled External Geometry for Unseen Tabular States**
2. **When Should Tabular Models Trust Feature Geometry?**
3. **Helpful or Harmful Geometry: Safe Cold-State Tabular Prediction**
4. **Beyond Unknown Tokens: Testing Metric Transfer to Unseen Tabular States**
5. **Known Geometry, Familiar Tools: A Stress Test of Metric-Space Tabular Encodings**

The first four titles are contingent on prospective confirmation and a real
risk-control contribution. The fifth is the honest title for the evidence
already completed.

## 27. Paper outline

1. **Introduction** — external value geometry under unseen-state shift and the
   danger of negative transfer.
2. **Typed metric fields and trust** — `(X,d)`, target-independent metadata,
   state-disjoint risk, and leakage boundaries.
3. **Representation candidates** — normalized weights, raw distances,
   similarity, Nyström, specialist encoders, and categorical fallback.
4. **Risk-controlled selection** — state-group cross-fitting, confidence-aware
   fallback, and an oracle/safety bound.
5. **Experimental protocol** — new frozen sources/cohorts, source clustering,
   sealed outcomes, and corrupted/irrelevant metrics.
6. **Real unseen-state benchmark** — neural and classical backbones with strong
   same-information and domain baselines.
7. **Transferability analysis** — when geometry helps, fails, or becomes
   harmful; support and smoothness are diagnostics rather than assumptions.
8. **Safety and calibration** — regret relative to fallback and oracle,
   degradation tails, and new-source reliability.
9. **Related work and novelty subtraction** — similarity encoding, anchor
   distances, knowledge-enriched tabular learning, cold-start gating, and graph
   transport.
10. **Conclusion** — geometry is metadata to trust conditionally, not a
   universally superior tokenizer.

## 28. Final recommendation

**PIVOT TO DIFFERENT METRIC METHOD**

Stop optimizing MPE depth, token width, bandwidth, landmark count, or another
linear embedding on the observed panel. Preserve MPE as a compact baseline and
theory vehicle. Promote the fixed five-fold Metric Trust Router only to a
prospectively frozen new-data test, alongside normalized weights, raw
distances, Similarity Encoding, Nyström, categorical fallback, and the best
domain specialists. Require at least 5% source-balanced neural gain, no source
degradation above 2%, correct fallback on at least two irrelevant metrics, and
superiority to the best same-information baseline. Without that confirmation
and a meaningful risk guarantee, do not submit this direction as an ICLR method
paper.

---

Evidence anchors:

- [Original MPE completion audit](../mpe_iclr/FINAL_AUDIT.md)
- [Original MPE main real results](../mpe_iclr/TABLE_3_MAIN_REAL_RESULTS.md)
- [Original MPE theory validation](../mpe_iclr/TABLE_9_THEOREM_VALIDATION.md)
- [Day-6 semantic arithmetic report](DAY6_FINAL_REPORT.md)
- [Native Feature Geometry report](native-feature-geometry/PILOT_FINAL_REPORT.md)
- [Synthetic MPE report](metric-partition-embeddings/FINAL_REPORT.md)
- [Metric-Field Transport results](metric-field-transport/RESULTS.md)
- [Metric Trust Router results](metric-trust-router/RESULTS.md)
- [Next-direction decision](NEXT_ICLR_DIRECTION.md)
