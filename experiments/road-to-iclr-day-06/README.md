# Road to ICLR — Day 6

Day 6 contains five connected experiment tracks.  They all belong to the same
research day; the topic directories below separate their code and results
without advancing the series counter.

| Track | Directory | Focus |
|---|---|---|
| Semantic arithmetic | this directory | finite-precision behavior under equivalent schema rewrites |
| Native Feature Geometry | `native-feature-geometry/` | learned-chart transport and metric-corruption controls |
| Metric Partition Embeddings | `metric-partition-embeddings/` | metric-aware tokenization, neural confirmation, and bike-demand controls |
| Metric-Field Transport | `metric-field-transport/` | post-MPE factorization and raw metric-coordinate screens |
| Metric Trust Router | `metric-trust-router/` | state-cross-fitted fallback for harmful metric transfer |

The public Day 6 essay draws on all five tracks.  Each topic directory keeps
its own frozen protocols, reports, scripts, and result artifacts so their
evidence trails remain reproducible.

The latest post-MPE research decision is in `NEXT_ICLR_DIRECTION.md`. It keeps
the strong spatial/hierarchical distance-coordinate effect, rejects an
always-on metric tokenizer, and recommends a separately confirmed
state-cross-fitted metric-trust direction.

Day 6 starts from the strongest unresolved mechanism in the Day-5 final
closure: exact functional matching closes schema paths for some architectures,
but semantically identical dense coordinates can still diverge during training.

The first hypothesis is **Semantic Arithmetic Amplification (SAA)**.  It asks
whether feature/category permutations inject only floating-point reduction-order
error at the first affine map, after which the optimizer amplifies that error.
The proposed intervention, **Interface Exact Accumulation (IEA)**, performs only
the schema-facing affine accumulation in float64 and immediately casts back to
float32.  Every later operation, parameter, optimizer state, minibatch, dropout
draw, and objective remains float32 and matched.

## Current evidence state

| ID | Idea | Status |
|---|---|---|
| H1 | Semantic Arithmetic Amplification / IEA64 | confirmed on 72 bundles; keep with narrow claim |
| H2 | nominal Precision-Delay Law | falsified; discard the law |
| H3 | all-row, 200-epoch closure | falsified on the complete 36-bundle / 288-path matrix |
| H4 | two-epoch Semantic Shadow forecast | falsified on 324 bundles / 972 paths |
| H5 | shadow-to-seed fragility transfer | falsified on the complete H4 tensor reuse |
| H6 | epoch-20 Semantic Lyapunov Screen | falsified: AUROC 1.0 but zero improvement over the raw level |
| H7 | Rounding-Cell Survival | supported on 31 prospective bundles / 93 paths |
| H8 | Level-or-Acceleration Semantic Screen | falsified: balanced accuracy .944 but zero improvement over H6 |
| H9 | Post-Breach Arithmetic Attenuation | supported on 25 prospective bundles / 75 paths |

The first partial H3 Bank/ResNet result is an important scope correction: it
looks stable at epoch 20 but becomes macroscopically divergent by epoch 50,
while IEA64 remains exactly closed.  Therefore the H1 architecture boundary is
currently a short-horizon observation, not a universal MLP/ResNet guarantee.

The final Day-6 ranking was issued only after the 06:21 KST seven-hour horizon
and completion of all frozen evidence.  OrbitCover is the 65/100 lead;
Semantic Arithmetic / IEA64 is the 58/100 alternative.  See
`DAY6_FINAL_REPORT.md` for the rubric and failure boundaries.

## Authoritative files

- `THEORY_FOUNDATIONS.md`: propositions and explicit claim boundaries;
- `RECENT_LITERATURE_AUDIT.md`: closest-work and novelty subtraction;
- `REVIEWER_ATTACK_AUDIT.md`: adversarial paper-readiness and baseline audit;
- `STATISTICAL_SCOPE_AUDIT.md`: replication units and anti-pseudoreplication
  interpretation policy;
- `FINAL_REPORT_CHECKLIST.md`: artifact-by-artifact final evidence contract;
- `EXTERNAL_CONFIRMATION_ROADMAP.md`: outcome-contingent next-stage experiments;
- `IDEA_LEDGER.md`: chronological keep/change/discard state;
- `H1_CONFIRMATION_REPORT.md`, `H2_FINAL_REPORT.md`, and
  `H3_FINAL_REPORT.md`–`H9_FINAL_REPORT.md`: final hypothesis reports;
- `DAY6_FINAL_REPORT.md`: final numeric ranking, lead, alternative, and discard
  list;
- `FINAL_RANKING_PROTOCOL.md`, `FINAL_RANKING_ADDENDUM_H8.md`, and
  `FINAL_RANKING_ADDENDUM_H9.md`: pre-outcome rubric and the no-weight-change
  placement of the later H8/H9 successors;
- `HYPOTHESIS_0*_PROTOCOL.md`: frozen hypothesis-specific gates;
- `semantic_arithmetic.py`, `precision_delay.py`,
  `fullscale_arithmetic.py`, and `semantic_shadow.py`: bundle runners;
- `analyze_*.py`: frozen primary and reuse analyses;
- `audit_day6_integrity.py`: hashes, menus, tensor-shape, and finiteness audit;
- `audit_day6_completion.py`: strict horizon, matrix, summary, report, and
  final-ranking completion audit;
- `RESEARCH_LOG.md`: timestamped decision and evidence history;
- `make_day6_figures.py`: declared H3–H9 trajectory, forecast, survival, and
  attenuation figures;
- `test_semantic_arithmetic.py`: construction and algebra tests.

Run one paired bundle with:

```bash
PYTHON=/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python
CUDA_VISIBLE_DEVICES=0 "$PYTHON" semantic_arithmetic.py \
  --dataset bank_marketing_subscription --model ft_transformer --seed 6101 \
  --device cuda:0
```

The experiment is resumable at the bundle level.  A result is never overwritten
unless `--force` is explicitly passed.

The completed run used a detached `run_day6_post_h3.sh` chain.  It waited for
H3's complete summary, evaluated prospective H6/H7/H8/H9, ran H4 only after H3
released both GPUs, then analyzed H4/H5 and reran integrity/tests.  It did not
issue a scientific verdict or alter any frozen gate.

Reanalyze and audit current stored evidence with:

```bash
cd experiments/road-to-iclr-day-06
"$PYTHON" audit_day6_integrity.py
PYTHONPATH=. "$PYTHON" analyze_fullscale_arithmetic.py
PYTHONPATH=. "$PYTHON" analyze_semantic_lyapunov.py
PYTHONPATH=. "$PYTHON" analyze_rounding_survival.py
PYTHONPATH=. "$PYTHON" analyze_semantic_acceleration.py
PYTHONPATH=. "$PYTHON" analyze_postbreach_attenuation.py
"$PYTHON" -m pytest -q
```
