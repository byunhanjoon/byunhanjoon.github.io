# Research-program completion audit

Updated: 2026-08-31. This is a scope-control checklist against `agent.md`, not an evidence report. A green implementation test does not substitute for an empirical gate, and later phases remain conditional on earlier gates.

| Requirement | Current evidence | Status / next proof required |
|---|---|---|
| Literature/novelty boundary | `NOVELTY_LEDGER.md`; repository/environment entry in `EXPERIMENT_LOG.md` | Implemented; refresh before final claims. |
| Reproducible environment and immutable per-job artifacts | Frozen YAML configs, append-only manifest, checksummed prediction bundles, package/checkpoint telemetry | Implemented for Phase I/II; final whole-repository reproduction remains open. |
| Exact transformation library and four-way protocol | Unit/integration tests; Phase I raw audits; 1,120/1,120 Phase II transform preflights | Implemented and verified. |
| Frozen pilot and Gate G1 | `PILOT_PANEL_FREEZE.md`, `PHASE1_KILL_TEST.md`, 3,276/3,276 validated jobs | Complete: narrow Route-B pass for TabICLv2; Route A failed. |
| Frozen 20-task development suite | `DEVELOPMENT_SUITE_FREEZE.md`, `configs/audit/main.yaml` | Complete before Phase II outcomes. |
| Phase II full audit | Frozen 13,440-job grid under source digest `02882b44384093b972708fe6edfa54a1340d8599988a5d35f9e851a042053b5e` | Running. Completion requires 13,440 checksum-valid records, hierarchical CIs, six prescribed figures, and `REPARAM_AUDIT.md`. |
| Phase III descriptor explanation | Train-only descriptor schema and grouped cross-dataset ridge/RF analyzer | Implemented but unevaluated. Completion requires held-out-dataset metrics, interpretation, and a Gate-G2 decision. |
| Synthetic S1–S6 | No generator/config/report yet | Forbidden to claim. Implement only after the Phase II audit and as required to resolve G2. |
| Mechanistic analysis | No internal-neighborhood/readout or causal-restoration result | TODO if descriptor/synthetic evidence alone does not establish G2, or as needed for a strong mechanism claim. |
| Gate G2 explanation | Open | Pass only with descriptor, representation/readout, or controlled-prior evidence; otherwise pivot and stop remedies. |
| Method ladder M1–M7 | No method-selection run; no `METHOD_FREEZE.md` | Correctly deferred until G2. If G2 passes, run cheapest baselines first and stop weak branches. |
| Pretraining/RSPF | Not implemented | Correctly deferred until a development-suite method survives. |
| Gate G3 remedy | Open | Requires large robustness reduction, unseen-transform transfer, and negligible clean loss. |
| Gate G4 current-model relevance | Not established broadly; Phase I currently supports TabICLv2 with Mitra heterogeneity | Requires two current strong TFMs or a new model that explains family differences. |
| Frozen confirmatory TabArena/BeyondArena benchmark | No method freeze or benchmark config | Correctly deferred. Must never tune on these outcomes. |
| Gate G5 benchmark relevance | Open | Requires competitive clean performance and robust frozen-benchmark results. |
| Theory target | Informal framing only | TODO if direction survives: finite-group prior symmetrization proposition plus explicit marginal-information tradeoff assumptions. |
| Required phase reports | Phase I report exists; Phase II–final reports absent | Each report must contain question, protocol, table/CIs, plots, alternatives, decision, and raw paths. |
| Final six reproducibility commands | Core audit command exists; synthetic/method/pretraining/final/artifact scripts absent | Not complete. Required only for surviving branches; a killed direction instead needs a rigorous negative-result/pivot handoff. |
| `ICLR_READINESS.md` and 12-point rubric | Absent | Final required artifact. It must recommend a pivot if evidence scores ≤7/12. |

## Current critical path

1. Finish and integrity-check the frozen Phase II grid.
2. Generate Phase II/III tables, figures, and group-held-out descriptor results.
3. Write `REPARAM_AUDIT.md`, update the claims matrix/log, and decide Gate G2.
4. If G2 fails, stop method/pretraining compute and write the strongest defensible pivot/readiness report. If G2 passes, proceed through synthetic controls and the method ladder without touching confirmatory data.

The program is not complete merely when Phase II terminates. Completion means either the surviving direction passes the remaining gates with all required artifacts, or a failed gate is documented honestly and the mandated expensive downstream branches are stopped.
