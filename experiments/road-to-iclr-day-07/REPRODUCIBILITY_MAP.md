# OrbitCover reproducibility map

This map points from each paper artifact to the frozen, audited source of
truth. Paths are relative to `experiments/road-to-iclr-day-07/`.

The compilable anonymous submission source is `paper/main.tex`; its official
ICLR 2027 style assets and build notes live in `paper/`.

## Frozen specification

| Artifact | Path |
| --- | --- |
| Prospective closure protocol | `../final_closure/FINAL_CLOSURE_PROTOCOL.md` |
| Frozen configuration | `../final_closure/final_closure_config.json` |
| Frozen hashes | `../final_closure/PROTOCOL_HASH.txt` |
| Recorded deviations | `../final_closure/PROTOCOL_DEVIATIONS.md` |
| Audit report | `../final_closure/FINAL_AUDIT.md` |
| Machine-readable audit | `../final_closure/final_audit_summary.json` |

## Authoritative results

| Paper section | Source |
| --- | --- |
| Independent-seed showdown and target distance | `../final_closure/summaries/experiment_a_summary.json`, `experiment_a_cells.csv` |
| Classical-model extension | `../final_closure/summaries/experiment_a_classical_summary.json` |
| Training scale, convergence, matched functions | `../final_closure/summaries/experiment_b_summary.json` |
| Interaction spectrum and complete failures | `../final_closure/summaries/experiment_c_summary.json`, `experiment_c_cells.csv` |
| Coupling mechanism | `../final_closure/summaries/experiment_d_summary.json`, `experiment_d_cells.csv` |
| Final selected claims | `../final_closure/summaries/final_claims_summary.json` |
| Human-readable complete report | `../final_closure/results.md` |

## Regeneration order

From `experiments/final_closure/`, using the environment recorded in its
README:

```bash
$PY analyze_experiment_a.py
$PY analyze_experiment_a_classical.py
$PY analyze_experiment_b.py
$PY analyze_experiment_c.py
$PY analyze_experiment_d.py
$PY make_final_claims.py
$PY audit_final_closure.py
$PY make_final_results.py
```

The audit checks mandatory manifests, prediction finiteness and alignment,
unique master seeds, registry completeness, frozen hashes, figures, tables,
and tests. The last audited run reports 140,592 complete unique fit keys and
116/116 passing tests.

## Main figures

1. `../final_closure/figures/figure_1_independent_seed_showdown.pdf`
2. `paper/figures/target_shift_summary.pdf` (from
   `make_submission_figures.py` and the authoritative Experiment-A reference
   table)
3. `../final_closure/figures/figure_10_coupling_mechanism.pdf`
4. `../final_closure/figures/figure_6_orbitcover_convergence.pdf`
5. `../final_closure/figures/figure_9_matched_convergence.pdf`

Appendix figures are the remaining files in `../final_closure/figures/`.
Every concept is stored as both PNG for Markdown inspection and PDF for the
submission source.

## Day-7 auxiliary evidence

The learned optional-structure PFN and the nested nuisance-value certificates
are exploratory extensions, not evidence for OrbitCover's headline claim.
Their frozen protocols and outcomes are:

- `LEARNED_PFN_PROTOCOL.md` and `LEARNED_PFN_RESULTS.md`;
- `NESTED_BACKBONE_PROTOCOL.md` and `BACKBONE_RESULTS.md`;
- `LITERATURE_AUDIT.md` for the novelty collision that removed the PFN from
  the paper lead.

Keeping these artifacts separate prevents post-hoc portfolio experiments from
silently changing the evidence grade of the frozen OrbitCover closure.
