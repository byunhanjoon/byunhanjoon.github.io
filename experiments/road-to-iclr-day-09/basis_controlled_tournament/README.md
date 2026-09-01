# Basis-Controlled Tabular Learning — Method Tournament

This directory implements the staged tournament specified by
`../AGENT.md — Basis-Controlled Tabular Learning_ Method Tournament.md`.

The prospective panel and protocol are locked before Stage 1. Development and
prospective runners are separate entry points; the prospective runner refuses to
load outcomes until finalist configurations have been frozen and hashed.

Generated artifacts are written under `results/raw`, `results/processed`, and
`figures`. The final scientific report is `results.md`.

The implementation reuses the frozen Day-9 confirmation code for dataset
splits, RBF blocks, orbit generation, natural basis pairs, metrics, and the five
model adapters. Tournament-specific code is confined to `tournament/` and
`scripts/`. All commands use the research environment at
`/home/byunhanjoon/miniconda3/bin/python` with `PYTHONPATH=.`.

The enforced execution order is:

1. `run_stage1.py`, then `analyze_stage1.py`.
2. Stage-2 optimizer and representation runners, anchor ablations, the matched
   optimizer audit, and `run_equal_hpo.py`.
3. `analyze_stage2.py` and `analyze_auxiliary.py`.
4. `freeze_finalists.py`, which refuses incomplete development evidence and
   writes `FINALIST_CONFIGS.json` plus its SHA without overwriting an existing
   freeze.
5. Natural-basis validation, `run_prospective.py`, and the separate
   `run_condition_exploratory.py`.
6. The corresponding analyzers, `make_figures.py`, `generate_report.py`, tests,
   and finally `audit_completion.py`.

`run_prospective.py` checks the finalist status, count, prospective-panel hash,
and finalist-file SHA before it resolves a prospective dataset. Cached raw
bundles also embed all frozen hashes and are rejected on drift.
