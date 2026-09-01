# Interaction-Aware Context Selection Kill Experiment

This directory is an isolated implementation of
`../AGENT.md — Kill Experiment 1_ Interaction-Aware Context Selection for Tabular Foundation Models.md`.
It does not reuse or modify the separate Day 9 coordinate–marginal program.

The experiment is intentionally resumable. Raw context evaluations are written
after every batch, and analysis never uses final-test labels for fitting or
selection.

```bash
PY=/home/byunhanjoon/miniconda3/bin/python
$PY experiments/run_pipeline.py evaluate --dataset adult --device cuda:0
$PY experiments/run_pipeline.py analyze --dataset adult --device cuda:0
$PY experiments/run_pipeline.py diagnostic --dataset credit-g --device cuda:0
$PY experiments/run_pipeline.py report
```

The final deliverables are `results.md`, tidy files under `results/`, and plots
under `plots/`.
