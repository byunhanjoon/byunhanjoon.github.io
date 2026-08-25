# Day 3 — Basis geometry experiments

The reader-facing synthesis is in `day3.md`. The concise experiment brief is
in `day3_agent.md`, and the exact research record is in `REPORT_DAY3.md`.
The study reuses the official TabPack arrays and splits from Day 1/2, freezes
the Day 2 MLP protocol, and stores every trained run as CSV before analysis.

Use the existing environment:

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m pytest -q
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite numeric
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite categorical
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite ordinal
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite whitening
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite regularizer
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite block
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_suite cyclic
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_residual_te
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.run_frequency_preconditioning
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_day3
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets adult --kappas 1 3000 --seeds 0 1 --models mlp --output results/day3/optimizer_remedies/screen_adult.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets diamond --kappas 1 3000 --seeds 0 1 --models mlp --output results/day3/optimizer_remedies/screen_diamond.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets adult --kappas 1 3000 --seeds 0 1 2 3 4 --models mlp resnet --remedies adamw whiten_sgd anchor_whiten_adamw anchor_whiten_sgd natural_hybrid_invariant_init natural_hybrid_invariant_init_lr01 --output results/day3/optimizer_remedies/confirm_adult.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets diamond --kappas 1 3000 --seeds 0 1 2 3 4 --models mlp resnet --remedies adamw whiten_sgd anchor_whiten_adamw anchor_whiten_sgd natural_hybrid_invariant_init natural_hybrid_invariant_init_lr01 --output results/day3/optimizer_remedies/confirm_diamond.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets california --kappas 1 3000 --seeds 0 1 2 3 4 --models mlp --remedies adamw whiten_sgd anchor_whiten_adamw anchor_whiten_sgd natural_hybrid_invariant_init natural_hybrid_invariant_init_lr01 --output results/day3/optimizer_remedies/confirm_california_mlp.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.optimizer_remedies --datasets california --kappas 1 3000 --seeds 0 1 2 3 4 --models resnet --remedies adamw whiten_sgd anchor_whiten_adamw anchor_whiten_sgd natural_hybrid_invariant_init natural_hybrid_invariant_init_lr01 --output results/day3/optimizer_remedies/confirm_california_resnet.csv
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_optimizer_remedies
```

The immutable hypotheses/config are in
`experiments/day3/configs/preregistered.json`. Raw results and figures are in
`results/day3/`; the research verdict is in `REPORT_DAY3.md`. The optimizer
follow-up is documented in `OPTIMIZER_REMEDIES_REPORT.md`, with raw artifacts in
`results/day3/optimizer_remedies/`.

## Broad benchmark and final audits

The final Day 3 work extends the original study with a frozen 25-dataset screen,
a separately frozen five-dataset replication, four architectures, five-seed
remedy confirmation, rank/ridge stress tests, and a same-table
distribution-shift comparison. The result and the honest novelty boundary are
in `BROAD_BENCHMARK_REPORT.md`.

The main reproduction/analysis entry points are:

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_broad_final
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_broad_extension
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_distribution_shift
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_broad_completion
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_extension_completion
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_distribution_shift
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.verify_day3_final
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_day3_goal
```

## Function-matched trajectory extension

The post-benchmark Day 3 extension decomposes basis sensitivity into initial
function mismatch and optimizer-induced trajectory drift. Its outcome-blind
protocol and final report are in `TRAJECTORY_DECOMPOSITION_PROTOCOL.md` and
`TRAJECTORY_DECOMPOSITION_REPORT.md`.

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.trajectory_decomposition --device cuda:0
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_trajectory_decomposition
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.audit_trajectory_decomposition
```

Important artifacts are under `results/day3/broad_benchmark/`:

- `combined_30_summary.json`: combined controlled result;
- `final_summary.json`: main and confirmation summary;
- `completion_audit.json`: cell coverage, failures, freeze, and invariants;
- `extension_completion_audit.json`: five-dataset replication audit;
- `distribution_shift_summary.json`: chronological versus random split;
- `canonicalization_numerical_audit.json`: exact/sketch floating-point audit;
- `final_verification.json` and `day3_goal_completion_audit.json`: fail-closed
  completion proof.

## Equivalent-basis orbit ensemble continuation

The latest Day 3 continuation uses cumulative/local exact-basis charts as a
structured diversity axis inside TabM. The frozen protocol is in
`ORBIT_ENSEMBLE_PROTOCOL.md`; the completed 297-run result and honest
prior-art boundary are in `ORBIT_ENSEMBLE_REPORT.md`.

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m pytest -q tests/test_orbit_ensemble.py
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.orbit_ensemble --stage screen --shard 0 --num-shards 2 --device cuda:0
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.orbit_ensemble --stage confirmation --shard 0 --num-shards 2 --device cuda:0
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python -m experiments.day3.analyze_orbit_ensemble
```

Orbit-TabM passed its frozen confirmation gate: +0.767% mean proper-loss
reduction over 21 confirmation datasets, with a dataset-cluster interval of
[+0.015%, +1.886%], 15/21 dataset wins, identical trainable parameter counts,
and no failures. The result is promising but primarily a proper-loss/calibration
gain rather than an Adult-identity-sized accuracy improvement.

## Mixed-measure Measure-Orbit continuation

Selective Measure-Orbit produced an Adult-sized internal result, but its broad
performance claim did **not** survive the required untouched test.
`MEASURE_ORBIT_PROTOCOL.md` records the hypothesis sequence and freezes;
`MEASURE_ORBIT_REPORT.md` and `EXTERNAL_MEASURE_ORBIT_REPORT.md` give the
internal and external results.

The development screen improved Adult accuracy by +1.013 percentage points.
The frozen 21-dataset internal confirmation reduced proper loss by +1.103% on
average. However, on seven untouched external datasets, Measure-Orbit was
0.521% worse than an exactly update-matched two-seed ordinary TabM ensemble;
the 95% dataset-bootstrap interval was entirely negative
([−0.831%, −0.195%]). It should therefore be treated as a diagnostic, not a
validated performance method.
