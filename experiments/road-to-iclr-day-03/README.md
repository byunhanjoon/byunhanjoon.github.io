# Day 3 — Basis geometry experiments

This directory implements the mechanism-first Day 3 study in `day3.md`.
It reuses the official TabPack arrays and splits from Day 1/2, freezes the Day
2 MLP protocol, and stores every trained run as CSV before analysis.

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

Important artifacts are under `results/day3/broad_benchmark/`:

- `combined_30_summary.json`: combined controlled result;
- `final_summary.json`: main and confirmation summary;
- `completion_audit.json`: cell coverage, failures, freeze, and invariants;
- `extension_completion_audit.json`: five-dataset replication audit;
- `distribution_shift_summary.json`: chronological versus random split;
- `canonicalization_numerical_audit.json`: exact/sketch floating-point audit;
- `final_verification.json` and `day3_goal_completion_audit.json`: fail-closed
  completion proof.
