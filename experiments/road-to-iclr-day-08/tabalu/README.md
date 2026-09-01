# TabALU

This directory is the isolated implementation of the research program in
`AGENT.md — TabALU_ Learn What to Compute, Execute It Exactly.md`.

Current status: **the staged study is complete and the proposed full
architecture is a NO-GO**. Exact execution survives; deep discovery, learned
operand correction, real temporal routing, context-conditioned coefficients,
and the unguarded residual do not. See `FINAL_DECISION.md` for the component
decision and cleared claim.

Run tests:

```bash
/home/byunhanjoon/miniconda3/bin/python -m pytest tabalu/tests -q
```

Run the cheap smoke panel:

```bash
/home/byunhanjoon/miniconda3/bin/python tabalu/scripts/run_phase_a.py \
  --config tabalu/configs/smoke.json
```

Run the frozen 100-task, five-seed pilot:

```bash
bash tabalu/scripts/reproduce_main_tables.sh
```

Each run stores its frozen config, generated task programs, selector histories,
compiled programs, failures, per-cell metrics, bootstrap summaries, gate audit,
and extrapolation figure under `tabalu/results/`.

The pilot gate is fixed before observing results:

- compiled TabALU extreme-OOD NRMSE at most 50% of MLP NRMSE;
- soft TabALU IID NRMSE at most 125% of MLP NRMSE;
- compiled extreme-OOD NRMSE at most 125% of soft TabALU NRMSE.

Failure means debugging or redesigning program induction before proceeding to
noise, regimes, temporal data, or broad real-data benchmarks.

The warm start exhaustively searches the short chain family used in Phase A.
This makes the initial test a clean exact-execution falsification, but it does
not establish general differentiable program induction. Randomly initialized
straight-through selection failed the first smoke diagnostic and is preserved
in `NEGATIVE_RESULTS.md`.

Run the direct execution intervention:

```bash
/home/byunhanjoon/miniconda3/bin/python \
  tabalu/scripts/run_exact_execution_ablation.py \
  --config tabalu/configs/exact_execution_ablation.json
```

See `PHASE_A_RESULTS.md` through `PHASE_G_RESULTS.md` and
`EXACT_EXECUTION_RESULTS.md`, `REAL_TEMPORAL_RESULTS.md`,
`GENERAL_PILOT_RESULTS.md`, and `SCALING_RESULTS.md` for scoped decisions and
limitations.

Audit all 13 main runs while keeping data integrity separate from scientific
gate failures:

```bash
/home/byunhanjoon/miniconda3/bin/python tabalu/scripts/audit_study.py
```

The audit covers 27,610 finite metric records. Install the recorded environment
with `requirements.txt`; `scripts/reproduce_study.sh` lists the full execution
order. The UCI archive is downloaded on demand and checksum-verified.
