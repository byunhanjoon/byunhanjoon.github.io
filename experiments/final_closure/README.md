# OrbitCover final closure

This directory is the frozen, restartable implementation of
`../road-to-iclr-day-05/AGENT.md — FINAL ICLR CLOSURE EXPERIMENTS.md`.

The protocol and config hashes are in `PROTOCOL_HASH.txt`.  Do not edit either
frozen file; record implementation corrections in `PROTOCOL_DEVIATIONS.md`.

## Execution

Use the TabM environment:

```bash
PY=/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python
```

Matrix scripts accept restartable shard indices.  For example, eight A shards
can be distributed four per H100:

```bash
CUDA_VISIBLE_DEVICES=0 run_experiment_a_matrix.sh cuda:0 0 8
CUDA_VISIBLE_DEVICES=0 run_experiment_a_matrix.sh cuda:0 1 8
CUDA_VISIBLE_DEVICES=1 run_experiment_a_matrix.sh cuda:0 4 8
```

Run the analogous `run_experiment_b_matrix.sh`,
`run_matched_convergence_matrix.sh`, and `run_experiment_d_matrix.sh` until
their frozen matrices close.  CatBoost/XGBoost use
`run_experiment_a_classical_matrix.sh` on CPU shards.

Then regenerate in this order:

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

The last command is intentionally after the read-only audit.  It writes the
same standalone report to this directory, the Day-5 directory, and repository
root.

## Persistence

Every prediction tensor has a completion mask.  Every physical fit is keyed in
`fit_registry.sqlite3` by dataset, split, model/config hash, schema hash, RNG,
training size/budget, and matched arm.  An interrupted matrix command validates
and resumes completed rows.  Estimator resampling never counts overlapping
cached draws as independent scientific units.
