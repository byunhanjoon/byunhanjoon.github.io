# Safe Basis Control — reproducibility guide

This directory is the self-contained implementation and artifact root for the
tail-robust gating, rank-adaptive Gram, and embedding-integration round.

The experimental boundary is deliberate:

1. `configs/NEW_TAIL_PROSPECTIVE_PANEL.json` was selected from OpenML metadata
   and locked before development outcomes were used for method selection.
2. Development reused frozen Raw/Gram predictions where available, then ran
   rank, failure-diagnosis, and numerical-embedding experiments.
3. `configs/TAIL_FINALISTS.json` was written and SHA256-locked before the new
   prospective runner was allowed to load its datasets.
4. The prospective loader rejects missing, mismatched, oversized, or
   non-frozen finalist configurations.

## Environment

- Python: `/home/byunhanjoon/miniconda3/bin/python`
- GPUs: two NVIDIA H100 NVL devices
- Run from this directory with `PYTHONPATH=.`

## Reproduce in protocol order

```bash
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python -m pytest
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_development_gates.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_rank_development.py --stage screen --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_rank_development.py --stage full --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_failure_diagnosis.py --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_descriptor_gate.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_embedding_integration.py --stage main --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_embedding_integration.py --stage dimension --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/freeze_finalists.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_prospective.py --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/analyze_prospective.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/run_natural_bases.py --device cuda:0
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/make_figures.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/generate_report.py
PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python scripts/audit_completion.py
```

Every expensive prediction bundle is content-guarded by the frozen protocol,
panel, and—after the freeze boundary—finalist hashes. Re-running a command
uses the matching cache and refuses drift rather than silently overwriting it.

## Artifact contract

- `results.md`: exact prescribed report outline and scientific verdict
- `results/raw/`: compressed predictions plus JSON provenance/telemetry
- `results/processed/`: cell tables, primary units, rankings, diagnostics,
  manifests, and completion audit
- `figures/`: Figures 1–8 as PNG and PDF
- `configs/`: panel, protocol, finalist locks, and SHA256 sidecars

The primary statistical unit is dataset × model. Seeds are aggregated within
that unit before medians and tail quantiles are computed. Denominator-sensitive
cells remain in every primary result and are additionally reported in a
secondary exclusion analysis.
