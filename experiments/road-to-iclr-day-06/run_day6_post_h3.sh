#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python}

echo "post-H3 chain waiting at $(date --iso-8601=seconds)"
while ! "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("results/h3_summary.json")
raise SystemExit(
    0 if path.exists() and json.loads(path.read_text()).get("status") == "complete" else 1
)
PY
do
  sleep 60
done

echo "H3 complete; starting audits at $(date --iso-8601=seconds)"
PYTHONPATH=. "$PYTHON_BIN" analyze_fullscale_arithmetic.py
PYTHONPATH=. "$PYTHON_BIN" analyze_fullscale_dynamics.py
PYTHONPATH=. "$PYTHON_BIN" analyze_semantic_lyapunov.py
PYTHONPATH=. "$PYTHON_BIN" analyze_rounding_survival.py
PYTHONPATH=. "$PYTHON_BIN" analyze_semantic_acceleration.py
PYTHONPATH=. "$PYTHON_BIN" analyze_postbreach_attenuation.py
PYTHONPATH=. "$PYTHON_BIN" make_day6_figures.py
"$PYTHON_BIN" audit_day6_integrity.py

echo "starting H4 at $(date --iso-8601=seconds)"
./run_h4_dual.sh

echo "H4 complete; starting H5 and final checks at $(date --iso-8601=seconds)"
PYTHONPATH=. "$PYTHON_BIN" analyze_semantic_shadow.py
PYTHONPATH=. "$PYTHON_BIN" analyze_cross_perturbation.py
PYTHONPATH=. "$PYTHON_BIN" analyze_semantic_acceleration.py
PYTHONPATH=. "$PYTHON_BIN" make_day6_figures.py
PYTHONPATH=. "$PYTHON_BIN" audit_analysis_reproducibility.py
"$PYTHON_BIN" audit_day6_integrity.py
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q
echo "post-H3 chain complete at $(date --iso-8601=seconds)"
