#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
base="experiments/final_closure"

"$python_bin" "$base/analyze_experiment_a.py"
"$python_bin" "$base/analyze_experiment_a_classical.py"
"$python_bin" "$base/analyze_experiment_b.py"
"$python_bin" "$base/analyze_experiment_c.py"
"$python_bin" "$base/analyze_experiment_d.py"
"$python_bin" "$base/make_final_claims.py"
"$python_bin" "$base/record_regeneration.py"
