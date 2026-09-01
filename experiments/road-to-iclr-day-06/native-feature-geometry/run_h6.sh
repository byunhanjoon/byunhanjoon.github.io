#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
domains=(cycle16 ordinal16 tree16 nominal16)
seeds=(7301 7302 7303)
index=0
for domain in "${domains[@]}"; do
  for seed in "${seeds[@]}"; do
    device=$((index % 2))
    "$PYTHON" transport_dose.py --domain "$domain" --seed "$seed" --device "cuda:${device}" &
    index=$((index + 1))
    if (( index % 2 == 0 )); then wait; fi
  done
done
wait
"$PYTHON" analyze_transport_dose.py

