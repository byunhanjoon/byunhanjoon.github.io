#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/byunhanjoon/miniconda3/bin/python
RUNNER=/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-09/interaction_context_selection/prospective_contextual_search/posthoc_round_audit.py

run_group() {
  local gpu=$1
  shift
  for dataset in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" run --dataset "$dataset" --device cuda:0
  done
}

run_group 0 adult credit-g electricity churn &
pid_a=$!
run_group 1 bank-marketing california_housing diamonds house_16H &
pid_b=$!
wait "$pid_a" "$pid_b"

"$PYTHON" "$RUNNER" report
