#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/byunhanjoon/miniconda3/bin/python
RUNNER=/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-09/interaction_context_selection/selection_aware_successor/run_successor.py

run_group() {
  local gpu=$1
  shift
  for dataset in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" run --dataset "$dataset" --device cuda:0
  done
}

run_group 0 breast-w blood-transfusion kin8nm cpu-act &
pid_a=$!
run_group 1 credit-approval sick puma32h elevators &
pid_b=$!
wait "$pid_a" "$pid_b"

"$PYTHON" "$RUNNER" report
