#!/usr/bin/env bash
set -euo pipefail

TASK_PY=/home/byunhanjoon/miniconda3/bin/python
TASK_ROOT=/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-09/interaction_context_selection
TASK_RUNNER="$TASK_ROOT/experiments/run_pipeline.py"

run_group() {
  local phase=$1
  local visible_gpu=$2
  shift 2
  for dataset in "$@"; do
    CUDA_VISIBLE_DEVICES="$visible_gpu" "$TASK_PY" "$TASK_RUNNER" "$phase" --dataset "$dataset" --device cuda:0
  done
}

run_group evaluate 0 adult credit-g electricity churn &
pid_a=$!
run_group evaluate 1 bank-marketing california_housing diamonds house_16H &
pid_b=$!
wait "$pid_a" "$pid_b"

run_group analyze 0 adult credit-g electricity churn &
pid_a=$!
run_group analyze 1 bank-marketing california_housing diamonds house_16H &
pid_b=$!
wait "$pid_a" "$pid_b"

run_group diagnostic 0 credit-g &
pid_a=$!
run_group diagnostic 1 diamonds &
pid_b=$!
wait "$pid_a" "$pid_b"

"$TASK_PY" "$TASK_RUNNER" report
