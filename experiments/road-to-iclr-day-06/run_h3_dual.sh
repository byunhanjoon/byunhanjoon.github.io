#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 SHARD_COUNT=2 DEVICE=cuda:0 ./run_h3_matrix.sh &
job0=$!
CUDA_VISIBLE_DEVICES=1 SHARD_INDEX=1 SHARD_COUNT=2 DEVICE=cuda:0 ./run_h3_matrix.sh &
job1=$!
wait "$job0" "$job1"
PYTHONPATH=. /home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  analyze_fullscale_arithmetic.py
