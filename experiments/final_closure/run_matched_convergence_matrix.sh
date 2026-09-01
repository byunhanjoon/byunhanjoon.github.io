#!/usr/bin/env bash
set -euo pipefail
device="${1:-cuda:0}"
shard="${2:-0}"
shards="${3:-1}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
datasets=(bank_marketing_subscription fremtpl_claim_count)
models=(mlp resnet ft_transformer tabm)
index=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if (( index % shards == shard )); then
      "$python_bin" experiments/final_closure/run_matched_convergence.py \
        --dataset "$dataset" --model "$model" --device "$device"
    fi
    index=$((index + 1))
  done
done
