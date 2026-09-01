#!/usr/bin/env bash
set -euo pipefail
device="${1:-cuda:0}"
shard="${2:-0}"
shards="${3:-1}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
datasets=(bank_marketing_subscription heloc_credit_risk fremtpl_claim_count kdd17_stock_return)
models=(mlp resnet ft_transformer tabm)
splits=(2026082801 2026082811 2026082821)
index=0
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for model in "${models[@]}"; do
      if (( index % shards == shard )); then
        "$python_bin" experiments/final_closure/run_experiment_d.py \
          --dataset "$dataset" --split-seed "$split" --model "$model" --device "$device"
      fi
      index=$((index + 1))
    done
  done
done
