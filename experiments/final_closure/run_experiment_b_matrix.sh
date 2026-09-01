#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
shard="${2:-0}"
shards="${3:-1}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
index=0
datasets=(
  bank_marketing_subscription credit_card_default heloc_credit_risk
  fremtpl_claim_count kdd17_stock_return openml-abalone-183
)
models=(mlp resnet ft_transformer tabm)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if (( index % shards == shard )); then
      "$python_bin" experiments/final_closure/run_experiment_b_bundle.py \
        --dataset "$dataset" --model "$model" --device "$device"
    fi
    index=$((index + 1))
  done
done
