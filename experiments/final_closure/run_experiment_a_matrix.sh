#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
shard="${2:-0}"
shards="${3:-1}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
index=0

datasets=(
  australian_credit_approval bank_marketing_subscription credit_card_default
  german_credit_risk heloc_credit_risk lendingclub_loan_default
  fremtpl_claim_count kdd17_stock_return openml-abalone-183
  openml-kin8nm-189 openml-pol-201 openml-puma32h-308
)
models=(mlp resnet ft_transformer tabm)
splits=(2026082801 2026082811 2026082821)

for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for model in "${models[@]}"; do
      if (( index % shards == shard )); then
        "$python_bin" experiments/final_closure/run_experiment_a.py \
          --dataset "$dataset" --split-seed "$split" --model "$model" --device "$device"
      fi
      index=$((index + 1))
    done
  done
done
