#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python}
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
DEVICE=${DEVICE:-cuda:0}
datasets=(bank_marketing_subscription credit_card_default fremtpl_claim_count)
models=(mlp resnet ft_transformer)
seeds=(8101 8202 8303 8404)

job=0
for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      if (( job % SHARD_COUNT == SHARD_INDEX )); then
        "$PYTHON_BIN" fullscale_arithmetic.py \
          --dataset "$dataset" --model "$model" --seed "$seed" --device "$DEVICE"
      fi
      job=$((job + 1))
    done
  done
done
