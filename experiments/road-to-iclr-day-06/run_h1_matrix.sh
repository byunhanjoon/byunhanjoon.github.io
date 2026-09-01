#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python}
PHASE=${PHASE:-pilot}
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
DEVICE=${DEVICE:-cuda:0}

if [[ "$PHASE" == "pilot" ]]; then
  seeds=(6101 6202)
elif [[ "$PHASE" == "confirmation" ]]; then
  seeds=(6303 6404 6505 6606 6707 6808)
elif [[ "$PHASE" == "all" ]]; then
  seeds=(6101 6202 6303 6404 6505 6606 6707 6808)
else
  echo "unknown PHASE=$PHASE" >&2
  exit 2
fi

datasets=(bank_marketing_subscription credit_card_default fremtpl_claim_count)
models=(mlp resnet ft_transformer)
job=0
for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      if (( job % SHARD_COUNT == SHARD_INDEX )); then
        "$PYTHON_BIN" semantic_arithmetic.py \
          --dataset "$dataset" --model "$model" --seed "$seed" --device "$DEVICE"
      fi
      job=$((job + 1))
    done
  done
done
