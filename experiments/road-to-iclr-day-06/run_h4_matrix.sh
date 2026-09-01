#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python}
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
DEVICE=${DEVICE:-cuda:0}
datasets=(bank_marketing_subscription credit_card_default fremtpl_claim_count)
models=(mlp resnet ft_transformer)
seeds=(9101 9202 9303)
learning_rates=(0.0003 0.001 0.003)
weight_decays=(0 0.0001)
batches=(128 256)

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      # Balanced parity assignment: for two shards, each physical GPU gets
      # exactly half of every batch, weight-decay, and learning-rate level
      # within each seed/dataset/model cell.
      for batch_index in "${!batches[@]}"; do
        batch=${batches[$batch_index]}
        for wd_index in "${!weight_decays[@]}"; do
          wd=${weight_decays[$wd_index]}
          for lr_index in "${!learning_rates[@]}"; do
            lr=${learning_rates[$lr_index]}
            assignment=$(( (batch_index + wd_index + lr_index) % SHARD_COUNT ))
            if (( assignment == SHARD_INDEX )); then
              config_id="lr${lr}__wd${wd}__b${batch}"
              "$PYTHON_BIN" semantic_shadow.py \
                --dataset "$dataset" --model "$model" --seed "$seed" \
                --config-id "$config_id" --device "$DEVICE"
            fi
          done
        done
      done
    done
  done
done
