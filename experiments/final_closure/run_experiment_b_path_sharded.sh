#!/usr/bin/env bash
set -uo pipefail

python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
path_shards="${1:-3}"
log_dir="experiments/final_closure/logs/b_path_shards"
mkdir -p "$log_dir"

datasets=(
  bank_marketing_subscription credit_card_default heloc_credit_risk
  fremtpl_claim_count kdd17_stock_return openml-abalone-183
)
models=(mlp resnet ft_transformer tabm)
pids=()
names=()

for dataset_index in "${!datasets[@]}"; do
  dataset="${datasets[$dataset_index]}"
  gpu=$((dataset_index % 2))
  for model in "${models[@]}"; do
    for ((path_shard = 0; path_shard < path_shards; path_shard++)); do
      name="${dataset}__${model}__shard${path_shard}of${path_shards}"
      env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_VISIBLE_DEVICES="$gpu" \
        "$python_bin" experiments/final_closure/run_experiment_b_bundle.py \
        --dataset "$dataset" --model "$model" --device cuda:0 \
        --path-shard "$path_shard" --path-shards "$path_shards" \
        >"$log_dir/$name.log" 2>&1 &
      pids+=("$!")
      names+=("$name")
    done
  done
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'FAILED %s (see %s/%s.log)\n' \
      "${names[$index]}" "$log_dir" "${names[$index]}" >&2
    status=1
  fi
done
exit "$status"
