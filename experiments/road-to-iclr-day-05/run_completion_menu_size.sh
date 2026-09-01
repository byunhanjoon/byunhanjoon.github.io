#!/usr/bin/env bash
set -u

gpu="${1:?gpu index required}"
shard="${2:?shard index required}"
shards="${3:-2}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
output_dir="results/completion_menu_size"
log_dir="results/completion_menu_size_logs"
mkdir -p "$output_dir" "$log_dir"

datasets=(australian_credit_approval fremtpl_claim_count)
models=(mlp resnet ft_transformer tabm)
task_index=0
failures=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    assigned=$((task_index % shards))
    task_index=$((task_index + 1))
    stem="${dataset}__${model}"
    if [[ -f "${output_dir}/${stem}.json" ]] || (( assigned != shard )); then
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$python_bin" completion_menu_size.py --dataset "$dataset" --model "$model" \
      --device cuda:0 >"${log_dir}/${stem}.log" 2>&1
    status=$?
    if (( status != 0 )); then
      failures=$((failures + 1))
      printf '%s\t%s\t%s\n' "$dataset" "$model" "$status" >>"${log_dir}/failures_shard${shard}.tsv"
    fi
  done
done
printf 'menu-size shard=%s failures=%s tasks_seen=%s\n' "$shard" "$failures" "$task_index"
exit "$failures"
