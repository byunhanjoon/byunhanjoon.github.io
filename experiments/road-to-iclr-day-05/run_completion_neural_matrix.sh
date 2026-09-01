#!/usr/bin/env bash
set -u

gpu="${1:?gpu index required}"
shard="${2:?shard index required}"
shards="${3:-2}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
output_dir="results/completion_neural"
log_dir="results/completion_neural_logs"
mkdir -p "$output_dir" "$log_dir"

datasets=(
  australian_credit_approval bank_marketing_subscription credit_card_default
  german_credit_risk heloc_credit_risk lendingclub_loan_default
  fremtpl_claim_count kdd17_stock_return openml-abalone-183 openml-kin8nm-189
  openml-pol-201 openml-puma32h-308
)
exact_datasets=(
  australian_credit_approval bank_marketing_subscription
  fremtpl_claim_count kdd17_stock_return
)
models=(mlp resnet ft_transformer tabm)
splits=(2026082801 2026082811 2026082821)
task_index=0
failures=0

run_task() {
  local dataset="$1" model="$2" mode="$3" split="${4:-}"
  local stem log_path output_path
  if [[ "$mode" == "matched" ]]; then
    stem="${dataset}__${model}__matched"
  else
    stem="${dataset}__${model}__split${split}__${mode}"
  fi
  output_path="${output_dir}/${stem}.json"
  log_path="${log_dir}/${stem}.log"
  assigned=$((task_index % shards))
  task_index=$((task_index + 1))
  if [[ -f "$output_path" ]]; then
    return
  fi
  if (( assigned != shard )); then
    return
  fi
  if [[ "$mode" == "matched" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$python_bin" completion_neural_panel.py --dataset "$dataset" --model "$model" \
      --mode matched --device cuda:0 >"$log_path" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$python_bin" completion_neural_panel.py --dataset "$dataset" --model "$model" \
      --mode "$mode" --split-seed "$split" --device cuda:0 >"$log_path" 2>&1
  fi
  status=$?
  if (( status != 0 )); then
    failures=$((failures + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$model" "$mode" "$split" "$status" \
      >> "${log_dir}/failures_shard${shard}.tsv"
  fi
}

for dataset in "${exact_datasets[@]}"; do
  for model in "${models[@]}"; do
    run_task "$dataset" "$model" matched
  done
done

for dataset in "${exact_datasets[@]}"; do
  for model in "${models[@]}"; do
    run_task "$dataset" "$model" exact "${splits[0]}"
  done
done

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for split in "${splits[@]}"; do
      run_task "$dataset" "$model" broad "$split"
    done
  done
done

printf 'shard=%s failures=%s tasks_seen=%s\n' "$shard" "$failures" "$task_index"
exit "$failures"
