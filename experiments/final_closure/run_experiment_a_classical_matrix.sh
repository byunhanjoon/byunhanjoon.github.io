#!/usr/bin/env bash
set -euo pipefail
shard="${1:-0}"
shards="${2:-1}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
datasets=(
  australian_credit_approval bank_marketing_subscription credit_card_default
  german_credit_risk heloc_credit_risk lendingclub_loan_default
  fremtpl_claim_count kdd17_stock_return openml-abalone-183
  openml-kin8nm-189 openml-pol-201 openml-puma32h-308
)
models=(catboost_native xgboost)
index=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if (( index % shards == shard )); then
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$python_bin" experiments/final_closure/run_experiment_a_classical.py \
        --dataset "$dataset" --model "$model"
    fi
    index=$((index + 1))
  done
done
