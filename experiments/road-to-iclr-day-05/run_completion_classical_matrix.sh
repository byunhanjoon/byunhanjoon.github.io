#!/usr/bin/env bash
set -u

python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
output_dir="results/completion_classical"
log_dir="results/completion_classical_logs"
mkdir -p "$output_dir" "$log_dir"

datasets=(
  australian_credit_approval bank_marketing_subscription credit_card_default
  german_credit_risk heloc_credit_risk lendingclub_loan_default
  fremtpl_claim_count kdd17_stock_return openml-abalone-183 openml-kin8nm-189
  openml-pol-201 openml-puma32h-308
)
models=(onehot_linear native_histgb catboost_native xgboost lightgbm)

run_one() {
  dataset="$1"; model="$2"
  stem="${dataset}__${model}"
  if [[ -f "results/completion_classical/${stem}.json" ]]; then
    exit 0
  fi
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
    completion_classical_panel.py --dataset "$dataset" --model "$model" \
    >"results/completion_classical_logs/${stem}.log" 2>&1
}
export -f run_one

tasks=()
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    tasks+=("$dataset $model")
  done
done
printf '%s\n' "${tasks[@]}" | xargs -P 8 -n 2 bash -c 'run_one "$0" "$1"'
