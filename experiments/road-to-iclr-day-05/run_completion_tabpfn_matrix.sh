#!/usr/bin/env bash
set -u

gpu="${1:?gpu index required}"
shard="${2:?shard required}"
shards="${3:-2}"
python_bin="/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
mkdir -p results/completion_tabpfn results/completion_tabpfn_logs
datasets=(
  australian_credit_approval bank_marketing_subscription credit_card_default
  german_credit_risk heloc_credit_risk lendingclub_loan_default
)
splits=(2026082801 2026082811 2026082821)
index=0
failures=0
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    stem="${dataset}__split${split}"
    if [[ -f "results/completion_tabpfn/${stem}.json" ]]; then
      index=$((index + 1)); continue
    fi
    if (( index % shards != shard )); then
      index=$((index + 1)); continue
    fi
    index=$((index + 1))
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$python_bin" completion_tabpfn.py --dataset "$dataset" --split-seed "$split" \
      --device cuda:0 >"results/completion_tabpfn_logs/${stem}.log" 2>&1
    status=$?
    if (( status != 0 )); then
      failures=$((failures + 1))
      printf '%s\t%s\t%s\n' "$dataset" "$split" "$status" >> "results/completion_tabpfn_logs/failures_shard${shard}.tsv"
    fi
  done
done
printf 'tabpfn shard=%s failures=%s\n' "$shard" "$failures"
exit "$failures"
