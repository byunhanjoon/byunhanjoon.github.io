#!/usr/bin/env bash
set -uo pipefail

shards="${1:-8}"
log_dir="experiments/final_closure/logs/d_shards"
mkdir -p "$log_dir"
pids=()

for ((shard = 0; shard < shards; shard++)); do
  gpu=$((shard * 2 / shards))
  env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_VISIBLE_DEVICES="$gpu" \
    bash experiments/final_closure/run_experiment_d_matrix.sh cuda:0 "$shard" "$shards" \
    >"$log_dir/shard${shard}of${shards}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'FAILED D shard %s (see %s/shard%sof%s.log)\n' \
      "$index" "$log_dir" "$index" "$shards" >&2
    status=1
  fi
done
exit "$status"
