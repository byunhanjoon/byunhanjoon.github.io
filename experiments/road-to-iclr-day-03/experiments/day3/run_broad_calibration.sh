#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?dataset}
device=${2:?device}
output=${3:?output csv}
python_bin=/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python
module=experiments.day3.run_broad_benchmark
common=(--datasets "$dataset" --representations controlled --kappas 1 1000 --models mlp --seeds 31415 --device "$device" --output "$output")

for learning_rate in 0.0003 0.001 0.003; do
  "$python_bin" -m "$module" "${common[@]}" --remedies adamw diagonal_adamw whiten_adamw anchor_whiten_adamw --learning-rate "$learning_rate"
done

for learning_rate in 0.003 0.01 0.03; do
  "$python_bin" -m "$module" "${common[@]}" --remedies input_natural --learning-rate "$learning_rate"
done

for learning_rate in 0.003 0.01 0.03; do
  for ridge in 0.0001 0.001 0.01; do
    "$python_bin" -m "$module" "${common[@]}" --remedies first_layer_kfac --learning-rate "$learning_rate" --ridge "$ridge"
  done
done

for remedy in shampoo soap; do
  for learning_rate in 0.003 0.01 0.03; do
    for frequency in 1 10; do
      "$python_bin" -m "$module" "${common[@]}" --remedies "$remedy" --learning-rate "$learning_rate" --precondition-frequency "$frequency"
    done
  done
done
