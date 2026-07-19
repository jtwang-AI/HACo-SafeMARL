#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=code

SEEDS=(20260429 20260430 20260431 20260432)
GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/review_marl_baselines/mappo_acoustic_shield}"
OUT_ROOT="${OUT_ROOT:-results/review_same_actor_shield_replay}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_ROOT"

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS_ARR[$((index % ${#GPUS_ARR[@]}))]}"
  checkpoint="${CHECKPOINT_ROOT}/seed_${seed}/checkpoint.pt"
  out="${OUT_ROOT}/seed_${seed}"
  mkdir -p "$out"
  echo "launch same-actor shield replay seed ${seed} on GPU ${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" code/evaluate_checkpoint_shield_replay.py \
      --checkpoint "$checkpoint" \
      --seed "$seed" \
      --eval-episodes "$EVAL_EPISODES" \
      --out "$out"
  ) > "$out/eval.log" 2>&1 &
done

wait
echo "same-actor shield replay complete: ${OUT_ROOT}"
