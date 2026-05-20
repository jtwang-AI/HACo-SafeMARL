#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-results/torch_mappo}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_DIR"

export PYTHONPATH=code
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python code/train_torch_mappo.py \
  --warm-policy results/cem_local/best_policy.json \
  --bc-episodes "${BC_EPISODES:-12}" \
  --bc-epochs "${BC_EPOCHS:-100}" \
  --ppo-updates "${PPO_UPDATES:-100}" \
  --rollout-episodes "${ROLLOUT_EPISODES:-8}" \
  --ppo-epochs "${PPO_EPOCHS:-4}" \
  --eval-episodes "${EVAL_EPISODES:-50}" \
  --out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"

python code/make_paper_artifacts.py
