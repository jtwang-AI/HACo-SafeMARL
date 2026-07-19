#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=code

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_OUT="${ROOT_OUT:-results/review_marl_baselines/mappo_no_gat_acoustic_shield}"
BC_EPOCHS="${BC_EPOCHS:-100}"
PPO_UPDATES="${PPO_UPDATES:-100}"
BC_EPISODES="${BC_EPISODES:-12}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SEEDS=(20260429 20260430 20260431 20260432)

mkdir -p "$ROOT_OUT" logs

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="$index"
  out_dir="$ROOT_OUT/seed_${seed}"
  mkdir -p "$out_dir"
  if [[ -s "$out_dir/aggregate_metrics.csv" ]]; then
    echo "skip completed no-GAT MAPPO seed ${seed}"
    continue
  fi
  echo "launch no-GAT MAPPO seed ${seed} on GPU ${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" code/train_torch_mappo.py \
      --warm-policy results/cem_local/best_policy.json \
      --seed "$seed" \
      --policy-name mappo_no_gat_acoustic_shield \
      --bc-episodes "$BC_EPISODES" \
      --bc-epochs "$BC_EPOCHS" \
      --ppo-updates "$PPO_UPDATES" \
      --rollout-episodes "$ROLLOUT_EPISODES" \
      --ppo-epochs "$PPO_EPOCHS" \
      --eval-episodes "$EVAL_EPISODES" \
      --no-graph-attention \
      --out "$out_dir"
  ) > "$out_dir/train.log" 2>&1 &
done

wait
"$PYTHON_BIN" code/summarize_remote_review_experiments.py \
  --root results/review_marl_baselines
echo "NO_GAT_BASELINE_COMPLETE"
