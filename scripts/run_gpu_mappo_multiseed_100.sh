#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(20260429 20260430 20260431 20260432)
fi
ROOT_OUT="${ROOT_OUT:-results/mappo_multiseed_100}"

mkdir -p "$ROOT_OUT"
export PYTHONPATH=code

for seed in "${SEEDS[@]}"; do
  echo "=== MAPPO 100-run seed ${seed} ==="
  out_dir="${ROOT_OUT}/seed_${seed}"
  mkdir -p "$out_dir"
  bc_epochs="${BC_EPOCHS:-100}"
  ppo_updates="${PPO_UPDATES:-100}"
  bc_episodes="${BC_EPISODES:-12}"
  rollout_episodes="${ROLLOUT_EPISODES:-8}"
  ppo_epochs="${PPO_EPOCHS:-4}"
  eval_episodes="${EVAL_EPISODES:-50}"
  python code/train_torch_mappo.py \
    --warm-policy results/cem_local/best_policy.json \
    --seed "$seed" \
    --bc-episodes "$bc_episodes" \
    --bc-epochs "$bc_epochs" \
    --ppo-updates "$ppo_updates" \
    --rollout-episodes "$rollout_episodes" \
    --ppo-epochs "$ppo_epochs" \
    --eval-episodes "$eval_episodes" \
    --out "$out_dir" \
    2>&1 | tee "$out_dir/train.log"
done

python code/make_paper_artifacts.py
