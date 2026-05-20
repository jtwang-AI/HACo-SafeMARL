#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=code

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(20260429 20260430 20260431 20260432)
fi

ROOT_OUT="${ROOT_OUT:-results/review_marl_baselines}"
BC_EPOCHS="${BC_EPOCHS:-100}"
PPO_UPDATES="${PPO_UPDATES:-100}"
BC_EPISODES="${BC_EPISODES:-12}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"

mkdir -p "$ROOT_OUT"

run_variant() {
  local variant="$1"
  local policy_name="$2"
  shift 2
  local extra_args=()
  if (( $# > 0 )); then
    extra_args=("$@")
  fi

  for seed in "${SEEDS[@]}"; do
    local out_dir="${ROOT_OUT}/${variant}/seed_${seed}"
    mkdir -p "$out_dir"
    echo "=== ${variant} seed ${seed} ==="
    if (( ${#extra_args[@]} > 0 )); then
      python code/train_torch_mappo.py \
        --warm-policy results/cem_local/best_policy.json \
        --seed "$seed" \
        --policy-name "$policy_name" \
        --bc-episodes "$BC_EPISODES" \
        --bc-epochs "$BC_EPOCHS" \
        --ppo-updates "$PPO_UPDATES" \
        --rollout-episodes "$ROLLOUT_EPISODES" \
        --ppo-epochs "$PPO_EPOCHS" \
        --eval-episodes "$EVAL_EPISODES" \
        --out "$out_dir" \
        "${extra_args[@]}" \
        2>&1 | tee "$out_dir/train.log"
    else
      python code/train_torch_mappo.py \
        --warm-policy results/cem_local/best_policy.json \
        --seed "$seed" \
        --policy-name "$policy_name" \
        --bc-episodes "$BC_EPISODES" \
        --bc-epochs "$BC_EPOCHS" \
        --ppo-updates "$PPO_UPDATES" \
        --rollout-episodes "$ROLLOUT_EPISODES" \
        --ppo-epochs "$PPO_EPOCHS" \
        --eval-episodes "$EVAL_EPISODES" \
        --out "$out_dir" \
        2>&1 | tee "$out_dir/train.log"
    fi
  done
}

# Minimal review-driven package:
# 1. MAPPO with acoustic features/reward and shield.
# 2. MAPPO without acoustic observations/reward, still shielded.
# 3. MAPPO with acoustic observations/reward but without shield.
# MADQN is intentionally excluded because the environment uses continuous actions.
run_variant "mappo_acoustic_shield" "mappo_acoustic_shield"
run_variant "mappo_no_acoustic_shield" "mappo_no_acoustic_shield" --no-acoustic-features --no-acoustic-reward
run_variant "mappo_acoustic_no_shield" "mappo_acoustic_no_shield" --disable-shield

python code/summarize_remote_review_experiments.py --root "$ROOT_OUT"
