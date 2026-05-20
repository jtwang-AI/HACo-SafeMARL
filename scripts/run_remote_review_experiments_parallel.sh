#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=code

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(20260429 20260430 20260431 20260432)
fi

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS_ARR[@]}}"
ROOT_OUT="${ROOT_OUT:-results/review_marl_baselines}"
BC_EPOCHS="${BC_EPOCHS:-100}"
PPO_UPDATES="${PPO_UPDATES:-100}"
BC_EPISODES="${BC_EPISODES:-12}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"

mkdir -p "$ROOT_OUT" logs

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while (( "$(running_jobs)" >= MAX_PARALLEL )); do
    sleep 20
  done
}

launch_job() {
  local gpu="$1"
  local variant="$2"
  local policy_name="$3"
  local seed="$4"
  shift 4
  local extra_args=("$@")
  local out_dir="${ROOT_OUT}/${variant}/seed_${seed}"
  local log_path="${out_dir}/train.log"
  mkdir -p "$out_dir"

  if [[ -s "${out_dir}/aggregate_metrics.csv" ]]; then
    echo "=== skip completed ${variant} seed ${seed} ==="
    return 0
  fi

  wait_for_slot
  echo "=== launch ${variant} seed ${seed} on GPU ${gpu} ==="
  if (( ${#extra_args[@]} > 0 )); then
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
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
        "${extra_args[@]}"
    ) > "$log_path" 2>&1 &
  else
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
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
        --out "$out_dir"
    ) > "$log_path" 2>&1 &
  fi
}

job_index=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPUS_ARR[$((job_index % ${#GPUS_ARR[@]}))]}"
  launch_job "$gpu" "mappo_acoustic_shield" "mappo_acoustic_shield" "$seed"
  job_index=$((job_index + 1))
done

for seed in "${SEEDS[@]}"; do
  gpu="${GPUS_ARR[$((job_index % ${#GPUS_ARR[@]}))]}"
  launch_job "$gpu" "mappo_no_acoustic_shield" "mappo_no_acoustic_shield" "$seed" --no-acoustic-features --no-acoustic-reward
  job_index=$((job_index + 1))
done

for seed in "${SEEDS[@]}"; do
  gpu="${GPUS_ARR[$((job_index % ${#GPUS_ARR[@]}))]}"
  launch_job "$gpu" "mappo_acoustic_no_shield" "mappo_acoustic_no_shield" "$seed" --disable-shield
  job_index=$((job_index + 1))
done

wait
python code/summarize_remote_review_experiments.py --root "$ROOT_OUT"
