# Experiment Manifest

## Current Completed Runs

- Pilot benchmark: `results/local_eval_20/`
  - Command: `python code/run_pilot_experiments.py --episodes 20 --out results/local_eval_20`
  - Output: 840 episode records.
- CEM warm-start search: `results/cem_local/`
  - Command: `python code/train_cem_warmstart.py --generations 8 --population 24 --episodes 3 --out results/cem_local`
  - Best communication-safety policy: `results/cem_local/best_policy.json`.
- Main final evaluation: `results/final_eval_local40/`
  - Command: `python code/evaluate_policy_params.py --policy results/cem_local/best_policy.json --episodes 40 --out results/final_eval_local40`
  - Output: 1680 episode records.
- Balanced preference evaluation: `results/final_eval_balanced40/`
  - Command: `python code/evaluate_policy_params.py --policy results/balanced_policy/best_policy.json --episodes 40 --out results/final_eval_balanced40`
  - Output: 1680 episode records.
- Task-priority preference evaluation: `results/task_priority_policy/`
  - Command: `PYTHONPATH=code python3 code/search_task_priority_policy.py --search-episodes 3 --eval-episodes 40 --grid-candidates 40 --random-candidates 60 --out results/task_priority_policy`
  - Output: 1680 episode records from remote server `happy`.

## Generated Artifacts Included in This Repository

- Emergency rescue animation: `figures/emergency_rescue_animation.svg`
- Framework diagram SVG: `figures/framework.svg`
- Framework preview image: `figures/framework_from_pptx_preview.png`
- Refined framework preview image: `figures/framework_standalone_refined.png`
- Review-driven MAPPO diagnostic summary: `results/review_marl_baselines_summary.json`

Regenerate experiment summaries:

```bash
PYTHONPATH=code python3 code/summarize_remote_review_experiments.py \
  --root results/review_marl_baselines
```

## GPU MAPPO Next Step

The PyTorch implementation is in:

- `code/haco/torch_policy.py`
- `code/train_torch_mappo.py`

Smoke test:

```bash
PYTHONPATH=code python code/train_torch_mappo.py \
  --bc-episodes 1 --bc-epochs 1 \
  --ppo-updates 1 --rollout-episodes 1 --ppo-epochs 1 \
  --eval-episodes 1 --eval-scenarios survey \
  --out results/mappo_smoke
```

Full GPU run:

```bash
./scripts/run_gpu_mappo.sh
```

Four-seed 100-epoch/update GPU run for the paper training curves:

```bash
./scripts/run_gpu_mappo_multiseed_100.sh
```

The training script writes incremental `results/torch_mappo/bc_history.json`,
`results/torch_mappo/ppo_history.json`, and final
`results/torch_mappo/training_history.json`. Model checkpoints and transient
logs are intentionally excluded from the public repository.

## Review-Driven Experiment Strengthening

The review on 2026-05-20 concluded that the paper should not be framed as a new
RL optimizer. The targeted remote experiment package therefore tests whether the
domain structure matters:

- MAPPO + acoustic features/reward + shield
- MAPPO without acoustic observations/reward + shield
- MAPPO + acoustic features/reward without shield

MADQN is intentionally excluded because the environment uses continuous USV/AUV
actions; discretizing actions would introduce an unfair and low-value baseline.

Run the focused remote package:

```bash
./scripts/run_remote_review_experiments.sh
```

The script writes:

- `results/review_marl_baselines/*/seed_*/`
- `results/review_marl_baselines_summary.json`

Use this diagnostic as evidence that the domain structure matters. The main
claim should remain acoustic-communication-aware safe USV-AUV planning, not
algorithmic superiority over all MARL variants.
