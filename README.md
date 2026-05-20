# HACo-SafeMARL

HACo-SafeMARL is a reproducible research workspace for heterogeneous acoustic-communication safe multi-agent reinforcement learning for cooperative surface-underwater robot path planning.

The project studies joint unmanned surface vehicle (USV) and autonomous underwater vehicle (AUV) planning under underwater acoustic communication loss, surface traffic, underwater obstacles, emergency events, and safety constraints.

## Repository Layout

- `code/`: simulator, policies, training, evaluation, and artifact-generation scripts.
- `code/haco/`: core environment, controllers, experiment utilities, and PyTorch policy implementation.
- `scripts/`: GPU and remote experiment launch scripts.
- `results/`: aggregate metrics, raw episode tables, training histories, and run configs.
- `figures/`: generated non-LaTeX visual artifacts, including SVG and PNG outputs.

## Environment

Install the minimal Python dependencies:

```bash
pip install -r requirements.txt
```

For neural MAPPO experiments, use a CUDA-enabled PyTorch installation matching your GPU driver.

## Reproduce Main Results

Run the warm-start search:

```bash
PYTHONPATH=code python code/train_cem_warmstart.py \
  --generations 8 --population 24 --episodes 3 \
  --out results/cem_local
```

Run the main held-out evaluation:

```bash
PYTHONPATH=code python code/evaluate_policy_params.py \
  --policy results/cem_local/best_policy.json \
  --episodes 40 \
  --out results/final_eval_local40
```

Run the focused review-driven MAPPO diagnostic ablations:

```bash
./scripts/run_remote_review_experiments_parallel.sh
```

This writes:

- `results/review_marl_baselines/*/seed_*/`
- `results/review_marl_baselines_summary.json`

## Regenerate Experiment Summaries

```bash
PYTHONPATH=code python3 code/summarize_remote_review_experiments.py \
  --root results/review_marl_baselines
```

The repository intentionally excludes manuscript PDF and LaTeX sources. It is a code, data, generated-figure, and experiment-result release.

## Notes on Claims

The paper should be read as an engineering AI framework for acoustic-communication-aware safe USV-AUV planning. MAPPO is used as a reproducible centralized-training/decentralized-execution optimizer and diagnostic carrier; the main contribution is the coupling of acoustic physics, heterogeneous graph policy learning, constrained optimization, and explicit safety shields.

## License

Add the final license before public release if required by the target venue or institution.
