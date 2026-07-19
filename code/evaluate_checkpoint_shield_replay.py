from __future__ import annotations

import argparse
from pathlib import Path

import torch

from haco.experiment import aggregate, run_episode, scenario_config, write_csv, write_json
from haco.torch_policy import TorchHACoPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one trained actor with and without the executed-action shield."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-scenarios",
        nargs="*",
        default=["survey", "traffic", "acoustic_degradation", "emergency", "generalization"],
    )
    parser.add_argument("--generalization-auvs", nargs="*", type=int, default=[2, 6, 8])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    policies = [
        TorchHACoPolicy(
            args.checkpoint,
            device=args.device,
            deterministic=True,
            use_shield=True,
            name="mappo_same_actor_shielded",
        ),
        TorchHACoPolicy(
            args.checkpoint,
            device=args.device,
            deterministic=True,
            use_shield=False,
            name="mappo_same_actor_raw",
        ),
    ]
    rows = []
    for scenario in args.eval_scenarios:
        counts = args.generalization_auvs if scenario == "generalization" else [3]
        for num_auvs in counts:
            for episode in range(args.eval_episodes):
                seed = args.seed + 900000 + 10000 * episode + 100 * num_auvs
                cfg = scenario_config(scenario, seed=seed, num_auvs=num_auvs)
                for policy in policies:
                    rows.append(run_episode(policy, cfg))

    summary = aggregate(rows)
    write_csv(args.out / "episode_metrics.csv", rows)
    write_csv(args.out / "aggregate_metrics.csv", summary)
    write_json(
        args.out / "run_config.json",
        {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "eval_episodes": args.eval_episodes,
            "eval_scenarios": args.eval_scenarios,
            "generalization_auvs": args.generalization_auvs,
            "comparison": "same actor, shield enabled versus disabled at execution",
        },
    )
    print(f"wrote {len(rows)} paired replay episodes to {args.out}")


if __name__ == "__main__":
    main()
