from __future__ import annotations

import argparse
import json
from pathlib import Path

from haco.experiment import aggregate, default_policies, run_episode, scenario_config, write_csv, write_json, write_latex_table
from haco.policies import HACoSafeHeuristicPolicy


def load_trained_policy(path: Path):
    payload = json.loads(path.read_text())
    params = payload.get("params", payload.get("best_params", {}))
    return HACoSafeHeuristicPolicy(params=params, name="haco_safemarl_trained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=Path("results/cem/best_policy.json"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/final_eval"))
    parser.add_argument("--scenarios", nargs="*", default=["survey", "traffic", "acoustic_degradation", "emergency", "generalization"])
    parser.add_argument("--generalization-auvs", nargs="*", type=int, default=[2, 6, 8])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trained = load_trained_policy(args.policy)
    policies = [p for p in default_policies() if p.name != "haco_safemarl_pilot"] + [trained]
    rows = []
    for scenario in args.scenarios:
        auv_counts = args.generalization_auvs if scenario == "generalization" else [3]
        for num_auvs in auv_counts:
            for ep in range(args.episodes):
                seed = 777000 + 10000 * ep + 100 * num_auvs + len(rows)
                cfg = scenario_config(scenario, seed=seed, num_auvs=num_auvs)
                for policy in policies:
                    rows.append(run_episode(policy, cfg))

    agg = aggregate(rows)
    write_csv(args.out / "episode_metrics.csv", rows)
    write_csv(args.out / "aggregate_metrics.csv", agg)
    write_json(args.out / "aggregate_metrics.json", agg)
    write_latex_table(Path("tables/final_main_results.tex"), agg)
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {args.out / 'aggregate_metrics.csv'}")
    print("latex table: tables/final_main_results.tex")


if __name__ == "__main__":
    main()
