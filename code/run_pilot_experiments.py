from __future__ import annotations

import argparse
import json
from pathlib import Path

from haco.experiment import aggregate, default_policies, run_episode, scenario_config, write_csv, write_latex_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    parser.add_argument("--scenarios", nargs="*", default=["survey", "traffic", "acoustic_degradation", "emergency", "generalization"])
    parser.add_argument("--generalization-auvs", nargs="*", type=int, default=[2, 6, 8])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in args.scenarios:
        auv_counts = args.generalization_auvs if scenario == "generalization" else [3]
        for num_auvs in auv_counts:
            for ep in range(args.episodes):
                seed = 1000 * (1 + len(rows)) + ep
                cfg = scenario_config(scenario, seed=seed, num_auvs=num_auvs)
                for policy in default_policies():
                    rows.append(run_episode(policy, cfg))

    agg = aggregate(rows)
    write_csv(args.out / "episode_metrics.csv", rows)
    write_csv(args.out / "aggregate_metrics.csv", agg)
    (args.out / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2))
    write_latex_table(Path("tables/pilot_main_results.tex"), agg)
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {args.out / 'aggregate_metrics.csv'}")
    print("latex table: tables/pilot_main_results.tex")


if __name__ == "__main__":
    main()
